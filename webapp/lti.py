"""Compile-time construction of vector-fitted LTI channel components.

Each catalog entry tagged ``"lti": <kind>`` is turned into a per-instance
state-space model here (sampled target response -> vf.vector_fit ->
realisation payload for catalog._lti_vt/_lti_field).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import vf

UPLOADS = Path(__file__).resolve().parent / "uploads"
_C0 = 299792458.0


def _key(kind: str, settings: dict) -> str:
    blob = json.dumps(settings, sort_keys=True, default=str)
    return f"lti:{kind}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def parse_touchstone(text: str):
    """Minimal .s2p reader -> (f_hz, S[n,2,2]). Handles # HZ..GHZ, RI/MA/DB."""
    funit, fmt = 1e9, "MA"
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].upper().split()
            for t in toks:
                if t in ("HZ", "KHZ", "MHZ", "GHZ"):
                    funit = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6,
                             "GHZ": 1e9}[t]
                if t in ("RI", "MA", "DB"):
                    fmt = t
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) == 9:
            rows.append(vals)
        elif rows and len(rows[-1]) < 9:
            rows[-1].extend(vals)          # wrapped continuation line
        else:
            rows.append(vals)
    data = np.asarray([r for r in rows if len(r) == 9], float)
    if data.size == 0:
        raise ValueError("no S-parameter rows found in the Touchstone file")
    f = data[:, 0] * funit
    pairs = data[:, 1:].reshape(-1, 4, 2)

    if fmt == "RI":
        S4 = pairs[:, :, 0] + 1j * pairs[:, :, 1]
    else:
        mag = pairs[:, :, 0] if fmt == "MA" else 10.0 ** (pairs[:, :, 0] / 20)
        S4 = mag * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    # touchstone column order: S11 S21 S12 S22
    S = np.empty((len(f), 2, 2), complex)
    S[:, 0, 0], S[:, 1, 0] = S4[:, 0], S4[:, 1]
    S[:, 0, 1], S[:, 1, 1] = S4[:, 2], S4[:, 3]
    return f, S


def build(kind: str, settings: dict, log: list) -> tuple[str, dict]:
    """-> (model_key, payload). payload: {"real": (A,B,C,D,zin)} or
    {"cplx": (poles, res, d)}."""
    key = _key(kind, settings)

    if kind == "chan":
        loss = float(settings.get("loss_db", 10.0))
        f_nyq = float(settings.get("f_nyq", 14e9))
        n_poles = int(settings.get("n_poles", 10))
        f = np.linspace(f_nyq / 1000, 3.5 * f_nyq, 400)
        mag = 10.0 ** (-loss * np.sqrt(f / f_nyq) / 20.0)
        H = vf.min_phase(f, mag)
        poles, res, d, err = vf.vector_fit(f, H, n_poles=n_poles)
        A, B, C, D = vf.realify(poles, res, d)
        log.append(f"channel fit: {n_poles} poles, rel err {err:.2e} "
                   f"({loss:g} dB at {f_nyq / 1e9:g} GHz)")
        return key, {"real": (A, B, C, D, 0.0)}

    if kind == "s2p":
        fid = str(settings.get("file", "")).strip()
        safe = "".join(c for c in fid if c.isalnum() or c in "._-")
        if not safe:
            raise ValueError("s2p_channel: upload a Touchstone file first "
                             "(inspector Upload button)")
        path = UPLOADS / safe
        if not path.exists():
            raise ValueError(f"s2p_channel: uploaded file {safe!r} not found "
                             "on the server — re-upload it")
        z0 = float(settings.get("z0", 50.0))
        n_poles = int(settings.get("n_poles", 12))
        f, S = parse_touchstone(path.read_text())
        H = S[:, 1, 0] / 2.0          # matched-termination insertion transfer
        sel = f > 0
        poles, res, d, err = vf.vector_fit(f[sel], H[sel], n_poles=n_poles)
        A, B, C, D = vf.realify(poles, res, d)
        log.append(f"s2p fit: {safe}, {sel.sum()} points to "
                   f"{f.max() / 1e9:.1f} GHz, {n_poles} poles, "
                   f"rel err {err:.2e}")
        return key, {"real": (A, B, C, D, z0)}

    if kind == "fiber":
        L = float(settings.get("length_km", 10.0)) * 1e3
        lam = float(settings.get("lambda_nm", 1550.0)) * 1e-9
        att = float(settings.get("atten_db_km", 0.2))
        bw = float(settings.get("fit_bw", 60e9))
        n_poles = int(settings.get("n_poles", 28))
        lam0_nm = float(settings.get("lambda0_nm", 0.0))
        S_ps = float(settings.get("S_ps", 0.0))       # slope, ps/nm^2/km
        if lam0_nm > 0.0:
            # O-band-correct G.652 profile around the zero-dispersion
            # wavelength: D(l) = S0/4*(l - l0^4/l^3), S(l) = dD/dl.
            # S_ps is read as S0 (the slope AT l0); D_ps is ignored.
            S0 = S_ps if S_ps > 0.0 else 0.092
            lnm, l0 = lam * 1e9, lam0_nm
            D_ps = S0 / 4.0 * (lnm - l0 ** 4 / lnm ** 3)
            S_ps = S0 / 4.0 * (1.0 + 3.0 * l0 ** 4 / lnm ** 4)
        else:
            D_ps = float(settings.get("D_ps", 17.0))
        D_si = D_ps * 1e-6                      # ps/nm/km  -> s/m^2
        S_si = S_ps * 1e3                       # ps/nm^2/km -> s/m^3
        beta2 = -D_si * lam ** 2 / (2.0 * np.pi * _C0)
        beta3 = lam ** 3 * (S_si * lam + 2.0 * D_si) / (2.0 * np.pi * _C0) ** 2
        b2L, b3L = beta2 * L, beta3 * L
        w_max = 2.0 * np.pi * bw
        # causal transit latency covering the worst group delay (beta2 term
        # plus the cubic beta3 term that dominates near lambda0)
        Td = 1.5 * (abs(b2L) * w_max + 0.5 * abs(b3L) * w_max ** 2)
        amp = 10.0 ** (-att * L / 1e3 / 20.0)

        def theta(w):
            return 0.5 * b2L * w ** 2 + b3L * w ** 3 / 6.0 + Td * w

        def group_delay(w):
            return b2L * w + 0.5 * b3L * w ** 2 + Td

        # Fit over +-bw plus a C1 out-of-band extension (flat |H|, group
        # delay frozen at the band edge). Without it a nearly-flat target
        # (fiber at the zero-dispersion wavelength) lets the LS pick a
        # degenerate d >> amp with huge cancelling residues — perfect in
        # band but wildly amplifying out of band.
        f_core = np.linspace(-bw, bw, 501)
        f_ext = np.linspace(bw, 1.5 * bw, 76)[1:]
        w_edge = 2.0 * np.pi * bw
        w_ep = 2.0 * np.pi * f_ext
        H_core = amp * np.exp(-1j * theta(2.0 * np.pi * f_core))
        H_hi = amp * np.exp(-1j * (theta(w_edge)
                                   + group_delay(w_edge) * (w_ep - w_edge)))
        H_lo = amp * np.exp(-1j * (theta(-w_edge)
                                   - group_delay(-w_edge) * (w_ep - w_edge)))
        f = np.concatenate([-f_ext[::-1], f_core, f_ext])
        H = np.concatenate([H_lo[::-1], H_core, H_hi])
        wgt = np.concatenate([0.3 * np.ones(f_ext.size),
                              np.ones(f_core.size),
                              0.3 * np.ones(f_ext.size)])
        poles, res, d, _ = vf.vector_fit(f, H, n_poles=n_poles, n_iter=30,
                                         complex_pairs=False, weight=wgt)
        H_fit = vf.eval_fit(poles, res, d, f_core)
        err = (np.linalg.norm(H_fit - H_core) / np.linalg.norm(H_core))
        log.append(f"fiber CD fit: D = {D_ps:.3f} ps/nm/km, "
                   f"beta2L = {b2L * 1e24:.1f} ps^2, "
                   f"beta3L = {b3L * 1e36:.2f} ps^3, latency "
                   f"{Td * 1e12:.0f} ps, {n_poles} poles, rel err {err:.2e}")
        if err > 0.05:
            log.append("fiber CD fit: WARNING error > 5% — raise n_poles or "
                       "lower fit_bw/length")
        return key, {"cplx": (poles, res, d)}

    if kind == "filter":
        # Real optical bandpass ON the coherent field: a Butterworth response
        # whose passband is the filter's optical transfer mapped to the
        # baseband envelope around the laser carrier. It therefore acts on the
        # modulation sidebands (narrow bandwidth -> cuts sidebands -> closes
        # the eye), not as a static per-wavelength scalar.
        center = float(settings.get("center_nm", 1310.0)) * 1e-9
        carrier = float(settings.get("lambda_nm", 1310.0)) * 1e-9
        bw_nm = max(float(settings.get("bandwidth_nm", 0.6)), 1e-6)
        order = int(round(float(settings.get("order", 3.0))))
        order = max(1, min(order, 8))
        il = float(settings.get("il_db", 0.0))

        f_center = _C0 / center
        f_carrier = _C0 / carrier
        # |df/dlam| = c/lam^2 -> bandwidth (FWHM) in Hz, and the passband's
        # offset from the carrier in baseband. Coherent envelopes use the
        # physics e^{-i w t} convention, so an optical tone at f_center sits at
        # baseband +(f_carrier - f_center); centering the passband there makes
        # center_nm select the true wavelength (matches the WDM laser offset).
        df_fwhm = _C0 / center ** 2 * (bw_nm * 1e-9)
        w_off = 2.0 * np.pi * (f_carrier - f_center)
        wc = 2.0 * np.pi * (df_fwhm / 2.0)             # -3 dB half-width
        # Butterworth low-pass prototype poles (radius wc, order n)
        k = np.arange(order)
        p_lp = wc * np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order))
        denom = [np.prod(p_lp[i] - np.delete(p_lp, i)) for i in range(order)]
        # drop = Butterworth lowpass H_d(s) = wc^n / B(s)
        res_d = np.array([wc ** order / denom[i] for i in range(order)],
                         complex)
        # thru = the SAME-POLE Butterworth highpass H_t(s) = s^n / B(s):
        # |H_d|^2 + |H_t|^2 = 1 exactly on the j-omega axis, so the pair is a
        # unitary (power-complementary) add-drop like a real lossless filter.
        # The naive thru 1 - H_d is NOT passive: its skirt peaks at +4 dB
        # because H_d's phase rotates through the band edge.
        # H_t = 1 + sum_k t_k/(s - p_k) with t_k = p_k^n / prod(p_k - p_j);
        # after the +j*w_off frequency shift the residues are unchanged
        # (numerator (s - j*w_off)^n evaluates to p_lp^n at the poles).
        res_t = np.array([p_lp[i] ** order / denom[i] for i in range(order)],
                         complex)
        a = 10.0 ** (-il / 20.0)
        poles = p_lp + 1j * w_off                      # shift to bandpass
        res_d = res_d * a                              # il on the drop path
        log.append(
            f"optical add-drop filter: Butterworth order {order}, FWHM "
            f"{df_fwhm / 1e9:.3g} GHz ({bw_nm * 1000:.3g} pm), detuning "
            f"{(f_center - f_carrier) / 1e9:.3g} GHz from carrier")
        return key, {"cplx_drop": (poles, res_d, res_t)}

    raise ValueError(f"unknown lti kind {kind!r}")
