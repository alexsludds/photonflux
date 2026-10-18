"""IBIS-AMI blocks in the web app: Tx waveform shaping and the Rx report.

Two schematic blocks wrap ``photonflux.ibisami`` models:

* **IBIS-AMI Tx** (``ami_tx``) is a pattern source whose baked waveform
  goes through the Tx model's ``AMI_GetWave`` at compile time
  (:func:`tx_waveform`): the PRBS / PAM4 shape (0..1, jitter included) is
  resampled to 32 samples/UI, mapped to the IBIS +/-0.5 V stimulus, run
  through the model and mapped back, so the source's v0/v1 levels still
  apply at run time and a level change does not recompile.
* **IBIS-AMI Rx** (``ami_rx``) is electrically a unity buffer (like the Rx
  FFE/DFE blocks); the link report runs the model on the "BER vs" probe
  (:func:`ami_report`):

  1. the channel the Rx's ``AMI_Init`` sees is estimated from the probe and
     the known symbols (``serdes.estimate_channel``: data-aided least
     squares -- for a nonlinear optical link, the best linear fit; the
     residual is reported);
  2. statistical flow: Rx ``AMI_Init`` on that channel -> statistical eye,
     BER, eye height/width, bathtubs;
  3. time-domain flow: Rx ``AMI_GetWave`` on the probe waveform itself (plus
     the block's input noise) -> recovered clock, counted BER, adaptation
     trace (every ``AMI_parameters_out``).
"""
from __future__ import annotations

import numpy as np

NSPUI = 32


def parse_overrides(text: str | None) -> dict:
    """``"ctle_mode=1 dfe_ntaps=8, Modulation=PAM4"`` -> dict (numbers
    parsed where they parse)."""
    out: dict = {}
    for tok in str(text or "").replace(",", " ").replace(";", " ").split():
        if "=" not in tok:
            raise ValueError(f"AMI parameter override {tok!r} is not name=value")
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            out[k] = v
    return out


def model_names(kind: str) -> list[str]:
    """Library models of one kind already built (for the catalog enum)."""
    try:
        from photonflux import ibisami

        names = [e.name for e in ibisami.library() if e.kind == kind
                 and (e.runs_here or e.runs_in_container)]
    except Exception:  # noqa: BLE001 - the catalog must import without PyAMI
        names = []
    dflt = "pf_tx" if kind == "tx" else "pf_rx"
    return [dflt] + [n for n in names if n != dflt]


def _model(name: str, kind: str, overrides: dict):
    """A fresh instance per run (sweep points may run concurrently, and a
    model instance is stateful); the caller closes it."""
    from photonflux import ibisami

    m = ibisami.load(name)
    if m.kind != kind:
        m.close()
        raise ValueError(f"{name} is an IBIS-AMI {m.kind.upper()} model; this "
                         f"block needs a {kind.upper()} model")
    m.params.update(overrides)
    return m


def _modulation(settings: dict) -> str:
    return "PAM4" if str(settings.get("mode", "nrz")) == "pam4" else "NRZ"


def tx_waveform(settings: dict, wt: np.ndarray, wv: np.ndarray,
                log: list) -> tuple[np.ndarray, np.ndarray]:
    """Unit-level pattern shape (0..1) -> the same shape after the Tx model."""
    from photonflux import serdes

    ui = float(settings.get("ui", 100e-12))
    ts = ui / NSPUI
    name = str(settings.get("model") or "pf_tx")
    with _model(name, "tx", parse_overrides(settings.get("ami_params"))) as tx:
        if "Modulation" in tx.params:
            tx.params["Modulation"] = _modulation(settings)
        t = np.arange(0.0, float(wt[-1]), ts)
        x = np.interp(t, wt, wv) - 0.5
        r = tx.init(serdes.Channel.ideal(ts, n=64 * NSPUI).impulse, ts, ui)
        if tx.getwave_exists:
            y = tx.getwave(x, block_bits=512).wave
        else:   # Init-only Tx: filter by its impulse response
            y = np.convolve(x, r.impulse[0])[: len(x)] * ts
        log.append(f"IBIS-AMI Tx {name} ({tx.where}): {r.message.strip()[:160]}")
    return t, y + 0.5


def _decimate(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[np.linspace(0, len(x) - 1, n).astype(int)]


def _finite(v):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(v, float)]


def ami_report(result: dict, meta: dict, cfg: dict, log: list) -> dict | None:
    """Run the placed IBIS-AMI Rx block on the link report's probe."""
    import wavesrc

    blocks = meta.get("ami_rx") or {}
    if not blocks:
        return None
    bname, blk = next(iter(blocks.items()))
    if len(blocks) > 1:
        log.append(f"IBIS-AMI: {len(blocks)} Rx blocks placed, using {bname}")
    patterns = meta.get("patterns") or {}
    pat_inst = cfg.get("pattern") or next(iter(patterns), None)
    pattern = patterns.get(pat_inst) if pat_inst else None
    probe = cfg.get("probe")
    tr = next((t for t in result.get("traces", []) if t["name"] == probe), None)
    if pattern is None or tr is None:
        log.append("IBIS-AMI report: needs a pattern source and the 'BER vs' "
                   "probe at the Rx block's input")
        return None
    if pattern.get("sequence", "prbs") not in ("prbs", "prbs+oma"):
        log.append("IBIS-AMI report: needs PRBS data (sequence prbs)")
        return None
    ui = float(pattern.get("ui", 100e-12))
    ts = ui / NSPUI
    mod = _modulation(pattern)
    t = np.asarray(result["x"], float)
    tu = np.arange(0.0, t[-1], ts)
    y = np.interp(tu, t, np.asarray(tr["values"], float))
    nsym = len(y) // NSPUI
    if nsym < 400:
        log.append(f"IBIS-AMI report: {nsym} UI is too short; run >= 400 UI")
        return None
    a = 2.0 * wavesrc._symbols(pattern, nsym + 8)[:nsym] - 1.0
    y = y - np.mean(y)
    with _model(str(blk["model"] or "pf_rx"), "rx",
                parse_overrides(blk["params"])) as rx:
        return _rx_report(rx, blk, bname, probe, mod, ui, ts, y, a, nsym, log)


def _rx_report(rx, blk, bname, probe, mod, ui, ts, y, a, nsym, log) -> dict:
    from photonflux import serdes

    ch, resid, res_w = serdes.estimate_channel(
        y, ts, ui, a, span_ui=min(64, nsym // 8), residual=True)
    sig_rms = float(np.std(y))
    noise = float(blk.get("noise_rms", 0.0))
    dens = noise * np.sqrt(2 * ts)
    # what the linear fit misses (square-law detection after dispersion,
    # modulator compression, ...) enters the statistical flow as noise,
    # shaped by the Rx's own linear front end (a Bussgang-style split)
    skip = min(rx.ignore_bits, nsym // 3)
    rx_in = y + (noise * np.random.default_rng(7).standard_normal(len(y))
                 if noise > 0 else 0.0)
    td = serdes.receive(rx, rx_in, a, ui, ts, ch.impulse, modulation=mod,
                        skip=skip)
    # ... measured where it matters: at the recovered clock's sampling phase
    g = serdes.rx_linear_response(rx, ch.impulse, ts, ui)
    dist = np.convolve(res_w, g)[: len(res_w)]
    ph = int(round(td.sampling_phase_ui * NSPUI)) % NSPUI
    at = dist[ph::NSPUI][skip:]
    at = at[np.abs(res_w[ph::NSPUI][skip:]) > 0]
    sig_dist = float(np.std(at)) if len(at) > 64 else 0.0
    lti = serdes.statistical_link(None, ch, rx, ui, modulation=mod,
                                  noise_density=dens, nv=301)
    st = serdes.statistical_link(None, ch, rx, ui, modulation=mod,
                                 noise_rms=sig_dist, noise_density=dens, nv=301)
    eye = st.eye
    summ = eye.summary(targets=(1e-6, 1e-12))
    # adaptation trace: every numeric scalar / vector the model reports
    tr_t = (td.rx_trace.trace_t / ui).tolist() if td.rx_trace else []
    series = {}
    if td.rx_trace and td.rx_trace.trace:
        keys = [k for k, v in td.rx_trace.trace[-1].items()
                if isinstance(v, (int, float, list)) and k != "symbols"]
        for k in keys:
            v0 = td.rx_trace.trace[-1][k]
            n = len(v0) if isinstance(v0, list) else 0
            if n:
                for i in range(min(n, 16)):
                    series[f"{k}[{i}]"] = _finite(td.adaptation(k, i)[1])
            else:
                series[k] = _finite(td.adaptation(k)[1])
    comb = np.log10(np.clip(eye.combined(), 1e-30, 0.5))
    ip = np.linspace(0, comb.shape[1] - 1, min(96, comb.shape[1])).astype(int)
    iv = np.linspace(0, comb.shape[0] - 1, min(80, comb.shape[0])).astype(int)
    cdr_at = eye.relative_phase(td.sampling_phase_ui) if td.rx_trace else 0.0
    tubs = []
    for e in range(eye.n_eyes):
        x, b = eye.bathtub_t(e)
        tubs.append({"phase_ui": _finite(x), "log10_ber": _finite(np.log10(b))})
    c = td.counted
    log.append(f"IBIS-AMI Rx {rx.name} ({rx.where}) on {probe}: counted BER "
               f"{c['ber']:.3g} ({c['bit_errors']}/{c['bits']} after "
               f"{td.skip} ignored), statistical BER {eye.bit_error_rate():.3g} "
               f"({lti.eye.bit_error_rate():.3g} treating the link as LTI; "
               f"linear-fit residual {100 * resid / max(sig_rms, 1e-30):.1f} %)")
    return {
        "block": bname, "model": rx.name, "where": rx.where, "probe": probe,
        "modulation": mod, "ui": ui, "init_message": st.rx_message.strip(),
        "init_params": {k: v for k, v in st.rx_params.items()
                        if isinstance(v, (int, float, list, str))},
        "final_params": (td.rx_trace.trace[-1] if td.rx_trace else {}),
        "linear_residual": resid / max(sig_rms, 1e-30),
        "distortion_at_slicer": sig_dist,
        "noise_rms": noise, "noise_at_slicer": st.noise_rms,
        "stat": {"ber": eye.bit_error_rate(), "ser": eye.ser(),
                 "ber_lti": lti.eye.bit_error_rate(),
                 "eyes": summ["eyes"],
                 "map": {"phase_ui": _finite(eye.phase_ui[ip]),
                         "v": _finite(eye.v[iv]),
                         "log10_ber": [_finite(r) for r in comb[np.ix_(iv, ip)]]},
                 "bathtubs": tubs},
        "td": {"counted": c, "qfit_ber": td.qfit.get("ber_est") if td.qfit.get("ok") else None,
               "skip": td.skip, "lag": td.lag,
               "sampling_phase_ui": cdr_at,
               "bathtub": {"phase_ui": _finite(td.bathtub["offset_ui"] + cdr_at),
                           "log10_ber": _finite(np.log10(td.bathtub["ber"]))},
               "trace_t_ui": tr_t, "series": series},
    }
