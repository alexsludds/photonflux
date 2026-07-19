"""APD photodetector (``webapp/catalog.py`` ``apd`` / ``_apd_noisy``): avalanche
gain, McIntyre excess noise, and the gain-bandwidth tradeoff.

The APD is a JAX bridge component (not a ``models/*.va`` device), so — unlike
the physics pins in ``test_edfa.py`` etc. — these drive the full webapp
simulation engine (``simulate.run``) and compare against the closed-form
receiver equations:

* DC: ``I = M(R P + Idk_bulk) + Idk_surf`` — the primary photocurrent and the
  bulk dark current are multiplied, the surface dark current is not.
* transient shot noise: multi-seed variance = ``2 q I_prim M^2 F(M) BW`` with
  the McIntyre excess-noise factor ``F(M) = k M + (2 - 1/M)(1 - k)`` — the same
  noise-bank machinery the PIN photodiode uses (``_photodiode_noisy``).
* sensitivity vs M: the thermal-limited -> excess-noise-limited crossover gives
  an interior noise-optimal ``M*``.

The APD current is read as a voltage by loading ``cat`` with a resistor to
ground (``V = -I_apd R_load``); ``Cj = f3db = gbp = 0`` so the load is purely
resistive and the transient noise rides the output current undistorted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

Q_E = 1.602176634e-19


def _receiver_sch(*, R=0.8, M=10.0, k_ion=0.3, Idk_bulk=0.0, Idk_surf=0.0,
                  power=1e-3, r_load=1000.0):
    """CW laser -> APD -> load resistor to ground; probe the APD output node.

    With Cj/f3db/gbp = 0 the load is resistive, so V(vout) = -I_apd * r_load.
    """
    return {
        "instances": {
            "LAS1": {"type": "cw_laser",
                     "settings": {"wavelength_nm": 1310.0, "power": power}},
            "APD1": {"type": "apd",
                     "settings": {"R": R, "M": M, "k_ion": k_ion,
                                  "Idk_bulk": Idk_bulk, "Idk_surf": Idk_surf,
                                  "Cj": 0.0, "f3db": 0.0, "gbp": 0.0}},
            "RL": {"type": "resistor", "settings": {"R": r_load}},
            "G1": {"type": "ground", "settings": {}},
            "G2": {"type": "ground", "settings": {}},
        },
        "wires": [
            {"from": "LAS1,p1", "to": "APD1,po_p"},
            {"from": "LAS1,p2", "to": "G1,p1"},
            {"from": "APD1,po_n", "to": "G2,p1"},
            {"from": "APD1,an", "to": "G2,p1"},
            {"from": "APD1,cat", "to": "RL,p1"},
            {"from": "RL,p2", "to": "G2,p1"},
        ],
        "probes": [{"name": "vout", "at": "APD1,cat"}],
    }


def _excess_noise(M: float, k: float) -> float:
    """McIntyre excess-noise factor F(M) = k M + (2 - 1/M)(1 - k)."""
    return k * M + (2.0 - 1.0 / M) * (1.0 - k)


@pytest.fixture(scope="module")
def sim():
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    return simulate


def _dc_vout(sim, sch) -> float:
    res = sim.run({"schematic": sch, "analysis": {"mode": "dc"}})
    assert res["ok"], res.get("log")
    row = next(r for r in res["rows"] if r["name"] == "vout")
    return float(row["value"])


# ---------------------------------------------------------------------------
# DC: avalanche gain and the multiplied/unmultiplied dark-current split
# ---------------------------------------------------------------------------
def test_apd_dc_photocurrent(sim):
    """Output photocurrent = R P M at DC (acceptance criterion 1)."""
    R, P, M, r_load = 0.8, 1e-3, 12.0, 1000.0
    v = _dc_vout(sim, _receiver_sch(R=R, M=M, power=P, r_load=r_load))
    assert abs(v) == pytest.approx(R * P * M * r_load, rel=1e-6)


def test_apd_dc_dark_current_split(sim):
    """Bulk dark current is multiplied by M, surface dark current is not:
    I = M * Idk_bulk + Idk_surf when dark (P = 0)."""
    M, bulk, surf, r_load = 8.0, 3e-9, 5e-9, 1e6
    v = _dc_vout(sim, _receiver_sch(M=M, power=0.0, Idk_bulk=bulk,
                                    Idk_surf=surf, r_load=r_load))
    assert abs(v) == pytest.approx((M * bulk + surf) * r_load, rel=1e-6)


# ---------------------------------------------------------------------------
# transient shot noise: variance = 2 q I_prim M^2 F(M) BW
# ---------------------------------------------------------------------------
def test_apd_shot_noise_variance(sim):
    """Multi-seed transient output-current variance matches the analytic
    multiplied shot noise 2 q I M^2 F(M) BW (acceptance criterion 2), and is
    distinguishable from the no-excess-noise (F = 1) case."""
    R, P, M, k, r_load, bw = 0.8, 1e-3, 12.0, 0.3, 1000.0, 50e9
    res = sim.run({
        "schematic": _receiver_sch(R=R, M=M, k_ion=k, power=P, r_load=r_load),
        "analysis": {"mode": "transient", "t_stop": 4e-9, "points": 1200,
                     "noise": {"seeds": 12, "bw": bw, "seed": 1}},
    })
    assert res["ok"], res.get("log")
    traces = [t for t in res["traces"] if t.get("probe") == "vout"]
    assert len(traces) == 12, "expected one trace per noise seed"
    V = np.array([t["values"] for t in traces])          # (seeds, points)
    var = float(np.var(V[:, 1:]))                         # pool seeds x time

    F = _excess_noise(M, k)
    i_prim = R * P
    analytic = 2.0 * Q_E * i_prim * M * M * F * bw * r_load ** 2
    assert var == pytest.approx(analytic, rel=0.12)
    # the excess-noise factor is really wired: F(M) ~ 4.9 here, so the measured
    # variance must sit far above the no-excess-noise (F = 1) shot floor
    no_excess = 2.0 * Q_E * i_prim * M * M * 1.0 * bw * r_load ** 2
    assert var > 3.0 * no_excess


# ---------------------------------------------------------------------------
# sensitivity vs M: the thermal -> excess-noise crossover gives an optimum M*
# ---------------------------------------------------------------------------
def test_apd_sensitivity_optimum():
    """A receiver whose noise is thermal (M-independent) + APD excess shot
    noise has an interior optimum gain M*: SNR rises while thermal-limited,
    then falls once the M^2 F(M) shot noise dominates (acceptance criterion 3).
    """
    R, P, k = 0.8, 1e-5, 0.3            # low power -> thermal-competitive
    bw = 20e9
    i_thermal2 = (15e-12) ** 2 * bw     # input-referred thermal noise power
    i_prim = R * P

    M = np.linspace(1.0, 100.0, 1000)
    F = _excess_noise(M, k)
    signal2 = (M * i_prim) ** 2
    noise2 = 2.0 * Q_E * i_prim * M ** 2 * F * bw + i_thermal2
    snr = signal2 / noise2

    m_star = M[int(np.argmax(snr))]
    assert 1.0 < m_star < 100.0, "expected an interior noise-optimal M*"
    # SNR at the optimum beats both the unity-gain (thermal-limited) and the
    # high-gain (excess-noise-limited) ends
    assert snr.max() > snr[0] * 1.5
    assert snr.max() > snr[-1] * 1.5
