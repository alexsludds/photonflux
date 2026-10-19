"""pf_tx PAM4 level pre-distortion (tx_pam4_lo / tx_pam4_hi)."""
import numpy as np
import pytest

pytest.importorskip("pyibisami")

UI = 1 / 53.125e9
NSPUI = 8


def _levels(**params):
    from photonflux import ibisami, serdes

    a = np.repeat([-1.0, -1 / 3, 1 / 3, 1.0], 8)
    x = serdes.stimulus(a, NSPUI)
    with ibisami.load("pf_tx") as tx:
        tx.params.update(params)
        tx.init(serdes.Channel.ideal(UI / NSPUI).impulse, UI / NSPUI, UI)
        y = tx.getwave(x).wave
    # mid-symbol samples of each 8-UI run, after the FIR delay (2 UI)
    return np.array([y[(8 * k + 6) * NSPUI] for k in range(4)])


def test_default_levels_are_unchanged():
    assert np.allclose(_levels(), [-0.5, -1 / 6, 1 / 6, 0.5], atol=1e-9)


def test_inner_levels_move_outer_levels_stay():
    lv = _levels(tx_pam4_lo=-0.25, tx_pam4_hi=0.4)
    assert np.allclose(lv, [-0.5, -0.125, 0.2, 0.5], atol=1e-9)


def test_out_of_range_level_is_rejected():
    from photonflux import ibisami

    with ibisami.load("pf_tx") as tx, pytest.raises(ValueError):
        tx.params.update(tx_pam4_hi=0.95)
