"""Engine + binding behaviour: analyses, vectors, errors, reuse."""
import numpy as np
import pytest

import photonflux as ls


def test_tran_rc(eng):
    r = eng.tran(
        "rc lowpass\nVin in 0 PULSE(0 1 1n 100p 100p 10n 20n)\nR1 in out 1k\nC1 out 0 1p\n",
        "50p", "9n",
    )
    assert "out" in r and len(r.t) > 50
    assert r["out"].max() > 0.99  # ~8 tau after the edge


def test_ac_complex_vectors(eng):
    r = eng.ac(
        "rc ac\nVin in 0 DC 0 AC 1\nR1 in out 1k\nC1 out 0 1p\n",
        1e6, 1e12, points=20,
    )
    h = r["out"]
    assert np.iscomplexobj(h)
    fc = 1 / (2 * np.pi * 1e3 * 1e-12)
    idx = int(np.argmin(np.abs(r.f - fc)))
    assert abs(h[idx]) == pytest.approx(1 / np.sqrt(2), rel=0.05)


def test_branch_current_accessor(eng):
    r = eng.op("divider\nV1 a 0 2\nR1 a b 1k\nR2 b 0 1k\n")
    assert float(r.i("v1")[0]) == pytest.approx(-1e-3, rel=1e-6)


def test_netlist_error_raises_and_engine_recovers(eng):
    # NB: ngspice treats some malformed element lines as warnings and just
    # drops them; a missing subcircuit is a genuine hard error.
    with pytest.raises(ls.NgSpiceError):
        eng.op("broken deck\nV1 a 0 1\nX1 a 0 no_such_subckt\n")
    r = eng.op("ok deck\nV1 a 0 1\nR1 a 0 1k\n")
    assert float(r["a"][0]) == pytest.approx(1.0)


def test_unknown_vector_message(eng):
    r = eng.op("trivial\nV1 a 0 1\nR1 a 0 1k\n")
    with pytest.raises(KeyError, match="available"):
        r["nope"]
