"""Circuit builder: cards, devices, params, validation."""
import pytest

import lightspice as ls


def test_device_emits_model_and_instance():
    ckt = ls.Circuit("t")
    ckt.device(ls.va("laser_dml"), "ld", "a", "0", "p", "0", tau=2e-12)
    deck = ckt.build()
    assert ".model mod_ld laser_dml (tau=2e-12)" in deck
    assert "Nld a 0 p 0 mod_ld" in deck
    assert ckt.osdi_files == [ls.va("laser_dml").osdi]


def test_unknown_param_rejected():
    ckt = ls.Circuit("t")
    with pytest.raises(ValueError, match="no parameter"):
        ckt.device(ls.va("laser_dml"), "ld", "a", "0", "p", "0", bogus=1)


def test_duplicate_name_rejected():
    ckt = ls.Circuit("t")
    ckt.device(ls.va("laser_dml"), "ld", "a", "0", "p", "0")
    with pytest.raises(ValueError, match="duplicate"):
        ckt.device(ls.va("laser_dml"), "ld", "b", "0", "q", "0")


def test_raw_dedents_and_end_card():
    ckt = ls.Circuit("t")
    ckt.raw(
        """
        V1 a 0 1
        R1 a 0 1k
        """
    )
    deck = ckt.build()
    assert "\nV1 a 0 1\nR1 a 0 1k\n" in deck
    assert deck.rstrip().endswith(".end")
