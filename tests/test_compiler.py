"""Compilation: caching, module-name parsing, and the collision guard."""
import time

import pytest

import photonflux as ls
from photonflux.compiler import compile_va, parse_va


def test_compile_all_models():
    mods = ls.compile_all()
    names = {m.name for m in mods}
    assert {"laser_dml", "laser_rate", "photodiode", "mzm"} <= names
    for m in mods:
        assert m.osdi.exists()
        assert m.descriptor_name() == m.name


def test_cache_skips_recompile():
    laser = ls.va("laser_dml")
    before = laser.osdi.stat().st_mtime_ns
    time.sleep(0.01)
    again = compile_va(laser.va)
    assert again.osdi.stat().st_mtime_ns == before, "cache miss on unchanged source"


def test_param_parsing():
    laser = ls.va("laser_dml")
    assert set(laser.params) == {"Ith", "slope", "Rs", "Von", "tau"}


def test_collision_guard(tmp_path):
    bad = tmp_path / "res.va"
    bad.write_text(
        '`include "discipline.h"\n'
        "module res(p, n);\n"
        "    inout p, n;\n"
        "    electrical p, n;\n"
        "    analog V(p, n) <+ 1k * I(p, n);\n"
        "endmodule\n"
    )
    with pytest.raises(ls.ModelNameCollision, match="incorrect model type"):
        compile_va(bad)


def test_parse_va_module_and_params(tmp_path):
    f = tmp_path / "thing.va"
    f.write_text(
        "// a comment with module fake(x) inside\n"
        "module my_thing(a, b);\n"
        "    parameter real gain = 2.5 from (0:inf);\n"
        "    parameter real off = 1m;\n"
        "endmodule\n"
    )
    name, params = parse_va(f)
    assert name == "my_thing"
    assert params == {"gain": "2.5", "off": "1m"}
