"""Verilog-A hierarchy: ring_mod.va assembled from its CMT sub-components.

ring_mod.va is a *hierarchical* Verilog-A module — it instantiates
directional_coupler.va + ring_phase_shifter.va + ring_waveguide.va on a shared
cavity node. openvaf drops child instances silently, so cx.va() flattens the
hierarchy source-level first (photonflux/va_hier.py). These tests pin

1. the flattener's output shape (flat module, ports preserved, no instances);
2. that flat models pass through untouched;
3. that the three sub-component .va models also lower and compose *standalone*
   at the circulax netlist level into the same ring — through-port field equal
   to the flattened cx.va("ring_mod") at machine precision;
4. that unsupported hierarchy constructs fail loudly instead of miscompiling.

The physics of the flattened ring itself (Lorentzian, coupling regimes,
equivalence to the Python decomposition) is pinned by tests/test_ring_mod.py
and tests/test_ring_decomposition.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource

from photonflux import cx, va_hier, toolchain

RING_VA = toolchain.MODELS_DIR / "optical_field" / "ring_mod.va"

RADIUS_UM = 7.5
N_G = 4.0
LOSS_DB_M = 7000.0
KAPPA2 = 0.10
DL_DV_PM = 45.0
LAM_NM = 1310.0
P_IN = 1e-3

SETTINGS = dict(lambda_nm=LAM_NM, lambda_res_nm=LAM_NM, radius_um=RADIUS_UM,
                n_g=N_G, n_eff=2.4, loss_db_m=LOSS_DB_M, kappa2=KAPPA2,
                dl_dv_pm=DL_DV_PM)


# ---------------------------------------------------------------------------
# flattener
# ---------------------------------------------------------------------------

def test_flattened_ring_mod_is_flat_and_port_identical():
    text = va_hier.flatten_va(RING_VA, toolchain.MODELS_DIR)
    assert text is not None, "ring_mod.va should be detected as hierarchical"
    # one module, same name and ports as the hierarchical source
    heads = re.findall(r"\bmodule\s+(\w+)\s*\(([^)]*)\)", text)
    assert len(heads) == 1 and heads[0][0] == "ring_mod"
    assert [p.strip() for p in heads[0][1].split(",")] == [
        "in_re", "in_im", "out_re", "out_im", "vp", "vn", "gnd"]
    # no child instances survive; the children's physics does
    for child in ("directional_coupler", "ring_phase_shifter", "ring_waveguide"):
        assert not re.search(rf"\b{child}\b\s*#", text)
    assert "ddt(V(are, gnd))" in text        # ring_waveguide storage
    assert "V(in_re, gnd) - V(aim, gnd)" in text  # coupler through port


def test_flat_model_passes_through_untouched():
    src = toolchain.MODELS_DIR / "optical_field" / "directional_coupler.va"
    assert va_hier.flatten_va(src, toolchain.MODELS_DIR) is None


def test_flat_models_outside_subset_pass_through():
    """The strict subset grammar must only ever gate *hierarchical* files:
    library models whose flat constructs the flattener does not parse
    (module name != stem, input/output directions) still load unchanged."""
    for stem in ("capacitor", "cart2pol"):
        src = next(p for p in toolchain.MODELS_DIR.glob(f"**/{stem}.va")
                   if "__jax__" not in p.parts)
        assert va_hier.flatten_va(src, toolchain.MODELS_DIR) is None
    assert cx.va("capacitor").__name__ == "Capacitor"
    assert cx.va("cart2pol").__name__.lower() == "cart2pol"


def test_missing_child_module_fails_loudly(tmp_path: Path):
    """A typo'd child name is still instance-shaped: it must raise, never
    fall through to openvaf (which would silently drop the instance)."""
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, gnd;\n"
        "    no_such_model u1(.a(x), .gnd(gnd));\n"
        "endmodule\n")
    with pytest.raises(FileNotFoundError, match="no_such_model"):
        va_hier.flatten_va(tmp_path / "top.va", tmp_path)


def test_non_literal_unoverridden_default_raises(tmp_path: Path):
    """A child default referencing sibling parameters would capture *parent*
    names when inlined verbatim — must raise, not miscompile."""
    (tmp_path / "leaf.va").write_text(
        '`include "discipline.h"\n'
        "module leaf(a, gnd);\n"
        "    inout a, gnd;\n"
        "    electrical a, gnd;\n"
        "    parameter real g = 1.0;\n"
        "    parameter real q = 2.0*g;\n"
        "    analog begin\n"
        "        I(a, gnd) <+ q * V(a, gnd);\n"
        "    end\n"
        "endmodule\n")
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, gnd;\n"
        "    leaf u1(.a(x), .gnd(gnd));\n"
        "endmodule\n")
    with pytest.raises(NotImplementedError, match="non-literal default"):
        va_hier.flatten_va(tmp_path / "top.va", tmp_path)


def test_scale_suffixed_default_is_normalised(tmp_path: Path):
    """Scale suffixes (50f) are legal on declarations but not in expressions;
    an unoverridden suffixed default must be inlined as a plain float."""
    (tmp_path / "leaf.va").write_text(
        '`include "discipline.h"\n'
        "module leaf(a, gnd);\n"
        "    inout a, gnd;\n"
        "    electrical a, gnd;\n"
        "    parameter real c = 50f;\n"
        "    analog begin\n"
        "        I(a, gnd) <+ c * ddt(V(a, gnd));\n"
        "    end\n"
        "endmodule\n")
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, gnd;\n"
        "    leaf u1(.a(x), .gnd(gnd));\n"
        "endmodule\n")
    text = va_hier.flatten_va(tmp_path / "top.va", tmp_path)
    assert "5e-14" in text and "50f" not in text


def test_mixed_electrical_declaration_not_duplicated(tmp_path: Path):
    """electrical decl mixing a port and an internal node must emit each
    net exactly once."""
    (tmp_path / "leaf.va").write_text(
        '`include "discipline.h"\n'
        "module leaf(a, gnd);\n"
        "    inout a, gnd;\n"
        "    electrical a, gnd;\n"
        "    analog begin\n"
        "        I(a, gnd) <+ ddt(V(a, gnd)) + V(a, gnd);\n"
        "    end\n"
        "endmodule\n")
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, hidden;\n"   # port + internal in one statement
        "    electrical gnd;\n"
        "    leaf u1(.a(hidden), .gnd(gnd));\n"
        "endmodule\n")
    text = va_hier.flatten_va(tmp_path / "top.va", tmp_path)
    decls = [ln for ln in text.splitlines()
             if re.search(r"\belectrical\b.*\bhidden\b", ln)]
    assert len(decls) == 1


def test_unsupported_positional_connections_raise(tmp_path: Path):
    (tmp_path / "leaf.va").write_text(
        '`include "discipline.h"\n'
        "module leaf(a, gnd);\n"
        "    inout a, gnd;\n"
        "    electrical a, gnd;\n"
        "    parameter real g = 1.0;\n"
        "    analog begin\n"
        "        I(a, gnd) <+ g * V(a, gnd);\n"
        "    end\n"
        "endmodule\n")
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, gnd;\n"
        "    leaf u1(x, gnd);\n"   # positional — outside the supported subset
        "endmodule\n")
    with pytest.raises(NotImplementedError, match="named port connections"):
        va_hier.flatten_va(tmp_path / "top.va", tmp_path)


def test_unconnected_child_port_raises(tmp_path: Path):
    (tmp_path / "leaf.va").write_text(
        '`include "discipline.h"\n'
        "module leaf(a, b, gnd);\n"
        "    inout a, b, gnd;\n"
        "    electrical a, b, gnd;\n"
        "    analog begin\n"
        "        I(a, gnd) <+ V(a, b);\n"
        "    end\n"
        "endmodule\n")
    (tmp_path / "top.va").write_text(
        '`include "discipline.h"\n'
        "module top(x, gnd);\n"
        "    inout x, gnd;\n"
        "    electrical x, gnd;\n"
        "    leaf u1(.a(x), .gnd(gnd));\n"   # b left dangling
        "endmodule\n")
    with pytest.raises(ValueError, match="must connect exactly the ports"):
        va_hier.flatten_va(tmp_path / "top.va", tmp_path)


# ---------------------------------------------------------------------------
# the sub-component .va models compose standalone into the same ring
# ---------------------------------------------------------------------------

def _field_terminator():
    @component(ports=("c",))
    def FieldTerminator(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"c": 0.0}, {}

    return FieldTerminator


def _base():
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": LAM_NM, "power": P_IN}},
        "VT": {"component": "vsrc", "settings": {"V": 0.0}},
        "TAP": {"component": "f2ri"},
        "JOIN": {"component": "ri2f"},
        "TO": {"component": "term"},
    }
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "vsrc": VoltageSource, "term": _field_terminator(),
              "f2ri": cx.field_to_ri(), "ri2f": cx.ri_to_field()}
    return inst, models


def _through(c, v):
    y = c.dc(params={"VT.V": jnp.asarray(v)})
    return np.asarray(c.port(y, "sout"))


def _monolithic_ring():
    inst, models = _base()
    inst["RING"] = {"component": "ring", "settings": SETTINGS}
    models["ring"] = cx.va("ring_mod")
    conn = {
        "LAS,p1": "TAP,c",
        "TAP,re": "RING,in_re", "TAP,im": "RING,in_im",
        "RING,out_re": "JOIN,re", "RING,out_im": "JOIN,im",
        "JOIN,c": "TO,c",
        "VT,p1": "RING,vp",
        "GND,p1": ("LAS,p2", "RING,vn", "RING,gnd", "VT,p2"),
    }
    net = {"instances": inst, "connections": conn, "ports": {"sout": "JOIN,c"}}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def _va_submodule_ring():
    """The three sub-component .va models wired at the *netlist* level —
    exactly the wiring ring_mod.va's hierarchy expresses, but composed by
    circulax instead of the flattener."""
    rates = cx.ring_cmt_rates(radius_um=RADIUS_UM, n_g=N_G,
                              loss_db_m=LOSS_DB_M, kappa2=KAPPA2)
    inst, models = _base()
    inst.update({
        "CPL": {"component": "va_coupler",
                "settings": {"inv_tau_e": rates["inv_tau_e"]}},
        "PS": {"component": "va_ps",
               "settings": {"lambda_nm": LAM_NM, "lambda_res_nm": LAM_NM,
                            "dl_dv_pm": DL_DV_PM, "cj": rates["cj"]}},
        "WG": {"component": "va_wg",
               "settings": {"inv_tau_i": rates["inv_tau_i"]}},
    })
    models.update({"va_coupler": cx.va("directional_coupler"),
                   "va_ps": cx.va("ring_phase_shifter"),
                   "va_wg": cx.va("ring_waveguide")})
    conn = {
        "LAS,p1": "TAP,c",
        "TAP,re": "CPL,in_re", "TAP,im": "CPL,in_im",
        "CPL,out_re": "JOIN,re", "CPL,out_im": "JOIN,im",
        "JOIN,c": "TO,c",
        # the shared cavity node
        "CPL,are": ("PS,are", "WG,are"),
        "CPL,aim": ("PS,aim", "WG,aim"),
        "VT,p1": "PS,vp",
        "GND,p1": ("LAS,p2", "PS,vn", "VT,p2",
                   "CPL,gnd", "PS,gnd", "WG,gnd"),
    }
    net = {"instances": inst, "connections": conn, "ports": {"sout": "JOIN,c"}}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def test_va_submodules_compose_to_ring_mod():
    """Netlist-composed sub-component .va models == flattened ring_mod.va."""
    v = np.linspace(-4.0, 4.0, 41)
    e_sub = _through(_va_submodule_ring(), v)
    e_mono = _through(_monolithic_ring(), v)
    assert np.max(np.abs(e_sub - e_mono)) < 1e-9
