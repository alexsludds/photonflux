"""Shared helpers for the circulax physics tests.

Every model in ``models/*.va`` lowers to a circulax component via ``cx.va``;
these helpers wrap the boilerplate of building a single-device netlist, driving
each input node with a DC voltage source, terminating unused optical outputs,
and reading output nodes back — the same pattern ``examples/*.py`` use, minus
the transient machinery.

``build`` returns the compiled circuit so tests can run vectorised parameter
sweeps (``c.dc(params={...})``); ``op`` is the single-point convenience.

Source-instance naming: the source driving DUT port ``P`` is instance
``SRC_P`` — sweep its level with ``params={"SRC_P.V": array}``. Device
parameters are addressed as ``DUT.<param>``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource


def terminator():
    """Infinite-impedance 2-node termination for driven-but-unused (re, im)
    optical outputs — circulax requires every port connected."""

    @component(ports=("re", "im"))
    def Terminator(signals: Signals, s: States):
        return {"re": 0.0, "im": 0.0}, {}

    return Terminator


def build(component, drives, *, settings=None, reads=(), terms=(),
          is_complex=False, backend="dense", **compile_kw):
    """Compile a one-device netlist.

    drives   {port: dc_voltage} — each gets a ``SRC_<port>`` voltage source
    settings {param: value}     — device parameters
    reads    [port, ...]        — output nodes exposed as circuit ports
    terms    [(re_port, im_port), ...] — unused optical outputs to terminate
    """
    insts = {"GND": {"component": "ground"},
             "DUT": {"component": "dut", "settings": dict(settings or {})}}
    gnd = ["DUT,gnd"]
    for port, val in drives.items():
        sn = f"SRC_{port}"
        insts[sn] = {"component": "vsrc", "settings": {"V": float(val)}}
        gnd.append(f"{sn},p2")
    conns = {f"SRC_{p},p1": f"DUT,{p}" for p in drives}
    for j, (re, im) in enumerate(terms):
        tn = f"TERM{j}"
        insts[tn] = {"component": "term"}
        conns[f"{tn},re"] = f"DUT,{re}"
        conns[f"{tn},im"] = f"DUT,{im}"
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {r: f"DUT,{r}" for r in reads}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "dut": component, "term": terminator()}
    return compile_circuit(net, models, is_complex=is_complex,
                           backend=backend, **compile_kw)


def op(component, drives, *, settings=None, reads=(), terms=(),
       is_complex=False, **compile_kw):
    """Single DC operating point → {read_port: complex value}."""
    c = build(component, drives, settings=settings, reads=reads, terms=terms,
              is_complex=is_complex, **compile_kw)
    y = c.dc()
    return {r: complex(c.port(y, r)) for r in reads}


def power(vals, re: str, im: str) -> float:
    """|E|^2 = re^2 + im^2 from an op() result dict."""
    return vals[re].real ** 2 + vals[im].real ** 2
