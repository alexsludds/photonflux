#!/usr/bin/env python3
"""Importable SKY130 CMOS-inverter driver flavors for the circulax testbenches.

A "driver" here is the electronic stage that swings a modulator electrode off
a digital input. circulax netlists are flat SAX dicts (no subcircuits), so a
driver can't be a single component — it is a *fragment* of instances +
connections + models that gets stitched between four external nets:

    input source  ->[ driver ]->  load (electrode)
                       |  |
                      VDD GND

Three flavors, all built from exact-BSIM4.8 SKY130 FETs (OSDI, via
``cx.sky130_fet``):

* ``single_stage_inverter`` — one CMOS inverter (pfet + nfet). Logically
  **inverting**: electrode = NOT(input). This is the classic ring/MZM driver.
* ``single_stage_neutralized_inverter`` — the single inverter plus a
  **Miller-neutralization cap** (resizable ``c_neut``) that cancels the output
  FETs' Cgd feedthrough onto the electrode. Needs a *complement input* to drive
  the cap (see :func:`stitch_driver`'s ``vinbar``); with an ideal complementary
  source the cancellation is skew-free.
* ``two_stage_inverter`` — two inverters back to back. Logically
  **non-inverting** (two inversions): electrode follows the input. The first
  stage presents a light gate load to the source; the second (optionally
  ``taper``-scaled wider) does the electrode driving.

Each builder returns a :class:`DriverParts`. Wire it into a netlist with
:func:`stitch_driver`, then read its ``inverting`` flag when interpreting the
electrode polarity downstream (eye mapping, etc.):

    parts = single_stage_inverter(w_p=22.0, w_n=12.0, l=0.18)
    vdrv = stitch_driver(parts, inst, conn, mdl, gnd,
                         vin="VIN,p1", vdd="VDD,p1", load="RING,vp")
    # `vdrv` is the electrode net anchor — use it as a probe port.

    # neutralized variant also needs a complement-input net:
    parts = single_stage_neutralized_inverter(w_p=22, w_n=12, l=0.18, c_neut=2e-15)
    vdrv = stitch_driver(parts, inst, conn, mdl, gnd, vin="VIN,p1",
                         vdd="VDD,p1", load="RING,vp", vinbar="VINB,p1")

Swapping between flavors is a one-line change at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass

from circulax.components.electronic import Capacitor

from lightspice import cx

__all__ = [
    "DriverParts",
    "single_stage_inverter",
    "single_stage_neutralized_inverter",
    "two_stage_inverter",
    "stitch_driver",
]


@dataclass
class DriverParts:
    """A driver netlist fragment, ready to stitch into a SAX netlist.

    ``instances``/``connections``/``models`` are the driver-internal pieces
    (FETs, parasitic caps, and any fully-internal nets like a two-stage
    mid-node). The ``*_members`` tuples list the terminals that must join the
    four *external* nets — the caller wires those, so the fragment stays
    independent of the surrounding circuit's node names. ``out_members[0]`` is
    the electrode-net anchor and doubles as the drive probe point.
    """

    instances: dict
    connections: dict
    models: dict
    in_members: tuple          # first-stage gates  -> input source net
    out_members: tuple         # last-stage output  -> load net (anchor = [0])
    vdd_members: tuple         # PMOS sources/bodies -> VDD net
    gnd_members: tuple         # NMOS sources/bodies + cap returns -> GND net
    inverting: bool            # electrode logic vs input (odd # of stages)
    inbar_members: tuple = ()  # Miller-neutralization plate -> complement-input net


def _fets(name: str, stage: int, w_p: float, w_n: float, l: float) -> dict:  # noqa: E741
    """SKY130 pfet/nfet model factories for one inverter stage, keyed uniquely.

    Identical geometries content-hash to the same OSDI object, so tapered and
    untapered stages both compile only the distinct (W, L) bins once.
    """
    return {
        f"{name}_pfet{stage}": cx.sky130_fet("pfet_01v8", w=w_p, l=l),
        f"{name}_nfet{stage}": cx.sky130_fet("nfet_01v8", w=w_n, l=l),
    }


def single_stage_inverter(
    *,
    w_p: float,
    w_n: float,
    l: float,  # noqa: E741
    name: str = "DRV",
    cw: float = 5e-15,
) -> DriverParts:
    """One CMOS inverter (pfet ``w_p`` over nfet ``w_n``, channel length ``l``).

    Inverting: electrode = NOT(input). ``cw`` is the electrode wiring parasitic
    (its own drain caps aside), which gives the output net a finite RC so BDF2
    marches with sane steps. All widths/length in **um** (SKY130 convention).
    """
    mp, mn, cwo = f"{name}_MP", f"{name}_MN", f"{name}_CWO"
    cap = f"{name}_cap"
    return DriverParts(
        instances={
            mp: {"component": f"{name}_pfet0"},
            mn: {"component": f"{name}_nfet0"},
            cwo: {"component": cap, "settings": {"C": cw}},
        },
        connections={},                       # every net is external here
        models={**_fets(name, 0, w_p, w_n, l), cap: Capacitor},
        in_members=(f"{mp},g", f"{mn},g"),
        out_members=(f"{mp},d", f"{mn},d", f"{cwo},p1"),
        vdd_members=(f"{mp},s", f"{mp},b"),
        gnd_members=(f"{mn},s", f"{mn},b", f"{cwo},p2"),
        inverting=True,
    )


def single_stage_neutralized_inverter(
    *,
    w_p: float,
    w_n: float,
    l: float,  # noqa: E741
    c_neut: float,
    name: str = "DRV",
    cw: float = 5e-15,
) -> DriverParts:
    """Single CMOS inverter with a Miller-neutralization cap of size ``c_neut``.

    Same inverter as :func:`single_stage_inverter` (inverting: electrode =
    NOT(input)) plus a neutralization capacitor ``CN`` that cancels the output
    FETs' Cgd feedthrough onto the electrode.

    The gate *is* the input, so Cgd injects ``+Cgd*dVin`` onto the electrode
    (drain). ``CN`` bridges the electrode to a *complement* input ``vin_bar``
    (= NOT(vin)); it injects ``+c_neut*dVin_bar = -c_neut*dVin``, so the two
    cancel when ``c_neut ~ Cgd_out``. This only works if ``vin_bar`` is
    time-aligned with ``vin`` — trivially true when both come from an ideal
    complementary source (as in the ring/MZM benches). ``c_neut`` is the tuning
    knob: 0 -> no neutralization, ``~Cgd`` -> full cancellation, ``>Cgd`` ->
    over-neutralized (opposite-sign glitch).

    The fragment exposes ``inbar_members`` for the ``vin_bar`` net; wire it with
    :func:`stitch_driver`'s ``vinbar`` argument. Widths/length in **um**.
    """
    mp, mn, cwo, cn = f"{name}_MP", f"{name}_MN", f"{name}_CWO", f"{name}_CN"
    cap = f"{name}_cap"
    return DriverParts(
        instances={
            mp: {"component": f"{name}_pfet0"},
            mn: {"component": f"{name}_nfet0"},
            cwo: {"component": cap, "settings": {"C": cw}},
            cn: {"component": cap, "settings": {"C": c_neut}},
        },
        connections={},                       # every net is external here
        models={**_fets(name, 0, w_p, w_n, l), cap: Capacitor},
        in_members=(f"{mp},g", f"{mn},g"),
        # electrode net carries the drains, wiring cap, and the CN electrode plate
        out_members=(f"{mp},d", f"{mn},d", f"{cwo},p1", f"{cn},p1"),
        vdd_members=(f"{mp},s", f"{mp},b"),
        gnd_members=(f"{mn},s", f"{mn},b", f"{cwo},p2"),
        inverting=True,
        inbar_members=(f"{cn},p2",),          # CN far plate -> complement input
    )


def two_stage_inverter(
    *,
    w_p: float,
    w_n: float,
    l: float,  # noqa: E741
    name: str = "DRV",
    taper: float = 1.0,
    cw: float = 5e-15,
    c_mid: float = 3e-15,
) -> DriverParts:
    """Two CMOS inverters back to back — a non-inverting buffer.

    Stage 1 is ``w_p``/``w_n``; stage 2 is scaled by ``taper``. A classic
    tapered buffer (``taper>1``) makes the output stage wider to drive a heavy
    load *from a weak upstream stage*. If the input source is already fast/ideal
    (as in the testbenches here) that trade-off inverts: a small first stage
    only bottlenecks the mid-node, so ``taper=1.0`` with both stages sized to
    drive the load is fastest. ``c_mid`` is the internal mid-node parasitic
    (kept fully inside the fragment); ``cw`` is the electrode wiring parasitic.
    Electrode follows the input (two inversions). Widths/length in **um**.

    Note on Cgd kickback: the output FETs couple their (mid-node) gate edge onto
    the electrode, overshooting the rails by ~Cgd_out/C_load of the swing
    (~0.3 V here). It shrinks with a larger ``cw`` or a smaller output stage, but
    both cost edge speed; a true Cgd-neutralization cap needs a *differential*
    driver (a single-ended input tap leads the mid-node by a stage delay and
    doesn't cancel). See the driver discussion in the module docstring.
    """
    mp1, mn1 = f"{name}_MP1", f"{name}_MN1"
    mp2, mn2 = f"{name}_MP2", f"{name}_MN2"
    cmid, cwo = f"{name}_CMID", f"{name}_CWO"
    cap = f"{name}_cap"
    wp2, wn2 = w_p * taper, w_n * taper
    return DriverParts(
        instances={
            mp1: {"component": f"{name}_pfet1"},
            mn1: {"component": f"{name}_nfet1"},
            mp2: {"component": f"{name}_pfet2"},
            mn2: {"component": f"{name}_nfet2"},
            cmid: {"component": cap, "settings": {"C": c_mid}},
            cwo: {"component": cap, "settings": {"C": cw}},
        },
        # mid-node is fully internal: stage-1 drains drive stage-2 gates
        connections={
            f"{mp1},d": (f"{mn1},d", f"{mp2},g", f"{mn2},g", f"{cmid},p1"),
        },
        models={
            **_fets(name, 1, w_p, w_n, l),
            **_fets(name, 2, wp2, wn2, l),
            cap: Capacitor,
        },
        in_members=(f"{mp1},g", f"{mn1},g"),
        out_members=(f"{mp2},d", f"{mn2},d", f"{cwo},p1"),
        vdd_members=(f"{mp1},s", f"{mp1},b", f"{mp2},s", f"{mp2},b"),
        gnd_members=(f"{mn1},s", f"{mn1},b", f"{mn2},s", f"{mn2},b",
                     f"{cmid},p2", f"{cwo},p2"),
        inverting=False,
    )


def stitch_driver(
    parts: DriverParts,
    instances: dict,
    connections: dict,
    models: dict,
    gnd: list,
    *,
    vin: str,
    vdd: str,
    load,
    vinbar: str | None = None,
) -> str:
    """Merge a :class:`DriverParts` into a SAX netlist, in place.

    ``instances``/``connections``/``models`` are extended with the fragment,
    and ``gnd`` (the running list of terminals on the global-ground net) is
    extended with the driver's ground returns — wire that list into the GND
    net yourself afterward. The input gates are tied to ``vin``, the PMOS
    supplies to ``vdd``, and the driver output to ``load`` (one terminal or a
    tuple/list of them, e.g. the modulator electrode).

    A neutralized driver also exposes ``inbar_members`` (the neutralization
    cap's far plate); pass ``vinbar`` — the complement-input source terminal —
    to wire it. Omitting ``vinbar`` for such a driver is an error.

    Returns the electrode-net anchor terminal, for use as a drive probe port.
    """
    instances.update(parts.instances)
    models.update(parts.models)
    connections.update(parts.connections)
    connections[vin] = tuple(parts.in_members)
    connections[vdd] = tuple(parts.vdd_members)
    load_terms = [load] if isinstance(load, str) else list(load)
    anchor, *rest = parts.out_members
    connections[anchor] = tuple([*rest, *load_terms])
    if parts.inbar_members:
        if vinbar is None:
            raise ValueError(
                "driver has a neutralization cap (inbar_members) but no "
                "'vinbar' complement-input net was given to stitch_driver")
        connections[vinbar] = tuple(parts.inbar_members)
    gnd.extend(parts.gnd_members)
    return anchor
