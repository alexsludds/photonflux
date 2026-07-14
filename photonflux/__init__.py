"""photonflux — analog + photonic co-simulation in Python.

Verilog-A photonics (compiled by OpenVAF to OSDI) and SPICE electronics
(PDK transistors included) solved together in one native ngspice matrix,
with numpy results and link-level analysis on top.

Quick start:

    import photonflux as ls

    ckt = ls.Circuit("photonic link")
    ckt.raw("Vdrv drv 0 PULSE(1.30 1.55 0.5n 50p 50p 0.5n 1.5n)")
    ckt.device(ls.va("laser_dml"), "laser", "drv", "0", "popt_tx", "0")
    ls.add_fiber(ckt, "fib", "popt_tx", "popt_rx", loss_db=3)
    ckt.device(ls.va("photodiode"), "pd", "popt_rx", "0", "pd_an", "pd_cat")
    ckt.raw("Rf vout pd_cat 200", "Eopamp vout 0 0 pd_cat 1e6", "Ranch pd_an 0 1")

    r = ls.Engine().tran(ckt, "10p", "4n")
    print(r["popt_tx"].max())   # optical power [W] is a node voltage
"""
from ._ngspice import NgSpice, NgSpiceError, NgSpiceFatalError
from .analysis import QStats, best_sampling, eye_fold, q_ber, sensitivity
from .circuit import Circuit
from .compiler import ModelNameCollision, VaModule, compile_all, compile_va
from .devices import add_fiber, attenuation_lin, dbm, library, va
from .engine import Engine, Result
from .signals import nrz_pwl, prbs, sample_centers
from .sweep import sweep
from .toolchain import doctor, doctor_report, sky130_lib

__version__ = "0.1.0"

__all__ = [
    "Circuit", "Engine", "Result",
    "va", "library", "compile_va", "compile_all", "VaModule",
    "add_fiber", "attenuation_lin", "dbm",
    "prbs", "nrz_pwl", "sample_centers",
    "q_ber", "QStats", "best_sampling", "eye_fold", "sensitivity",
    "sweep",
    "doctor", "doctor_report", "sky130_lib",
    "NgSpice", "NgSpiceError", "NgSpiceFatalError", "ModelNameCollision",
]
