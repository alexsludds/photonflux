"""Shared pytest configuration for the circulax test suite.

Every model in ``models/*.va`` is lowered to a JAX component by ``cx.va`` and
solved by circulax; there is no ngspice simulation flow. Steady-state physics
is pinned here; the transient studies (lasing turn-on, four-wave mixing, eye
diagrams) live in ``examples/*.py``, which carry their own analytic asserts.
"""
import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def sky130_available() -> bool:
    from photonflux import cx, sky130_lib

    try:
        sky130_lib()
    except FileNotFoundError:
        return False
    return cx.openvaf_ir_path().exists()
