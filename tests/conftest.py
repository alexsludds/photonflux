import pytest

import photonflux as ls


@pytest.fixture(scope="session")
def eng() -> ls.Engine:
    """One Engine for the whole session (libngspice is a process singleton);
    this also exercises the circuit-reload path between tests."""
    return ls.Engine()


@pytest.fixture(scope="session")
def sky130_available() -> bool:
    try:
        ls.sky130_lib()
        return True
    except FileNotFoundError:
        return False
