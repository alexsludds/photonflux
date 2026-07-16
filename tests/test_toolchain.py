"""Toolchain prerequisites: if these fail, nothing else can pass."""
from photonflux import doctor


def test_doctor_core_checks_pass():
    # the SKY130 PDK is optional (only the FET path needs it); everything else
    # — openvaf-ir, the VA includes, libngspice for card extraction — must pass
    failures = [c for c in doctor() if not c.ok and "PDK" not in c.name]
    assert not failures, "\n".join(f"{c.name}: {c.detail}" for c in failures)
