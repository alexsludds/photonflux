"""Toolchain prerequisites: if these fail, nothing else can pass."""
import lightspice as ls


def test_doctor_core_checks_pass():
    failures = [
        c for c in ls.doctor() if not c.ok and "PDK" not in c.name
    ]
    assert not failures, "\n".join(f"{c.name}: {c.detail}" for c in failures)
