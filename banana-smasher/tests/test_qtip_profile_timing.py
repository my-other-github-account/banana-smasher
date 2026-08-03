from __future__ import annotations

import pytest

from banana_smasher.solver_qtip_profile import _split_solve_and_conformance_seconds


def test_split_solve_and_conformance_seconds_keeps_mandatory_audit_in_solve() -> None:
    solve, conformance = _split_solve_and_conformance_seconds(
        4.0,
        {"phase_seconds": {"packed_decode_conformance": 2.25}},
    )

    assert solve == pytest.approx(4.0)
    assert conformance == pytest.approx(2.25)


@pytest.mark.parametrize("conformance", [-1.0, 4.01, float("nan")])
def test_split_solve_and_conformance_seconds_rejects_invalid_audit(
    conformance: float,
) -> None:
    with pytest.raises(RuntimeError, match="packed decode conformance timing"):
        _split_solve_and_conformance_seconds(
            4.0,
            {"phase_seconds": {"packed_decode_conformance": conformance}},
        )