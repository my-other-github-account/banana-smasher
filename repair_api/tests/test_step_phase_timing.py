import json

import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import _record_step_phase


def test_step_phase_receipt_is_optional(tmp_path) -> None:
    _record_step_phase(
        {}, rank=0, update=21, phase="forward", boundary="start"
    )
    assert list(tmp_path.iterdir()) == []


def test_step_phase_receipt_appends_identity_bound_boundaries(tmp_path) -> None:
    receipt = tmp_path / "STEP_PHASE.rank{rank}.jsonl"
    config = {
        "step_phase_receipt": str(receipt),
        "task_id": "t_fixture",
        "basis_sha256": "b" * 64,
        "canonical_git_pin": "c" * 40,
    }
    _record_step_phase(
        config, rank=1, update=21, phase="optimizer", boundary="start"
    )
    _record_step_phase(
        config,
        rank=1,
        update=21,
        phase="optimizer",
        boundary="complete",
        elapsed_seconds=1.25,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "STEP_PHASE.rank1.jsonl").read_text().splitlines()
    ]
    assert [(row["phase"], row["boundary"]) for row in rows] == [
        ("optimizer", "start"),
        ("optimizer", "complete"),
    ]
    assert rows[1]["elapsed_seconds"] == 1.25
    assert rows[1]["update"] == 21
    assert rows[1]["rank"] == 1
    assert rows[1]["basis_sha256"] == "b" * 64
    assert rows[1]["canonical_git_pin"] == "c" * 40


def test_step_phase_rejects_unknown_boundary(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="start or complete"):
        _record_step_phase(
            {"step_phase_receipt": str(tmp_path / "phase.jsonl")},
            rank=0,
            update=21,
            phase="forward",
            boundary="entered",
        )
