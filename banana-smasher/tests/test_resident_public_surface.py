from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import banana_smasher
from banana_smasher.cli import _parser
from banana_smasher import persistent, resident_training, update_service


def _command_names() -> set[str]:
    names: set[str] = set()
    pending = [_parser()]
    while pending:
        parser = pending.pop()
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                names.update(choices)
                pending.extend(choices.values())
    return names


def test_only_resident_session_is_exported_for_training() -> None:
    assert "ResidentTrainingSession" in banana_smasher.__all__
    assert "ResidentTrainer" not in banana_smasher.__all__
    assert "ResidentTrainer" not in resident_training.__all__
    assert not hasattr(resident_training, "ResidentTrainer")
    retired_commands = {
        "train",
        "train-status",
        "checkpoint-info",
        "update-enqueue",
        "update-status",
    }
    assert retired_commands.isdisjoint(_command_names())
    assert "UpdateQueue" not in banana_smasher.__all__
    assert "serve_persistent_updates" not in banana_smasher.__all__
    assert not hasattr(persistent, "UpdateQueue")
    assert not hasattr(persistent, "serve_queue")
    assert not hasattr(update_service, "serve_persistent_updates")

    for retired in retired_commands:
        with pytest.raises(SystemExit) as raised:
            _parser().parse_args([retired])
        assert raised.value.code == 2


def test_spark_resident_step_performance_receipt() -> None:
    """Physical Spark jobs opt in with their API-produced timing receipt."""

    receipt_path = os.environ.get("BANANA_SMASHER_SPARK_TRAINING_RECEIPT")
    if receipt_path is None:
        pytest.skip(
            "set BANANA_SMASHER_SPARK_TRAINING_RECEIPT on the Spark integration rail"
        )
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution_rail"] == "resident-in-memory"
    assert len(set(receipt["model_instance_ids"])) == 1
    steps = receipt["steps"]
    assert steps
    required_phases = {
        "forward",
        "backward",
        "communication",
        "optimizer",
        "update_total",
    }
    for step in steps:
        assert required_phases <= set(step["phase_seconds"])
        assert 360.0 <= float(step["phase_seconds"]["update_total"]) <= 420.0
