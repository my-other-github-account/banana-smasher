from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from banana_smasher.cli import _parser
from banana_smasher.gate_only_trainer import (
    CLASSES,
    GateTrainingConfig,
    load_training_manifests,
    one_cell_sign_step,
    project_exact_budget,
    train_gate_only,
)


def _manifest(path: Path, split: str, prefix: str) -> Path:
    rows = [
        {
            "window_id": f"{prefix}-{name}",
            "class": name,
            "teacher_logits": [2.0, -2.0],
        }
        for name in CLASSES
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-gate-data-v1",
                "split": split,
                "class_counts": {name: 1 for name in CLASSES},
                "rows": rows,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_optimizer_allowlist_frozen_digest_and_cli_surface(tmp_path: Path) -> None:
    train_path = _manifest(tmp_path / "TRAIN.json", "TRAIN", "train")
    dev_path = _manifest(tmp_path / "DEV.json", "DEV", "dev")
    train, dev = load_training_manifests(train_path, dev_path)
    runtime = TinyRuntime()

    result = train_gate_only(
        runtime,
        train,
        dev,
        GateTrainingConfig(
            cell_count=2,
            whole_model_target_bytes=10,
            fixed_dense_metadata_bytes=6,
            expert_envelope_bytes=4,
            repair_budget_bytes=0,
            steps=1,
            learning_rate=2.0,
            dev_every=1,
        ),
    )

    assert list(dict(result.model.named_parameters())) == ["tier_logits"]
    assert result.optimizer_parameter_names == ["tier_logits"]
    assert result.frozen_state_digest_before == result.frozen_state_digest_after
    assert result.receipt["frozen_state_digest_before"] == result.receipt[
        "frozen_state_digest_after"
    ]
    commands = next(
        action.choices for action in _parser()._actions if getattr(action, "choices", None)
    )
    assert "train-gates" in commands


def test_lower_kld_branch_probability_increases_without_sign_normalization() -> None:
    before, after, gradient = one_cell_sign_step(
        branch_kld=torch.tensor([1.0, 0.1, 2.0]),
        initial_logits=torch.zeros(3),
        learning_rate=0.5,
    )

    assert after[1] > before[1]
    assert gradient[1] < 0.0
    assert not torch.allclose(gradient.abs(), torch.ones_like(gradient))


def test_exact_projection_uses_actual_wire_bytes() -> None:
    projection = project_exact_budget(
        tier_logits=torch.tensor([[0.0, 4.0, 1.0], [0.0, 3.0, 5.0]]),
        cell_ids=["cell-0", "cell-1"],
        tier_bytes=torch.tensor([[2, 1, 3], [2, 1, 3]], dtype=torch.int64),
        expert_envelope_bytes=4,
    )

    assert projection.hard_expert_bytes == 4
    assert projection.tier_indices.tolist() == [1, 2]
    assert projection.tier_counts == {"native_mxfp4": 0, "qtip2": 1, "qtip3": 1}


def test_train_dev_overlap_and_holdout_are_rejected(tmp_path: Path) -> None:
    train_path = _manifest(tmp_path / "TRAIN.json", "TRAIN", "same")
    dev_path = _manifest(tmp_path / "DEV.json", "DEV", "same")
    with pytest.raises(ValueError, match="overlap"):
        load_training_manifests(train_path, dev_path)

    holdout_path = _manifest(tmp_path / "HOLDOUT.json", "DEV", "holdout")
    with pytest.raises(ValueError, match="HOLDOUT"):
        load_training_manifests(train_path, holdout_path)


class TinyRuntime:
    layers = (0, 1)
    cell_ids = ("L000.E000.down", "L001.E000.down")
    cell_layers = (0, 1)
    tier_bytes = torch.tensor([[2, 1, 3], [2, 1, 3]], dtype=torch.int64)

    def __init__(self) -> None:
        self.frozen = {
            "router": torch.tensor([0.25, -0.5]),
            "teacher": torch.tensor([2.0, -2.0]),
            "planes": torch.arange(6, dtype=torch.float32),
        }
        self.layer_stage_calls: list[int] = []

    def frozen_state(self):
        return self.frozen

    def initial_tier_logits(self):
        return torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def batches(self, manifest):
        return manifest.rows

    def initial(self, batch):
        return torch.tensor(0.0)

    @contextmanager
    def layer_stage(self, layer: int):
        self.layer_stage_calls.append(layer)
        branch_values = (
            torch.tensor([0.0, 1.5, -1.0])
            if layer == 0
            else torch.tensor([0.0, -1.0, 1.5])
        )

        def forward(activation, *, gates, hard_tiers, window_id):
            del hard_tiers, window_id
            return activation + torch.sum(gates[0] * branch_values)

        yield forward

    def final_logits(self, activation, *, window_id):
        del window_id
        return torch.stack((activation, -activation))

    def teacher_logits(self, batch):
        return torch.tensor(batch["teacher_logits"])


def test_tiny_gate_training_decreases_soft_loss_and_flips_hard_cells(tmp_path: Path) -> None:
    train_path = _manifest(tmp_path / "TRAIN.json", "TRAIN", "train")
    dev_path = _manifest(tmp_path / "DEV.json", "DEV", "dev")
    train, dev = load_training_manifests(train_path, dev_path)
    runtime = TinyRuntime()

    result = train_gate_only(
        runtime,
        train,
        dev,
        GateTrainingConfig(
            cell_count=2,
            whole_model_target_bytes=10,
            fixed_dense_metadata_bytes=6,
            expert_envelope_bytes=4,
            repair_budget_bytes=0,
            steps=6,
            learning_rate=2.0,
            temperature=1.0,
            dev_every=1,
        ),
    )

    checkpoints = result.receipt["checkpoints"]
    assert checkpoints[-1]["soft_train_kld"] < checkpoints[0]["soft_train_kld"]
    assert max(row["moved_cells"] for row in checkpoints[1:]) >= 1
    assert checkpoints[-1]["hard_expert_bytes"] == 4
    assert checkpoints[-1]["hard_whole_model_bytes"] == 10
    assert len(runtime.layer_stage_calls) % 2 == 0
    assert all(
        runtime.layer_stage_calls[index : index + 2] == [0, 1]
        for index in range(0, len(runtime.layer_stage_calls), 2)
    )
    assert result.receipt["objective"] == "frozen-own-base-final-logit-teacher-kld"
    assert result.receipt["repair_budget_bytes"] == 0
