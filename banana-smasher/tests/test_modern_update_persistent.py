from __future__ import annotations

import json
import hashlib
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher.cli import _parser, main
from banana_smasher.persistent import PersistentUpdateQueue
from banana_smasher.production import FrozenAttentionDescriptor, ProductionTrainableSurface
from banana_smasher.update import build_token_window, plan_token_window
from banana_smasher.update_engine import run_segmented_update

GIB = 1024**3


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "code_sha256": "4" * 64,
    }


def test_update_and_evaluate_are_first_class_cli_verbs(tmp_path: Path, capsys) -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert "update" in action.choices
    assert "evaluate" in action.choices

    queue = tmp_path / "queue"
    assert main(["update", "--serve", "--queue", str(queue)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["command"] == "update"
    assert receipt["state"] == "WAITING"
    assert json.loads((queue / "state.json").read_text())["state"] == "WAITING"


def test_update_dependency_extra_is_exactly_pinned() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["update"] == ["torch==2.11.0"]


def test_cli_reference_update_runs_exact_physical_window(tmp_path: Path, capsys) -> None:
    source = tmp_path / "window.npz"
    np.savez(
        source,
        input_ids=np.arange(6, dtype=np.int64),
        teacher_mask=np.ones(6, dtype=np.bool_),
        positions=np.arange(6, dtype=np.int64),
        features=np.arange(18, dtype=np.float32).reshape(6, 3) / 10,
        targets=np.zeros((6, 3), dtype=np.float32),
    )
    identity = tmp_path / "identity.json"
    bound = _identity()
    bound["content_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    identity.write_text(json.dumps(bound))
    output = tmp_path / "update.pt"

    assert (
        main(
            [
                "update",
                "--input",
                str(source),
                "--output",
                str(output),
                "--identity",
                str(identity),
                "--tokens",
                "4",
                "--segments",
                "2",
                "--depth",
                "2",
                "--reference",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS_UPDATE"
    assert receipt["observed_tensor_shape"] == [1, 4, 3]
    assert receipt["physical_tokens"] == 4
    assert receipt["segments"] == 2
    assert receipt["optimizer_steps"] == 1
    assert receipt["execution"] == {
        "forward": True,
        "backward": True,
        "optimizer": True,
    }
    assert receipt["peak_memory_bytes"] >= 0


def test_tokens_are_exact_physical_batch_one_geometry() -> None:
    ids = np.arange(12, dtype=np.int64)
    teacher_mask = np.array([True] * 11 + [False])
    positions = np.arange(100, 112, dtype=np.int64)

    window = build_token_window(
        ids,
        teacher_mask=teacher_mask,
        positions=positions,
        tokens=8,
    )

    assert window.input_ids.shape == (1, 8)
    assert window.teacher_mask.shape == (1, 8)
    assert window.positions.shape == (1, 8)
    assert window.receipt == {
        "observed_tensor_shape": [1, 8],
        "teacher_mask_shape": [1, 8],
        "teacher_mask_true": 8,
        "position_shape": [1, 8],
        "position_first": 100,
        "position_last": 107,
        "physical_tokens": 8,
    }
    with pytest.raises(ValueError, match="exactly 13 physical tokens"):
        build_token_window(ids, teacher_mask=teacher_mask, positions=positions, tokens=13)


def test_auto_token_sizing_reserves_four_gib_os_floor_and_fails_unknown_capacity() -> None:
    assert plan_token_window(
        requested_tokens=None,
        bytes_per_token=GIB,
        available_os_bytes=7 * GIB,
        available_device_bytes=2 * GIB,
    ) == 2
    with pytest.raises(RuntimeError, match="cannot determine OS memory capacity"):
        plan_token_window(
            requested_tokens=1,
            bytes_per_token=1,
            available_os_bytes=None,
            available_device_bytes=GIB,
        )
    with pytest.raises(MemoryError, match="4 GiB OS floor"):
        plan_token_window(
            requested_tokens=4,
            bytes_per_token=GIB,
            available_os_bytes=7 * GIB,
            available_device_bytes=8 * GIB,
        )


def test_full_depth_surface_trains_every_declared_layer() -> None:
    surface = ProductionTrainableSurface(depth=4, width=3)
    value = torch.randn(1, 5, 3)
    surface(value).sum().backward()
    census = surface.trainable_census()
    assert census["depth"] == 4
    assert census["layers_with_gradients"] == [0, 1, 2, 3]
    assert census["all_gradients_finite"] is True


def test_frozen_attention_descriptor_is_lazy_hash_bound_and_relocation_safe(
    tmp_path: Path,
) -> None:
    member = tmp_path / "attention.npy"
    index = tmp_path / "attention.index.json"
    np.save(member, np.arange(12, dtype=np.float32).reshape(3, 4))
    index.write_text('{"layer.0.attention":"attention.npy"}\n')
    descriptor = FrozenAttentionDescriptor.from_file(
        member,
        key="layer.0.attention",
        member_sha256=None,
        index_path=index,
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        execution_device="cpu",
    )
    assert "array" not in vars(descriptor)
    assert descriptor.execution_device == "cpu"
    assert descriptor.open().shape == (3, 4)
    member.write_bytes(member.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="member SHA-256 mismatch"):
        descriptor.open()

    np.save(member, np.arange(12, dtype=np.float32).reshape(3, 4))
    index.write_text("{}\n")
    with pytest.raises(RuntimeError, match="index SHA-256 mismatch"):
        descriptor.open()


def test_checkpoint_resume_after_relocation_rebinds_atomically_and_preserves_state(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    segments = [torch.ones(2, 2), torch.full((3, 2), 2.0)]
    output = tmp_path / "first" / "update.pt"

    def interrupt(index: int, _manifest: dict[str, object]) -> None:
        if index == 0:
            raise KeyboardInterrupt("test interruption")

    with pytest.raises(KeyboardInterrupt):
        run_segmented_update(
            parameters=list(model.parameters()),
            optimizer=optimizer,
            segments=segments,
            item_count=lambda segment: segment.shape[0],
            loss_sum=lambda segment: model(segment).square().sum(),
            output=output,
            identity=_identity(),
            on_segment_committed=interrupt,
        )

    relocated = tmp_path / "relocated"
    shutil.move(str(output.parent), relocated)
    relocated_output = relocated / "update.pt"
    resumed_model = torch.nn.Linear(2, 1, bias=False)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    receipt = run_segmented_update(
        parameters=list(resumed_model.parameters()),
        optimizer=resumed_optimizer,
        segments=segments,
        item_count=lambda segment: segment.shape[0],
        loss_sum=lambda segment: resumed_model(segment).square().sum(),
        output=relocated_output,
        identity=_identity(),
    )

    assert receipt["status"] == "PASS_UPDATE"
    assert receipt["optimizer_steps"] == 1
    assert receipt["resumed_segments"] == 1
    rebind = json.loads(
        (Path(f"{relocated_output}.checkpoint") / "rebind-receipt.json").read_text()
    )
    assert rebind["status"] == "ATOMIC_REBIND"
    assert rebind["identity"] == _identity()
    assert "/first/" not in json.dumps(rebind)


def test_checkpoint_content_drift_is_fail_closed_after_relocation(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    output = tmp_path / "update.pt"
    receipt = run_segmented_update(
        parameters=list(model.parameters()),
        optimizer=optimizer,
        segments=[torch.ones(1, 1)],
        item_count=lambda segment: 1,
        loss_sum=lambda segment: model(segment).square().sum(),
        output=output,
        identity=_identity(),
    )
    assert receipt["status"] == "PASS_UPDATE"
    payload = next(Path(f"{output}.checkpoint").glob("payload-*.pt"))
    payload.write_bytes(payload.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="payload SHA-256 mismatch"):
        run_segmented_update(
            parameters=list(model.parameters()),
            optimizer=optimizer,
            segments=[torch.ones(1, 1)],
            item_count=lambda segment: 1,
            loss_sum=lambda segment: model(segment).square().sum(),
            output=output,
            identity=_identity(),
        )


def test_persistent_queue_waits_durably_and_starts_clock_at_segment_start(
    tmp_path: Path,
) -> None:
    times = iter([10.0, 25.0])
    queue = PersistentUpdateQueue(tmp_path / "queue", clock=lambda: next(times))
    waiting = queue.waiting()
    assert waiting["state"] == "WAITING"
    assert "segment_started_at" not in waiting

    queued = queue.enqueue({"tokens": 8})
    claimed = queue.claim_next(worker="worker-a")
    assert claimed["job_id"] == queued["job_id"]
    assert "segment_started_at" not in claimed

    started = queue.segment_start(claimed["job_id"], worker="worker-a")
    assert started["state"] == "SEGMENT_START"
    assert started["segment_started_at"] == 25.0
    assert json.loads((tmp_path / "queue" / "state.json").read_text()) == started
