from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import pytest
import torch

import banana_smasher.update_engine as update_engine_module
from banana_smasher.update_checkpoint import commit_segment_checkpoint
from banana_smasher.update_engine import run_segmented_update


class Interrupted(RuntimeError):
    pass


def _write_marker_and_return_payload(marker: str) -> dict:
    Path(marker).write_text("unsafe global executed")
    return {}


class _MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return _write_marker_and_return_payload, (self.marker,)


def _identity(fill: str = "a") -> dict[str, str]:
    return {
        "content_sha256": fill * 64,
        "config_sha256": "b" * 64,
        "assignment_sha256": "c" * 64,
        "aot_sha256": "d" * 64,
        "runtime_sha256": "e" * 64,
        "code_sha256": "f" * 64,
    }


def _run(
    output: Path,
    *,
    stop_after: int | None = None,
    forbidden: bool = False,
    identity: dict[str, str] | None = None,
) -> dict:
    parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1e-2)
    segments = [
        torch.tensor([float(i), float(i) + 0.5], dtype=torch.float64) for i in range(3)
    ]

    def loss_sum(values: torch.Tensor) -> torch.Tensor:
        if forbidden:
            raise AssertionError("completed replay executed forward")
        return ((parameter - values) ** 2).sum()

    def committed(index: int, _manifest: dict) -> None:
        if stop_after == index:
            raise Interrupted

    return run_segmented_update(
        parameters=[parameter],
        optimizer=optimizer,
        segments=segments,
        item_count=lambda values: int(values.numel()),
        loss_sum=loss_sum,
        output=output,
        receipt=output.with_suffix(".json"),
        identity=_identity() if identity is None else identity,
        physical_tokens=2,
        observed_input_shape=[1, 2],
        teacher_geometry={
            "target_shape": [2],
            "mask_shape": [1, 2],
            "position_shape": [1, 2],
        },
        peak_memory_bytes=lambda: 1024,
        on_segment_committed=committed,
    )


def _interrupted_child(output: str) -> None:
    _run(Path(output), stop_after=0)


def test_interrupted_process_resumes_contiguous_prefix(tmp_path: Path) -> None:
    output = tmp_path / "process.pt"
    process = multiprocessing.get_context("spawn").Process(
        target=_interrupted_child, args=(str(output),)
    )
    process.start()
    process.join(60)
    assert process.exitcode not in (None, 0)
    assert not output.exists()

    resumed = _run(output)
    assert resumed["resumed_segments"] == 1
    assert resumed["completed_segments"] == 3
    assert resumed["optimizer_steps"] == 1


def test_interrupted_resume_moves_directory_and_rebinds_atomically(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    try:
        _run(original / "update.pt", stop_after=0)
    except Interrupted:
        pass
    else:  # pragma: no cover - test guard
        raise AssertionError("fixture did not interrupt")

    manifest_before = json.loads(
        (original / "update.pt.checkpoint" / "manifest.json").read_text()
    )
    assert not Path(manifest_before["payload_path"]).is_absolute()

    moved = tmp_path / "moved"
    shutil.move(original, moved)
    resumed = _run(moved / "update.pt")

    assert resumed["resumed_segments"] == 1
    assert resumed["optimizer_steps"] == 1
    checkpoint = moved / "update.pt.checkpoint"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    rebind_path = checkpoint / manifest["last_rebind_receipt"]["path"]
    rebind = json.loads(rebind_path.read_text())
    assert rebind["status"] == "PASS_REBIND"
    assert rebind["old_root_sha256"] != rebind["new_root_sha256"]


def test_completed_replay_is_idempotent_and_does_not_compute(tmp_path: Path) -> None:
    output = tmp_path / "complete.pt"
    first = _run(output)
    replay = _run(output, forbidden=True)

    assert replay == first
    assert replay["forward_count"] == 3
    assert replay["backward_count"] == 3
    assert replay["optimizer_steps"] == 1


def test_replay_rejects_tampered_atomic_rebind_receipt(tmp_path: Path) -> None:
    original = tmp_path / "first"
    original.mkdir()
    try:
        _run(original / "update.pt", stop_after=0)
    except Interrupted:
        pass
    moved = tmp_path / "second"
    shutil.move(original, moved)
    _run(moved / "update.pt")

    checkpoint = moved / "update.pt.checkpoint"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    rebind = checkpoint / manifest["last_rebind_receipt"]["path"]
    rebind.write_bytes(rebind.read_bytes() + b"tamper")

    try:
        _run(moved / "update.pt", forbidden=True)
    except RuntimeError as exc:
        assert "rebind receipt" in str(exc)
    else:  # pragma: no cover - test guard
        raise AssertionError("tampered rebind receipt was accepted")


def test_joint_rebind_receipt_and_manifest_replacement_fails_authentication(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    with pytest.raises(Interrupted):
        _run(original / "update.pt", stop_after=0)
    moved = tmp_path / "moved"
    shutil.move(original, moved)
    _run(moved / "update.pt")

    checkpoint = moved / "update.pt.checkpoint"
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rebind_path = checkpoint / manifest["last_rebind_receipt"]["path"]
    rebind = json.loads(rebind_path.read_text())
    rebind["old_root_sha256"] = "0" * 64
    rebind_path.write_text(json.dumps(rebind, indent=2, sort_keys=True) + "\n")
    rebind_bytes = rebind_path.read_bytes()
    manifest["last_rebind_receipt"]["bytes"] = len(rebind_bytes)
    manifest["last_rebind_receipt"]["sha256"] = hashlib.sha256(rebind_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        RuntimeError, match="checkpoint manifest authentication mismatch"
    ):
        _run(moved / "update.pt", forbidden=True)


def test_corrupt_checkpoint_payload_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "corrupt.pt"
    with pytest.raises(Interrupted):
        _run(output, stop_after=0)
    checkpoint = Path(f"{output}.checkpoint")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    payload = checkpoint / manifest["payload_path"]
    payload.write_bytes(payload.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="payload (byte count|SHA-256) mismatch"):
        _run(output)


def test_manifest_replacement_cannot_authorize_malicious_torch_global(
    tmp_path: Path,
) -> None:
    output = tmp_path / "malicious.pt"
    with pytest.raises(Interrupted):
        _run(output, stop_after=0)
    checkpoint = Path(f"{output}.checkpoint")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = checkpoint / manifest["payload_path"]
    marker = tmp_path / "unsafe-global-executed"
    torch.save(_MaliciousPayload(marker), payload)
    payload_bytes = payload.read_bytes()
    manifest["payload_bytes"] = len(payload_bytes)
    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="checkpoint payload cannot be loaded"):
        _run(output)
    assert not marker.exists()


def test_joint_manifest_and_safe_payload_replacement_fails_authentication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "substitution.pt"
    with pytest.raises(Interrupted):
        _run(output, stop_after=0)
    checkpoint = Path(f"{output}.checkpoint")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload_path = checkpoint / manifest["payload_path"]
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload["parameters"][0].add_(1000)
    torch.save(payload, payload_path)
    payload_bytes = payload_path.read_bytes()
    manifest["payload_bytes"] = len(payload_bytes)
    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="checkpoint manifest authentication"):
        _run(output)


def test_checkpoint_run_id_cannot_escape_payload_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="run_id"):
        commit_segment_checkpoint(
            checkpoint,
            {
                "run_id": "../escaped",
                "next_segment_index": 1,
                "state": "accumulating",
                "completed_segments": [0],
            },
            identity=_identity(),
            backend="accelerated",
            segment_plan=[1],
        )
    assert not (tmp_path / "escaped-0001-accumulating.pt").exists()


@pytest.mark.parametrize("invalid", [True, 1.0, 1.5])
def test_checkpoint_segment_plan_requires_integer_geometry(
    tmp_path: Path, invalid: object
) -> None:
    with pytest.raises(TypeError, match=r"segment_plan\[0\] must be an integer"):
        commit_segment_checkpoint(
            tmp_path / "checkpoint",
            {
                "run_id": "a" * 32,
                "next_segment_index": 1,
                "state": "optimizer_pending",
                "completed_segments": [0],
            },
            identity=_identity(),
            backend="accelerated",
            segment_plan=[invalid],
        )


@pytest.mark.parametrize("physical_tokens", [True, 1.0, 1.5])
def test_update_engine_geometry_requires_integer_physical_tokens(
    tmp_path: Path, physical_tokens: object
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(TypeError, match="physical_tokens must be an integer"):
        run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=[torch.ones(1)],
            item_count=lambda values: values.numel(),
            loss_sum=lambda values: (parameter * values).sum(),
            output=tmp_path / "invalid-physical.pt",
            identity=_identity(),
            physical_tokens=physical_tokens,
            observed_input_shape=[1, physical_tokens],
            teacher_geometry={
                "target_shape": [1, physical_tokens],
                "mask_shape": [1, physical_tokens],
                "position_shape": [1, physical_tokens],
            },
            peak_memory_bytes=0,
        )


def test_resume_after_optimizer_checkpoint_preserves_step_and_timing(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "post-step.pt"
    original_finalize = update_engine_module.finalize_checkpoint
    original_step = torch.optim.Adam.step
    finalize_calls = 0
    step_calls = 0

    def interrupt_finalize(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise Interrupted("crash after durable optimizer checkpoint")
        return original_finalize(*args, **kwargs)

    def counted_step(self, *args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(update_engine_module, "finalize_checkpoint", interrupt_finalize)
    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    with pytest.raises(Interrupted, match="durable optimizer checkpoint"):
        _run(output)

    receipt = _run(output)
    assert step_calls == 1
    assert receipt["optimizer_steps"] == 1
    assert receipt["timing"]["optimizer_started_unix"] is not None
    assert receipt["timing"]["optimizer_completed_unix"] is not None
    assert receipt["timing"]["optimizer_seconds"] >= 0


def test_terminal_timing_includes_durable_receipt_and_manifest_publication(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "durable-timing.pt"
    original_finalize = update_engine_module.finalize_checkpoint
    finalized_at: list[float] = []

    def observed_finalize(*args, **kwargs):
        result = original_finalize(*args, **kwargs)
        finalized_at.append(time.time())
        return result

    monkeypatch.setattr(update_engine_module, "finalize_checkpoint", observed_finalize)
    result = _run(output)

    assert finalized_at
    assert result["timing"]["durable_completed_unix"] >= finalized_at[0]
    persisted = json.loads(output.with_suffix(".json").read_text())
    assert persisted["timing"] == result["timing"]


@pytest.mark.parametrize("invalid", [True, 1.0, 1.5])
def test_peak_memory_probe_requires_strict_integer(
    tmp_path: Path, invalid: object
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(TypeError, match="peak_memory_bytes must be an integer"):
        run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=[torch.ones(1)],
            item_count=lambda values: values.numel(),
            loss_sum=lambda values: (parameter * values).sum(),
            output=tmp_path / "invalid-memory.pt",
            identity=_identity(),
            physical_tokens=1,
            observed_input_shape=[1, 1],
            teacher_geometry={
                "target_shape": [1, 1],
                "mask_shape": [1, 1],
                "position_shape": [1, 1],
            },
            peak_memory_bytes=invalid,  # type: ignore[arg-type]
        )


def test_completed_replay_binds_verified_output_to_requested_path(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "expected.pt"
    receipt = output.with_suffix(".json")
    receipt.write_text('{"status":"PASS_UPDATE"}\n')
    decoy = tmp_path / "external-decoy.pt"
    decoy.write_bytes(b"decoy")
    checkpoint = Path(f"{output}.checkpoint")
    checkpoint.mkdir()

    monkeypatch.setattr(
        update_engine_module,
        "load_checkpoint",
        lambda *args, **kwargs: ({}, {"status": "COMPLETE"}),
    )
    monkeypatch.setattr(
        update_engine_module,
        "verify_completed_files",
        lambda *args, **kwargs: (decoy.resolve(), receipt.resolve()),
    )

    with pytest.raises(RuntimeError, match="artifact path mismatch"):
        _run(output, forbidden=True)


@pytest.mark.parametrize(
    "field",
    [
        "content_sha256",
        "config_sha256",
        "assignment_sha256",
        "aot_sha256",
        "runtime_sha256",
        "code_sha256",
    ],
)
def test_resume_rejects_every_immutable_hash_drift(tmp_path: Path, field: str) -> None:
    output = tmp_path / f"drift-{field}.pt"
    with pytest.raises(Interrupted):
        _run(output, stop_after=0)
    changed = _identity()
    changed[field] = "9" * 64

    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        _run(output, identity=changed)
