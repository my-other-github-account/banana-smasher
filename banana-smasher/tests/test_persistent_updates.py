from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

import banana_smasher.persistent as persistent_module
from banana_smasher.persistent import (
    DuplicateSegment,
    _UpdateQueue as UpdateQueue,
    _atomic_json,
    _serve_queue as serve_queue,
    recover_committed_cycle,
    request_identity,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _request(request_id: str, input_sha: str, *, config_sha: str, aot_sha: str) -> dict[str, object]:
    return {
        "schema": "banana-smasher-update-request-v1",
        "segment_id": request_id,
        "input_checkpoint": f"/checkpoints/{input_sha}.pt",
        "input_checkpoint_sha256": input_sha,
        "config_sha256": config_sha,
        "aot_sha256": aot_sha,
        "payload": {"fixture": True},
    }


def _cycle_result(status: str, output_sha: str) -> dict[str, object]:
    return {
        "status": status,
        "output_checkpoint": f"/checkpoints/{output_sha}.pt",
        "output_checkpoint_sha256": output_sha,
        "phase_seconds": {
            "decode": 0.0,
            "staging_resident_layout": 0.0,
            "kernel_forward": 0.01,
            "backward": 0.02,
            "optimizer": 0.01,
            "checkpoint": 0.01,
            "total": 0.05,
        },
        "rchar_delta": 0,
        "aot_engaged": True,
        "loss": 1.0,
        "memory_floor_bytes": 1024,
        "fallback_used": False,
    }


def test_persistent_worker_initializes_once_for_multiple_cycles(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    initial_sha = _sha("checkpoint-0")
    next_sha = _sha("checkpoint-1")
    final_sha = _sha("checkpoint-2")
    queue.enqueue(_request("cycle-1", initial_sha, config_sha=config_sha, aot_sha=aot_sha))
    queue.enqueue(_request("cycle-2", next_sha, config_sha=config_sha, aot_sha=aot_sha))

    calls = {"initialize": 0, "cycle": 0}

    def initialize() -> dict[str, object]:
        calls["initialize"] += 1
        return {"checkpoint_sha256": initial_sha, "decoded_once": True}

    def cycle(worker: dict[str, object], request: dict[str, object]) -> dict[str, object]:
        calls["cycle"] += 1
        output_sha = next_sha if request["request_id"] == "cycle-1" else final_sha
        worker["checkpoint_sha256"] = output_sha
        return {
            "status": "PASS",
            "output_checkpoint": f"/checkpoints/{output_sha}.pt",
            "output_checkpoint_sha256": output_sha,
            "phase_seconds": {
                "decode": 0.0,
                "staging_resident_layout": 0.0,
                "kernel_forward": 0.01,
                "backward": 0.02,
                "optimizer": 0.01,
                "checkpoint": 0.01,
                "total": 0.05,
            },
            "rchar_delta": 0,
            "aot_engaged": True,
            "loss": 1.25,
            "memory_floor_bytes": 1024,
            "fallback_used": False,
        }

    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=initialize,
        cycle=cycle,
        stop_after=2,
        poll_seconds=0.0,
    )

    assert calls == {"initialize": 1, "cycle": 2}
    assert summary["status"] == "PASS_STOP_AFTER"
    assert summary["worker_pid"] == os.getpid()
    assert summary["cycles_completed"] == 2
    assert queue.status("cycle-1")["state"] == "PASS"
    assert queue.status("cycle-2")["state"] == "PASS"
    assert queue.status("cycle-1")["worker_pid"] == queue.status("cycle-2")["worker_pid"]
    assert queue.status("cycle-2")["output_checkpoint_sha256"] == final_sha
    init = __import__("json").loads((queue.root / "INIT.json").read_text())
    assert init["checkpoint_sha256"] == initial_sha


def test_enqueue_refuses_completed_segment_without_rewriting_receipt(
    tmp_path: Path,
) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    request = _request(
        "dedup",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )
    queue.enqueue(request)
    queue.write_state(request, "RUNNING", worker_pid=123)
    queue.write_state(request, "PASS", output_checkpoint_sha256=_sha("checkpoint-1"))
    receipt = queue.requests / "dedup.json"
    before = receipt.read_bytes()

    with pytest.raises(DuplicateSegment, match="state=COMPLETED"):
        queue.enqueue(request)

    observed = queue.status("dedup")
    assert observed["state"] == "PASS"
    assert receipt.read_bytes() == before
    assert queue.ledger()["segments"]["dedup"]["state"] == "COMPLETED"


def test_identity_mismatch_fails_loud_without_running_cycle(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    initial_sha = _sha("checkpoint-0")
    queue.enqueue(
        _request(
            "wrong-config",
            initial_sha,
            config_sha=_sha("other-config"),
            aot_sha=aot_sha,
        )
    )
    cycle_calls = 0

    def cycle(_worker: dict[str, object], _request: dict[str, object]) -> dict[str, object]:
        nonlocal cycle_calls
        cycle_calls += 1
        raise AssertionError("identity mismatch must not run a cycle")

    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=lambda: {"checkpoint_sha256": initial_sha},
        cycle=cycle,
        poll_seconds=0.001,
        idle_timeout_seconds=0.01,
    )

    status = queue.status("wrong-config")
    assert summary["status"] == "FAIL_IDLE_TIMEOUT"
    assert status["state"] == "FAIL"
    assert status["error_type"] == "IdentityMismatch"
    assert "config identity mismatch" in status["error"]
    assert cycle_calls == 0


def test_request_identity_binds_adapter_visible_top_level_fields() -> None:
    request = _request(
        "bound-fields",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )

    assert request_identity({**request, "learning_rate": 0.001}) != request_identity(
        {**request, "learning_rate": 0.01}
    )


def test_cycle_cannot_pass_without_advancing_resident_checkpoint(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    initial_sha = _sha("checkpoint-0")
    output_sha = _sha("checkpoint-1")
    queue.enqueue(
        _request("stale-resident", initial_sha, config_sha=config_sha, aot_sha=aot_sha)
    )

    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=lambda: {"checkpoint_sha256": initial_sha},
        cycle=lambda _worker, _request: _cycle_result("PASS", output_sha),
        stop_after=1,
        poll_seconds=0.0,
    )

    assert summary["status"] == "FAIL_STOP_AFTER"
    status = queue.status("stale-resident")
    assert status["state"] == "FAIL"
    assert "resident checkpoint identity mismatch" in status["error"]


def test_cycle_result_cannot_override_canonical_receipt_identity(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    initial_sha = _sha("checkpoint-0")
    output_sha = _sha("checkpoint-1")
    queue.enqueue(
        _request("canonical-receipt", initial_sha, config_sha=config_sha, aot_sha=aot_sha)
    )

    def cycle(worker: dict[str, object], _request: dict[str, object]) -> dict[str, object]:
        worker["checkpoint_sha256"] = output_sha
        return {
            **_cycle_result("PASS", output_sha),
            "segment_id": "forged-segment",
            "request_identity_sha256": "0" * 64,
        }

    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=lambda: {"checkpoint_sha256": initial_sha},
        cycle=cycle,
        stop_after=1,
        poll_seconds=0.0,
    )

    assert summary["status"] == "FAIL_STOP_AFTER"
    status = queue.status("canonical-receipt")
    assert status["state"] == "FAIL"
    assert status["segment_id"] == "canonical-receipt"
    assert status["request_identity_sha256"] != "0" * 64
    assert "reserved receipt fields" in status["error"]


def test_crash_restart_finalizes_committed_running_request_without_replay(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    initial_sha = _sha("checkpoint-0")
    output_sha = _sha("checkpoint-1")
    request = _request("crash-cycle", initial_sha, config_sha=config_sha, aot_sha=aot_sha)
    queue.enqueue(request)
    queue.write_state(request, "RUNNING", worker_pid=999, started_unix=1.0)
    cycle_calls = 0

    def cycle(_worker: dict[str, object], _request: dict[str, object]) -> dict[str, object]:
        nonlocal cycle_calls
        cycle_calls += 1
        raise AssertionError("a checkpointed request must not replay")

    recovered = _cycle_result("PASS", output_sha)
    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=lambda: {"checkpoint_sha256": output_sha},
        cycle=cycle,
        recover=lambda _worker, row: recovered if row["request_id"] == "crash-cycle" else None,
        stop_after=1,
        poll_seconds=0.0,
    )

    assert summary["cycles_completed"] == 1
    assert cycle_calls == 0
    status = queue.status("crash-cycle")
    assert status["state"] == "PASS"
    assert status["recovered_after_crash"] is True


def test_generic_recovery_seals_committed_wall_failure_without_replay(
    tmp_path: Path,
) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    config_sha = _sha("config")
    aot_sha = _sha("aot")
    input_sha = _sha("checkpoint-0")
    output_sha = _sha("checkpoint-1")
    request = _request(
        "crash-wall-failure",
        input_sha,
        config_sha=config_sha,
        aot_sha=aot_sha,
    )
    queue.enqueue(request)
    queue.write_state(request, "RUNNING", worker_pid=999)

    summary = serve_queue(
        queue,
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=lambda: {"checkpoint_sha256": output_sha},
        cycle=lambda _worker, _request: (_ for _ in ()).throw(
            AssertionError("committed wall failure must not replay")
        ),
        recover=lambda _worker, _request: _cycle_result(
            "FAIL_MAX_SEGMENT_SECONDS", output_sha
        ),
        stop_after=1,
        poll_seconds=0.0,
    )

    assert summary["cycles_completed"] == 0
    assert summary["cycles_failed"] == 1
    status = queue.status("crash-wall-failure")
    assert status["state"] == "FAIL"
    assert status["error_type"] == "SegmentWallExceeded"
    entry = queue.ledger()["segments"]["crash-wall-failure"]
    assert len(entry["attempts"]) == 1


def test_exclusive_atomic_json_does_not_overwrite_after_stale_existence_check(
    tmp_path: Path, monkeypatch,
) -> None:
    destination = tmp_path / "request.json"
    destination.write_text('{"owner":"first"}\n')
    original_exists = Path.exists

    def stale_exists(path: Path) -> bool:
        if path == destination:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", stale_exists)

    try:
        _atomic_json(destination, {"owner": "second"}, exclusive=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("exclusive atomic write replaced an existing request")

    assert destination.read_text() == '{"owner":"first"}\n'


def test_concurrent_submitters_admit_exactly_one_segment(
    tmp_path: Path,
) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    request = _request(
        "concurrent",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )
    barrier = threading.Barrier(8)

    def submit(_index: int) -> str:
        barrier.wait(timeout=5)
        try:
            return str(queue.enqueue(request)["state"])
        except DuplicateSegment:
            return "DUPLICATE_REFUSED"

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(submit, range(8)))

    assert rows.count("QUEUED") == 1
    assert rows.count("DUPLICATE_REFUSED") == 7
    assert queue.get("concurrent")["input_checkpoint_sha256"] == request[
        "input_checkpoint_sha256"
    ]
    assert list(queue.ledger()["segments"]) == ["concurrent"]


def test_queue_worker_lock_prevents_concurrent_servers_and_releases_cleanly(
    tmp_path: Path,
) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    first = queue.acquire_worker_lock()
    try:
        try:
            queue.acquire_worker_lock()
        except RuntimeError as exc:
            assert "already has a live worker" in str(exc)
        else:
            raise AssertionError("a second persistent worker acquired the same queue")
    finally:
        first.close()

    restarted = queue.acquire_worker_lock()
    restarted.close()


def test_status_rejects_request_id_path_traversal(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    try:
        queue.status("../outside")
    except ValueError as exc:
        assert "invalid request_id" in str(exc)
    else:
        raise AssertionError("status accepted a traversal request id")


def test_pending_recovers_from_canonical_ledger_without_projection_files(tmp_path: Path) -> None:
    queue = UpdateQueue(tmp_path / "queue")
    request = _request(
        "orphan",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )
    queue.enqueue(request)
    (queue.requests / "orphan.json").unlink()
    (queue.receipts / "orphan.QUEUED.json").unlink()

    pending = queue.pending()

    assert [item["request_id"] for item in pending] == ["orphan"]
    assert queue.status("orphan")["state"] == "QUEUED"


def test_resume_reuses_inflight_attempt_without_allocating_a_second_attempt(
    tmp_path: Path,
) -> None:
    request = _request(
        "resume-one-attempt",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )
    first = UpdateQueue(tmp_path / "run")
    first.enqueue(request)
    claimed = first.claim_segment(request)

    resumed = UpdateQueue(tmp_path / "run").claim_segment(request)

    assert claimed["resumed"] is False
    assert resumed["resumed"] is True
    assert resumed["attempt_id"] == claimed["attempt_id"]
    entry = first.ledger()["segments"]["resume-one-attempt"]
    assert entry["state"] == "INFLIGHT"
    assert len(entry["attempts"]) == 1


def test_worker_exit_after_ledger_seal_cannot_reopen_completed_segment(
    tmp_path: Path, monkeypatch,
) -> None:
    request = _request(
        "sealed-before-receipt",
        _sha("checkpoint-0"),
        config_sha=_sha("config"),
        aot_sha=_sha("aot"),
    )
    queue = UpdateQueue(tmp_path / "run")
    queue.enqueue(request)
    queue.write_state(request, "RUNNING", worker_pid=456)
    real_atomic_json = persistent_module._atomic_json
    pass_receipt = queue.receipts / "sealed-before-receipt.PASS.json"

    def exit_after_seal(path: Path, value: object, *, exclusive: bool = False) -> str:
        if path == pass_receipt:
            raise OSError("simulated worker exit after ledger seal")
        return real_atomic_json(path, value, exclusive=exclusive)

    monkeypatch.setattr(persistent_module, "_atomic_json", exit_after_seal)
    with pytest.raises(OSError, match="worker exit"):
        queue.write_state(request, "PASS", output_checkpoint_sha256=_sha("checkpoint-1"))
    monkeypatch.setattr(persistent_module, "_atomic_json", real_atomic_json)

    restarted = UpdateQueue(tmp_path / "run")
    recovered_status = restarted.status("sealed-before-receipt")
    assert recovered_status["state"] == "PASS"
    assert recovered_status["output_checkpoint_sha256"] == _sha("checkpoint-1")
    with pytest.raises(DuplicateSegment, match="state=COMPLETED"):
        restarted.enqueue(request)
    entry = restarted.ledger()["segments"]["sealed-before-receipt"]
    assert len(entry["attempts"]) == 1


def test_legacy_projection_queue_migrates_without_orphan_or_replay(
    tmp_path: Path,
) -> None:
    queue = UpdateQueue(tmp_path / "run")
    legacy_request = {
        **_request(
            "legacy-complete",
            _sha("checkpoint-0"),
            config_sha=_sha("config"),
            aot_sha=_sha("aot"),
        ),
        "queued_unix": 1.0,
    }
    legacy_request["request_id"] = legacy_request.pop("segment_id")
    _atomic_json(queue.requests / "legacy-complete.json", legacy_request, exclusive=True)
    _atomic_json(
        queue.receipts / "legacy-complete.PASS.json",
        {
            "schema": "banana-smasher-update-queue-receipt-v1",
            "state": "PASS",
            "status": "PASS",
            "request_id": "legacy-complete",
            "worker_pid": 789,
            "updated_unix": 2.0,
        },
        exclusive=True,
    )

    migrated = queue.ledger()
    entry = migrated["segments"]["legacy-complete"]
    assert entry["state"] == "COMPLETED"
    assert len(entry["attempts"]) == 1
    with pytest.raises(DuplicateSegment, match="state=COMPLETED"):
        queue.enqueue(legacy_request)

    worker_lock = queue.acquire_worker_lock()
    worker_lock.close()
    assert queue.ledger_path.is_file()
    persisted = json.loads(queue.ledger_path.read_text())
    assert persisted["migration"]["request_projections"] == 1
    assert persisted["segments"]["legacy-complete"]["state"] == "COMPLETED"


def test_recover_committed_cycle_uses_restored_update_index_sidecar_not_latest_alias(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    output_sha = _sha("checkpoint-1")
    result = {
        "status": "PASS",
        "output_checkpoint": str(checkpoints / "UPDATE_001.pt"),
        "output_checkpoint_sha256": output_sha,
        "phase_seconds": {
            "decode": 0.0,
            "staging_resident_layout": 0.0,
            "kernel_forward": 0.01,
            "backward": 0.02,
            "optimizer": 0.01,
            "checkpoint": 0.01,
            "total": 0.05,
        },
        "rchar_delta": 0,
        "aot_engaged": True,
        "loss": 1.0,
        "memory_floor_bytes": 1024,
        "fallback_used": False,
    }
    (checkpoints / "UPDATE_001.json").write_text(
        __import__("json").dumps(
            {
                "persistent_request_id": "cycle-1",
                "persistent_result": result,
            }
        )
        + "\n"
    )

    observed = recover_committed_cycle(
        checkpoints,
        update=1,
        request_id="cycle-1",
        checkpoint_sha256=output_sha,
    )

    assert observed == result

    failed_result = {
        **result,
        "status": "FAIL_MAX_SEGMENT_SECONDS",
        "output_checkpoint": str(checkpoints / "UPDATE_002.pt"),
        "output_checkpoint_sha256": _sha("checkpoint-2"),
    }
    (checkpoints / "UPDATE_002.json").write_text(
        json.dumps(
            {
                "persistent_request_id": "cycle-2",
                "persistent_result": failed_result,
            }
        )
        + "\n"
    )
    recovered_failure = recover_committed_cycle(
        checkpoints,
        update=2,
        request_id="cycle-2",
        checkpoint_sha256=_sha("checkpoint-2"),
    )
    assert recovered_failure == failed_result