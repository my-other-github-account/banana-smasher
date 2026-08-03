from __future__ import annotations

import copy
import hashlib
import shutil
import sys
import types

import numpy as np
import pytest

from banana_smasher.checkpoint_rebind import (
    build_checkpoint_identity_rebind_receipt,
    validate_checkpoint_identity_rebind,
)
from banana_smasher.cli import _parser, main
from banana_smasher.persistent import UpdateQueue
from banana_smasher.update_inputs import prepare_physical_batch
from banana_smasher.update_checkpoint import commit_segment_checkpoint, load_checkpoint


def _identity(root: str, claim: str = "c") -> dict[str, object]:
    return {
        "assignment": f"/{root}/inputs/assignment.json",
        "assignment_sha256": "a" * 64,
        "assignment_stat": {
            "bytes": 17,
            "mtime_ns": 111,
            "ctime_ns": 222,
            "device": 1,
            "inode": 2,
        },
        "config": f"/{root}/config/runtime.json",
        "config_sha256": "b" * 64,
        "claim_sha256": claim * 64,
    }


def test_public_parser_exposes_update_evaluate_and_queue_verbs() -> None:
    parser = _parser()
    help_text = parser.format_help()
    for command in ("update", "evaluate", "update-enqueue", "update-status"):
        assert command in help_text


def test_physical_batch_preserves_tokens_masks_and_positions() -> None:
    batch = prepare_physical_batch(
        np.array([[11, 12, 13], [21, 22, 0]], dtype=np.int64),
        attention_mask=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int8),
        position_ids=np.array([[4, 5, 6], [8, 9, 0]], dtype=np.int64),
        teacher_mask=np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int8),
    )
    assert batch.input_ids.tolist() == [[11, 12, 13], [21, 22, 0]]
    assert batch.attention_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.position_ids.tolist() == [[4, 5, 6], [8, 9, 0]]
    assert batch.teacher_mask.tolist() == [[False, True, True], [True, False, False]]
    assert batch.physical_token_count == 5
    assert batch.teacher_token_count == 3


def test_physical_batch_defaults_are_explicit_and_fail_closed() -> None:
    batch = prepare_physical_batch(np.array([[3, 4, 5]], dtype=np.int32))
    assert batch.attention_mask.tolist() == [[True, True, True]]
    assert batch.position_ids.tolist() == [[0, 1, 2]]
    assert batch.teacher_mask.tolist() == [[True, True, True]]
    with pytest.raises(ValueError, match="teacher mask selects a non-attended token"):
        prepare_physical_batch(
            np.array([[3, 4]]),
            attention_mask=np.array([[1, 0]]),
            teacher_mask=np.array([[1, 1]]),
        )


def test_checkpoint_relocation_requires_exact_receipt() -> None:
    old = _identity("old")
    current = copy.deepcopy(old)
    current["assignment"] = "/new/inputs/assignment.json"
    current["config"] = "/new/config/runtime.json"
    current["claim_sha256"] = "d" * 64
    current["assignment_stat"]["ctime_ns"] = 333
    current["assignment_stat"]["device"] = 4
    current["assignment_stat"]["inode"] = 5
    sidecar_sha = hashlib.sha256(b"sidecar").hexdigest()
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    with pytest.raises(RuntimeError, match="approved checkpoint identity rebind receipt required"):
        validate_checkpoint_identity_rebind(
            old_identity=old,
            current_identity=current,
            receipt=None,
            sidecar_sha256=sidecar_sha,
            checkpoint_sha256=checkpoint_sha,
        )
    receipt = build_checkpoint_identity_rebind_receipt(
        old_identity=old,
        current_identity=current,
        sidecar_sha256=sidecar_sha,
        checkpoint_sha256=checkpoint_sha,
        task_id="t_6fe6b7a6",
    )
    validate_checkpoint_identity_rebind(
        old_identity=old,
        current_identity=current,
        receipt=receipt,
        sidecar_sha256=sidecar_sha,
        checkpoint_sha256=checkpoint_sha,
    )


def test_checkpoint_rebind_rejects_immutable_drift() -> None:
    old = _identity("old")
    current = copy.deepcopy(old)
    current["assignment_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="immutable checkpoint identity drift"):
        build_checkpoint_identity_rebind_receipt(
            old_identity=old,
            current_identity=current,
            sidecar_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
            task_id="t_6fe6b7a6",
        )


def test_queue_heartbeat_reports_waiting(tmp_path) -> None:
    queue = UpdateQueue(tmp_path / "run")
    row = queue.heartbeat("WAITING", initialized=True)
    assert row["state"] == "WAITING"
    assert row["segment_queue"].endswith("SEGMENT_QUEUE.json")


def test_public_update_serve_initializes_once_and_exposes_waiting(
    tmp_path, monkeypatch, capsys
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.json"
    aot = tmp_path / "kernel.so"
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"{}\n")
    aot.write_bytes(b"aot")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    calls = {"initialize": 0}
    adapter = types.ModuleType("banana_smasher.test_update_adapter")

    def initialize(**_kwargs):
        calls["initialize"] += 1
        return {"checkpoint_sha256": sha(checkpoint)}

    def cycle(*_args):
        raise AssertionError("empty queue must not execute a cycle")

    setattr(adapter, "initialize", initialize)
    setattr(adapter, "cycle", cycle)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    assert main(
        [
            "update",
            "--serve",
            "--queue-root",
            str(tmp_path / "run"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--aot",
            str(aot),
            "--adapter",
            adapter.__name__,
            "--config-sha256",
            sha(config),
            "--aot-sha256",
            sha(aot),
            "--idle-timeout-seconds",
            "0",
        ]
    ) == 0
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["status"] == "PASS_IDLE_TIMEOUT"
    assert calls["initialize"] == 1
    assert UpdateQueue(tmp_path / "run").heartbeat("WAITING")["state"] == "WAITING"


def test_segment_checkpoint_survives_directory_relocation(tmp_path) -> None:
    source = tmp_path / "source" / "checkpoint"
    identity = {"model_index_sha256": "a" * 64}
    commit_segment_checkpoint(
        source,
        {
            "run_id": "portable",
            "state": "accumulating",
            "next_segment_index": 1,
            "completed_segments": [0],
            "optimizer_steps": 0,
            "value": 17,
        },
        identity=identity,
        backend="accelerated",
        segment_plan=[3, 3],
    )
    relocated = tmp_path / "relocated" / "checkpoint"
    shutil.copytree(source, relocated)
    shutil.rmtree(source.parent)
    payload, manifest = load_checkpoint(
        relocated,
        expected_identity=identity,
        expected_backend="accelerated",
        expected_segment_plan=[3, 3],
    )
    assert payload["value"] == 17
    assert manifest["payload_path"].startswith("payload-")
    assert not __import__("pathlib").Path(manifest["payload_path"]).is_absolute()
