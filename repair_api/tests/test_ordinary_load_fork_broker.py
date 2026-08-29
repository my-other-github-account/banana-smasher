import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any

import pytest
import torch

from repair_api.balanced64 import ArtifactError
from repair_api.ordinary_load_fork_broker import (
    prepare_ordinary_checkpoint_payload,
    run_forked_rank_children,
    run_same_process_dual_shard,
)
from repair_api.official_k2_resident_score import (
    OfficialK2LocalDualShardEngine,
    OfficialK2ResidentRankEngine,
    _ORDINARY_FORK_PAYLOADS,
    _load_score_checkpoint,
    _release_or_retain_checkpoint_payload,
    _unique_tensor_storage_bytes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_ordinary_materialization_is_inherited_read_only_by_both_ranks(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint.pt"
        torch.save({"state": {"weight": torch.arange(32)}}, checkpoint)
        expected_sha = _sha256(checkpoint)
        calls = root / "ordinary-load-calls.jsonl"
        real_load = torch.load

        def recording_load(path, **kwargs):
            with calls.open("a") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "path": str(path)}) + "\n")
            return real_load(path, **kwargs)

        monkeypatch.setattr(
            "repair_api.official_k2_resident_score._load_torch",
            lambda path: recording_load(path, map_location="cpu", weights_only=True),
        )
        receipt = prepare_ordinary_checkpoint_payload(checkpoint, expected_sha)
        assert receipt["materialization_pid"] == os.getpid()
        assert receipt["checkpoint_sha256"] == expected_sha

        read_fd, write_fd = os.pipe()
        children = []
        for rank in (0, 1):
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    payload = _load_score_checkpoint(
                        checkpoint,
                        expected_sha,
                        {"checkpoint_mmap": False, "ordinary_load_fork_broker": True},
                    )
                    with pytest.raises(TypeError):
                        payload["new"] = rank
                    row = {
                        "rank": rank,
                        "pid": os.getpid(),
                        "parent_pid": os.getppid(),
                        "value": payload["state"]["weight"].tolist(),
                    }
                    os.write(write_fd, (json.dumps(row) + "\n").encode())
                    os._exit(0)
                except BaseException:
                    os._exit(91)
            children.append(pid)
        os.close(write_fd)
        rows = [json.loads(line) for line in os.fdopen(read_fd).read().splitlines()]
        statuses = [os.waitpid(pid, 0)[1] for pid in children]

        assert all(os.waitstatus_to_exitcode(status) == 0 for status in statuses)
        assert [row["rank"] for row in sorted(rows, key=lambda row: row["rank"])] == [0, 1]
        assert all(row["parent_pid"] == os.getpid() for row in rows)
        assert all(row["value"] == list(range(32)) for row in rows)
        load_rows = [json.loads(line) for line in calls.read_text().splitlines()]
        assert load_rows == [{"pid": os.getpid(), "path": str(checkpoint.resolve())}]


def test_broker_refuses_wrong_or_changed_checkpoint_hash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({"state": {"weight": torch.arange(8)}}, checkpoint)
        expected_sha = _sha256(checkpoint)
        with pytest.raises(ArtifactError, match="checkpoint source SHA mismatch"):
            prepare_ordinary_checkpoint_payload(checkpoint, "0" * 64)

        prepare_ordinary_checkpoint_payload(checkpoint, expected_sha)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
        with pytest.raises(ArtifactError, match="checkpoint source SHA mismatch"):
            _load_score_checkpoint(
                checkpoint,
                expected_sha,
                {"checkpoint_mmap": False, "ordinary_load_fork_broker": True},
            )


def test_inherited_read_only_payload_is_retained_without_child_clear() -> None:
    payload = MappingProxyType({"state": MappingProxyType({"weight": torch.arange(4)})})

    result = _release_or_retain_checkpoint_payload(payload, ordinary_load_fork_broker=True)

    assert result == "inherited_read_only"
    assert payload["state"]["weight"].tolist() == [0, 1, 2, 3]


def test_registered_broker_identity_is_the_read_only_authority() -> None:
    checkpoint_sha = "a" * 64
    payload = {"state": MappingProxyType({"weight": torch.arange(4)})}
    _ORDINARY_FORK_PAYLOADS[checkpoint_sha] = {"payload": payload}
    try:
        result = _release_or_retain_checkpoint_payload(
            payload,
            ordinary_load_fork_broker=True,
            checkpoint_sha256=checkpoint_sha,
        )
        assert result == "inherited_read_only"

        adapted: dict[str, Any] = dict(payload)
        adapted["identity"] = MappingProxyType({"checkpoint_loaded": True})
        assert _release_or_retain_checkpoint_payload(
            adapted,
            ordinary_load_fork_broker=True,
            checkpoint_sha256=checkpoint_sha,
        ) == "inherited_read_only"

        with pytest.raises(ArtifactError, match="identity mismatch"):
            _release_or_retain_checkpoint_payload(
                {"state": MappingProxyType(dict(payload["state"]))},
                ordinary_load_fork_broker=True,
                checkpoint_sha256=checkpoint_sha,
            )
    finally:
        _ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)


def test_registered_broker_accepts_authenticated_state_mapping_view() -> None:
    checkpoint_sha = "c" * 64
    identity = MappingProxyType({"schema": "published-pre", "checkpoint_loaded": True})
    state = MappingProxyType({
        "luts": MappingProxyType({"L000": torch.arange(4)}),
        "norms": MappingProxyType({"L000": torch.arange(3)}),
        "outputs": MappingProxyType({"L000": torch.arange(2)}),
    })
    registered = MappingProxyType({"identity": identity, "state": state, "next_update": 0})
    _ORDINARY_FORK_PAYLOADS[checkpoint_sha] = {"payload": registered}
    try:
        viewed = {"identity": identity, "state": dict(state), "next_update": registered["next_update"]}

        assert _release_or_retain_checkpoint_payload(
            viewed,
            ordinary_load_fork_broker=True,
            checkpoint_sha256=checkpoint_sha,
        ) == "inherited_read_only"

        viewed["state"]["luts"] = MappingProxyType(dict(state["luts"]))
        with pytest.raises(ArtifactError, match="identity mismatch"):
            _release_or_retain_checkpoint_payload(
                viewed,
                ordinary_load_fork_broker=True,
                checkpoint_sha256=checkpoint_sha,
            )
    finally:
        _ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)


def test_registered_broker_accepts_authenticated_state_surface_projection() -> None:
    checkpoint_sha = "b" * 64
    luts = MappingProxyType({"L000": torch.arange(4)})
    norms = MappingProxyType({"L000": torch.arange(3)})
    outputs = MappingProxyType({"L000": torch.arange(2)})
    planes = MappingProxyType({"L028": torch.arange(5)})
    registered_state = MappingProxyType({
        "luts": luts,
        "norms": norms,
        "outputs": outputs,
        "expert_planes_l028_su_sv": planes,
    })
    registered = MappingProxyType({
        "state": registered_state,
        "identity": MappingProxyType({"checkpoint_loaded": True}),
        "next_update": 1,
    })
    _ORDINARY_FORK_PAYLOADS[checkpoint_sha] = {"payload": registered}
    try:
        projected = {
            "state": {surface: registered_state[surface] for surface in ("luts", "norms", "outputs")},
            "identity": MappingProxyType({"checkpoint_loaded": True}),
            "next_update": registered["next_update"],
        }

        assert _release_or_retain_checkpoint_payload(
            projected,
            ordinary_load_fork_broker=True,
            checkpoint_sha256=checkpoint_sha,
        ) == "inherited_read_only"

        projected["state"]["luts"] = MappingProxyType(dict(luts))
        with pytest.raises(ArtifactError, match="identity mismatch"):
            _release_or_retain_checkpoint_payload(
                projected,
                ordinary_load_fork_broker=True,
                checkpoint_sha256=checkpoint_sha,
            )
    finally:
        _ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)


def test_same_process_rank_boundary_releases_allocator_before_rank1() -> None:
    source = inspect.getsource(OfficialK2LocalDualShardEngine.__init__)
    rank0 = source.index("self.rank0 = OfficialK2ResidentRankEngine(")
    release = source.index("self.rank0._release_transient_resident_load_workspace()")
    receipt = source.index("coordinator.post_rank0_workspace_release = {")
    rank1 = source.index("self.rank1 = OfficialK2ResidentRankEngine(")
    assert rank0 < release < receipt < rank1
    assert '"required_cuda_free_bytes"' in source
    assert '"margin_bytes"' in source


def test_rank1_preflight_deducts_exact_brokered_storage_from_incremental_peak() -> None:
    owner = torch.arange(16)
    alias = owner.view(4, 4)
    distinct = torch.arange(7)
    assert _unique_tensor_storage_bytes({
        "owner": owner,
        "aliases": (alias, owner),
        "distinct": distinct,
    }) == owner.untyped_storage().nbytes() + distinct.untyped_storage().nbytes()

    source = inspect.getsource(OfficialK2ResidentRankEngine._preflight_memory)
    assert 'self.config.get("gpu_resident_storage_broker", False)' in source
    assert "self.gpu_resident_storage_broker" not in source
    inventory = source.index("_unique_tensor_storage_bytes")
    deduction = source.index(
        "incremental_peak = max(incremental_expected, peak - brokered_resident_bytes)"
    )
    required = source.index("required = incremental_peak + reserve")
    assert inventory < deduction < required


def test_nonbroker_payload_still_transfers_ownership_by_clear() -> None:
    payload = {"state": {"weight": torch.arange(4)}}

    result = _release_or_retain_checkpoint_payload(payload, ordinary_load_fork_broker=False)

    assert result == "cleared_child_owned"
    assert payload == {}


def test_same_process_dual_shard_owns_both_local_ranks_without_fork(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(os, "fork", lambda: (_ for _ in ()).throw(AssertionError("forked")))

    result = run_same_process_dual_shard(lambda rank: calls.append((os.getpid(), rank)) or 0)

    assert calls == [(os.getpid(), 1)]
    assert result["pids"] == {0: os.getpid(), 1: os.getpid()}
    assert result["same_process_dual_shard"] is True
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({"state": {"weight": torch.arange(4)}}, checkpoint)
        expected_sha = _sha256(checkpoint)
        prepare_ordinary_checkpoint_payload(checkpoint, expected_sha)
        payload = _load_score_checkpoint(checkpoint, expected_sha, {
            "ordinary_load_fork_broker": True,
            "checkpoint_mmap": False,
            "same_process_dual_shard": True,
        })
        assert payload["state"]["weight"].tolist() == [0, 1, 2, 3]
    source = inspect.getsource(OfficialK2ResidentRankEngine._score_window)
    assert "if (shared_cuda and not local_cuda) or network_pipeline:" in source


def test_failed_rank_terminates_and_reaps_its_peer_without_orphaning() -> None:
    observed: dict[int, int] = {}

    def rank_main(rank: int) -> int:
        if rank == 0:
            return 17
        time.sleep(30)
        return 0

    result = run_forked_rank_children(rank_main, ranks=(0, 1))
    observed.update(result["exit_codes"])

    assert observed[0] == 17
    assert observed[1] < 0
    assert result["status"] == "FAIL"
    assert result["all_children_reaped"] is True
    for pid in result["pids"].values():
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def test_rank_exception_is_reported_before_child_failure(capfd) -> None:
    def rank_main(rank: int) -> int:
        if rank == 0:
            raise RuntimeError("ordinary fork rank boom")
        time.sleep(30)
        return 0

    result = run_forked_rank_children(rank_main, ranks=(0, 1))
    captured = capfd.readouterr()

    assert result["exit_codes"][0] == 125
    assert "RuntimeError: ordinary fork rank boom" in captured.err
