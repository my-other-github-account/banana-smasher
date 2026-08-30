import hashlib
import importlib
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
    _attach_checkpoint_identity_envelope,
    _load_score_checkpoint,
    _release_or_retain_checkpoint_payload,
    _unique_tensor_storage_bytes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_broker_registry_survives_canonical_module_reload() -> None:
    from repair_api import official_k2_resident_score as resident_score

    checkpoint_sha = "d" * 64
    resident_score._ORDINARY_FORK_PAYLOADS[checkpoint_sha] = {"payload": object()}
    original_registry = resident_score._ORDINARY_FORK_PAYLOADS
    try:
        importlib.reload(resident_score)
        assert resident_score._ORDINARY_FORK_PAYLOADS is original_registry
        assert checkpoint_sha in resident_score._ORDINARY_FORK_PAYLOADS
    finally:
        resident_score._ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)


def test_loaded_broker_payload_remains_release_authority_after_registry_rebind() -> None:
    from repair_api import official_k2_resident_score as resident_score

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({"state": {"weight": torch.arange(8)}}, checkpoint)
        checkpoint_sha = _sha256(checkpoint)
        prepare_ordinary_checkpoint_payload(checkpoint, checkpoint_sha)
        loaded = _load_score_checkpoint(
            checkpoint,
            checkpoint_sha,
            {"checkpoint_mmap": False, "ordinary_load_fork_broker": True,
             "same_process_dual_shard": True},
        )
        rebound = MappingProxyType({
            "state": MappingProxyType({"weight": torch.arange(8)}),
        })
        resident_score._ORDINARY_FORK_PAYLOADS[checkpoint_sha]["payload"] = rebound
        rebound_loaded = _load_score_checkpoint(
            checkpoint,
            checkpoint_sha,
            {"checkpoint_mmap": False, "ordinary_load_fork_broker": True,
             "same_process_dual_shard": True},
        )
        assert rebound_loaded is rebound
        try:
            assert _release_or_retain_checkpoint_payload(
                loaded,
                ordinary_load_fork_broker=True,
                checkpoint_sha256=checkpoint_sha,
            ) == "inherited_read_only"
            copied = MappingProxyType({
                "state": MappingProxyType({"weight": torch.arange(8)}),
            })
            with pytest.raises(ArtifactError, match="identity mismatch"):
                _release_or_retain_checkpoint_payload(
                    copied,
                    ordinary_load_fork_broker=True,
                    checkpoint_sha256=checkpoint_sha,
                )
        finally:
            resident_score._ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)
            getattr(resident_score, "_ORDINARY_FORK_PAYLOAD_LEASES", {}).pop(
                checkpoint_sha, None
            )


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


def test_identity_envelope_promotion_preserves_exact_broker_release_authority() -> None:
    from repair_api import official_k2_resident_score as resident_score

    checkpoint_sha = "f" * 64
    state = MappingProxyType({"weight": torch.arange(4)})
    payload = MappingProxyType({"state": state})
    resident_score._ORDINARY_FORK_PAYLOADS[checkpoint_sha] = {"payload": payload}
    resident_score._ORDINARY_FORK_PAYLOAD_LEASES[checkpoint_sha] = [payload]
    envelope = {"checkpoint_loaded": True, "checkpoint_sha256": checkpoint_sha}
    try:
        adapted = _attach_checkpoint_identity_envelope(
            payload,
            envelope=envelope,
            checkpoint_sha256=checkpoint_sha,
        )

        assert adapted is resident_score._ORDINARY_FORK_PAYLOADS[checkpoint_sha]["payload"]
        assert resident_score._ORDINARY_FORK_PAYLOAD_LEASES[checkpoint_sha] == [adapted]
        assert adapted["state"] is state
        assert adapted["identity"]["checkpoint_sha256"] == checkpoint_sha
        assert _release_or_retain_checkpoint_payload(
            adapted,
            ordinary_load_fork_broker=True,
            checkpoint_sha256=checkpoint_sha,
        ) == "inherited_read_only"
    finally:
        resident_score._ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)
        resident_score._ORDINARY_FORK_PAYLOAD_LEASES.pop(checkpoint_sha, None)


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


def test_local_dual_shard_rebases_read_counters_after_both_ranks_load() -> None:
    source = inspect.getsource(OfficialK2LocalDualShardEngine.__init__)
    rank1 = source.index("self.rank1 = OfficialK2ResidentRankEngine(")
    rank1_loaded = source.index("self.rank0.ready_counter =", rank1)
    rank0_rebased = source.index(
        "self.rank0.read_counter.mark_resident_ready()", rank1_loaded
    )
    rank1_rebased = source.index(
        "self.rank1.read_counter.mark_resident_ready()", rank0_rebased
    )
    scoring_ready = source.index("ready = [self.rank0.local_ready", rank1_rebased)
    assert rank1 < rank1_loaded <= rank0_rebased < rank1_rebased < scoring_ready


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


def test_dual_shard_scorer_load_stays_broker_bound_through_parent_release(monkeypatch) -> None:
    """Reproduce the ordinary-load fork lifecycle across scorer load + release.

    The direct public scorer materializes one hash-bound payload before its
    local dual-shard engine acquires the checkpoint and rank 0 performs its
    post-bind parent release.  ``OfficialK2LocalDualShardEngine`` forces
    ``ordinary_load_fork_broker`` onto its rank configs, so the release
    validates against the broker registry.  The sealed manifest config does not
    carry that flag, so before the fix the load took the ordinary path and
    produced an unleased second payload that the parent release rejected.
    """
    from repair_api import official_k2_resident_score as resident_score

    monkeypatch.setenv("BANANA_SMASHER_SAME_PROCESS_DUAL_SHARD", "1")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "UPDATE_000.pt"
        torch.save({
            "identity": {"checkpoint_loaded": True},
            "next_update": 0,
            "state": {
                "luts": {"L000": torch.arange(4)},
                "norms": {"L000": torch.arange(3)},
                "outputs": {"L000": torch.arange(2)},
            },
        }, checkpoint)
        checkpoint_sha = _sha256(checkpoint)

        scorer = resident_score.OfficialK2ResidentScorer.__new__(
            resident_score.OfficialK2ResidentScorer
        )
        # Sealed manifest config: no broker flags, as delivered to score().
        scorer.config = {"checkpoint_path": str(checkpoint)}
        scorer._checkpoint_loads = 0
        scorer._engine = None
        assert scorer._engine_type() is resident_score.OfficialK2LocalDualShardEngine
        try:
            assert checkpoint_sha not in _ORDINARY_FORK_PAYLOADS
            scorer._align_checkpoint_load_lifecycle(checkpoint_sha)
            payload = _load_score_checkpoint(checkpoint, checkpoint_sha, scorer.config)
            # The acquired payload is the exact registered broker object and is
            # leased, so the ordinary-load fork pages stay shared.
            assert payload is _ORDINARY_FORK_PAYLOADS[checkpoint_sha]["payload"]
            assert any(
                candidate is payload
                for candidate in resident_score._ORDINARY_FORK_PAYLOAD_LEASES[checkpoint_sha]
            )

            # Rank-0 post-bind parent release, with the broker lifecycle the
            # dual-shard engine forces onto its rank configs.
            assert _release_or_retain_checkpoint_payload(
                payload,
                ordinary_load_fork_broker=True,
                checkpoint_sha256=checkpoint_sha,
            ) == "inherited_read_only"

            # A genuinely foreign payload is still rejected.
            with pytest.raises(ArtifactError, match="identity mismatch"):
                _release_or_retain_checkpoint_payload(
                    MappingProxyType({
                        "identity": MappingProxyType({"checkpoint_loaded": True}),
                        "next_update": 0,
                        "state": MappingProxyType({
                            "luts": MappingProxyType({"L000": torch.arange(4)}),
                            "norms": MappingProxyType({"L000": torch.arange(3)}),
                            "outputs": MappingProxyType({"L000": torch.arange(2)}),
                        }),
                    }),
                    ordinary_load_fork_broker=True,
                    checkpoint_sha256=checkpoint_sha,
                )

            # Without a brokered materialization the loader selection is
            # untouched, so non-broker configurations keep their behavior.
            monkeypatch.delenv("BANANA_SMASHER_SAME_PROCESS_DUAL_SHARD")
            other_scorer = resident_score.OfficialK2ResidentScorer.__new__(
                resident_score.OfficialK2ResidentScorer
            )
            other_scorer.config = {"checkpoint_path": str(checkpoint)}
            other_scorer._align_checkpoint_load_lifecycle("e" * 64)
            assert other_scorer.config == {"checkpoint_path": str(checkpoint)}
        finally:
            _ORDINARY_FORK_PAYLOADS.pop(checkpoint_sha, None)
            resident_score._ORDINARY_FORK_PAYLOAD_LEASES.pop(checkpoint_sha, None)
