import hashlib
import json
from pathlib import Path

from repair_api.balanced64 import ScoreResult
from repair_api import static_w28_resident
from repair_api.api import _localize_official_k2_rank_seat


def _write_reference(path: Path) -> str:
    path.write_text(json.dumps({
        "schema": "banana-smasher-sealed-2x2-cell-v1",
        "status": "PASS",
        "basis_sha256": static_w28_resident.BASIS_SHA256,
        "loaded_sha": static_w28_resident.CHECKPOINT_SHA256,
        "kld_mean": 0.1364830042977786,
        "top1": 880,
        "windows": 1,
    }))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_static_w28_calls_existing_resident_scorer_once_and_seals_truth(monkeypatch, tmp_path) -> None:
    reference = tmp_path / "B2_PUBLISHED_PRE.json"
    reference_sha = _write_reference(reference)
    seen = {}

    class API:
        def score(self, checkpoint, windows):
            seen["checkpoint"] = checkpoint
            seen["windows"] = tuple(windows)
            return ScoreResult(
                checkpoint=checkpoint,
                windows=(28,),
                positions=1024,
                support=8192,
                kld=0.1364830042977786,
                top1=880,
                top1_rate=880 / 1024,
                artifact_root=str(tmp_path / "artifact"),
                spec="balanced64-v1",
                candidate_dir="fully-resident-official-k2",
                execution_mode="resident_in_memory",
                resident_load_seconds=179.0,
                timed_wall_seconds=23.0,
                identity={
                    "checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256,
                    "model_index_sha256": static_w28_resident.BASIS_SHA256,
                },
                runtime_counters={
                    "resident_engine_loads": 1,
                    "resident_checkpoint_rebinds": 0,
                    "timed_score_file_reads": 0,
                    "resident_ready": [{"checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256}],
                },
            )

    monkeypatch.setattr(
        static_w28_resident.ResidentRepairAPI,
        "open",
        lambda root, official_rank_seat=None: API(),
    )
    monkeypatch.setattr(
        static_w28_resident.sealed_pre_forward,
        "source_binding",
        lambda: {"status": "PASS", "builder_sha256": "builder", "known_value_fixture": {
            "window": 28, "kld_mean": 0.1364830042977786, "top1": 880,
        }},
    )

    receipt = static_w28_resident.run_static_w28_resident_acceptance(
        task="t_test",
        root=tmp_path / "run",
        artifact_root=tmp_path / "artifact",
        checkpoint="PRE",
        canonical_pin="deadbeef",
        reference_receipt=reference,
        reference_sha256=reference_sha,
    )

    assert seen == {"checkpoint": "PRE", "windows": (28,)}
    assert receipt["status"] == "PASS"
    assert receipt["resident_state_persisted"] is True
    assert receipt["measurement"]["kld_mean"] == 0.1364830042977786
    assert receipt["measurement"]["top1"] == 880
    assert receipt["measurement"]["timed_wall_seconds"] == 23.0
    assert receipt["full64_launched"] is False
    assert receipt["sealed_truth_receipt_sha256"] == reference_sha
    assert receipt["source_binding"]["builder_sha256"] == "builder"


def test_rank_seat_localization_changes_only_runtime_rendezvous() -> None:
    original = {
        "rank": 0,
        "world_size": 2,
        "master_addr": "192.168.200.1",
        "master_port": 30391,
        "qsfp_host_ip_by_rank": {"0": "192.168.200.1", "1": "192.168.200.3"},
        "checkpoint_sha256": "sealed",
    }
    seat = {
        "rank": 0,
        "host": "spark-2",
        "local_qsfp_ip": "192.168.200.2",
        "peer_rank": 1,
        "peer_host": "spark-3",
        "peer_qsfp_ip": "192.168.200.3",
    }

    localized = _localize_official_k2_rank_seat(original, seat)

    assert original["master_addr"] == "192.168.200.1"
    assert localized["master_addr"] == "192.168.200.2"
    assert localized["qsfp_host_ip_by_rank"]["0"] == "192.168.200.2"
    assert localized["qsfp_host_ip_by_rank"]["1"] == "192.168.200.3"
    assert {
        key for key in localized if localized[key] != original[key]
    } == {"master_addr", "qsfp_host_ip_by_rank"}


def test_rank_seat_localization_refuses_peer_or_scientific_drift() -> None:
    original = {
        "rank": 0,
        "master_addr": "192.168.200.1",
        "qsfp_host_ip_by_rank": {"0": "192.168.200.1", "1": "192.168.200.3"},
    }
    seat = {
        "rank": 0,
        "local_qsfp_ip": "192.168.200.2",
        "peer_rank": 1,
        "peer_qsfp_ip": "192.168.200.4",
    }
    try:
        _localize_official_k2_rank_seat(original, seat)
    except Exception as exc:
        assert "authorized peer" in str(exc)
    else:
        raise AssertionError("peer drift was accepted")

    try:
        _localize_official_k2_rank_seat(
            original,
            {**seat, "peer_qsfp_ip": "192.168.200.3", "lr": 1e-4},
        )
    except Exception as exc:
        assert "fields refused" in str(exc)
    else:
        raise AssertionError("scientific field drift was accepted")


def test_static_w28_resident_acceptance_is_public_api() -> None:
    import repair_api

    assert (
        repair_api.run_static_w28_resident_acceptance
        is static_w28_resident.run_static_w28_resident_acceptance
    )
    assert "run_static_w28_resident_acceptance" in repair_api.__all__


def test_static_w28_refuses_nonresident_or_slow_result(monkeypatch, tmp_path) -> None:
    reference = tmp_path / "B2_PUBLISHED_PRE.json"
    reference_sha = _write_reference(reference)

    class API:
        def score(self, checkpoint, windows):
            return ScoreResult(
                checkpoint=checkpoint,
                windows=(28,), positions=1024, support=8192,
                kld=0.1364830042977786, top1=880, top1_rate=880 / 1024,
                artifact_root=str(tmp_path), spec="balanced64-v1", candidate_dir="candidate",
                execution_mode="resident_in_memory", resident_load_seconds=1.0,
                timed_wall_seconds=300.001,
                identity={"checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256,
                          "model_index_sha256": static_w28_resident.BASIS_SHA256},
                runtime_counters={"resident_engine_loads": 1, "timed_score_file_reads": 0,
                                  "resident_ready": [{}]},
            )

    monkeypatch.setattr(
        static_w28_resident.ResidentRepairAPI,
        "open",
        lambda root, official_rank_seat=None: API(),
    )
    monkeypatch.setattr(static_w28_resident.sealed_pre_forward, "source_binding", lambda: {"status": "PASS"})

    try:
        static_w28_resident.run_static_w28_resident_acceptance(
            task="t_test", root=tmp_path / "run", artifact_root=tmp_path,
            checkpoint="PRE", canonical_pin="deadbeef",
            reference_receipt=reference, reference_sha256=reference_sha,
        )
    except RuntimeError as exc:
        assert "STATIC_W28_RESIDENT_RED" in str(exc)
    else:
        raise AssertionError("slow resident score must fail")
