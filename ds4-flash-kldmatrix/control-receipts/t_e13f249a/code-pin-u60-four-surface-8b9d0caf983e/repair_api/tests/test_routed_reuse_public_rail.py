from __future__ import annotations

import json
from pathlib import Path
import tempfile

from repair_api import ResidentRepairAPI
from repair_api.balanced64 import ScoreResult
from repair_api.official_k2_resident_score import (
    ALTERNATE_PRE_CHECKPOINT_SHA256,
    OfficialK2ResidentScorer,
    ROUTED_K2_CLOSURE,
    ROUTED_K2_ROUTE_KIND,
)

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
WINDOWS = tuple(range(64))


def _route() -> dict[str, object]:
    return {
        **ROUTED_K2_CLOSURE,
        "route_kind": ROUTED_K2_ROUTE_KIND,
        "pre_checkpoint_identity_sha256": "pre-identity",
        "post_checkpoint_identity_sha256": "post-identity",
        "post_parent_checkpoint_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
        "teacher_manifest_sha256": "teacher-manifest",
        "corpus_manifest_sha256": "corpus-manifest",
        "window_manifest_sha256": "window-manifest",
    }


def _artifact(root: Path) -> None:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "PRE.pt").write_bytes(b"pre")
    (checkpoints / "POST.pt").write_bytes(b"post")
    manifest = {
        "schema": "repair-artifact-v1",
        "artifact_id": "routed-reuse-test",
        "identity": {
            "basis_sha256": BASIS,
            "builder_eval_corpus_sha256": "builder",
            "train_score_corpus_sha256": "train",
            "teacher_inventory": "teacher",
        },
        "checkpoints": {
            "PRE": {
                "path": "checkpoints/PRE.pt",
                "sha256": ROUTED_K2_CLOSURE["pre_checkpoint_sha256"],
                "identity_sha256": "pre-identity",
                "parent_sha256": None,
                "next_update": 0,
            },
            "POST": {
                "path": "checkpoints/POST.pt",
                "sha256": ROUTED_K2_CLOSURE["post_checkpoint_sha256"],
                "identity_sha256": "post-identity",
                "parent_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                "next_update": 1,
            },
        },
        "score": {
            "spec": "balanced64-v1",
            "teacher_dir": "score/teacher",
            "candidate_dir_template": "score/candidates/{checkpoint}",
            "window_ids": list(WINDOWS),
            "positions_per_window": 1024,
            "support": 8192,
            "official_k2_resident": {
                "basis_sha256": BASIS,
                "teacher_manifest_sha256": "teacher-manifest",
                "corpus_manifest_sha256": "corpus-manifest",
                "window_manifest_sha256": "window-manifest",
            },
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest))


class CachedBackend:
    def __init__(self, artifact):
        self.artifact = artifact
        self.bound = []
        self.calls = []

    def bind_routed_k2(self, route):
        self.bound.append(dict(route))

    def score(self, checkpoint, windows):
        self.calls.append(checkpoint)
        terminal = [
            {
                "rank": rank,
                "timed_score_file_reads": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "reference_fwht_calls": 0,
                "cpu_relay_bytes": 0,
            }
            for rank in (0, 1)
        ]
        return ScoreResult(
            checkpoint=checkpoint,
            windows=tuple(windows),
            positions=64 * 1024,
            support=8192,
            kld=0.2 if checkpoint == "PRE" else 0.3,
            top1=60000,
            top1_rate=60000 / (64 * 1024),
            artifact_root=str(self.artifact.root),
            spec="balanced64-v1",
            candidate_dir="resident",
            execution_mode="resident_in_memory",
            resident_load_seconds=0.0,
            timed_wall_seconds=1.0,
            runtime_counters={
                "timed_score_file_reads": 0,
                "file_reads_during_timed_score": 0,
                "resident_ready": [{"rank": 0}, {"rank": 1}],
                "rank_terminal": terminal,
                "payload_model_file_read_delta": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "reference_fwht_calls": 0,
                "cpu_relay_bytes": 0,
            },
        )


def test_routed_score_reuses_cached_public_resident_backend() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _artifact(root)
        api = ResidentRepairAPI.open(root)
        backend = CachedBackend(api.artifact)
        api._official_backends[WINDOWS] = backend

        result = api.score_routed_k2("PRE", "POST", route=_route(), windows=WINDOWS)

        assert result["resident_reused_from_public_score"] is True
        assert len(backend.bound) == 1
        assert backend.calls == ["PRE", "POST"]


def test_official_backend_route_bind_preserves_existing_engine() -> None:
    scorer = object.__new__(OfficialK2ResidentScorer)
    engine = object()
    scorer._engine = engine
    scorer.config = {"basis_sha256": BASIS}

    scorer.bind_routed_k2(_route())

    assert scorer._engine is engine
    assert scorer.config["route_kind"] == ROUTED_K2_ROUTE_KIND
    assert scorer.config["pre_checkpoint_sha256"] == ALTERNATE_PRE_CHECKPOINT_SHA256
