from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from repair_api import ArtifactError, ResidentRepairAPI


CANONICAL = (
    "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
    "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
)
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT = "7978d1002d7e4ecfa280f646f70cc76638c0e7bd833cc3cc13a2de999050133f"
TEACHER = "e494b7fd83bcce7ee0bbf14371bee2d87005ea846cdd178dbd69379c2c336a82"
CANDIDATE = "9ff57dceb525fdb695596145fd527e24e84852069c977d6ef4237d57cd3dcc78"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _open_api(root: Path) -> ResidentRepairAPI:
    (root / "checkpoints").mkdir()
    (root / "score" / "teacher").mkdir(parents=True)
    (root / "checkpoints" / "UPDATE_000.pt").write_bytes(b"checkpoint")
    manifest = {
        "schema": "repair-artifact-v1",
        "identity": {
            "basis_sha256": BASIS,
            "builder_eval_corpus_sha256": _sha("builder-corpus"),
            "train_score_corpus_sha256": _sha("score-corpus"),
            "teacher_inventory_sha256": _sha("teacher-inventory"),
        },
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": CHECKPOINT,
                "identity_sha256": _sha("checkpoint-identity"),
                "parent_sha256": None,
                "next_update": 0,
            }
        },
        "score": {
            "spec": "balanced64-v1",
            "teacher_dir": "score/teacher",
            "candidate_dir_template": "score/candidates/{checkpoint}",
            "window_ids": [328, *range(63)],
            "positions_per_window": 1024,
            "support": 8192,
            "official_k2_resident": {"basis_sha256": BASIS},
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest))
    return ResidentRepairAPI.open(root)


def _tap(name: str) -> dict[str, object]:
    return {
        "sha256": _sha("current-" + name),
        "dtype": "torch.bfloat16",
        "shape": [1],
        "sample": [0.0],
    }


def _current_trace() -> dict[str, object]:
    return {
        "taps": {name: _tap(name) for name in CANONICAL},
        "runtime_counters": {
            "timed_model_payload_reads": 0,
            "fallback_calls": 0,
            "reconstruction_calls": 0,
            "reference_fwht_calls": 0,
            "cpu_relay_bytes": 0,
            "layer_streaming_calls": 0,
            "resident_ready": [{"rank": 0}, {"rank": 1}],
        },
    }


def _fixture(api: ResidentRepairAPI, taps: dict[str, dict[str, object]]) -> dict[str, object]:
    identity = api._identity("UPDATE_000", (328,))
    return {
        "schema": "banana-smasher-independent-parity-fixture-v1",
        "identity": {
            "basis_sha256": identity["basis_sha256"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "builder_eval_corpus_sha256": identity["builder_eval_corpus_sha256"],
            "train_score_corpus_sha256": identity["train_score_corpus_sha256"],
            "teacher_inventory_sha256": identity["teacher_inventory"],
            "window": 328,
            "teacher_sha256": TEACHER,
            "candidate_sha256": CANDIDATE,
        },
        "taps": taps,
    }


def test_hash_only_w328_fixture_reports_q_lp_and_marks_earlier_taps_unavailable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        api = _open_api(Path(directory))
        sealed_q = {**_tap("q_lp_at_ref"), "sha256": _sha("sealed-q")}
        fixture = _fixture(api, {"q_lp_at_ref": sealed_q})
        with patch.object(api, "parity_tap", return_value=_current_trace()):
            result = api.compare_parity_fixture(
                "UPDATE_000", window=328, fixture=fixture,
                teacher_sha256=TEACHER, candidate_sha256=CANDIDATE,
            )
    assert result["first_comparable_tap"] == "q_lp_at_ref"
    assert result["first_mismatch"] == "q_lp_at_ref"
    assert result["unavailable_before_first_comparable"] == list(CANONICAL[:48])
    assert result["quality_status"] == "DIAGNOSTIC_ONLY_UNPROMOTED"


def test_full_fixture_walks_canonical_order_and_returns_first_mismatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        api = _open_api(Path(directory))
        sealed = {name: dict(_tap(name)) for name in CANONICAL}
        sealed["L007"]["sha256"] = _sha("sealed-L007")
        sealed["q_lp_at_ref"]["sha256"] = _sha("sealed-q")
        fixture = _fixture(api, sealed)
        with patch.object(api, "parity_tap", return_value=_current_trace()):
            result = api.compare_parity_fixture(
                "UPDATE_000", window=328, fixture=fixture,
                teacher_sha256=TEACHER, candidate_sha256=CANDIDATE,
            )
    assert result["first_comparable_tap"] == "ids"
    assert result["first_mismatch"] == "L007"
    assert [row["tap"] for row in result["comparisons"]] == list(CANONICAL)


@pytest.mark.parametrize(
    "field",
    [
        "basis_sha256", "checkpoint_sha256", "builder_eval_corpus_sha256",
        "train_score_corpus_sha256", "teacher_inventory_sha256", "window",
        "teacher_sha256", "candidate_sha256",
    ],
)
def test_fixture_identity_gate_rejects_every_bound_field(field: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        api = _open_api(Path(directory))
        fixture = _fixture(api, {"q_lp_at_ref": _tap("q_lp_at_ref")})
        fixture = copy.deepcopy(fixture)
        fixture["identity"][field] = 999 if field == "window" else _sha("wrong-" + field)
        with patch.object(api, "parity_tap", side_effect=AssertionError("must reject before tap")):
            with pytest.raises(ArtifactError, match=field):
                api.compare_parity_fixture(
                    "UPDATE_000", window=328, fixture=fixture,
                    teacher_sha256=TEACHER, candidate_sha256=CANDIDATE,
                )


def test_shared_code_mode_equality_cannot_promote_independent_fixture_parity_or_score() -> None:
    with tempfile.TemporaryDirectory() as directory:
        api = _open_api(Path(directory))
        fixture = _fixture(api, {"q_lp_at_ref": _tap("q_lp_at_ref")})
        with patch.object(api, "parity_tap", return_value=_current_trace()) as parity, \
             patch.object(api, "score", side_effect=AssertionError("full64 score forbidden")) as score:
            result = api.compare_parity_fixture(
                "UPDATE_000", window=328, fixture=fixture,
                teacher_sha256=TEACHER, candidate_sha256=CANDIDATE,
                mode="sealed_reference",
            )
    parity.assert_called_once()
    score.assert_not_called()
    assert result["shared_code_mode_parity_is_not_independent_parity"] is True
    assert result["independent_fixture"] is True
    assert result["quality_status"] == "DIAGNOSTIC_ONLY_UNPROMOTED"
    assert "target_kld" not in result and "target_ladder" not in result
