from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock(corpus_sha: str, model_index_sha: str) -> dict:
    lock = {
        "schema": "banana-smasher.balanced64-suite-lock.v1",
        "name": "BALANCED64_FIXTURE_V1",
        "positions": 65536,
        "positions_per_window": 1024,
        "support": 8192,
        "window_count": 64,
        "window_population_sha256": "a" * 64,
        "source_windows_sha256": corpus_sha,
        "teacher_bank": "TEACHER_FIXTURE_BALANCED64_V1",
        "teacher_source_model_index_sha256": model_index_sha,
        "metrics": {
            "kld": {
                "direction": "KL(teacher||candidate)",
                "per_position_dtype": "IEEE-754 binary64",
                "reduction": "math.fsum over per-position values in ascending window ordinal then position order; divide once by 65536",
                "support": "teacher top-8192; teacher and candidate renormalized on identical ordered support",
            },
            "top1": {"tie_break": "deterministic first-index argmax"},
        },
        "windows": [
            {"ordinal": ordinal, "window_id": 1000 + ordinal, "source_class": "fixture"}
            for ordinal in range(64)
        ],
    }
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    lock["suite_lock_sha256"] = hashlib.sha256(canonical).hexdigest()
    return lock


class _FixtureRuntime:
    runtime_id = "fixture-model-neutral-v1"

    def capture_teacher(self, *, source, suite_lock, corpus, output, windows):
        rows = []
        output.mkdir(parents=True)
        for window in windows:
            path = output / f"teacher-{window['ordinal']:02d}.bin"
            path.write_bytes(f"teacher:{window['window_id']}".encode())
            rows.append(
                {
                    **window,
                    "positions": 1024,
                    "support": 8192,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
        return {"rows": rows, "runtime_counters": {"fallback": 0, "relay": 0, "reconstruction": 0, "streaming": 0}}

    def score_pre(self, *, artifact, teacher_capture, suite_lock, corpus):
        return {
            "rows": [
                {
                    **window,
                    "positions": 1024,
                    "kld_values": ["0.0"] * 1024,
                    "top1_matches": 1024,
                }
                for window in suite_lock["windows"]
            ],
            "resident_ready": True,
            "timed_wall_seconds": 0.01,
            "runtime_counters": {
                "timed_payload_reads": 0,
                "timed_model_reads": 0,
                "fallback": 0,
                "relay": 0,
                "reconstruction": 0,
                "streaming": 0,
            },
        }


def test_public_model_neutral_teacher_capture_and_score_pre(tmp_path: Path) -> None:
    from banana_smasher import capture_balanced64_teacher, score_balanced64_pre

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "fixture_moe"}) + "\n")
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fixture")
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}) + "\n")
    corpus = tmp_path / "balanced64.json"
    corpus.write_text("fixture frozen population\n")
    suite_lock = _lock(_sha(corpus), _sha(index))
    lock_path = tmp_path / "suite-lock.json"
    lock_path.write_text(json.dumps(suite_lock, sort_keys=True) + "\n")
    runtime = _FixtureRuntime()

    canary = capture_balanced64_teacher(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        suite_lock=lock_path,
        corpus=corpus,
        output=tmp_path / "teacher-canary",
        receipt_path=tmp_path / "TEACHER_CANARY.json",
        windows=[1000],
        runtime=runtime,
    )
    assert canary["status"] == "PASS_DIAGNOSTIC"
    assert canary["artifact_admissible"] is False
    assert canary["row_count"] == 1

    teacher = capture_balanced64_teacher(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        suite_lock=lock_path,
        corpus=corpus,
        output=tmp_path / "teacher",
        receipt_path=tmp_path / "TEACHER_CAPTURE.json",
        runtime=runtime,
    )
    artifact = {
        "schema": "banana-smasher-hf-moe-uniform-artifact-v1",
        "status": "PASS",
        "reload_verified": True,
        "source": {"model_index_sha256": _sha(index)},
        "accounting": {"routed_tensor_count": 1, "native_tensor_count": 1},
        "mechanisms": {"fallback": 0, "relay": 0, "reconstruction": 0, "streaming": 0},
    }
    pre = score_balanced64_pre(
        artifact,
        teacher_capture=teacher,
        suite_lock=lock_path,
        corpus=corpus,
        receipt_path=tmp_path / "PRE.json",
        runtime=runtime,
    )

    assert teacher["status"] == "PASS"
    assert teacher["api"] == {"method": "capture_balanced64_teacher", "version": 1}
    assert teacher["row_count"] == 64
    assert teacher["positions"] == 65536
    assert teacher["support"] == 8192
    assert teacher["source"]["model_index_sha256"] == _sha(index)
    assert pre["status"] == "PASS"
    assert pre["api"] == {"method": "score_balanced64_pre", "version": 1}
    assert pre["rows_sealed"] == 64
    assert pre["positions"] == 65536
    assert pre["support"] == 8192
    assert pre["mean_kld"] == 0.0
    assert pre["top1_matches"] == 65536
    assert pre["resident_ready"] is True
    assert pre["runtime_counters"] == {
        "timed_payload_reads": 0,
        "timed_model_reads": 0,
        "fallback": 0,
        "relay": 0,
        "reconstruction": 0,
        "streaming": 0,
    }
    assert json.loads((tmp_path / "TEACHER_CAPTURE.json").read_text()) == teacher
    assert json.loads((tmp_path / "PRE.json").read_text()) == pre


def test_glm_suite_lock_and_pending_evals_are_model_local() -> None:
    repository = Path(__file__).parents[2]
    lock_path = repository / "Evals/configs/glm-5.3-flash-balanced64-v1.json"
    lock = json.loads(lock_path.read_text())
    declared = lock.pop("suite_lock_sha256")
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
    assert declared == "dc5e1a78d0b1ae0975d52b89ee6cfbdc7f8d3207784fe0d7fd5afd3abe844846"
    assert lock["teacher_source_model_index_sha256"] == "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
    assert lock["window_population_sha256"] == "24089eea1b3e5650265b971930571dbf249aba0b2f62e954a9628dcbfd182f09"
    assert len(lock["windows"]) == 64

    result = json.loads(
        (repository / "Evals/results/glm-5.3-flash-balanced64-v1.json").read_text()
    )
    row = result["rows"][0]
    assert result["status"] == "PENDING_PRE"
    assert row["rows_sealed"] == 0
    assert row["kld"] is None
    assert row["top1_matches"] is None
    assert row["exact_accounting_bytes"] is None
    assert row["comparison_bpw"] is None

    worked = (repository / "WORKED_EXAMPLE.md").read_text()
    assert "ledger = build_balanced64_token_ledger(" in worked
    assert 'source_manifest="/local/authenticated-balanced64-source-text.json"' in worked
    assert 'bound_suite_lock="/local/eval/model-balanced64-suite-lock.json"' in worked
    assert 'suite_lock="/local/eval/model-balanced64-suite-lock.json"' in worked
    assert 'corpus="/local/eval/model-balanced64-token-ledger.json"' in worked
    assert "capture_balanced64_teacher(" in worked
    assert "score_balanced64_pre(" in worked
    assert "runtime=" not in worked.split("teacher = capture_balanced64_teacher(", 1)[1].split("pre =", 1)[0]
