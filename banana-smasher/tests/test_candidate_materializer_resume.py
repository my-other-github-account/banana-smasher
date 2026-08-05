from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from banana_smasher.anchor import (
    AnchorEvaluationError,
    build_bank_manifest,
    materialize_candidate_producer,
)


BASIS = "a" * 64


def _manifest() -> dict:
    identity = {"status": "resolved", "sha256": "b" * 64, "uri": "fixture://source"}
    classes = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
    return build_bank_manifest(
        bank_id="balanced64",
        role="train_balanced64",
        windows=[{"id": index, "class": classes[index % len(classes)]} for index in range(64)],
        parent_corpus=identity,
        identities={
            "corpus": identity,
            "tokenizer": {"status": "resolved", "sha256": "c" * 64, "uri": "fixture://tokenizer"},
            "teacher": {"status": "unresolved", "reason": "not needed"},
            "scorer": {"status": "resolved", "sha256": "d" * 64, "uri": "fixture://scorer"},
        },
        split_lineage={"split": "train"},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, dict]:
    run_root = tmp_path / "run"
    bank_path = run_root / "banks" / "balanced64.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(
        "".join(
            json.dumps({"window_id": index, "tokens": [index + 1]}, sort_keys=True) + "\n"
            for index in range(64)
        )
    )
    model = tmp_path / "model"
    receipt = model / "provenance" / "LAYER_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    (model / "BANANA_PACK_MANIFEST.json").write_text('{"schema":"fixture"}\n')
    receipt.write_text(json.dumps({"layer": 0, "tier": "d4_k4096", "basis_sha256": BASIS}) + "\n")

    import banana_smasher.contract as contract

    monkeypatch.setattr(contract, "verify_pack", lambda _root: {"status": "PASS"})
    monkeypatch.setattr(contract, "load_manifest", lambda _root: {"layers": [0]})

    calls = tmp_path / "calls.txt"
    failed = tmp_path / "failed-once"
    producer = tmp_path / "offline_layerwise.py"
    producer.write_text(
        "import argparse,json,pathlib,sys\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--model'); p.add_argument('--config'); p.add_argument('--bank')\n"
        "p.add_argument('--output'); p.add_argument('--basis-sha256')\n"
        "a=p.parse_args()\n"
        f"calls=pathlib.Path({str(calls)!r}); failed=pathlib.Path({str(failed)!r})\n"
        "count=int(calls.read_text())+1 if calls.exists() else 1\n"
        "calls.write_text(str(count))\n"
        "if count == 2 and not failed.exists(): failed.write_text('1'); sys.exit(9)\n"
        "rows=[json.loads(line) for line in open(a.bank) if line.strip()]\n"
        "with open(a.output,'w') as sink:\n"
        "  for row in rows: sink.write(json.dumps({'window_id':row['window_id'],'logits':[float(row['window_id']),0.0]},sort_keys=True)+'\\n')\n"
    )
    config = tmp_path / "producer.json"
    config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-candidate-producer-v1",
                "producer": "offline-layerwise",
                "command": [sys.executable, str(producer)],
                "parameters": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    return run_root, model, config, _manifest()


def test_partial_resume_and_completed_rerun_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, model, config, manifest = _fixture(tmp_path, monkeypatch)

    with pytest.raises(AnchorEvaluationError, match="candidate producer failed with exit 9"):
        materialize_candidate_producer(
            run_root,
            manifest,
            candidate_id="candidate-a",
            model_root=model,
            producer_config=config,
            basis_sha256=BASIS,
            execution_mode="offline-layerwise",
            chunk_size=8,
        )

    progress_path = run_root / "materializations" / "candidate" / "candidate-a" / "balanced64.progress.json"
    progress = json.loads(progress_path.read_text())
    assert progress["status"] == "IN_PROGRESS"
    assert progress["completed_windows"] == 8
    assert progress["execution_mode"] == "offline-layerwise"

    resumed = materialize_candidate_producer(
        run_root,
        manifest,
        candidate_id="candidate-a",
        model_root=model,
        producer_config=config,
        basis_sha256=BASIS,
        execution_mode="offline-layerwise",
        chunk_size=8,
    )
    assert resumed["coverage"] == "64/64"
    assert resumed["resumed_windows"] == 8
    assert resumed["computed_windows"] == 56
    assert resumed["completed_rerun"] is False
    imported = run_root / resumed["relative_path"]
    original = imported.read_bytes()
    call_count = int((tmp_path / "calls.txt").read_text())

    complete = materialize_candidate_producer(
        run_root,
        manifest,
        candidate_id="candidate-a",
        model_root=model,
        producer_config=config,
        basis_sha256=BASIS,
        execution_mode="auto",
        chunk_size=8,
    )
    assert complete["completed_rerun"] is True
    assert complete["resumed_windows"] == 64
    assert complete["computed_windows"] == 0
    assert imported.read_bytes() == original
    assert int((tmp_path / "calls.txt").read_text()) == call_count

    tampered = [json.loads(line) for line in imported.read_text().splitlines()]
    tampered[0]["logits"] = [999.0, 0.0]
    imported.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tampered)
    )
    with pytest.raises(AnchorEvaluationError, match="producer SHA-256 mismatch"):
        materialize_candidate_producer(
            run_root,
            manifest,
            candidate_id="candidate-a",
            model_root=model,
            producer_config=config,
            basis_sha256=BASIS,
            execution_mode="auto",
            chunk_size=8,
        )


def test_resume_rejects_nonprefix_or_changed_interim_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, model, config, manifest = _fixture(tmp_path, monkeypatch)
    staging = run_root / "materializations" / "candidate" / "candidate-a"
    staging.mkdir(parents=True)
    (staging / "balanced64.interim.jsonl").write_text(
        json.dumps({"window_id": 3, "logits": [1.0, 0.0]}) + "\n"
    )

    with pytest.raises(AnchorEvaluationError, match="ordered prefix"):
        materialize_candidate_producer(
            run_root,
            manifest,
            candidate_id="candidate-a",
            model_root=model,
            producer_config=config,
            basis_sha256=BASIS,
            execution_mode="offline-layerwise",
            chunk_size=8,
        )
