from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.anchor import build_bank_manifest, materialize_candidate_producer
from banana_smasher.fixed_d4 import produce_fixed_d4_layerwise_logits


BASIS = "b" * 64


def _manifest() -> dict:
    identity = {"status": "resolved", "sha256": "c" * 64, "uri": "fixture://source"}
    classes = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
    return build_bank_manifest(
        bank_id="balanced64",
        role="train_balanced64",
        windows=[
            {"id": index, "class": classes[index % len(classes)]}
            for index in range(64)
        ],
        parent_corpus=identity,
        identities={
            "corpus": identity,
            "tokenizer": {
                "status": "resolved",
                "sha256": "d" * 64,
                "uri": "fixture://tokenizer",
            },
            "teacher": {"status": "unresolved", "reason": "not needed"},
            "scorer": {
                "status": "resolved",
                "sha256": "e" * 64,
                "uri": "fixture://scorer",
            },
        },
        split_lineage={"split": "train"},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )


def test_auto_dispatches_builtin_offline_layerwise_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    bank_path = run_root / "banks" / "balanced64.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(
        "".join(
            json.dumps({"window_id": index, "tokens": [index + 1]}) + "\n"
            for index in range(64)
        )
    )
    model = tmp_path / "model"
    layer_receipt = model / "provenance" / "LAYER_RECEIPT.json"
    layer_receipt.parent.mkdir(parents=True)
    (model / "BANANA_PACK_MANIFEST.json").write_text('{"schema":"fixture"}\n')
    layer_receipt.write_text(
        json.dumps({"layer": 0, "tier": "d4_k4096", "basis_sha256": BASIS})
        + "\n"
    )
    import banana_smasher.contract as contract

    monkeypatch.setattr(contract, "verify_pack", lambda _root: {"status": "PASS"})
    monkeypatch.setattr(contract, "load_manifest", lambda _root: {"layers": [0]})

    config = tmp_path / "offline-layerwise.json"
    config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-candidate-producer-v1",
                "producer": "fixed-d4-offline-layerwise",
                "parameters": {
                    "input_field": "tokens",
                    "batch_size": 16,
                    "engine": {"cpu_offload_gb": 64, "enforce_eager": True},
                },
            }
        )
    )
    calls: list[list[int]] = []

    def fake_layerwise(
        model_root: Path,
        producer_config: Path,
        bank_path: Path,
        output_path: Path,
        *,
        basis_sha256: str,
    ) -> dict[str, object]:
        rows = [json.loads(line) for line in bank_path.read_text().splitlines()]
        calls.append([row["window_id"] for row in rows])
        output_path.write_text(
            "".join(
                json.dumps(
                    {"window_id": row["window_id"], "logits": [1.0, 0.0]}
                )
                + "\n"
                for row in rows
            )
        )
        return {"status": "PASS", "rows": len(rows)}

    monkeypatch.setattr(
        "banana_smasher.fixed_d4.produce_fixed_d4_layerwise_logits",
        fake_layerwise,
    )
    receipt = materialize_candidate_producer(
        run_root,
        _manifest(),
        candidate_id="layerwise-auto",
        model_root=model,
        producer_config=config,
        basis_sha256=BASIS,
        execution_mode="auto",
        chunk_size=16,
    )

    assert calls == [
        list(range(0, 16)),
        list(range(16, 32)),
        list(range(32, 48)),
        list(range(48, 64)),
    ]
    assert receipt["execution_mode"] == "offline-layerwise"
    assert receipt["producer_backend"] == "fixed-d4-offline-layerwise"
    assert receipt["computed_windows"] == 64


def test_builtin_offline_layerwise_requires_public_vllm_weight_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "offline-layerwise.json"
    config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-candidate-producer-v1",
                "producer": "fixed-d4-offline-layerwise",
                "parameters": {
                    "input_field": "tokens",
                    "batch_size": 1,
                    "engine": {"enforce_eager": True},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="cpu_offload_gb"):
        produce_fixed_d4_layerwise_logits(
            tmp_path / "model",
            config,
            tmp_path / "bank.jsonl",
            tmp_path / "output.jsonl",
            basis_sha256=BASIS,
        )

    value = json.loads(config.read_text())
    value["parameters"]["engine"]["cpu_offload_gb"] = 64
    config.write_text(json.dumps(value))
    called: dict[str, object] = {}

    def fake_vllm(*args: object, **kwargs: object) -> dict[str, object]:
        called["args"] = args
        called["kwargs"] = kwargs
        return {"status": "PASS"}

    monkeypatch.setattr("banana_smasher.fixed_d4.produce_fixed_d4_logits", fake_vllm)
    receipt = produce_fixed_d4_layerwise_logits(
        tmp_path / "model",
        config,
        tmp_path / "bank.jsonl",
        tmp_path / "output.jsonl",
        basis_sha256=BASIS,
    )

    assert receipt == {"status": "PASS"}
    assert called["kwargs"] == {
        "basis_sha256": BASIS,
        "_expected_producer": "fixed-d4-offline-layerwise",
    }
