from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pytest
import torch

from banana_smasher.anchor import build_bank_manifest, materialize_candidate_producer
from banana_smasher.anchor_sidecars import (
    CandidateSidecarWriter,
    write_teacher_support_manifest,
)
from banana_smasher.fixed_d4 import produce_fixed_d4_layerwise_logits
from banana_smasher.offline_layerwise import _load_runtime


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


def test_auto_dispatches_builtin_offline_layerwise_as_one_model_pass(
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

    packaged_config = (
        Path(__file__).parents[1]
        / "producer_configs"
        / "fixed_d4_offline_layerwise.json"
    )
    config = tmp_path / "offline-layerwise.json"
    config.write_bytes(packaged_config.read_bytes())
    teacher_manifest = tmp_path / "teacher_support.json"
    teacher_idx = torch.tensor([[1, 0]], dtype=torch.int32)
    teacher_lp = torch.tensor([[-0.1, -1.0]], dtype=torch.float16)
    bank_sha256 = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    write_teacher_support_manifest(
        teacher_manifest,
        windows=[
            {"window_id": index, "idx": teacher_idx, "logprob": teacher_lp}
            for index in range(64)
        ],
        bank_sha256=bank_sha256,
        teacher_sha256="a" * 64,
    )
    calls: list[list[int]] = []

    def fake_layerwise(
        model_root: Path,
        producer_config: Path,
        bank_path: Path,
        output_path: Path,
        *,
        basis_sha256: str,
        verified_pack_receipt: dict | None = None,
    ) -> dict[str, object]:
        assert verified_pack_receipt is not None
        assert verified_pack_receipt["status"] == "PASS"
        rows = [json.loads(line) for line in bank_path.read_text().splitlines()]
        calls.append([row["window_id"] for row in rows])
        writer = CandidateSidecarWriter(
            output_path,
            teacher_manifest_path=teacher_manifest,
            window_ids=[row["window_id"] for row in rows],
            basis_sha256=basis_sha256,
            bank_sha256=bank_sha256,
            model_id="fixture-model",
            pack_sha256=hashlib.sha256(
                (model_root / "BANANA_PACK_MANIFEST.json").read_bytes()
            ).hexdigest(),
        )
        for row in rows:
            writer.write_window(
                row["window_id"],
                q_lp_at_ref=torch.tensor([[-0.2, -1.2]], dtype=torch.float16),
                q_argmax=torch.tensor([1], dtype=torch.int32),
            )
        return {
            "status": "PASS",
            "rows": len(rows),
            "output_format": "torch-sidecars-with-json-manifest",
        }

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

    assert calls == [list(range(64))]
    assert receipt["execution_mode"] == "offline-layerwise"
    assert receipt["producer_backend"] == "fixed-d4-offline-layerwise"
    assert receipt["computed_windows"] == 64
    assert receipt["output_format"] == "torch-sidecars-with-json-manifest"
    assert receipt["classification"] == "backend-smoke-only"


def test_builtin_offline_layerwise_does_not_delegate_to_resident_vllm() -> None:
    source = inspect.getsource(produce_fixed_d4_layerwise_logits)

    assert "cpu_offload_gb" not in source
    assert "produce_fixed_d4_logits(" not in source


def test_packaged_offline_layerwise_config_binds_genuine_adapter() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "producer_configs"
        / "fixed_d4_offline_layerwise.json"
    )
    config = json.loads(config_path.read_text())

    assert config["producer"] == "fixed-d4-offline-layerwise"
    assert set(config["parameters"]) == {
        "execution_mode",
        "input_field",
        "layers",
        "physical_limits",
        "positions",
        "runtime_adapter",
        "teacher_support",
    }
    assert config["parameters"]["layers"] == list(range(43))
    adapter = _load_runtime(
        config["parameters"]["runtime_adapter"], root=config_path.parent
    )
    assert adapter.__name__ == "DeepseekV4D4Runtime"
    assert adapter.__module__ == "banana_smasher.hf_deepseek_v4_d4_adapter"
    assert "vllm" not in config_path.read_text()
    assert "cpu_offload" not in config_path.read_text()
