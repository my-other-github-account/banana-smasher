from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.qtip_periodic_signal import (
    FF0731_MODEL_INDEX_SHA256,
    TEACHER_TOP8192_MANIFEST_SHA256,
    TRAIN64_BANK_MANIFEST_SHA256,
    TRAIN8_ROW_IDS,
)
from banana_smasher.qtip_periodic_train8 import (
    MANIFEST_SCHEMA,
    arm_cell_sources,
    load_manifest,
    matched_measurements,
    write_manifest,
)


def _file(path: Path, payload: bytes, *, declared_sha: str | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": declared_sha or hashlib.sha256(payload).hexdigest(),
    }


def _cell(control: str, expert: int, root: Path) -> dict[str, object]:
    weights = 8_388_608
    code_bits = weights * 5 // 2
    return {
        "control": control,
        "identity": {"layer": 0, "expert": expert, "projection": "down"},
        "control_unit": _file(root / f"{control}.pt", control.encode()),
        "periodic_codes": _file(root / f"{control}.npy", b"periodic-" + control.encode()),
        "source_weight_sha256": f"{expert + 1:x}" * 64,
        "accounting": {"weights": weights, "code_bits": code_bits},
        "direct_error": {
            "control_sse": 10.0 + expert,
            "periodic_sse": 2.0 + expert,
        },
    }


def _manifest(tmp_path: Path) -> dict[str, object]:
    claim = _file(
        tmp_path / "HOST_CLAIM.json",
        json.dumps({"task_id": "t_7002ac79", "state": "CLAIMED"}).encode(),
    )
    shards = _file(
        tmp_path / "SHARDS.json",
        json.dumps({"intended_basis": FF0731_MODEL_INDEX_SHA256}).encode(),
    )
    config = {
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "vocab_size": 129280,
        "n_routed_experts": 256,
    }
    weight_map = {
        "embed.weight": "model-00001-of-00048.safetensors",
        "head.weight": "model-00045-of-00048.safetensors",
        "norm.weight": "model-00045-of-00048.safetensors",
        **{
            f"layers.{layer}.attn_norm.weight": f"model-{layer + 2:05d}-of-00048.safetensors"
            for layer in range(43)
        },
    }
    model_config = _file(tmp_path / "model/config.json", json.dumps(config).encode())
    model_index_path = tmp_path / "model/model.safetensors.index.json"
    model_index_path.write_text(json.dumps({"weight_map": weight_map}))
    model_index = {
        "path": str(model_index_path),
        "bytes": model_index_path.stat().st_size,
        "sha256": FF0731_MODEL_INDEX_SHA256,
    }
    bank = _file(tmp_path / "train8/bank.json", b"bank", declared_sha=TRAIN64_BANK_MANIFEST_SHA256)
    teacher_manifest = _file(
        tmp_path / "train8/teacher.json",
        b"teacher",
        declared_sha=TEACHER_TOP8192_MANIFEST_SHA256,
    )
    teacher_rows = [
        {"row_id": row_id, **_file(tmp_path / f"train8/t8192_win{row_id}.pt", row_id.encode())}
        for row_id in TRAIN8_ROW_IDS
    ]
    corpus = _file(tmp_path / "train8/corpus.json", b"[]")
    source_root = tmp_path / "source-model"
    source_root.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    package = tmp_path / "package"
    package.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    qtip = tmp_path / "qtip"
    qtip.mkdir()
    return {
        "schema": MANIFEST_SCHEMA,
        "task_id": "t_7002ac79",
        "basis_sha256": FF0731_MODEL_INDEX_SHA256,
        "row_ids": list(TRAIN8_ROW_IDS),
        "attention_implementation": "eager",
        "position_cutoff": 1024,
        "support_width": 8192,
        "claim": claim,
        "shards": shards,
        "model": {
            "config": model_config,
            "index": model_index,
            "source_root": str(source_root),
        },
        "corpus": corpus,
        "bank": bank,
        "teacher": {"manifest": teacher_manifest, "rows": teacher_rows},
        "cells": [_cell("qtip_k2", 0, tmp_path), _cell("qtip_k3", 1, tmp_path)],
        "runtime": {
            "banana_smasher_source": str(package),
            "public_site": str(public),
            "qtip_root": str(qtip),
            "shard_stage_dir": str(stage),
            "microbatch": 2,
            "readout_chunk_positions": 16,
        },
        "output": {"root": str(tmp_path / "output")},
    }


def test_two_cell_arm_plan_and_measurements_are_matched() -> None:
    cells = [
        {
            "control": "qtip_k2",
            "accounting": {"weights": 8, "code_bits": 20},
            "direct_error": {"control_sse": 10.0, "periodic_sse": 2.0},
        },
        {
            "control": "qtip_k3",
            "accounting": {"weights": 8, "code_bits": 20},
            "direct_error": {"control_sse": 11.0, "periodic_sse": 3.0},
        },
    ]

    errors, bits = matched_measurements(cells)

    assert arm_cell_sources() == {
        "qtip_k2": ("qtip_k2", "source"),
        "qtip_k3": ("source", "qtip_k3"),
        "qtip25_avg_member": ("qtip_k2", "qtip_k3"),
        "qtip25_periodic_23": ("periodic", "periodic"),
    }
    assert errors == {
        "qtip_k2": 10.0,
        "qtip_k3": 11.0,
        "qtip25_avg_member": 21.0,
        "qtip25_periodic_23": 5.0,
    }
    assert bits == {
        "qtip_k2": 16,
        "qtip_k3": 24,
        "qtip25_avg_member": 40,
        "qtip25_periodic_23": 40,
    }


def test_manifest_preflight_binds_current_ff0731_and_exact_two_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "MANIFEST.json"
    write_manifest(path, manifest)

    real_sha = hashlib.sha256

    def accepted_sha(path_value: str | Path) -> str:
        path_object = Path(path_value)
        if path_object.name == "model.safetensors.index.json":
            return FF0731_MODEL_INDEX_SHA256
        if path_object.name == "bank.json":
            return TRAIN64_BANK_MANIFEST_SHA256
        if path_object.name == "teacher.json":
            return TEACHER_TOP8192_MANIFEST_SHA256
        digest = real_sha()
        digest.update(path_object.read_bytes())
        return digest.hexdigest()

    monkeypatch.setattr(
        "banana_smasher.qtip_periodic_train8.sha256_file", accepted_sha
    )
    observed, derived = load_manifest(path, verify_files=True)

    assert observed["basis_sha256"] == FF0731_MODEL_INDEX_SHA256
    assert derived["nominal_code_bits"] == {
        "qtip_k2": 16_777_216,
        "qtip_k3": 25_165_824,
        "qtip25_avg_member": 41_943_040,
        "qtip25_periodic_23": 41_943_040,
    }
    assert derived["direct_error"]["qtip25_avg_member"] == 21.0
    assert derived["direct_error"]["qtip25_periodic_23"] == 5.0


def test_manifest_rejects_any_non_eager_forward(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["attention_implementation"] = "sdpa"
    path = tmp_path / "MANIFEST.json"
    write_manifest(path, manifest)

    with pytest.raises(ValueError, match="eager attention"):
        load_manifest(path, verify_files=False)
