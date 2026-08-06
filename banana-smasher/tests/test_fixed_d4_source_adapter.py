from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.backpack import (
    BackpackPlan,
    BackpackPlanError,
    generate_backpack_candidates,
    inspect_backpack,
)
from banana_smasher.cli import main


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, value) in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + payload)


def _write_native_model(model: Path, *, scale_byte: int) -> str:
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    weight_map: dict[str, str] = {}
    for expert in range(256):
        for weight in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{weight}"
            tensors[f"{prefix}.weight"] = ("I8", [1, 16], bytes([0x42]) * 16)
            tensors[f"{prefix}.scale"] = ("F8_E8M0", [1, 1], bytes([scale_byte]))
            weight_map[f"{prefix}.weight"] = shard.name
            weight_map[f"{prefix}.scale"] = shard.name
    _write_safetensors(shard, tensors)
    basis_index = model / "model.safetensors.index.json"
    basis_index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return hashlib.sha256(basis_index.read_bytes()).hexdigest()


def test_public_prepare_solve_streams_native_mxfp4_source_into_bound_config(
    tmp_path: Path, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    weight_map: dict[str, str] = {}
    # One 32-value MXFP4 block per weight: packed low/high E2M1 nibbles are
    # 1.0 and 2.0, with a raw E8M0 scale byte. The real model uses the same
    # dtypes and simply has larger matrix shapes.
    for expert in range(256):
        for weight in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{weight}"
            tensors[f"{prefix}.weight"] = ("I8", [1, 16], bytes([0x42]) * 16)
            tensors[f"{prefix}.scale"] = ("F8_E8M0", [1, 1], bytes([127]))
            weight_map[f"{prefix}.weight"] = shard.name
            weight_map[f"{prefix}.scale"] = shard.name
    _write_safetensors(shard, tensors)
    basis_index = model / "model.safetensors.index.json"
    basis_index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    basis = hashlib.sha256(basis_index.read_bytes()).hexdigest()

    prepared = tmp_path / "prepared"

    assert (
        main(
            [
                "fixed-d4",
                "prepare-solve",
                "--model",
                str(model),
                "--tier",
                "d4_k2048",
                "--layer",
                "0",
                "--output",
                str(prepared),
                "--basis-sha256",
                basis,
                "--chunk-vectors",
                "32",
                "--reserve-bytes",
                "0",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["basis_sha256"] == basis
    assert receipt["source_dtype"] == "packed-mxfp4-e2m1-with-e8m0-scales"
    assert receipt["codebook_source"] == "source-frequency-top-k"
    config_path = Path(receipt["config"])
    config = json.loads(config_path.read_text())
    assert config["schema"] == "banana-smasher-fixed-d4-exact-solve-v1"
    assert config["vector_domain"] == "mxfp4_e2m1"
    assert config["basis_index"] == "model.safetensors.index.json"

    down = np.load(
        prepared / config["projections"]["down"]["normalized_vectors"]["path"]
    )
    fused13 = np.load(
        prepared / config["projections"]["fused13"]["normalized_vectors"]["path"]
    )
    assert down.shape == (256, 8, 4)
    assert fused13.shape == (256, 16, 4)
    assert np.all(down == [1.0, 2.0, 1.0, 2.0])
    assert np.all(fused13 == [1.0, 2.0, 1.0, 2.0])

    solve = tmp_path / "solve"
    assert (
        main(
            [
                "fixed-d4",
                "solve",
                "--config",
                str(config_path),
                "--output",
                str(solve),
                "--basis-sha256",
                basis,
            ]
        )
        == 0
    )
    solved = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(solved["manifest"]).read_text())
    for projection in ("down", "fused13"):
        assignments = np.load(
            Path(solved["manifest"]).parent
            / manifest["projections"][projection]["assignments"]["path"]
        )
        assert np.all(assignments == 0)


def test_public_plan_generates_both_fixed_d4_tiers_from_native_source(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    basis = _write_native_model(model, scale_byte=128)
    cells: list[dict[str, object]] = []
    weights: list[np.ndarray] = []
    for expert_ids in (range(128), range(128, 256)):
        for projection in ("fused13", "down"):
            width = 64 if projection == "fused13" else 32
            value = np.tile(
                np.asarray([2.0, 4.0], dtype=np.float32),
                len(expert_ids) * width // 2,
            ).reshape(len(expert_ids), width)
            path = f"cell-{len(cells)}.npy"
            np.save(model / path, value, allow_pickle=False)
            start = sum(array.size for array in weights)
            weights.append(value)
            cells.append(
                {
                    "cell_id": f"cell-{len(cells)}",
                    "path": path,
                    "feature_slice": [start, start + value.size],
                    "layer": 0,
                    "projection": projection,
                    "expert_ids": list(expert_ids),
                }
            )
    (model / "BACKPACK_MODEL.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-model-v1",
                "revision": "fixed-d4-native-r1",
                "weight_count": sum(array.size for array in weights),
                "dense_bytes": 0,
                "metadata_bytes": 0,
                "repair_bytes": 0,
                "cells": cells,
            }
        )
        + "\n"
    )
    bank = tmp_path / "anchor64.npz"
    np.savez(
        bank,
        features=np.zeros((64, sum(array.size for array in weights)), dtype=np.float32),
        classes=np.asarray(
            [
                ("agentic", "chat", "code", "multilingual", "prose", "reasoning")[
                    index % 6
                ]
                for index in range(64)
            ]
        ),
    )
    plan = BackpackPlan.from_mapping(
        {
            "schema": "banana-smasher-backpack-plan-v1",
            "model": {"root": str(model), "revision": "fixed-d4-native-r1"},
            "target": {"exact_bytes": 40_000_000},
            "tiers": [
                {
                    "id": "d4-k2048",
                    "family": "vector_vq",
                    "dimension": 4,
                    "codebook_size": 2048,
                    "provider": "d4-k2048",
                },
                {
                    "id": "d4-k4096",
                    "family": "vector_vq",
                    "dimension": 4,
                    "codebook_size": 4096,
                    "provider": "d4-k4096",
                },
            ],
            "anchor": {"bank": str(bank), "teacher": "model"},
            "prediction": {
                "class_caps": {
                    name: 1.0
                    for name in (
                        "agentic",
                        "chat",
                        "code",
                        "multilingual",
                        "prose",
                        "reasoning",
                    )
                }
            },
            "repair": {"method": "none"},
            "output": {
                "pack": str(tmp_path / "pack"),
                "model_id": "fixed-d4-native",
                "instance_id": "fixed-d4-native-r1",
            },
        }
    )

    inspect_backpack(plan, run_root=tmp_path / "run")
    generated = generate_backpack_candidates(plan, run_root=tmp_path / "run")

    assert {row["tier"] for row in generated["candidate_tiers"]} == {
        "d4-k2048",
        "d4-k4096",
    }
    for tier in generated["candidate_tiers"]:
        for cell in tier["cells"]:
            receipt = json.loads(Path(cell["receipt"]).read_text())
            assert receipt["algorithm"] == "exact-native-mxfp4-d4"
            assert receipt["basis_sha256"] == basis
            assert receipt["source_dtype"] == "packed-mxfp4-e2m1-with-e8m0-scales"
            decoded = np.load(Path(cell["receipt"]).parent / "decoded.npy")
            assert np.array_equal(decoded, np.load(model / f"{cell['cell_id']}.npy").reshape(-1))

    basis_index = model / "model.safetensors.index.json"
    changed = json.loads(basis_index.read_text())
    changed["metadata"] = {"revision": "changed"}
    basis_index.write_text(json.dumps(changed))
    with pytest.raises(BackpackPlanError, match="identity mismatch"):
        generate_backpack_candidates(plan, run_root=tmp_path / "run")
