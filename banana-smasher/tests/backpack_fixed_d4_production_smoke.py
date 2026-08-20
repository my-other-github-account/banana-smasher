from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import banana_smasher
from banana_smasher import BackpackPlan, verify_pack
from banana_smasher.backpack import _build_backpack as build_backpack

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


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


def _candidate_payload_bytes(*, codebook_size: int, vectors_per_expert: int) -> int:
    bits = 11 if codebook_size == 2048 else 12
    records = 256
    return (
        records * ((vectors_per_expert * bits + 7) // 8)
        + records * codebook_size * 4 * np.dtype(np.float16).itemsize
        + records * (vectors_per_expert // 8)
        + records * np.dtype(np.int16).itemsize
        + (records + 1) * 3 * np.dtype(np.int64).itemsize
        + records * 32
        + records * 3 * np.dtype(np.int32).itemsize
        + records * 8
    )


def _fixed_d4_plan(root: Path, *, codebook_size: int) -> dict[str, object]:
    model = root / f"model-k{codebook_size}"
    model.mkdir(parents=True)
    shard = model / "model-00001-of-00001.safetensors"
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    weight_map: dict[str, str] = {}
    scale_byte = 128  # E8M0 factor 2.0; non-unity catches dropped-scale decoding.
    normalized = np.tile(np.asarray([1.0, 2.0], dtype=np.float32), 16)
    down_rows: list[np.ndarray] = []
    fused_rows: list[np.ndarray] = []
    for expert in range(256):
        expert_parts: dict[str, np.ndarray] = {}
        for weight in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{weight}"
            tensors[f"{prefix}.weight"] = ("I8", [1, 16], bytes([0x42]) * 16)
            tensors[f"{prefix}.scale"] = ("F8_E8M0", [1, 1], bytes([scale_byte]))
            weight_map[f"{prefix}.weight"] = shard.name
            weight_map[f"{prefix}.scale"] = shard.name
            expert_parts[weight] = normalized * 2.0
        down_rows.append(expert_parts["w2"])
        fused_rows.append(np.concatenate([expert_parts["w1"], expert_parts["w3"]]))
    _write_safetensors(shard, tensors)
    basis_index = model / "model.safetensors.index.json"
    basis_index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    down = np.stack(down_rows).astype(np.float32)
    fused = np.stack(fused_rows).astype(np.float32)
    np.save(model / "down.npy", down, allow_pickle=False)
    np.save(model / "fused13.npy", fused, allow_pickle=False)
    cells = [
        {
            "cell_id": "layer-0-down",
            "path": "down.npy",
            "feature_slice": [0, down.size],
            "layer": 0,
            "projection": "down",
            "expert_ids": list(range(256)),
        },
        {
            "cell_id": "layer-0-fused13",
            "path": "fused13.npy",
            "feature_slice": [down.size, down.size + fused.size],
            "layer": 0,
            "projection": "fused13",
            "expert_ids": list(range(256)),
        },
    ]
    (model / "BACKPACK_MODEL.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-model-v1",
                "revision": f"native-fixed-d4-k{codebook_size}",
                "weight_count": down.size + fused.size,
                "dense_bytes": 0,
                "metadata_bytes": 0,
                "repair_bytes": 0,
                "cells": cells,
            }
        )
        + "\n"
    )
    rng = np.random.default_rng(codebook_size)
    bank = root / f"anchor64-k{codebook_size}.npz"
    np.savez(
        bank,
        features=rng.normal(scale=1e-3, size=(64, down.size + fused.size)).astype(
            np.float32
        ),
        classes=np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)]),
    )
    exact_bytes = (
        256
        + _candidate_payload_bytes(codebook_size=codebook_size, vectors_per_expert=8)
        + _candidate_payload_bytes(codebook_size=codebook_size, vectors_per_expert=16)
    )
    return {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {
            "root": str(model),
            "revision": f"native-fixed-d4-k{codebook_size}",
        },
        "target": {"exact_bytes": exact_bytes},
        "tiers": [
            {
                "id": f"d4-k{codebook_size}",
                "family": "vector_vq",
                "provider": f"d4-k{codebook_size}",
                "dimension": 4,
                "codebook_size": codebook_size,
            }
        ],
        "anchor": {"bank": str(bank), "teacher": "model"},
        "prediction": {"class_caps": {name: 1.0 for name in CLASSES}},
        "repair": {"method": "none"},
        "output": {
            "pack": str(root / f"final-pack-k{codebook_size}"),
            "model_id": f"native-fixed-d4-k{codebook_size}",
            "instance_id": f"native-fixed-d4-k{codebook_size}",
        },
    }


def _run_tier(root: Path, codebook_size: int) -> dict[str, object]:
    plan = BackpackPlan.from_mapping(
        _fixed_d4_plan(root / f"k{codebook_size}", codebook_size=codebook_size)
    )
    run_root = root / f"run-k{codebook_size}"
    result = build_backpack(plan, run_root=run_root)
    receipts = [
        json.loads(path.read_text())
        for path in sorted(
            (run_root / "candidates" / f"d4-k{codebook_size}").glob("*/RECEIPT.json")
        )
    ]
    if len(receipts) != 2:
        raise RuntimeError(
            f"K{codebook_size} did not produce both projection candidates"
        )
    basis = hashlib.sha256(
        (Path(plan.model["root"]) / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    for receipt in receipts:
        if (
            receipt.get("algorithm") != "exact-native-mxfp4-d4"
            or receipt.get("source_dtype") != "packed-mxfp4-e2m1-with-e8m0-scales"
            or receipt.get("basis_sha256") != basis
            or receipt.get("codebook_size") != codebook_size
        ):
            raise RuntimeError(
                f"K{codebook_size} candidate lost production fixed-D4 identity"
            )
        decoded = np.load(Path(receipt["decoded"]["path"]), allow_pickle=False)
        source = np.load(
            Path(plan.model["root"])
            / ("down.npy" if receipt["projection"] == "down" else "fused13.npy"),
            allow_pickle=False,
        )
        if not np.array_equal(decoded, source.reshape(-1)):
            raise RuntimeError(f"K{codebook_size} candidate dropped its E8M0 scales")
    if result["pre_repair_anchor"] != result["final_anchor"]:
        raise RuntimeError(
            f"K{codebook_size} exported-wire score drifted from its candidate"
        )
    verified = verify_pack(Path(result["final_pack"]))
    return {
        "algorithm": receipts[0]["algorithm"],
        "basis_sha256": basis,
        "candidate_cells": len(receipts),
        "final_status": result["status"],
        "pack_status": verified["status"],
        "whole_model_bytes": result["byte_accounting"]["whole_model_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    tiers = {str(size): _run_tier(root, size) for size in (2048, 4096)}
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "banana_smasher.cli",
            "backpack",
            "status",
            "--run-root",
            str(root / "run-k2048"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = {
        "status": "PASS",
        "module": str(Path(banana_smasher.__file__).resolve()),
        "python": sys.executable,
        "cli_status": json.loads(cli.stdout)["status"],
        "nonunity_e8m0_scale": 128,
        "tiers": tiers,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
