from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .gate_only_trainer import CLASSES, TIERS

BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
_QTIP2_CONTAINER_INFLATION_BYTES = 2_709
_EXPERT_ENVELOPE_PADDING_BYTES = 2
_EXPERT_ENVELOPE_PADDING_SHA256 = "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gzip(payload: bytes) -> bytes:
    result = io.BytesIO()
    with gzip.GzipFile(fileobj=result, mode="wb", filename="", mtime=0) as handle:
        handle.write(payload)
    return result.getvalue()


def _native_artifact(layer: int, expert: int, projection: str) -> tuple[dict[str, Any], int]:
    prefix = f"layers.{layer}.ffn.experts.{expert}"
    if projection == "down":
        weights = [(f"{prefix}.w2.weight", 4_194_304), (f"{prefix}.w2.scale", 262_144)]
    else:
        weights = [
            (f"{prefix}.w1.weight", 4_194_304),
            (f"{prefix}.w1.scale", 262_144),
            (f"{prefix}.w3.weight", 4_194_304),
            (f"{prefix}.w3.scale", 262_144),
        ]
    components = [
        {
            "bytes": byte_count,
            "name": name,
            "storage": "MXFP4_E2M1" if name.endswith("weight") else "E8M0",
        }
        for name, byte_count in weights
    ]
    identity = {
        "basis_sha256": BASIS_SHA256,
        "components": components,
        "identity_kind": "basis-indexed-native-tensors",
    }
    wire_bytes = sum(value for _, value in weights)
    return (
        {
            "artifact_id": f"native:{layer:03d}:{expert:03d}:{projection}",
            "components": components,
            "identity_kind": identity["identity_kind"],
            "identity_sha256": _sha256_bytes(_canonical(identity)),
        },
        wire_bytes,
    )


def _qtip2_component_bytes(projection: str) -> dict[str, int]:
    if projection == "down":
        return {"SU": 4_096, "SV": 8_192, "Wscale": 4, "tlut": 4_096, "trellis": 2_097_152}
    return {"SU": 8_192, "SV": 8_192, "Wscale": 4, "tlut": 4_096, "trellis": 4_194_304}


def _load_qtip2(archive: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for layer in range(43):
            member = f"handoff/verification/L{layer:03d}_UNITS.jsonl"
            rows = [
                json.loads(line)
                for line in bundle.extractfile(member).read().decode().splitlines()  # type: ignore[union-attr]
                if line.strip()
            ]
            if len(rows) != 512:
                raise ValueError(f"QTIP2 {member} must contain 512 units")
            for row in rows:
                expert = int(row["expert"])
                projection = str(row["projection"])
                key = (layer, expert, projection)
                if key in result or row.get("basis_sha256") != BASIS_SHA256:
                    raise ValueError(f"QTIP2 identity/basis failure at {key}")
                component_bytes = _qtip2_component_bytes(projection)
                logical_bytes = sum(component_bytes.values())
                if int(row["artifact_bytes"]) != logical_bytes + _QTIP2_CONTAINER_INFLATION_BYTES:
                    raise ValueError(f"QTIP2 physical container byte mismatch at {key}")
                artifact = {
                    "artifact_id": f"qtip2:{layer:03d}:{expert:03d}:{projection}",
                    "canonical_packed_sha256": row["canonical_packed_sha256"],
                    "components": [
                        {"bytes": value, "name": name}
                        for name, value in sorted(component_bytes.items())
                    ],
                    "container_bytes": row["artifact_bytes"],
                    "container_sha256": row["artifact_sha256"],
                    "identity_kind": "physical-qtip2-unit-container",
                    "identity_sha256": row["artifact_sha256"],
                }
                result[key] = {"artifact": artifact, "wire_bytes": logical_bytes}
    if len(result) != 22_016:
        raise ValueError(f"QTIP2 archive covers {len(result)} cells, not 22,016")
    return result


def _load_qtip3(archive: Path) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as bundle:
        index_member = "results/qtip3-shipping-weight-index.json"
        index_payload = bundle.extractfile(index_member).read()  # type: ignore[union-attr]
        index = json.loads(index_payload)
    if index.get("basis_model_index_sha256") != BASIS_SHA256:
        raise ValueError("QTIP3 shipping index basis mismatch")
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in index.get("qtip_units", []):
        layer = int(row["layer"])
        expert = int(row["expert"])
        projection = str(row["projection"])
        key = (layer, expert, projection)
        if key in result:
            raise ValueError(f"duplicate QTIP3 cell {key}")
        components = row["components"]
        wire_bytes = sum(int(component["bytes"]) for component in components.values())
        identity = {
            "basis_sha256": BASIS_SHA256,
            "components": components,
            "name": row["name"],
            "shape": row["shape"],
        }
        artifact = {
            "artifact_id": f"qtip3:{layer:03d}:{expert:03d}:{projection}",
            "components": components,
            "identity_kind": "physical-qtip3-component-index",
            "identity_sha256": _sha256_bytes(_canonical(identity)),
        }
        result[key] = {"artifact": artifact, "wire_bytes": wire_bytes}
    if len(result) != 22_016:
        raise ValueError(f"QTIP3 archive covers {len(result)} cells, not 22,016")
    metadata = {
        "candidate_lineage": index["candidate_lineage"],
        "shared_tlut": index["shared_tlut"],
        "shipping_index_sha256": _sha256_bytes(index_payload),
    }
    return result, metadata


def _load_windows(qtip3_archive: Path) -> list[dict[str, Any]]:
    with tarfile.open(qtip3_archive, "r:gz") as bundle:
        member = bundle.extractfile("results/PUBLIC_HANDOFF.json")
        if member is None:
            raise ValueError("QTIP3 public handoff is missing")
        handoff = json.load(member)
    windows = handoff["public_lock"]["ordered_windows"]
    if len(windows) != 64:
        raise ValueError("QTIP3 public lock must contain 64 windows")
    return windows


def _data_manifest(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in CLASSES}
    result_rows = []
    for row in rows:
        class_name = str(row["source_class"])
        counts[class_name] += 1
        result_rows.append(
            {
                "class": class_name,
                "public_lock_ordinal": int(row["ordinal"]),
                "source_window_id": int(row["window_id"]),
                "window_id": f"balanced64-v1:{int(row['window_id']):03d}",
            }
        )
    return {
        "schema": "banana-smasher-gate-data-v1",
        "split": split,
        "class_counts": counts,
        "holdout": False,
        "rows": result_rows,
    }


def build_inputs(*, qtip2_archive: Path, qtip3_archive: Path, output_dir: Path) -> dict[str, Any]:
    qtip2_archive = qtip2_archive.expanduser().resolve()
    qtip3_archive = qtip3_archive.expanduser().resolve()
    qtip2 = _load_qtip2(qtip2_archive)
    qtip3, qtip3_metadata = _load_qtip3(qtip3_archive)
    cells = []
    tier_totals = {tier: 0 for tier in TIERS}
    for layer in range(43):
        for expert in range(256):
            for projection in ("down", "fused13"):
                key = (layer, expert, projection)
                native, native_bytes = _native_artifact(layer, expert, projection)
                tiers = {
                    "native_mxfp4": {
                        "artifacts": [native],
                        "wire_bytes": native_bytes,
                    },
                    "qtip2": {
                        "artifacts": [qtip2[key]["artifact"]],
                        "wire_bytes": qtip2[key]["wire_bytes"],
                    },
                    "qtip3": {
                        "artifacts": [qtip3[key]["artifact"]],
                        "wire_bytes": qtip3[key]["wire_bytes"],
                    },
                }
                for tier in TIERS:
                    tier_totals[tier] += int(tiers[tier]["wire_bytes"])
                cells.append(
                    {
                        "cell_id": f"L{layer:03d}.E{expert:03d}.{projection}",
                        "expert": expert,
                        "layer": layer,
                        "projection": projection,
                        "tiers": tiers,
                    }
                )
    if len(cells) != 22_016:
        raise AssertionError(len(cells))
    expected_totals = {
        "native_mxfp4": 147_169_738_752,
        "qtip2": 69_662_234_624,
        "qtip3": 104_200_230_912,
    }
    if tier_totals != expected_totals:
        raise ValueError(f"tier byte totals mismatch: {tier_totals} != {expected_totals}")
    physical = {
        "schema": "banana-smasher-ff0731-three-tier-cells-v1",
        "status": "PASS",
        "basis_sha256": BASIS_SHA256,
        "tiers": list(TIERS),
        "cell_count": len(cells),
        "tier_wire_bytes": tier_totals,
        "global_artifacts": {
            "qtip3_shared_tlut": qtip3_metadata["shared_tlut"],
            "expert_envelope_padding": {
                "artifact_id": "expert-envelope-alignment-padding-v1",
                "bytes": _EXPERT_ENVELOPE_PADDING_BYTES,
                "content_hex": "0000",
                "sha256": _EXPERT_ENVELOPE_PADDING_SHA256,
            },
        },
        "sources": {
            "qtip2_archive": {
                "bytes": qtip2_archive.stat().st_size,
                "sha256": _sha256_file(qtip2_archive),
            },
            "qtip3_archive": {
                "bytes": qtip3_archive.stat().st_size,
                "sha256": _sha256_file(qtip3_archive),
                **qtip3_metadata,
            },
        },
        "cells": cells,
    }
    physical_json = _canonical(physical)
    physical_gzip = _gzip(physical_json)
    physical_path = output_dir / "ff0731-three-tier-cells.json.gz"
    _atomic_write(physical_path, physical_gzip)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_windows(qtip3_archive):
        grouped[str(row["source_class"])].append(row)
    if set(grouped) != set(CLASSES) or any(len(grouped[name]) < 7 for name in CLASSES):
        raise ValueError("public lock cannot provide five TRAIN plus two DEV rows per class")
    train_rows = [row for name in CLASSES for row in grouped[name][:5]]
    dev_rows = [row for name in CLASSES for row in grouped[name][5:7]]
    train = _data_manifest("TRAIN", train_rows)
    dev = _data_manifest("DEV", dev_rows)
    train_path = output_dir / "FF0731_GATE_TRAIN.json"
    dev_path = output_dir / "FF0731_GATE_DEV.json"
    _atomic_write(train_path, _canonical(train))
    _atomic_write(dev_path, _canonical(dev))

    receipt = {
        "schema": "banana-smasher-ff0731-gate-input-seal-v1",
        "status": "PASS",
        "basis_sha256": BASIS_SHA256,
        "physical_manifest": {
            "path": physical_path.name,
            "bytes": physical_path.stat().st_size,
            "json_bytes": len(physical_json),
            "sha256": _sha256_file(physical_path),
            "cell_count": len(cells),
            "tier_wire_bytes": tier_totals,
        },
        "train_manifest": {
            "path": train_path.name,
            "bytes": train_path.stat().st_size,
            "sha256": _sha256_file(train_path),
            "class_counts": train["class_counts"],
        },
        "dev_manifest": {
            "path": dev_path.name,
            "bytes": dev_path.stat().st_size,
            "sha256": _sha256_file(dev_path),
            "class_counts": dev["class_counts"],
        },
        "train_dev_overlap": [],
        "holdout_untouched": True,
        "source_receipts": physical["sources"],
        "shared_artifacts_accounting": {
            "qtip3_shared_tlut_bytes": physical["global_artifacts"]["qtip3_shared_tlut"][
                "bytes"
            ],
            "treatment": "fixed metadata, excluded from per-cell expert tier bytes",
            "expert_envelope_padding_bytes": physical["global_artifacts"][
                "expert_envelope_padding"
            ]["bytes"],
            "expert_envelope_padding_sha256": physical["global_artifacts"][
                "expert_envelope_padding"
            ]["sha256"],
            "expert_envelope_padding_treatment": "physical expert-segment alignment bytes added after exact selected-cell payloads",
        },
    }
    receipt_path = output_dir / "INPUT_SEAL.json"
    _atomic_write(receipt_path, _canonical(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtip2-archive", type=Path, required=True)
    parser.add_argument("--qtip3-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_inputs(
        qtip2_archive=args.qtip2_archive,
        qtip3_archive=args.qtip3_archive,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
