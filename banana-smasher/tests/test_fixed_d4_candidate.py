from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.anchor import build_bank_manifest, materialize_bank, register_bank
from banana_smasher.cli import main
from banana_smasher.contract import export_pack
from banana_smasher.fixed_d4 import persist_fixed_d4_solve


def _bound_array(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / name
    np.save(path, value, allow_pickle=False)
    payload = path.read_bytes()
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.parametrize(
    ("tier", "k", "bits", "packed_row_bytes"),
    (("d4_k2048", 2048, 11, 7), ("d4_k4096", 4096, 12, 8)),
)
def test_fixed_d4_payload_can_produce_and_import_a_balanced64_candidate(
    tmp_path: Path,
    capsys,
    tier: str,
    k: int,
    bits: int,
    packed_row_bytes: int,
) -> None:
    basis_index = tmp_path / "model.safetensors.index.json"
    basis_index.write_text('{"weight_map": {}}')
    basis = hashlib.sha256(basis_index.read_bytes()).hexdigest()
    solve = tmp_path / "solve"
    projections: dict[str, dict[str, np.ndarray]] = {}
    for projection in ("fused13", "down"):
        assignments = np.tile(
            np.asarray([[0, k - 1, 1, 2, 3]], dtype=np.int16), (256, 1)
        )
        scales = np.full((256, 1), 127, dtype=np.uint8)
        codebook = np.arange(k * 4, dtype=np.float32).reshape(k, 4)
        projections[projection] = {
            "assignments": assignments,
            "scales": scales,
            "codebook": codebook,
        }
    persisted = persist_fixed_d4_solve(
        solve,
        tier=tier,
        layer=0,
        basis_index=basis_index,
        basis_sha256=basis,
        projections=projections,
    )
    manifest_path = Path(persisted["manifest"])
    persisted_manifest = json.loads(manifest_path.read_text())
    assert persisted["assignment_count"] == 5 * 256 * 2
    assert persisted_manifest["basis_sha256"] == basis
    for projection in ("fused13", "down"):
        assignment_binding = persisted_manifest["projections"][projection]["assignments"]
        assignment_path = manifest_path.parent / assignment_binding["path"]
        assert assignment_path.stat().st_size == assignment_binding["bytes"]
        assert hashlib.sha256(assignment_path.read_bytes()).hexdigest() == assignment_binding["sha256"]
    wire = tmp_path / "wire"

    assert main(
        [
            "fixed-d4",
            "materialize",
            "--manifest",
            str(manifest_path),
            "--output",
            str(wire),
            "--basis-sha256",
            basis,
        ]
    ) == 0
    materialized = json.loads(capsys.readouterr().out)
    layer = wire / "layer_000"
    receipt = json.loads((layer / "LAYER_RECEIPT.json").read_text())
    assert materialized["basis_sha256"] == receipt["basis_sha256"] == basis
    assert receipt["tier"] == tier
    assert receipt["assignment_count"] == 5 * 256 * 2
    packed = (layer / f"{tier}.down.codes.le{bits}.bin").read_bytes()
    assert len(packed) == 256 * packed_row_bytes

    model = tmp_path / "model"
    pack_manifest = export_pack(
        source_root=wire,
        output=model,
        model_id=f"fixture-{tier}",
        instance_id=f"fixture-{tier}-1",
        runtime_floor_bytes=0,
        link_mode="copy",
    )
    assert pack_manifest["source_format"] == "banana_smasher-materialized-wire-v1"

    parent_rows = [
        {"window_id": index, "class": "fixture", "tokens": [index]}
        for index in range(64)
    ]
    parent = tmp_path / "balanced64.jsonl"
    parent.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in parent_rows)
    )
    parent_payload = parent.read_bytes()
    identity = {
        "status": "resolved",
        "sha256": hashlib.sha256(parent_payload).hexdigest(),
        "uri": "fixture://balanced64",
    }
    bank = build_bank_manifest(
        bank_id="balanced64",
        role="train_balanced64",
        windows=[{"id": row["window_id"], "class": row["class"]} for row in parent_rows],
        parent_corpus=identity,
        identities={
            "corpus": identity,
            "tokenizer": {"status": "resolved", "sha256": "1" * 64, "uri": "fixture://tokenizer"},
            "teacher": {"status": "unresolved", "reason": "not needed for production"},
            "scorer": {"status": "resolved", "sha256": "2" * 64, "uri": "fixture://scorer"},
        },
        split_lineage={"split": "train", "parent_bank_id": None},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )
    run_root = tmp_path / "anchor-run"
    register_bank(run_root, bank)
    materialize_bank(bank, parent, run_root / "banks" / "balanced64.jsonl")

    producer = tmp_path / "producer.py"
    producer.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--model', required=True)\n"
        "p.add_argument('--config', required=True)\n"
        "p.add_argument('--bank', required=True)\n"
        "p.add_argument('--output', required=True)\n"
        "p.add_argument('--basis-sha256', required=True)\n"
        "a=p.parse_args()\n"
        "rows=[json.loads(line) for line in open(a.bank) if line.strip()]\n"
        "with open(a.output, 'w') as sink:\n"
        "  for row in rows:\n"
        "    sink.write(json.dumps({'window_id': row['window_id'], 'logits': [1.0, 0.0]}, sort_keys=True)+'\\n')\n"
    )
    producer_config = tmp_path / "producer.json"
    producer_config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-candidate-producer-v1",
                "command": [sys.executable, str(producer)],
                "parameters": {"temperature": 0},
            }
        )
    )

    assert main(
        [
            "anchor",
            "materialize-candidate",
            "--run-root",
            str(run_root),
            "--bank",
            "balanced64",
            "--candidate-id",
            tier,
            "--model",
            str(model),
            "--config",
            str(producer_config),
            "--basis-sha256",
            basis,
        ]
    ) == 0
    produced = json.loads(capsys.readouterr().out)
    assert produced["coverage"] == "64/64"
    assert produced["basis_sha256"] == basis
    imported = run_root / produced["relative_path"]
    assert len(imported.read_text().splitlines()) == 64

    (model / "config.json").write_text("{}")
    assert main(
        [
            "anchor",
            "materialize-candidate",
            "--run-root",
            str(run_root),
            "--bank",
            "balanced64",
            "--candidate-id",
            f"tampered-{tier}",
            "--model",
            str(model),
            "--config",
            str(producer_config),
            "--basis-sha256",
            basis,
        ]
    ) == 2
    assert "candidate model pack verification failed" in capsys.readouterr().err


def test_fixed_d4_materialization_refuses_basis_index_mismatch(
    tmp_path: Path, capsys
) -> None:
    actual_index = tmp_path / "model.safetensors.index.json"
    actual_index.write_text('{"weight_map": {}}')
    basis = "4" * 64
    projections: dict[str, dict[str, object]] = {}
    for projection in ("fused13", "down"):
        projections[projection] = {
            "assignments": _bound_array(
                tmp_path,
                f"{projection}.assignments.npy",
                np.zeros((256, 1), dtype=np.int16),
            ),
            "scales": _bound_array(
                tmp_path,
                f"{projection}.scales.npy",
                np.zeros((256, 1), dtype=np.uint8),
            ),
            "codebook": _bound_array(
                tmp_path,
                f"{projection}.codebook.npy",
                np.zeros((4096, 4), dtype=np.float16),
            ),
        }
    manifest = tmp_path / "materialize.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-fixed-d4-materialization-v1",
                "tier": "d4_k4096",
                "layer": 0,
                "basis_sha256": basis,
                "basis_index": {"path": str(actual_index), "sha256": basis},
                "projections": projections,
            }
        )
    )

    assert main(
        [
            "fixed-d4",
            "materialize",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "wire"),
            "--basis-sha256",
            basis,
        ]
    ) == 2
    assert "basis_index SHA-256 mismatch" in capsys.readouterr().err
