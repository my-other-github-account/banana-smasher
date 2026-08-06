from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from banana_smasher import (
    BackpackFamilyBinding,
    BackpackPlan,
    build_backpack,
    generate_backpack_candidates,
    inspect_backpack,
    list_backpack_family_bindings,
    price_backpack_selection,
    verify_pack,
)


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _fixed_assignment_plan(tmp_path: Path) -> tuple[dict[str, object], Path]:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    cells: list[dict[str, object]] = []
    weights: list[np.ndarray] = []
    for expert_ids in (range(128), range(128, 256)):
        for projection in ("fused13", "down"):
            index = len(cells)
            value = np.zeros((len(expert_ids), 16), dtype=np.float32)
            np.save(model / f"cell{index}.npy", value, allow_pickle=False)
            start = sum(array.size for array in weights)
            weights.append(value)
            cells.append(
                {
                    "cell_id": f"cell{index}",
                    "path": f"cell{index}.npy",
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
                "revision": "fixed-fixture-r1",
                "weight_count": sum(value.size for value in weights),
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
        features=np.zeros((64, sum(value.size for value in weights)), dtype=np.float32),
        classes=np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)]),
    )

    package = tmp_path / "packaged-qtip"
    package.mkdir()
    member = package / "member.npz"
    np.savez(
        member,
        trellis=np.asarray([[0, 1]], dtype=np.uint16),
        SU=np.ones(4, dtype=np.float16),
        SV=np.ones(4, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float16),
        tlut=np.ones((4, 2), dtype=np.float16),
        reconstructed_weight=np.zeros(16, dtype=np.float16),
    )
    artifact = {
        "path": member.name,
        "bytes": member.stat().st_size,
        "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
    }
    manifest = package / "members.jsonl"
    with manifest.open("w") as stream:
        for projection in ("fused13", "down"):
            for expert in range(256):
                stream.write(
                    json.dumps(
                        {
                            "layer": 0,
                            "expert": expert,
                            "projection": projection,
                            "tier": "fixture-qtip2",
                            "geometry": {"L": 16, "K": 2, "V": 2},
                            "artifact": artifact,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan: dict[str, object] = {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {"root": str(model), "revision": "fixed-fixture-r1"},
        "target": {"exact_bytes": 1_000_000},
        "tiers": [
            {
                "id": "fixture-qtip2",
                "family": "qtip",
                "provider": "packaged_qtip",
                "runtime_family": "qtip2",
                "bpw": 2.0,
                "backend": "packaged_qtip",
                "fixed_assignment": {
                    "path": str(manifest),
                    "sha256": manifest_sha,
                    "member_root": str(package),
                },
            }
        ],
        "anchor": {"bank": str(bank), "teacher": "model"},
        "prediction": {"class_caps": {name: 100.0 for name in CLASSES}},
        "repair": {"method": "none"},
        "output": {
            "pack": str(tmp_path / "final-pack"),
            "model_id": "generic-fixed-qtip",
            "instance_id": "generic-fixed-qtip-1",
        },
    }
    return plan, manifest


def test_non_ff_generic_fixed_assignment_packaged_qtip_uses_public_lifecycle(
    tmp_path: Path,
) -> None:
    plan, manifest = _fixed_assignment_plan(tmp_path)
    parsed = BackpackPlan.from_mapping(plan)
    probe_root = tmp_path / "probe"
    inspected = inspect_backpack(parsed, run_root=probe_root)
    generated = generate_backpack_candidates(parsed, run_root=probe_root)
    assignments = [
        {"cell_id": cell_id, "tier": "fixture-qtip2"}
        for cell_id in inspected["cell_ids"]
    ]
    priced = price_backpack_selection(
        parsed,
        assignment=assignments,
        candidates=generated,
    )
    plan["target"] = {
        "exact_bytes": inspected["fixed_total_bytes"] + priced["payload_bytes"]
    }

    result = build_backpack(plan, run_root=tmp_path / "run")

    assert result["status"] == "PASS"
    assert result["family_counts"] == {"packaged_qtip": 4}
    assert priced["activation_bytes"] == manifest.stat().st_size
    assert len(priced["activation_artifacts"]) == 1
    assert verify_pack(Path(result["final_pack"]))["status"] == "PASS"
    pack_manifest = json.loads(
        (Path(result["final_pack"]) / "BANANA_PACK_MANIFEST.json").read_text()
    )
    accounting = pack_manifest["backpack_byte_accounting"]
    assert accounting["provider_activation_bytes"] == manifest.stat().st_size
    assert accounting["whole_model_bytes"] == plan["target"]["exact_bytes"]
    resumed = build_backpack(plan, run_root=tmp_path / "run")
    assert resumed["resumed_stages"] == resumed["stages"]


def test_public_provider_registry_has_real_generic_builtins() -> None:
    bindings = {row.provider: row for row in list_backpack_family_bindings()}
    assert {
        "native_mxfp4",
        "packaged_qtip",
        "d4_k2048",
        "d4_k4096",
    } <= set(bindings)
    assert all(isinstance(row, BackpackFamilyBinding) for row in bindings.values())
    assert all(
        callable(function)
        for row in bindings.values()
        for function in (row.generate, row.materialize, row.price, row.predict, row.verify)
    )
