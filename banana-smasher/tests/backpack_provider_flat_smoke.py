from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import banana_smasher
from banana_smasher import (
    BackpackPlan,
    anchor_backpack_candidates,
    bind_native_mxfp4_backpack_candidate,
    build_backpack,
    builtin_backpack_family_providers,
    generate_backpack_candidates,
    inspect_backpack,
    predict_backpack,
    price_backpack_candidate,
    qtip1_5_provider_declaration,
    solve_backpack,
    verify_pack,
)

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def fixture_plan(root: Path) -> dict[str, object]:
    model = root / "model"
    model.mkdir(parents=True)
    rng = np.random.default_rng(7)
    cells: list[dict[str, object]] = []
    weights: list[np.ndarray] = []
    for expert_ids in (range(128), range(128, 256)):
        for projection in ("fused13", "down"):
            index = len(cells)
            value = rng.normal(size=(len(expert_ids), 16)).astype(np.float32)
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
    weight_count = sum(array.size for array in weights)
    (model / "BACKPACK_MODEL.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-model-v1",
                "revision": "flat-smoke-r1",
                "weight_count": weight_count,
                "dense_bytes": 0,
                "metadata_bytes": 0,
                "repair_bytes": 0,
                "cells": cells,
            }
        )
        + "\n"
    )
    bank = root / "anchor64.npz"
    np.savez(
        bank,
        features=rng.normal(size=(64, weight_count)).astype(np.float32),
        classes=np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)]),
    )
    return {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {"root": str(model), "revision": "flat-smoke-r1"},
        "target": {"exact_bytes": 53344},
        "tiers": [
            {"id": "d4-k4", "family": "vector_vq", "dimension": 4, "bits": 2},
            {"id": "d8-k4", "family": "vector_vq", "dimension": 8, "bits": 2},
            {
                "id": "qtip-2.0",
                "family": "qtip",
                "bpw": 2.0,
                "backend": "fixture_reference",
            },
        ],
        "anchor": {"bank": str(bank), "teacher": "model"},
        "prediction": {"class_caps": {name: 100.0 for name in CLASSES}},
        "repair": {"method": "none"},
        "output": {
            "pack": str(root / "final-pack"),
            "model_id": "backpack-flat-smoke",
            "instance_id": "backpack-flat-smoke-v1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    plan = BackpackPlan.from_mapping(fixture_plan(root))
    run_root = root / "run"
    inspect_backpack(plan, run_root=run_root)
    generated = generate_backpack_candidates(plan, run_root=run_root)
    anchor_backpack_candidates(plan, run_root=run_root)
    predicted = predict_backpack(plan, run_root=run_root)
    solved = solve_backpack(plan, run_root=run_root)
    verified = verify_pack(Path(solved["pre_repair_pack"]))
    completed = build_backpack(plan, run_root=run_root)

    native_receipt = bind_native_mxfp4_backpack_candidate(
        run_root,
        tier={"id": "native", "family": "native_mxfp4"},
        cell={"cell_id": "source-cell"},
    )
    native_price = price_backpack_candidate(native_receipt)
    providers = builtin_backpack_family_providers()
    qtip15 = qtip1_5_provider_declaration()
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "banana_smasher.cli",
            "backpack",
            "status",
            "--run-root",
            str(run_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_status = json.loads(cli.stdout)
    result = {
        "status": "PASS",
        "module": str(Path(banana_smasher.__file__).resolve()),
        "python": sys.executable,
        "providers": sorted(providers),
        "qtip15_provider": qtip15.tier,
        "native_incremental_bytes": native_price.full_wire_bytes,
        "candidate_tiers": [row["tier"] for row in generated["candidate_tiers"]],
        "prediction_rows": len(predicted["rows"]),
        "assignment_count": len(solved["assignment"]),
        "pack_status": verified["status"],
        "final_status": completed["status"],
        "cli_status": cli_status["status"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
