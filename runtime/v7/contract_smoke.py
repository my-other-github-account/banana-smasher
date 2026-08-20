#!/usr/bin/env python3
"""Fast V7 runtime-closure check; never loads the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}: not a local regular file: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{label}: not a local directory: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--member-roster", type=Path, required=True)
    ap.add_argument("--expected-member-roster-sha256", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--delta-dir", type=Path, required=True)
    ap.add_argument("--vq3b-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)

    ap.add_argument("--lp4-pack", type=Path, required=True)
    ap.add_argument("--lp4-selection", type=Path)
    ap.add_argument("--compile-native", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    runner = root / "runner"
    vendor = root / "vendor"
    for rel in (
        "runner/fast_two_node_v7.py",
        "runner/fast_k2_grouped.py",
        "runner/fast_k2_grouped.cpp",
        "runner/fast_k2_grouped_kernel.cu",
        "runner/fast_v7_expert_base.py",
        "runner/base_binrepair_e2e.py",
        "runner/qtip_v7_repair.py",
        "runner/joint_v7_runtime_adapter.py",
        "runner/joint_v7_expert_base.py",
        "vendor/src_lp4/lp4_train.py",
        "vendor/src_lp4/lp4_pack.py",
        "vendor/src/t8192_ds4_build_v3.py",
        "vendor/site/banana_smasher/qtip_k2.py",
        "code/JOINT_REPAIR_ADMISSION.json",
    ):
        require_file(root / rel, rel)

    for path, label in (
        (args.model_root, "model root"),
        (args.vq3b_dir, "VQ3B compatibility root"),
        (args.lp4_pack, "LP4 pack"),
        (args.teacher, "teacher root"),
    ):
        require_dir(path.resolve(), label)
    for path, label in (
        (args.manifest, "LP4 manifest"),
        (args.delta_dir / "DELTA_PACK.COMPLETE", "delta completion marker"),
        (args.corpus, "TRAIN corpus"),
        (args.member_roster, "all-layer member roster"),
    ):
        require_file(path.resolve(), label)
    if (
        hashlib.sha256(args.member_roster.read_bytes()).hexdigest()
        != args.expected_member_roster_sha256
    ):
        raise RuntimeError("all-layer member roster SHA-256 drift")
    roster = json.loads(args.member_roster.read_text())
    rows = roster.get("members")
    expected_coordinates = {
        (layer, expert, projection)
        for layer in range(43)
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
    }
    observed_coordinates = (
        {
            (int(row["layer"]), int(row["expert"]), str(row["projection"]))
            for row in rows
        }
        if isinstance(rows, list)
        else set()
    )
    if (
        roster.get("schema") != "banana-smasher-qtip2-v7-selected-wire-roster-v2"
        or roster.get("member_count") != len(expected_coordinates)
        or len(rows or ()) != len(expected_coordinates)
        or observed_coordinates != expected_coordinates
    ):
        raise RuntimeError("all-layer member roster composition drift")
    if args.lp4_selection is not None:
        require_file(args.lp4_selection.resolve(), "LP4 selection")

    index = args.model_root.resolve() / "model.safetensors.index.json"
    require_file(index, "model index")
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        require_file(args.model_root.resolve() / shard, f"native shard {shard}")

    os.environ.setdefault("V7_LP4_ROOT", str(vendor))
    os.environ.setdefault("V7_VENDOR_ROOT", str(vendor))
    os.environ.setdefault("BANANA_SMASHER_PUBLIC_SRC", str(vendor / "site"))
    os.environ.setdefault("BR_MANIFEST", str(args.manifest.resolve()))
    os.environ.setdefault("BR_DELTA_DIR", str(args.delta_dir.resolve()))
    os.environ.setdefault("BR_VQ3B_DIR", str(args.vq3b_dir.resolve()))
    os.environ.setdefault("BR_CORPUS", str(args.corpus.resolve()))
    os.environ.setdefault("BR_TEACH", str(args.teacher.resolve()))
    os.environ.setdefault("BR_TRAIN", ",".join(str(x) for x in range(20, 84)))
    os.environ.setdefault("BR_PROBE", "20")
    sys.path[:0] = [
        str(runner.resolve()),
        str(vendor / "src_lp4"),
        str(vendor / "src"),
        str(vendor / "site"),
    ]

    import fast_k2_grouped  # noqa: PLC0415
    import banana_smasher.qtip_k2  # noqa: PLC0415,F401
    import lp4_train  # noqa: PLC0415,F401
    import lp4_pack  # noqa: PLC0415,F401
    import t8192_ds4_build_v3  # noqa: PLC0415,F401

    if args.compile_native:
        fast_k2_grouped._cuda_extension()

    print(
        json.dumps(
            {
                "schema": "banana-smasher-v7-runtime-closure-v1",
                "status": "PASS_NO_MODEL_LOAD",
                "root": str(root),
                "runner": str(runner / "fast_two_node_v7.py"),
                "native_sources": ["fast_k2_grouped.cpp", "fast_k2_grouped_kernel.cu"],
                "model_shards": len(shards),
                "vendored_modules": True,
                "native_extension": bool(args.compile_native),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
