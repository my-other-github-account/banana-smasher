from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from repair_api.sealed_pre_forward import BASIS_SHA256, atomic_json, sha256

WIRE_BYTES = 2_109_444


def member_set(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        file_sha = sha256(path)
        digest.update(path.name.encode() + b"\0" + str(WIRE_BYTES).encode() + b"\0" + file_sha.encode() + b"\n")
    return digest.hexdigest()


def stage_verified_cache(*, source_root: Path, cache_root: Path, freeze_path: Path,
                         receipt_path: Path) -> dict:
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if hashlib.sha256(freeze_raw).hexdigest() != "2dcc28497deb834164be26e267fdf4c30cc951342c73f47ce78b207354275fc9":
        raise RuntimeError("FREEZE_MANIFEST_IDENTITY_REFUSED")
    if freeze.get("basis_sha256") != BASIS_SHA256:
        raise RuntimeError("FREEZE_BASIS_REFUSED")
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    rows = []
    for source_row in freeze["layers"]:
        layer = int(source_row["layer"])
        if layer == 34:
            continue
        source = source_root / f"L{layer:03d}"
        files = sorted(path for path in source.iterdir() if path.is_file())
        expected_names = {f"E{expert:03d}_w{projection}" for expert in range(256) for projection in (1, 2, 3)}
        observed_names = {path.stem for path in files}
        if len(files) != 768 or observed_names != expected_names:
            raise RuntimeError(f"LAYER_INVENTORY_REFUSED_L{layer:03d}:{len(files)}")
        if any(path.stat().st_size != WIRE_BYTES for path in files):
            raise RuntimeError(f"LAYER_SIZE_REFUSED_L{layer:03d}")
        observed_set = member_set(files)
        physical = source_row["physical_census"]
        expected_set = physical.get("compact_member_set_sha256")
        if expected_set and observed_set != expected_set:
            raise RuntimeError(f"LAYER_MEMBER_SET_REFUSED_L{layer:03d}:{observed_set}")
        sentinels = physical.get("sentinels", [])
        sentinel_results = []
        by_stem = {path.stem: path for path in files}
        for sentinel in sentinels:
            if "relative_path" not in sentinel:
                continue
            relative = Path(sentinel["relative_path"])
            stem = f"{relative.parent.name}_{relative.stem}" if relative.parent.name.startswith("E") else relative.stem
            path = by_stem[stem]
            observed = sha256(path)
            if observed != sentinel["sha256"]:
                raise RuntimeError(f"LAYER_SENTINEL_REFUSED_L{layer:03d}:{stem}")
            sentinel_results.append({"member": stem, "sha256": observed})
        destination = cache_root / f"L{layer:03d}"
        wire = destination / "wire"
        wire.mkdir(parents=True)
        layout = source_row.get("route", {}).get("layout", "flat")
        for path in files:
            if layout == "nested":
                expert, projection = path.stem.split("_", 1)
                target = wire / expert / f"{projection}{path.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                target = wire / path.name
            os.link(path, target)
        marker = {
            "schema": "banana-smasher-verified-plane-cache-v1",
            "status": "PASS", "layer": layer, "basis_sha256": BASIS_SHA256,
            "files": 768, "wire_bytes": 768 * WIRE_BYTES,
            "compact_member_set_sha256": observed_set,
            "expected_compact_member_set_sha256": expected_set,
            "source_terminal_sha256": source_row.get("terminal_sha256"),
            "freeze_manifest_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        }
        atomic_json(destination / "CACHE_ADMISSION.json", marker)
        rows.append({
            "layer": layer, "status": "PASS", "source": str(source),
            "cache": str(destination), "files": 768, "wire_bytes": 768 * WIRE_BYTES,
            "compact_member_set_sha256": observed_set,
            "expected_compact_member_set_sha256": expected_set,
            "sentinels_verified": sentinel_results,
            "source_terminal_sha256": source_row.get("terminal_sha256"),
        })
    if len(rows) != 42:
        raise RuntimeError(f"LAYER_COVERAGE_REFUSED:{len(rows)}")
    receipt = {
        "schema": "banana-smasher-42-layer-plane-reachability-hash-v1",
        "status": "PASS", "basis_sha256": BASIS_SHA256,
        "freeze_manifest_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "source_root": str(source_root), "cache_root": str(cache_root),
        "layers_verified": 42, "layer_34": "SEPARATE_PRESERVED_PROVIDER",
        "rows": rows, "created_unix": time.time(),
    }
    receipt["receipt_sha256"] = atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_verified_cache(source_root=args.source_root, cache_root=args.cache_root,
                                          freeze_path=args.freeze, receipt_path=args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
