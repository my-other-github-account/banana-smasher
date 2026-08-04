#!/usr/bin/env python3
"""Rebuild retained custom kernels into an isolated directory and verify them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = Path(__file__).resolve().parent
SOURCES = KERNEL_ROOT / "sources"
LEDGER = ROOT / "notes/GOLDEN_CLOSURE_REUSE_LEDGER.json"
CUBIT_REPOSITORY = "https://github.com/kacper-daftcode/cubit.git"
CUBIT_REVISION = "c139df8b34f1dcab607f8ccb685fdea948f3ae4d"
CUBIT_CARGO_LOCK_SHA256 = "12af4b62563b9b8812ce87e015d63c9871e3e5d4b6371538260441a81963ed58"
CUBIT_SM120_TABLE_SHA256 = "fb28d81187e56fbe1e468e6f0356267c162d0fa12262e40d21069563e7fc918f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}"
        )


def verify_inputs(
    ledger: dict[str, object],
    artifacts: list[dict[str, object]],
    selected_families: set[str],
) -> None:
    families = ledger["families"]
    assert isinstance(families, dict)
    for family_name in selected_families:
        family = families[family_name]
        assert isinstance(family, dict)
        source_root = ROOT / str(family["source_root"])
        retained_hashes = family.get("retained_hashes", {})
        assert isinstance(retained_hashes, dict)
        for relative, expected in retained_hashes.items():
            verify_hash(source_root / str(relative), str(expected), "retained source")

    for artifact in artifacts:
        path = ROOT / str(artifact["path"])
        source = ROOT / str(artifact["source"])
        verify_hash(path, str(artifact["sha256"]), "admitted binary")
        verify_hash(source, str(artifact["source_sha256"]), "producing source")
        for key, label in (
            ("reference_sass", "reference SASS"),
            ("mercury_stub", "Mercury stub"),
        ):
            relative = artifact.get(key)
            if relative and not (ROOT / str(relative)).is_file():
                raise SystemExit(f"missing {label}: {relative}")


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def checked_output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def prepare_output(raw: str) -> Path:
    output = Path(raw).expanduser().resolve()
    protected = (ROOT.resolve(), KERNEL_ROOT.resolve(), SOURCES.resolve())
    if any(output == path or path in output.parents for path in protected):
        raise SystemExit(f"refusing output inside the source repository: {output}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_cubit(output: Path) -> Path:
    checkout = output / "toolchain/cubit"
    checkout.parent.mkdir(parents=True)
    run(["git", "init", str(checkout)])
    run(["git", "-C", str(checkout), "remote", "add", "origin", CUBIT_REPOSITORY])
    run(["git", "-C", str(checkout), "fetch", "--depth=1", "origin", CUBIT_REVISION])
    run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"])
    actual = checked_output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != CUBIT_REVISION:
        raise SystemExit(f"Cubit basis mismatch: expected {CUBIT_REVISION}, got {actual}")
    verify_hash(checkout / "Cargo.lock", CUBIT_CARGO_LOCK_SHA256, "Cubit Cargo.lock")
    verify_hash(
        checkout / "tables/sm120.json",
        CUBIT_SM120_TABLE_SHA256,
        "Cubit SM120 ISA table",
    )
    run(["cargo", "build", "-q", "--release"], cwd=checkout)
    binary = checkout / "target/release/cubit"
    if not binary.is_file():
        raise SystemExit(f"Cubit build did not produce {binary}")
    return binary


def cubit_root(binary: Path) -> Path:
    for candidate in binary.parents:
        if (candidate / "tables/sm120.json").is_file():
            return candidate
    raise SystemExit(f"cannot locate Cubit tables above {binary}")


def generate_sass(artifact: dict[str, object], output: Path) -> Path:
    family = str(artifact["family"])
    relative = Path(str(artifact["path"]))
    sass = output / "generated-sass" / relative.with_suffix(".sass").name
    sass.parent.mkdir(parents=True, exist_ok=True)
    k = int(str(artifact["k"]))
    variant = str(artifact["variant"])
    env = os.environ.copy()
    if family == "W2":
        generator = SOURCES / "w2/gen/gen_moe_w2.py"
        env["MOEW2_MC"] = {"base": "1", "mc2": "2", "mc4": "4", "mc4afrag": "4"}[variant]
        env["MOEW2_AFRAG"] = "1" if variant == "mc4afrag" else "0"
    elif family == "W4":
        generator = SOURCES / "w2/gen/gen_moe_w4.py"
    elif family == "W3":
        generator = SOURCES / "w3/gen_moe_w3.py"
        env["MOEW3_MC"] = "4" if variant != "base" else "1"
        env["MOEW3_AFRAG"] = "1" if variant == "mc4afrag" else "0"
        env["MOEW3_LUT_LO"] = "0xb6bfc6cd"
        env["MOEW3_LUT_HI"] = "0x4d463c21"
    else:
        raise SystemExit(f"no generator for family {family}")
    run([sys.executable, str(generator), str(sass), str(k)], env=env)
    reference_raw = artifact.get("reference_sass")
    if reference_raw:
        reference = ROOT / str(reference_raw)
        if sass.read_bytes() != reference.read_bytes():
            raise SystemExit(f"generated SASS differs from retained reference: {reference_raw}")
    return sass


def assemble(artifact: dict[str, object], cubit: Path, output: Path) -> dict[str, str]:
    relative = Path(str(artifact["path"]))
    destination = output / "cubins" / relative.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    family = str(artifact["family"])
    if family == "MLA":
        sass = ROOT / str(artifact["source"])
    else:
        sass = generate_sass(artifact, output)
    command = [
        str(cubit),
        "asm",
        str(sass),
        "-o",
        str(destination),
        "--kernel",
        str(artifact["kernel"]),
    ]
    stub_raw = artifact.get("mercury_stub")
    if stub_raw:
        command += ["--mercury-stub", str(ROOT / str(stub_raw))]
    run(command, cwd=cubit_root(cubit))
    actual = sha256(destination)
    expected = str(artifact["sha256"])
    if actual != expected:
        raise SystemExit(
            f"hash mismatch for {relative}: expected {expected}, got {actual}"
        )
    return {
        "path": str(relative),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": "MATCH",
    }


def build_m4(output: Path) -> dict[str, str]:
    destination = output / "m4"
    build_lib = destination / "lib"
    build_temp = destination / "temp"
    destination.mkdir(parents=True)
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    env["MAX_JOBS"] = "2"
    run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(build_lib),
            "--build-temp",
            str(build_temp),
        ],
        cwd=SOURCES / "m4",
        env=env,
    )
    products = sorted(build_lib.glob("vq_warp_gemv/_C*.so"))
    if len(products) != 1:
        raise SystemExit(f"expected one M4 extension, found {len(products)}")
    return {
        "path": str(products[0].relative_to(output)),
        "actual_sha256": sha256(products[0]),
        "status": "BUILT",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new or empty output directory outside this repository")
    parser.add_argument(
        "--family",
        action="append",
        choices=("M4", "W2", "W3", "W4", "MLA"),
        help="family to rebuild; repeat as needed (default: all admitted families)",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        help="rebuild only this repository-relative cubin path; repeat as needed",
    )
    parser.add_argument(
        "--cubit-bin",
        help="prebuilt Cubit executable; requires --cubit-revision matching the ledger pin",
    )
    parser.add_argument("--cubit-revision", help="full source revision of --cubit-bin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = prepare_output(args.output)
    ledger = json.loads(LEDGER.read_text())
    selected_families = set(args.family or ledger["default_rebuild_families"])
    requested = set(args.artifact or [])
    artifacts = [
        item
        for item in ledger["artifacts"]
        if str(item["family"]) in selected_families
        and (not requested or str(item["path"]) in requested)
    ]
    unknown = requested - {str(item["path"]) for item in artifacts}
    if unknown:
        raise SystemExit(f"unknown or family-excluded artifacts: {sorted(unknown)}")

    verify_inputs(ledger, artifacts, selected_families)

    results: list[dict[str, str]] = []
    cubit_families = {"W2", "W3", "W4", "MLA"}
    if any(str(item["family"]) in cubit_families for item in artifacts):
        if args.cubit_bin:
            if args.cubit_revision != CUBIT_REVISION:
                raise SystemExit(
                    f"external Cubit revision must be exact pin {CUBIT_REVISION}"
                )
            cubit = Path(args.cubit_bin).expanduser().resolve()
            if not cubit.is_file():
                raise SystemExit(f"Cubit binary not found: {cubit}")
        else:
            cubit = build_cubit(output)
        for artifact in artifacts:
            results.append(assemble(artifact, cubit, output))

    if "M4" in selected_families and not requested:
        results.append(build_m4(output))

    receipt = {
        "schema": "banana-smasher-kernel-rebuild-v1",
        "cubit_cargo_lock_sha256": CUBIT_CARGO_LOCK_SHA256,
        "cubit_revision": CUBIT_REVISION,
        "cubit_sm120_table_sha256": CUBIT_SM120_TABLE_SHA256,
        "ledger_sha256": sha256(LEDGER),
        "results": results,
        "runtime_family_mapping": ledger["runtime_registration"]["family_mapping"],
        "selected_families": sorted(selected_families),
        "status": "PASS",
    }
    receipt_path = output / "rebuild-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(results)} outputs; receipt={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
