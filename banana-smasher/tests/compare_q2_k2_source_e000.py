from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

MEMBERS = ("w1", "w2", "w3")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "count": -1,
            "first": None,
        }
    mismatch = np.argwhere(actual != expected)
    return {
        "actual_sha256": data_sha256(actual),
        "expected_sha256": data_sha256(expected),
        "count": int(mismatch.shape[0]),
        "first": mismatch[0].tolist() if mismatch.size else None,
    }


def atomic_json(path: Path, value: Any) -> str:
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", required=True)
    args = parser.parse_args()

    candidate_terminal = json.loads((args.candidate / "TERMINAL.json").read_bytes())
    candidate_manifest_path = args.candidate / "CANDIDATE_MANIFEST.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_bytes())
    if (
        candidate_terminal.get("status") != "PASS"
        or candidate_terminal.get("members") != 3
        or candidate_terminal.get("fallback_calls") != 0
        or candidate_manifest.get("basis_sha256") != args.basis
        or candidate_manifest.get("source_only") is not True
        or candidate_manifest.get("comparator_inputs") != 0
        or candidate_manifest.get("artifact_seed_inputs") != 0
        or candidate_manifest.get("external_state_map") is not False
    ):
        raise RuntimeError("candidate seal refused")

    comparator_manifest_path = args.comparator / "COMPARATOR_MANIFEST.json"
    comparator_manifest = json.loads(comparator_manifest_path.read_bytes())
    if comparator_manifest.get("basis_sha256") != args.basis:
        raise RuntimeError("comparator basis refused")
    for row in comparator_manifest["files"]:
        path = args.comparator / row["relative_path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"comparator seal refused: {path}")

    candidate_by_name = {row["member"]: row for row in candidate_manifest["members"]}
    results: dict[str, Any] = {}
    all_exact = True
    for name in MEMBERS:
        candidate = candidate_by_name[name]
        expected_root = args.comparator / name
        oracle_receipt = json.loads((expected_root / "MEMBER_RECEIPT.json").read_bytes())

        states = np.load(args.candidate / "members" / f"{name}.states.npy", allow_pickle=False)
        expected_states = np.load(expected_root / "trellis_encoded_unpacked_int16.npy", allow_pickle=False)
        packed = np.load(args.candidate / "members" / f"{name}.codes.npy", allow_pickle=False)
        expected_packed = np.load(expected_root / "trellis_packed_int16.npy", allow_pickle=False)
        su = np.load(args.candidate / "members" / f"{name}.su.npy", allow_pickle=False)
        expected_su = np.load(expected_root / "SU_final_fp32.npy", allow_pickle=False)
        sv = np.load(args.candidate / "members" / f"{name}.sv.npy", allow_pickle=False)
        expected_sv = np.load(expected_root / "SV_final_fp32.npy", allow_pickle=False)
        suh = np.load(args.candidate / "members" / f"{name}.suh.npy", allow_pickle=False)
        svh = np.load(args.candidate / "members" / f"{name}.svh.npy", allow_pickle=False)
        physical = np.fromfile(args.candidate / "members" / f"{name}.physical.bf16.bin", dtype=np.uint16).reshape(candidate["artifacts"]["physical_bfloat16"]["shape"])
        expected_physical = np.load(expected_root / "decoded_physical_out_in_bfloat16_bits.npy", allow_pickle=False)

        comparisons = {
            "states": exact_comparison(states.view(np.uint16), expected_states.view(np.uint16)),
            "packed": exact_comparison(packed.view(np.uint16), expected_packed.view(np.uint16)),
            "su": exact_comparison(su, expected_su),
            "sv": exact_comparison(sv, expected_sv),
            "suh": exact_comparison(suh, expected_su.astype(np.float16)),
            "svh": exact_comparison(svh, expected_sv.astype(np.float16)),
            "physical_bfloat16": exact_comparison(physical, expected_physical),
        }
        proxy = candidate["objective_proxy_error"]
        expected_proxy = oracle_receipt["objective"]["oracle_proxy_error"]
        objective = {
            "actual": proxy,
            "expected": expected_proxy,
            "relation": "EXACT" if proxy == expected_proxy else "MISMATCH",
        }
        boundary_expectations = {
            "states_sha256": oracle_receipt["encoded_comparison"]["expected_sha256"],
            "packed_sha256": oracle_receipt["packed_comparison"]["expected_sha256"],
            "physical_fp32_sha256": oracle_receipt["physical_fp32_comparison"]["expected_sha256"],
            "physical_bfloat16_sha256": oracle_receipt["physical_bfloat16_comparison"]["expected_sha256"],
        }
        boundary_comparison = {
            key: {
                "actual": candidate["boundaries"][key],
                "expected": expected,
                "relation": "EXACT" if candidate["boundaries"][key] == expected else "MISMATCH",
            }
            for key, expected in boundary_expectations.items()
        }
        exact = (
            all(row["count"] == 0 for row in comparisons.values())
            and objective["relation"] == "EXACT"
            and all(row["relation"] == "EXACT" for row in boundary_comparison.values())
            and candidate["solver_counters"]["fallback_calls"] == 0
            and candidate["solver_counters"]["cuda_calls"] > 0
        )
        all_exact &= exact
        results[name] = {
            "status": "PASS_EXACT" if exact else "FAIL_MISMATCH",
            "comparisons": comparisons,
            "boundary_comparison": boundary_comparison,
            "objective": objective,
            "cuda_calls": candidate["solver_counters"]["cuda_calls"],
            "fallback_calls": candidate["solver_counters"]["fallback_calls"],
        }

    receipt = {
        "schema": "banana-smasher-q2-source-e000-post-comparator-v1",
        "status": "PASS_EXACT_3_OF_3" if all_exact else "FAIL_MISMATCH",
        "basis_sha256": args.basis,
        "candidate_root": str(args.candidate),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_terminal_sha256": sha256_file(args.candidate / "TERMINAL.json"),
        "candidate_source_only": True,
        "candidate_comparator_inputs": 0,
        "candidate_artifact_seed_inputs": 0,
        "comparator_manifest_sha256": sha256_file(comparator_manifest_path),
        "members_exact": sum(row["status"] == "PASS_EXACT" for row in results.values()),
        "members_required": 3,
        "results": results,
        "ended_unix": time.time(),
    }
    receipt_sha = atomic_json(args.output, receipt)
    print(canonical({"status": receipt["status"], "receipt": str(args.output), "receipt_sha256": receipt_sha}).decode(), end="")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
