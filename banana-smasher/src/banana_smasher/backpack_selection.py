from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load(path: str | Path) -> tuple[Path, dict[str, Any], bytes]:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {resolved}")
    return resolved, value, raw


def _evidence(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _atomic_json(path: Path, value: object) -> None:
    raw = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _arm(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    arms = receipt.get("arms")
    arm = arms.get(name) if isinstance(arms, dict) else None
    if not isinstance(arm, dict):
        raise ValueError(f"solve receipt has no arm {name!r}")
    tiers = arm.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError(f"solve arm {name!r} has no tier menu")
    assignment_sha = arm.get("assignment_map_sha256")
    if not isinstance(assignment_sha, str) or len(assignment_sha) != 64:
        raise ValueError(f"solve arm {name!r} has no assignment SHA-256")
    return arm


def _score(value: dict[str, Any], label: str) -> dict[str, Any]:
    if value.get("status") != "PASS":
        raise ValueError(f"{label} score is not terminal PASS")
    if value.get("windows") != 64 or value.get("positions") != 65536:
        raise ValueError(f"{label} score is not the exact64 rail")
    mean_kld = value.get("mean_kld")
    top1 = value.get("top1_matches")
    if isinstance(mean_kld, bool) or not isinstance(mean_kld, (int, float)) or mean_kld < 0:
        raise ValueError(f"{label} score has invalid mean_kld")
    if isinstance(top1, bool) or not isinstance(top1, int) or not 0 <= top1 <= 65536:
        raise ValueError(f"{label} score has invalid top1_matches")
    return {"mean_kld": float(mean_kld), "top1_matches": top1}


def select_measured_nonworse(
    solve_receipt_path: str | Path,
    baseline_score_path: str | Path,
    expanded_score_path: str | Path,
    output_path: str | Path,
    *,
    baseline_arm: str,
    expanded_arm: str,
) -> dict[str, Any]:
    """Publish the expanded menu only when exact64 KLD and Top-1 are non-worse.

    The integer solver remains a proposal generator. This measured outer gate makes
    menu expansion monotonic even when additive per-cell predictions misorder the
    physical end-to-end candidates.
    """

    solve_path, solve, solve_raw = _load(solve_receipt_path)
    baseline_path, baseline_score_raw_value, baseline_raw = _load(baseline_score_path)
    expanded_path, expanded_score_raw_value, expanded_raw = _load(expanded_score_path)
    baseline = _arm(solve, baseline_arm)
    expanded = _arm(solve, expanded_arm)
    baseline_tiers = set(baseline["tiers"])
    expanded_tiers = set(expanded["tiers"])
    if not baseline_tiers < expanded_tiers:
        raise ValueError(
            "expanded arm must be a strict tier-menu superset of the baseline arm"
        )
    basis = solve.get("basis_sha256")
    if not isinstance(basis, str) or len(basis) != 64:
        raise ValueError("solve receipt has no basis SHA-256")
    rail_identity_keys = ("basis_sha256", "bank_sha256", "windows", "positions", "support_width")
    for key in rail_identity_keys:
        left = baseline_score_raw_value.get(key)
        right = expanded_score_raw_value.get(key)
        if left != right:
            raise ValueError(f"score rail mismatch for {key}: {left!r} != {right!r}")
    if baseline_score_raw_value.get("basis_sha256") != basis:
        raise ValueError("score basis does not match solve basis")
    baseline_measurement = _score(baseline_score_raw_value, "baseline")
    expanded_measurement = _score(expanded_score_raw_value, "expanded")
    kld_delta = expanded_measurement["mean_kld"] - baseline_measurement["mean_kld"]
    top1_delta = expanded_measurement["top1_matches"] - baseline_measurement["top1_matches"]
    accepted = kld_delta <= 0.0 and top1_delta >= 0
    chosen_name = expanded_arm if accepted else baseline_arm
    chosen = expanded if accepted else baseline
    proxy_baseline = float(baseline.get("objective", {}).get("value", float("nan")))
    proxy_expanded = float(expanded.get("objective", {}).get("value", float("nan")))
    receipt = {
        "schema": "banana-smasher-backpack-measured-monotonic-selection-v1",
        "status": "PASS",
        "decision": "ACCEPT_EXPANDED" if accepted else "RETAIN_BASELINE",
        "reason": (
            "expanded exact64 KLD and Top-1 are both non-worse"
            if accepted
            else "expanded menu violated measured non-worsening despite proxy proposal"
        ),
        "basis_sha256": basis,
        "baseline_arm": baseline_arm,
        "expanded_arm": expanded_arm,
        "baseline_tiers": sorted(baseline_tiers),
        "expanded_tiers": sorted(expanded_tiers),
        "chosen_arm": chosen_name,
        "chosen_assignment_map_sha256": chosen["assignment_map_sha256"],
        "measured": {
            "baseline": baseline_measurement,
            "expanded": expanded_measurement,
            "expanded_minus_baseline": {
                "mean_kld": kld_delta,
                "top1_matches": top1_delta,
            },
        },
        "proxy": {
            "baseline_objective": proxy_baseline,
            "expanded_objective": proxy_expanded,
            "expanded_minus_baseline": proxy_expanded - proxy_baseline,
            "ordering_agrees_with_measurement": (proxy_expanded <= proxy_baseline) == (kld_delta <= 0.0),
        },
        "inputs": {
            "solve_receipt": _evidence(solve_path, solve_raw),
            "baseline_score": _evidence(baseline_path, baseline_raw),
            "expanded_score": _evidence(expanded_path, expanded_raw),
        },
    }
    output = Path(output_path).expanduser().resolve()
    _atomic_json(output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measured monotonic Backpack menu selection")
    parser.add_argument("--solve-receipt", type=Path, required=True)
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--expanded-arm", required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--expanded-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            select_measured_nonworse(
                args.solve_receipt,
                args.baseline_score,
                args.expanded_score,
                args.output,
                baseline_arm=args.baseline_arm,
                expanded_arm=args.expanded_arm,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
