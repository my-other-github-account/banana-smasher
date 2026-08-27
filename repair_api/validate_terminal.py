"""Deterministic terminal gate for the Modern Green four-target goal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

GREEN_KLD = 0.226162314683653
GREEN_TOP1 = 56700
U3_KLD = 0.22103965283948
MAX_SECONDS = 1200.0
REQUIRED_ANCHORS = ("UPDATE_000", "UPDATE_003", "UPDATE_016")


def _fail(errors: list[str]) -> int:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2, sort_keys=True), file=sys.stderr)
    return 1


def _finite(value: Any, label: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{label} is not finite")
        return None
    return float(value)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _green_reference(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing Green U3 reference: {path}")
        return None
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"invalid Green U3 reference {path}: {exc}")
        return None
    if value.get("accepted_rows") != 64 or value.get("positions") != 65536:
        errors.append("Green U3 reference is not complete 64/64 and 65,536 positions")
    if value.get("direction") != "KL(teacher||candidate)" or value.get("dtype") != "binary64":
        errors.append("Green U3 reference has wrong direction/dtype")
    kld = _finite(value.get("kld_mean_binary64"), "Green U3 KLD", errors)
    if kld is not None and abs(kld - GREEN_KLD) > 1e-15:
        errors.append(f"Green U3 KLD mismatch: {kld!r} != {GREEN_KLD!r}")
    rows = value.get("rows")
    top1 = sum(int(row.get("top1_matches", 0)) for row in rows) if isinstance(rows, list) else -1
    if top1 != GREEN_TOP1:
        errors.append(f"Green U3 Top-1 mismatch: {top1} != {GREEN_TOP1}")
    return {"path": str(path), "sha256": _sha(path), "kld": kld, "top1": top1}


def validate(terminal_path: Path, green_path: Path, artifact_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if not terminal_path.is_file():
        return {"status": "FAIL", "errors": [f"missing terminal: {terminal_path}"]}
    try:
        terminal = json.loads(terminal_path.read_text())
    except Exception as exc:
        return {"status": "FAIL", "errors": [f"invalid terminal JSON: {exc}"]}
    if terminal.get("schema") != "modern-green-resident-api-terminal-v1":
        errors.append("wrong terminal schema")
    if terminal.get("status") != "PASS":
        errors.append(f"terminal status is {terminal.get('status')!r}, not PASS")
    green = _green_reference(green_path, errors)
    side = terminal.get("side_by_side")
    if not isinstance(side, dict):
        errors.append("missing side_by_side")
        side = {}
    for name in ("pre_repair", "green_u3_reference", "modern_u3", "modern_u16"):
        _finite(side.get(name), f"side_by_side.{name}", errors)
    if isinstance(side.get("green_u3_reference"), (int, float)) and abs(float(side["green_u3_reference"]) - GREEN_KLD) > 1e-15:
        errors.append("side_by_side Green U3 does not match the sealed reference")
    if isinstance(side.get("modern_u3"), (int, float)) and abs(float(side["modern_u3"]) - U3_KLD) > 1e-12:
        errors.append(f"Modern U3 calibration mismatch: {side['modern_u3']!r} != {U3_KLD!r}")
    anchors = terminal.get("anchors")
    if not isinstance(anchors, dict):
        errors.append("missing anchors")
        anchors = {}
    mapping = {"UPDATE_000": "pre_repair", "UPDATE_003": "modern_u3", "UPDATE_016": "modern_u16"}
    for key in REQUIRED_ANCHORS:
        receipt = anchors.get(key)
        if not isinstance(receipt, dict):
            errors.append(f"missing anchor receipt: {key}")
            continue
        if receipt.get("status") != "PASS":
            errors.append(f"{key} receipt status is not PASS")
        if receipt.get("score_execution_mode") != "resident_in_memory":
            errors.append(f"{key} was not scored resident_in_memory")
        if receipt.get("under_20_minute_anchor") is not True:
            errors.append(f"{key} did not pass the <1200-second anchor gate")
        score = receipt.get("score")
        if not isinstance(score, dict):
            errors.append(f"{key} missing score object")
            continue
        if score.get("execution_mode") != "resident_in_memory":
            errors.append(f"{key} score execution_mode is not resident_in_memory")
        if score.get("positions") != 65536 or score.get("support") != 8192:
            errors.append(f"{key} has wrong positions/support")
        windows = score.get("windows")
        if not isinstance(windows, list) or len(windows) != 64 or len(set(windows)) != 64:
            errors.append(f"{key} does not contain 64 unique windows")
        timed = _finite(score.get("timed_wall_seconds"), f"{key}.timed_wall_seconds", errors)
        if timed is not None and timed >= MAX_SECONDS:
            errors.append(f"{key} timed score is {timed:.3f}s >= {MAX_SECONDS:.0f}s")
        _finite(receipt.get("generation_wall_seconds"), f"{key}.generation_wall_seconds", errors)
        kld = _finite(score.get("kld_mean"), f"{key}.kld_mean", errors)
        expected_name = mapping[key]
        expected = side.get(expected_name)
        if kld is not None and isinstance(expected, (int, float)) and abs(kld - float(expected)) > 1e-15:
            errors.append(f"{key} score disagrees with side_by_side.{expected_name}")
        if not receipt.get("checkpoint_sha256") or not receipt.get("checkpoint_identity_sha256"):
            errors.append(f"{key} lacks checkpoint SHA/identity binding")
        generation = receipt.get("generation")
        if not isinstance(generation, dict) or generation.get("status") != "PASS":
            errors.append(f"{key} lacks PASS candidate-generation receipt")
    if artifact_root is not None:
        release = artifact_root / "receipts" / "CLAIM_RELEASE.json"
        if not release.is_file():
            errors.append(f"missing claim release receipt: {release}")
        else:
            try:
                if json.loads(release.read_text()).get("status") != "RELEASED":
                    errors.append("claim release receipt is not RELEASED")
            except Exception as exc:
                errors.append(f"invalid claim release receipt: {exc}")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "terminal": str(terminal_path),
        "green_reference": green,
        "required_anchors": list(REQUIRED_ANCHORS),
        "errors": errors,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("terminal", type=Path)
    parser.add_argument("--green-reference", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = validate(args.terminal, args.green_reference, args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
