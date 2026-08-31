#!/usr/bin/env python3
"""Apply a measured calibration and run the canonical exact-102GB solve."""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
from banana_smasher.provenance_wire import run_full_wire_provenance_solve
from banana_smasher.sensitivity_calibration import apply_calibration_to_rows

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def descriptor(p: Path) -> dict: return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True); ap.add_argument("--fixed", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--baseline-assignment", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mip-rel-gap", type=float, default=0.001)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    calibration = json.loads(a.calibration.read_text()); manifest = json.loads(a.manifest.read_text())
    if calibration.get("basis_sha256") != BASIS or manifest.get("basis_sha256") != BASIS: raise ValueError("basis mismatch")
    if sha(a.ledger) != calibration.get("source_option_ledger_sha256"): raise ValueError("source ledger SHA mismatch")
    source = [json.loads(line) for line in a.ledger.read_text().splitlines() if line.strip()]
    calibrated = apply_calibration_to_rows(source, calibration, manifest["stratification"]["layers_by_band"])
    a.output.mkdir(parents=True)
    out_ledger = a.output / "CALIBRATED_OPTION_LEDGER.jsonl"
    with out_ledger.open("w") as f:
        for row in calibrated: f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    receipt = run_full_wire_provenance_solve(
        out_ledger, a.fixed, a.output / "ASSIGNMENT.json", a.output / "SOLVE_RECEIPT.json",
        expected_option_ledger_sha256=sha(out_ledger), expected_fixed_accounting_sha256=sha(a.fixed),
        shipping_bytes_cap=102_000_000_000, class_weights={c: 1.0 for c in CLASSES},
        allowed_tiers=("native_mxfp4", "qtip2", "qtip3"), exact_envelope=False,
        padding_policy="metadata_reserve", mip_rel_gap=a.mip_rel_gap)
    assignment = json.loads((a.output / "ASSIGNMENT.json").read_text())
    baseline = json.loads(a.baseline_assignment.read_text())
    tiers = dict(sorted(collections.Counter(x["tier"] for x in assignment["assignments"]).items()))
    old = {x["cell_id"]: x["tier"] for x in baseline["assignments"]}
    moved = sum(old.get(x["cell_id"]) != x["tier"] for x in assignment["assignments"])
    terminal = {"schema": "banana-smasher-calibrated-102gb-resolve-v1", "status": "PASS",
        "basis_sha256": BASIS, "calibration_table": descriptor(a.calibration), "source_ledger": descriptor(a.ledger),
        "calibrated_ledger": descriptor(out_ledger), "assignment": descriptor(a.output / "ASSIGNMENT.json"),
        "solve_receipt": descriptor(a.output / "SOLVE_RECEIPT.json"), "tier_counts": tiers, "cells_moved": moved,
        "predicted_mean_kld": math.fsum(assignment["predicted_class_totals"][c] for c in CLASSES) / len(CLASSES),
        "whole_model_accounting": receipt["whole_model_accounting"], "solver": receipt["solver"]}
    (a.output / "TERMINAL.json").write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(terminal, sort_keys=True))
if __name__ == "__main__": main()
