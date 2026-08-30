from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from banana_smasher.knapsack import solve_class_balanced_options

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
LEDGER_SHA = "45b124e40a0f41a10e25949efdf32cc11a4271f24cf1331c6dbab6deacd813ee"
FIXED_SHA = "5d720e0a6e182d366c39168bce49b516c1ff779567518b92457ad8d54cc55043"
TIERS = ("native_mxfp4", "qtip2", "qtip3")
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
SHIPPING_BYTES = 102_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(ledger: Path, fixed: Path):
    if sha256(ledger) != LEDGER_SHA:
        raise ValueError("option-ledger SHA mismatch")
    if sha256(fixed) != FIXED_SHA:
        raise ValueError("fixed-accounting SHA mismatch")
    fixed_value = json.loads(fixed.read_text())
    components = fixed_value["components"]
    fixed_bytes = sum(int(components[key]) for key in ("dense_nonrouted_bytes", "repair_bytes", "metadata_bytes"))
    rows = {}
    with ledger.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("basis_sha256") != BASIS:
                raise ValueError(f"basis mismatch at line {line_number}")
            tier = row.get("tier")
            if tier not in TIERS:
                continue
            cell = row.get("cell_id")
            if (cell, tier) in rows:
                raise ValueError(f"duplicate option {(cell, tier)}")
            rows[(cell, tier)] = row
    cells = sorted({cell for cell, _ in rows})
    if set(rows) != {(cell, tier) for cell in cells for tier in TIERS}:
        raise ValueError("three-tier matrix incomplete")
    return rows, cells, fixed_bytes


def expert(cell: str) -> str:
    layer, expert_id, _projection = cell.split(":")
    return f"{layer}:{expert_id}"


def solve_expert(rows, cells, envelope):
    members = collections.defaultdict(list)
    for cell in cells:
        members[expert(cell)].append(cell)
    if any(len(value) != 2 for value in members.values()):
        raise ValueError("one-tier-per-expert groups must contain exactly two projections")
    groups = sorted(members)
    bytes_by_option = {}
    costs_by_option = {}
    for group in groups:
        for tier in TIERS:
            source = [rows[(cell, tier)] for cell in members[group]]
            bytes_by_option[(group, tier)] = sum(int(row["physical_bytes"]) for row in source)
            costs_by_option[(group, tier)] = {
                name: math.fsum(float(row["prediction_by_class"][name]) for row in source)
                for name in CLASSES
            }
    caps = {
        name: math.fsum(max(costs_by_option[(group, tier)][name] for tier in TIERS) for group in groups)
        for name in CLASSES
    }
    return solve_class_balanced_options(
        cells=groups,
        tiers=list(TIERS),
        bytes_by_option=bytes_by_option,
        class_costs_by_option=costs_by_option,
        activation_artifacts_by_option={(group, tier): () for group in groups for tier in TIERS},
        envelope_bytes=envelope,
        class_caps=caps,
        class_weights={name: 1.0 for name in CLASSES},
        exact_envelope=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--baseline-assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, cells, fixed_bytes = load(args.ledger, args.fixed)
    baseline = json.loads(args.baseline_assignment.read_text())
    if baseline.get("basis_sha256") != BASIS or baseline.get("source_option_ledger", {}).get("sha256") != LEDGER_SHA:
        raise ValueError("baseline assignment basis/ledger mismatch")
    baseline_totals = baseline["predicted_class_totals"]
    baseline_mean = math.fsum(float(baseline_totals[name]) for name in CLASSES) / len(CLASSES)
    result = solve_expert(rows, cells, SHIPPING_BYTES - fixed_bytes)
    constrained_totals = result["predicted_class_totals"]
    constrained_mean = math.fsum(float(constrained_totals[name]) for name in CLASSES) / len(CLASSES)
    output = {
        "schema": "banana-smasher-projection-granularity-comparison-v1",
        "status": "PASS",
        "basis_sha256": BASIS,
        "option_ledger_sha256": LEDGER_SHA,
        "fixed_accounting_sha256": FIXED_SHA,
        "shipping_bytes": SHIPPING_BYTES,
        "per_projection": {
            "assignment_sha256": sha256(args.baseline_assignment),
            "tier_counts": dict(collections.Counter(row["tier"] for row in baseline["assignments"])),
            "predicted_class_totals": baseline_totals,
            "predicted_mean_kld": baseline_mean,
            "selected_expert_bytes": baseline["whole_model_accounting"]["selected_expert_bytes"],
        },
        "one_tier_per_expert": {
            "tier_counts": {tier: 2 * sum(row["tier"] == tier for row in result["assignments"]) for tier in TIERS},
            "predicted_class_totals": constrained_totals,
            "predicted_mean_kld": constrained_mean,
            "selected_expert_bytes": result["assigned_bytes"],
            "solver": result["solver"],
        },
        "cost": {
            "predicted_mean_kld_delta": constrained_mean - baseline_mean,
            "predicted_mean_kld_ratio": constrained_mean / baseline_mean,
            "extra_bytes": result["assigned_bytes"] - baseline["whole_model_accounting"]["selected_expert_bytes"],
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["cost"], sort_keys=True))


if __name__ == "__main__":
    main()
