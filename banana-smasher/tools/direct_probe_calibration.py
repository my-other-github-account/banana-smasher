#!/usr/bin/env python3
"""Fit per-stratum median multiplicative corrections from raw probes."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
MANIFEST_SHA = "ddcdd9421215dcfffe5c2c3dad374e3bd114a1072956e294d7724620e818090b"
PAIRS = {"qtip2->qtip3", "qtip3->native_mxfp4"}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.manifest) != MANIFEST_SHA:
        raise ValueError("probe manifest SHA mismatch")
    manifest = json.loads(a.manifest.read_text())
    probes = {x["probe_id"]: x for x in manifest["probes"]}
    raw_count = 0
    accepted: dict[str, dict] = {}
    duplicates = 0
    for path in a.raw:
        for line in path.read_text().splitlines():
            raw_count += 1
            envelope = json.loads(line)
            outer = envelope.get("payload", envelope)
            x = outer.get("measurement", outer)
            probe_id = x.get("probe_id", outer.get("probe_id"))
            probe = probes.get(probe_id)
            if not probe or probe.get("role", "treatment") != "treatment" or probe.get("replicate_of"):
                continue
            measured = x.get("measured_delta_mean_kld", x.get("measured_delta_from_w28_baseline"))
            basis = x.get("basis_sha256", outer.get("canonical_model_basis_sha256"))
            if basis != BASIS or measured is None or not math.isfinite(float(measured)):
                continue
            row = {**probe, "measured_delta_mean_kld": float(measured)}
            if probe_id in accepted:
                duplicates += 1
                if accepted[probe_id]["measured_delta_mean_kld"] != row["measured_delta_mean_kld"]:
                    raise ValueError(f"conflicting replicate copies for {probe_id}")
            else:
                accepted[probe_id] = row
    groups = defaultdict(list)
    for row in accepted.values():
        predicted = float(row["predicted_delta_mean_kld"])
        if row["tier_pair"] in PAIRS and predicted:
            groups[(row["layer_band"], row["tier_pair"])].append(row)
    rows = []
    for (band, pair), values in sorted(groups.items()):
        ratios = [x["measured_delta_mean_kld"] / float(x["predicted_delta_mean_kld"]) for x in values]
        factor = statistics.median(ratios)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(f"invalid median factor for {(band, pair)}: {factor}")
        rows.append({"layer_band": band, "tier_pair": pair, "factor": factor,
                     "probes": len(values), "zero_measurements": sum(x["measured_delta_mean_kld"] == 0 for x in values),
                     "median_measured_over_predicted": factor,
                     "probe_ids": sorted(x["probe_id"] for x in values)})
    expected = {(b, p) for b in manifest["stratification"]["layers_by_band"] for p in PAIRS}
    if {(x["layer_band"], x["tier_pair"]) for x in rows} != expected:
        raise ValueError("incomplete calibration strata")
    table = {"schema": "banana-smasher-sensitivity-calibration-table-v1", "status": "PASS",
             "basis_sha256": BASIS, "probe_manifest_sha256": MANIFEST_SHA,
             "source_option_ledger_sha256": manifest["source_option_ledger_sha256"],
             "fit": "median(measured_delta_mean_kld / predicted_delta_mean_kld) per (layer_band,tier_pair)",
             "raw_rows": raw_count, "unique_treatment_probes": len(accepted), "duplicate_copies_filtered": duplicates,
             "null_control_and_replicate_rows_filtered": raw_count - len(accepted) - duplicates, "rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(table, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"output": str(a.output), "sha256": sha(a.output), "factors": rows}, sort_keys=True))
if __name__ == "__main__": main()
