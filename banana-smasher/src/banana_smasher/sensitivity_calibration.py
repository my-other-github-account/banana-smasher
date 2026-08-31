"""Measured multiplicative recalibration for mixed Backpack sensitivity ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
PAIR_Q2_Q3 = "qtip2->qtip3"
PAIR_Q3_NATIVE = "qtip3->native_mxfp4"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def fit_multiplicative_calibration(
    probe_manifest: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Fit one through-origin multiplier per ``(layer_band, tier_pair)``.

    The factor is ``sum(measured_delta) / sum(predicted_delta)``. Both deltas use
    target-minus-source convention, so a correctly directed improvement produces
    a positive multiplier even though both sums are negative.
    """

    if probe_manifest.get("schema") not in {
        "banana-smasher-sensitivity-probe-manifest-v1",
        "banana-smasher-sensitivity-probe-manifest-v2",
    }:
        raise ValueError("unsupported sensitivity probe manifest")
    probes = probe_manifest.get("probes")
    expected_count = probe_manifest.get("probe_count")
    if not isinstance(probes, list) or not probes or expected_count != len(probes):
        raise ValueError("probe manifest count mismatch")
    by_id = {}
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise ValueError("probe row must be an object")
        probe_id = probe.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id or probe_id in by_id:
            raise ValueError("probe ids must be unique non-empty strings")
        predicted = _finite(probe.get("predicted_delta_mean_kld"), f"{probe_id} predicted delta")
        is_fit_probe = probe.get("role", "treatment") == "treatment"
        if is_fit_probe and predicted == 0:
            raise ValueError(f"{probe_id} predicted delta must be nonzero")
        band = probe.get("layer_band")
        pair = probe.get("tier_pair")
        if not isinstance(band, str) or (is_fit_probe and pair not in {PAIR_Q2_Q3, PAIR_Q3_NATIVE}):
            raise ValueError(f"{probe_id} calibration stratum is invalid")
        by_id[probe_id] = probe
    measured_by_id = {}
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError("measurement row must be an object")
        if measurement.get("schema") not in {
            "banana-smasher-sensitivity-w28-probe-v1",
            "banana-smasher-sensitivity-w28-probe-v2",
        } or measurement.get("status") != "PASS":
            raise ValueError("measurements must be sensitivity W28 probe PASS rows")
        probe_id = measurement.get("probe_id")
        if probe_id not in by_id or probe_id in measured_by_id:
            raise ValueError(f"unknown or duplicate probe measurement: {probe_id!r}")
        if measurement.get("basis_sha256") != probe_manifest.get("basis_sha256"):
            raise ValueError(f"{probe_id} basis mismatch")
        if measurement.get("cell_id") != by_id[probe_id].get("cell_id"):
            raise ValueError(f"{probe_id} cell mismatch")
        measured_by_id[probe_id] = measurement
    fit_ids = {probe_id for probe_id, probe in by_id.items() if probe.get("role", "treatment") == "treatment"}
    measured_fit_ids = set(measured_by_id) & fit_ids
    if not allow_partial and measured_fit_ids != fit_ids:
        missing = sorted(fit_ids - measured_fit_ids)
        raise ValueError(f"missing {len(missing)} probe measurements; first={missing[:1]}")

    grouped = defaultdict(list)
    for probe_id in sorted(measured_fit_ids):
        probe = by_id[probe_id]
        measurement = measured_by_id[probe_id]
        measured = _finite(measurement.get("measured_delta_mean_kld"), f"{probe_id} measured delta")
        predicted = float(probe["predicted_delta_mean_kld"])
        grouped[(str(probe["layer_band"]), str(probe["tier_pair"]))].append(
            (predicted, measured, probe_id)
        )
    rows = []
    for (band, pair), values in sorted(grouped.items()):
        predicted_sum = math.fsum(value[0] for value in values)
        measured_sum = math.fsum(value[1] for value in values)
        if predicted_sum == 0:
            raise ValueError(f"calibration stratum {(band, pair)} has zero predicted sum")
        factor = measured_sum / predicted_sum
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"calibration stratum {(band, pair)} has non-positive multiplier {factor}")
        residuals = [measured - factor * predicted for predicted, measured, _ in values]
        rows.append(
            {
                "layer_band": band,
                "tier_pair": pair,
                "factor": factor,
                "probes": len(values),
                "predicted_delta_sum": predicted_sum,
                "measured_delta_sum": measured_sum,
                "mean_absolute_residual": math.fsum(abs(value) for value in residuals) / len(residuals),
                "probe_ids": [value[2] for value in values],
            }
        )
    expected_groups = {
        (band, pair)
        for band in probe_manifest.get("stratification", {}).get("layers_by_band", {})
        for pair in (PAIR_Q2_Q3, PAIR_Q3_NATIVE)
    }
    fitted_groups = {(row["layer_band"], row["tier_pair"]) for row in rows}
    if fitted_groups != expected_groups:
        raise ValueError("calibration table does not cover every declared band/tier pair")
    coverage_rows = []
    for band, pair in sorted(expected_groups):
        required = sum(
            probe.get("role", "treatment") == "treatment"
            and probe.get("layer_band") == band
            and probe.get("tier_pair") == pair
            for probe in by_id.values()
        )
        accepted = sum(
            by_id[probe_id].get("layer_band") == band and by_id[probe_id].get("tier_pair") == pair
            for probe_id in measured_fit_ids
        )
        coverage_rows.append(
            {
                "layer_band": band,
                "tier_pair": pair,
                "accepted": accepted,
                "required": required,
                "missing": required - accepted,
                "complete": accepted == required,
            }
        )
    return {
        "schema": "banana-smasher-sensitivity-calibration-table-v1",
        "status": "PASS",
        "basis_sha256": probe_manifest.get("basis_sha256"),
        "source_option_ledger_sha256": probe_manifest.get("source_option_ledger_sha256"),
        "source_assignment_sha256": probe_manifest.get("source_assignment_sha256"),
        "probe_manifest_sha256": _sha256(_canonical(probe_manifest)),
        "probe_count": len(measured_fit_ids),
        "manifest_fit_probe_count": len(fit_ids),
        "partial_coverage": measured_fit_ids != fit_ids,
        "coverage": coverage_rows,
        "fit": "through-origin ratio-of-sums per (layer_band,tier_pair)",
        "delta_convention": "target_mean_kld - source_mean_kld",
        "rows": rows,
    }


def apply_calibration_to_rows(
    rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any], layers_by_band: Mapping[str, Sequence[int]]
) -> list[dict[str, Any]]:
    """Apply calibrated Q3 damage and Q2-minus-Q3 gaps; leave other tiers unchanged."""

    if calibration.get("schema") != "banana-smasher-sensitivity-calibration-table-v1" or calibration.get("status") != "PASS":
        raise ValueError("calibration table must be v1 PASS")
    factor = {
        (str(row["layer_band"]), str(row["tier_pair"])): _finite(row["factor"], "calibration factor")
        for row in calibration.get("rows", [])
    }
    band_by_layer = {}
    band_starts = []
    for band, layers in layers_by_band.items():
        normalized_layers = [int(layer) for layer in layers]
        if not normalized_layers:
            raise ValueError(f"calibration band {band!r} has no sampled layers")
        band_starts.append((min(normalized_layers), str(band)))
        for layer in normalized_layers:
            if layer in band_by_layer:
                raise ValueError(f"layer {layer} appears in multiple bands")
            band_by_layer[layer] = str(band)
    band_starts.sort()
    for raw in rows:
        raw_layer = raw.get("layer")
        if isinstance(raw_layer, bool) or not isinstance(raw_layer, int):
            raise ValueError("ledger layer must be an integer")
        eligible = [band for start, band in band_starts if start <= raw_layer]
        if not eligible:
            raise ValueError(f"layer {raw_layer} precedes every calibration band")
        band_by_layer.setdefault(raw_layer, eligible[-1])
    by_cell_tier = {(str(row["cell_id"]), str(row["tier"])): row for row in rows}
    output = []
    for raw in rows:
        row = dict(raw)
        if row.get("basis_sha256") != calibration.get("basis_sha256"):
            raise ValueError("ledger/calibration basis mismatch")
        tier = str(row.get("tier"))
        raw_layer = row.get("layer")
        if isinstance(raw_layer, bool) or not isinstance(raw_layer, int):
            raise ValueError("ledger layer must be an integer")
        layer = raw_layer
        band = band_by_layer.get(layer)
        if band is None:
            raise ValueError(f"layer {layer} has no calibration band")
        predictions = row.get("prediction_by_class")
        if not isinstance(predictions, Mapping) or set(predictions) != set(CLASSES):
            raise ValueError("ledger prediction classes mismatch")
        if tier == "qtip3":
            q3_factor = factor[(band, PAIR_Q3_NATIVE)]
            calibrated = {name: max(0.0, q3_factor * float(predictions[name])) for name in CLASSES}
        elif tier == "qtip2":
            q3 = by_cell_tier.get((str(row["cell_id"]), "qtip3"))
            if q3 is None:
                raise ValueError(f"qtip2 row lacks qtip3 peer: {row.get('cell_id')}")
            q3_predictions = q3.get("prediction_by_class")
            if not isinstance(q3_predictions, Mapping) or set(q3_predictions) != set(CLASSES):
                raise ValueError(f"qtip3 peer prediction classes mismatch: {row.get('cell_id')}")
            q3_factor = factor[(band, PAIR_Q3_NATIVE)]
            gap_factor = factor[(band, PAIR_Q2_Q3)]
            calibrated = {
                name: max(
                    0.0,
                    q3_factor * float(q3_predictions[name])
                    + gap_factor * (float(predictions[name]) - float(q3_predictions[name])),
                )
                for name in CLASSES
            }
        elif tier == "native_mxfp4":
            calibrated = {name: 0.0 for name in CLASSES}
        else:
            calibrated = {name: float(predictions[name]) for name in CLASSES}
        row["uncalibrated_prediction_by_class"] = dict(predictions)
        row["prediction_by_class"] = calibrated
        row["calibration_table_sha256"] = _sha256(_canonical(calibration))
        row["calibration_layer_band"] = band
        output.append(row)
    return output


def run_sensitivity_calibration(
    probe_manifest_path: str | Path,
    measurements_path: str | Path,
    option_ledger_path: str | Path,
    *,
    output_table: str | Path,
    output_ledger: str | Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(probe_manifest_path).expanduser().resolve()
    measurement_path = Path(measurements_path).expanduser().resolve()
    ledger_path = Path(option_ledger_path).expanduser().resolve()
    manifest_raw = manifest_path.read_bytes()
    measurement_raw = measurement_path.read_bytes()
    ledger_raw = ledger_path.read_bytes()
    manifest = json.loads(manifest_raw)
    measurements = [json.loads(line) for line in measurement_raw.splitlines() if line.strip()]
    rows = [json.loads(line) for line in ledger_raw.splitlines() if line.strip()]
    if _sha256(ledger_raw) != manifest.get("source_option_ledger_sha256"):
        raise ValueError("option ledger SHA mismatch")
    table = fit_multiplicative_calibration(manifest, measurements, allow_partial=allow_partial)
    calibrated = apply_calibration_to_rows(
        rows, table, manifest["stratification"]["layers_by_band"]
    )
    table_path = Path(output_table).expanduser().resolve()
    output_path = Path(output_ledger).expanduser().resolve()
    table_raw = _canonical(table)
    output_raw = b"".join(_canonical(row) for row in calibrated)
    _atomic(table_path, table_raw)
    _atomic(output_path, output_raw)
    return {
        "schema": "banana-smasher-sensitivity-calibration-receipt-v1",
        "status": "PASS",
        "basis_sha256": table["basis_sha256"],
        "probe_count": table["probe_count"],
        "calibration_table": {"path": str(table_path), "bytes": len(table_raw), "sha256": _sha256(table_raw)},
        "calibrated_ledger": {"path": str(output_path), "bytes": len(output_raw), "sha256": _sha256(output_raw), "rows": len(calibrated)},
    }
