from __future__ import annotations

import pytest

from banana_smasher.sensitivity_calibration import (
    CLASSES,
    apply_calibration_to_rows,
    fit_multiplicative_calibration,
)

BASIS = "a" * 64


def _manifest() -> dict:
    probes = []
    index = 0
    for band, layer in (("early", 0), ("mid", 1), ("late", 2)):
        for pair in ("qtip2->qtip3", "qtip3->native_mxfp4"):
            probes.append(
                {
                    "probe_id": f"p{index}",
                    "cell_id": f"L{layer:03d}:E000:down",
                    "layer_band": band,
                    "tier_pair": pair,
                    "predicted_delta_mean_kld": -0.01,
                }
            )
            index += 1
    return {
        "schema": "banana-smasher-sensitivity-probe-manifest-v1",
        "basis_sha256": BASIS,
        "source_option_ledger_sha256": "b" * 64,
        "source_assignment_sha256": "c" * 64,
        "probe_count": len(probes),
        "stratification": {"layers_by_band": {"early": [0], "mid": [1], "late": [2]}},
        "probes": probes,
    }


def _measurements(manifest: dict) -> list[dict]:
    return [
        {
            "schema": "banana-smasher-sensitivity-w28-probe-v1",
            "status": "PASS",
            "probe_id": probe["probe_id"],
            "cell_id": probe["cell_id"],
            "basis_sha256": BASIS,
            "measured_delta_mean_kld": probe["predicted_delta_mean_kld"]
            * (3.0 if probe["tier_pair"] == "qtip2->qtip3" else 5.0),
        }
        for probe in manifest["probes"]
    ]


def test_fit_and_apply_simple_band_pair_multipliers() -> None:
    manifest = _manifest()
    table = fit_multiplicative_calibration(manifest, _measurements(manifest))
    factors = {(row["layer_band"], row["tier_pair"]): row["factor"] for row in table["rows"]}
    assert set(factors.values()) == {3.0, 5.0}
    rows = []
    for layer in range(3):
        common = {
            "basis_sha256": BASIS,
            "cell_id": f"L{layer:03d}:E000:down",
            "layer": layer,
            "expert": 0,
            "projection": "down",
        }
        rows.extend(
            [
                {**common, "tier": "native_mxfp4", "prediction_by_class": {name: 0.0 for name in CLASSES}},
                {**common, "tier": "qtip3", "prediction_by_class": {name: 0.01 for name in CLASSES}},
                {**common, "tier": "qtip2", "prediction_by_class": {name: 0.03 for name in CLASSES}},
            ]
        )
    calibrated = apply_calibration_to_rows(
        rows, table, manifest["stratification"]["layers_by_band"]
    )
    by_key = {(row["cell_id"], row["tier"]): row for row in calibrated}
    for layer in range(3):
        cell = f"L{layer:03d}:E000:down"
        assert set(by_key[cell, "qtip3"]["prediction_by_class"].values()) == {0.05}
        assert all(
            value == pytest.approx(0.11)
            for value in by_key[cell, "qtip2"]["prediction_by_class"].values()
        )
        assert set(by_key[cell, "native_mxfp4"]["prediction_by_class"].values()) == {0.0}


def test_fit_refuses_incomplete_measurements() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="missing 1 probe measurements"):
        fit_multiplicative_calibration(manifest, _measurements(manifest)[:-1])
