from __future__ import annotations

import hashlib
import json
from pathlib import Path

from banana_smasher.backpack_runtime_exact64 import _validate_whole_model_accounting
from banana_smasher.sensitivity_probe import materialize_sensitivity_candidate


def _raw(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_materialize_sensitivity_candidate_changes_one_cell_and_preserves_equations(
    tmp_path: Path,
) -> None:
    basis = "a" * 64
    assignment = {"0": {"0": {"down": "qtip3"}}}
    index = {
        "activation_artifact_ids": [],
        "cell_id": "L000:E000:down",
        "expert": 0,
        "layer": 0,
        "physical_artifact_sha256": "b" * 64,
        "physical_bytes": 30,
        "physical_receipt_path": "/source.json",
        "physical_receipt_sha256": "c" * 64,
        "projection": "down",
        "selection_group": "L000:E000:down",
        "source_key": "qtip3",
        "tier": "qtip3",
    }
    assignment_raw = _raw(assignment)
    index_raw = _raw(index)
    (tmp_path / "ASSIGNMENT.json").write_bytes(assignment_raw)
    (tmp_path / "MATERIALIZATION_INDEX.jsonl").write_bytes(index_raw)
    accounting = {
        "expert_physical_wire_bytes": 30,
        "dense_nonrouted_bytes": 50,
        "repair_bytes": 0,
        "metadata_bytes": 10,
        "fixed_nonexpert_bytes": 60,
        "padding_bytes": 10,
        "whole_shipping_bytes": 100,
        "shipping_bytes_cap": 100,
        "shipping_slack_bytes": 0,
        "logical_base_parameters": 100,
        "whole_model_bpw_numerator_bits": 800,
        "whole_model_bpw_exact_ratio": "800/100",
        "whole_model_bpw_decimal": "8",
    }
    manifest = {
        "schema": "banana-smasher-backpack-virtual-assignment-v1",
        "status": "PASS_LOGICAL_FULL_WIRE",
        "basis_sha256": basis,
        "assignment_map_sha256": hashlib.sha256(assignment_raw).hexdigest(),
        "assignment": {"file": "ASSIGNMENT.json", "bytes": len(assignment_raw), "rows": 1, "sha256": hashlib.sha256(assignment_raw).hexdigest()},
        "materialization_index": {"file": "MATERIALIZATION_INDEX.jsonl", "bytes": len(index_raw), "rows": 1, "sha256": hashlib.sha256(index_raw).hexdigest()},
        "tier_counts": {"qtip3": 1, "native_mxfp4": 0},
        "byte_accounting": {"payload_bytes": 30, "assigned_expert_bytes": 30, "assigned_package_bytes": 100, "tier_payload_bytes": {"qtip3": 30, "native_mxfp4": 0}},
        "expert_parameter_denominator": 100,
        "expert_wire_bpw": 2.4,
        "whole_model_accounting": accounting,
    }
    (tmp_path / "BACKPACK_VIRTUAL_MANIFEST.json").write_bytes(_raw(manifest))
    ledger = {
        "basis_sha256": basis,
        "cell_id": "L000:E000:down",
        "tier": "native_mxfp4",
        "physical_bytes": 40,
        "physical_producer": {
            "artifact_sha256": basis,
            "path": "/native.csv",
            "sha256": "d" * 64,
            "root_map_path": "/target/root-map.json",
            "root_map_sha256": "e" * 64,
        },
    }
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_bytes(_raw(ledger))
    receipt = materialize_sensitivity_candidate(
        tmp_path / "BACKPACK_VIRTUAL_MANIFEST.json",
        ledger_path,
        {"probe_id": "p0", "cell_id": "L000:E000:down", "source_tier": "qtip3", "target_tier": "native_mxfp4", "predicted_delta_mean_kld": -0.01},
        output_root=tmp_path / "candidate",
    )
    candidate = json.loads(Path(receipt["manifest_path"]).read_text())
    assert candidate["whole_model_accounting"]["whole_shipping_bytes"] == 110
    assert candidate["whole_model_accounting"]["shipping_slack_bytes"] == -10
    _validate_whole_model_accounting(candidate, allow_over_cap=True)
    changed = json.loads((tmp_path / "candidate/MATERIALIZATION_INDEX.jsonl").read_text())
    assert changed["tier"] == "native_mxfp4"
    assert changed["physical_bytes"] == 40
    assert receipt["target_root_map_path"] == "/target/root-map.json"
    assert receipt["target_root_map_sha256"] == "e" * 64
