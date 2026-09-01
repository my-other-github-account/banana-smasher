import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.q4_direct_objective import build_direct_objective_ledger, rewrite_q4_predictions


def _write_json(path: Path, value: dict) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_build_and_rewrite_direct_mse(tmp_path: Path) -> None:
    basis = "b" * 64
    cell = tmp_path / "receipts" / "L000_E000_down" / "CELL_RECEIPT.json"
    receipt = {
        "status": "PASS",
        "basis_sha256": basis,
        "source": {"sha256": "s" * 64},
        "artifacts": {"decoded": {"sha256": "d" * 64}, "codes": {"sha256": "c" * 64}},
        "direct_error": {"sse": 2.0, "mse": 0.5},
        "accounting": {"weights": 4},
        "cuda": {"fallback_calls": 0},
    }
    api_sha = _write_json(cell, receipt)
    physical = tmp_path / "physical.jsonl"
    physical.write_text(json.dumps({
        "cell": "L000/E000_down", "layer": 0, "expert": 0, "projection": "down",
        "api_receipt_sha256": api_sha, "codes_sha256": "c" * 64, "fallback_calls": 0,
    }, sort_keys=True) + "\n")
    objective = tmp_path / "objective.jsonl"
    terminal = build_direct_objective_ledger(physical, [tmp_path / "receipts"], objective, basis, expected_rows=1)
    assert terminal["status"] == "PASS"
    row = json.loads(objective.read_text())
    assert row["direct_reconstruction_mse"] == 0.5
    assert set(row["prediction_by_class"].values()) == {0.5}

    expanded = tmp_path / "expanded.jsonl"
    expanded.write_text(json.dumps({"cell_id": "L000:E000:down", "tier": "qtip4_v7", "prediction_by_class": {"chat": 99.0}}) + "\n")
    rewritten = tmp_path / "rewritten.jsonl"
    out = rewrite_q4_predictions(expanded, objective, rewritten, expected_q4_rows=1)
    assert out["qtip4_rows_rewritten"] == 1
    assert json.loads(rewritten.read_text())["prediction_by_class"]["chat"] == 0.5


def test_rejects_non_direct_or_fallback_receipt(tmp_path: Path) -> None:
    physical = tmp_path / "physical.jsonl"
    physical.write_text(json.dumps({"cell": "L000/E000_down", "api_receipt_sha256": "0" * 64, "fallback_calls": 1}) + "\n")
    with pytest.raises(ValueError, match="fallback"):
        build_direct_objective_ledger(physical, [tmp_path], tmp_path / "out.jsonl", "b" * 64, expected_rows=1)


def test_accepts_hash_bound_embedded_receipt_census(tmp_path: Path) -> None:
    basis = "b" * 64
    receipt = {
        "status": "PASS", "basis_sha256": basis,
        "source": {"sha256": "s" * 64},
        "artifacts": {"decoded": {"sha256": "d" * 64}, "codes": {"sha256": "c" * 64}},
        "direct_error": {"sse": 2.0, "mse": 0.5},
        "accounting": {"weights": 4},
        "cuda": {"fallback_calls": 0},
    }
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    receipt_sha = hashlib.sha256(raw).hexdigest()
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"json_receipts": {
        "outputs/q4/L000_E000_down/CELL_RECEIPT.json": {
            "stat": {"path": "/sealed/L000_E000_down/CELL_RECEIPT.json", "sha256": receipt_sha},
            "object": receipt,
        }
    }}))
    physical = tmp_path / "physical.jsonl"
    physical.write_text(json.dumps({
        "cell": "L000/E000_down", "api_receipt_sha256": receipt_sha,
        "codes_sha256": "c" * 64, "fallback_calls": 0,
    }) + "\n")
    terminal = build_direct_objective_ledger(
        physical, [], tmp_path / "out.jsonl", basis, expected_rows=1,
        embedded_censuses=[census],
    )
    assert terminal["rows"] == 1
