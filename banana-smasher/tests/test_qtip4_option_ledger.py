import hashlib
import json

import pytest

from banana_smasher.qtip4_option_ledger import emit_qtip4_option_row, emit_root_option_ledger


BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


def _fixture():
    codes_sha = "a" * 64
    api = {
        "status": "PASS",
        "basis_sha256": BASIS,
        "geometry": {"B": 16, "L": 16, "V": 4},
        "accounting": {"exact_code_bpw": 4.0, "exact_code_bits": 128, "weights": 32},
        "optimization": {"selected_factor": 1.0, "selected_scale": 0.125},
        "installed_cuda_decode": {"counters": {"fallback_calls": 0}},
        "artifacts": {"codes": {"sha256": codes_sha}},
    }
    api_raw = json.dumps(api, sort_keys=True, separators=(",", ":")).encode()
    public = {
        "schema": "banana-smasher-qtip3-v7-public-api-producer-v1-cell",
        "status": "PASS",
        "task_id": "t_fixture",
        "basis_sha256": BASIS,
        "cell": "L000/E000_down",
        "layer": 0,
        "expert": 0,
        "projection": "down",
        "bpw": 4.0,
        "backend": "cuda",
        "cuda_decode_calls": 1,
        "fallback_calls": 0,
        "provider": "qtip-native-v6@4.00",
        "geometry": {"B": 16, "L": 16, "V": 4},
        "scale_factors": [0.5, 1.0, 2.0],
        "objective": 1.25,
        "api_receipt_sha256": hashlib.sha256(api_raw).hexdigest(),
    }
    public_raw = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    expected = {
        "schema": "qtip4-v7-option-ledger-row-v1",
        "task_id": "t_fixture",
        "cell": "L000/E000_down",
        "layer": 0,
        "expert": 0,
        "projection": "down",
        "tier": "qtip4",
        "bpw": 4.0,
        "provider": "qtip-native-v6@4.00",
        "geometry": {"B": 16, "L": 16, "V": 4},
        "scale_factors": [0.5, 1.0, 2.0],
        "selected_factor": 1.0,
        "selected_scale": 0.125,
        "objective": 1.25,
        "fallback_calls": 0,
        "exact_code_bits": 128,
        "weights": 32,
        "codes_sha256": codes_sha,
        "api_receipt_sha256": public["api_receipt_sha256"],
        "public_receipt_sha256": hashlib.sha256(public_raw).hexdigest(),
    }
    expected_raw = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    return public_raw, api_raw, codes_sha, expected_raw


def test_emit_qtip4_option_row_has_exact_sealed_schema_bytes():
    public_raw, api_raw, codes_sha, expected_raw = _fixture()
    assert emit_qtip4_option_row(public_raw, api_raw, codes_sha) == expected_raw


def test_emit_qtip4_option_row_refuses_basis_drift():
    public_raw, api_raw, codes_sha, _ = _fixture()
    public = json.loads(public_raw)
    public["basis_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="basis"):
        emit_qtip4_option_row(
            json.dumps(public, sort_keys=True, separators=(",", ":")).encode(),
            api_raw,
            codes_sha,
        )


def test_emit_root_option_ledger_is_sorted_and_exact(tmp_path):
    public_raw, api_raw, _, _ = _fixture()
    for index in (1, 0):
        cell_dir = tmp_path / "outputs" / "q4" / f"L000_E{index:03d}_down"
        cell_dir.mkdir(parents=True)
        codes_raw = f"codes-{index}".encode()
        codes_path = cell_dir / "codes.npy"
        codes_path.write_bytes(codes_raw)
        codes_sha = hashlib.sha256(codes_raw).hexdigest()
        api = json.loads(api_raw)
        api["artifacts"]["codes"] = {"path": str(codes_path), "sha256": codes_sha}
        current_api_raw = json.dumps(api, sort_keys=True, separators=(",", ":")).encode()
        api_path = cell_dir / "CELL_RECEIPT.json"
        api_path.write_bytes(current_api_raw)
        public = json.loads(public_raw)
        public.update(
            {
                "cell": f"L000/E{index:03d}_down",
                "expert": index,
                "api_receipt": str(api_path),
                "api_receipt_sha256": hashlib.sha256(current_api_raw).hexdigest(),
            }
        )
        (cell_dir / "PUBLIC_CELL_RECEIPT.json").write_text(
            json.dumps(public, sort_keys=True, separators=(",", ":"))
        )

    ledger = emit_root_option_ledger(tmp_path, expected_cells=2)
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert [row["cell"] for row in rows] == ["L000/E000_down", "L000/E001_down"]
    assert all(row["schema"] == "qtip4-v7-option-ledger-row-v1" for row in rows)


def test_emit_root_option_ledger_accepts_authenticated_physical_cells(tmp_path):
    public_raw, api_raw, _, _ = _fixture()
    cell_dir = tmp_path / "outputs" / "q4" / "L000_E000_down"
    cell_dir.mkdir(parents=True)
    codes_sha = "c" * 64
    api = json.loads(api_raw)
    api["artifacts"]["codes"] = {"path": str(cell_dir / "absent.npy"), "sha256": codes_sha}
    current_api_raw = json.dumps(api, sort_keys=True, separators=(",", ":")).encode()
    api_path = cell_dir / "CELL_RECEIPT.json"
    api_path.write_bytes(current_api_raw)
    public = json.loads(public_raw)
    public.update(
        {
            "api_receipt": str(api_path),
            "api_receipt_sha256": hashlib.sha256(current_api_raw).hexdigest(),
        }
    )
    current_public_raw = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    (cell_dir / "PUBLIC_CELL_RECEIPT.json").write_bytes(current_public_raw)
    authenticated = {
        public["cell"]: {
            "public_receipt_sha256": hashlib.sha256(current_public_raw).hexdigest(),
            "cell_receipt_sha256": hashlib.sha256(current_api_raw).hexdigest(),
            "codes_sha256": codes_sha,
        }
    }

    ledger = emit_root_option_ledger(
        tmp_path, expected_cells=1, authenticated_cells=authenticated
    )
    assert json.loads(ledger)["codes_sha256"] == codes_sha
