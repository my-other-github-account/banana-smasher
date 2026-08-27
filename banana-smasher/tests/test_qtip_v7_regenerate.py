from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.qtip_v7_regenerate import (
    BASIS_SCHEMA,
    _ordered_sha,
    _validate_request,
)


def _request(tmp_path: Path) -> dict[str, object]:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "model-00001-of-00001.safetensors"}}))
    basis = hashlib.sha256(index.read_bytes()).hexdigest()
    rows = [
        {
            "layer": 34,
            "expert": 161,
            "projection": projection,
            "raw_hessian": {
                "path": f"/sealed/{projection}.npy",
                "sha256": "1" * 64,
                "data_sha256": "2" * 64,
                "count": 512000,
            },
        }
        for projection in ("w1", "w2", "w3")
    ]
    return {
        "schema": BASIS_SCHEMA,
        "task_id": "t_example",
        "board_run_id": 1,
        "basis_sha256": basis,
        "canonical_commit_sha": "3" * 40,
        "model_root": str(model),
        "output_root": str(tmp_path / "output"),
        "members": rows,
        "expected_identities": [
            {"layer": row["layer"], "expert": row["expert"], "projection": row["projection"]}
            for row in rows
        ],
    }


def test_request_binds_model_index_and_exact_member_set(tmp_path: Path) -> None:
    request = _request(tmp_path)
    bound = _validate_request(request)
    assert bound["basis_sha256"] == request["basis_sha256"]
    assert len(bound["members"]) == 3

    request["basis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="intended basis"):
        _validate_request(request)


def test_request_rejects_duplicate_or_drifted_member_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["members"] = [request["members"][0], request["members"][0]]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_request(request)

    request = _request(tmp_path)
    request["expected_identities"] = request["expected_identities"][:-1]
    with pytest.raises(ValueError, match="expected_identities"):
        _validate_request(request)


def test_ordered_sha_binds_member_hash_and_bytes() -> None:
    rows = [
        {"member": "L034/E161/w1", "sha256": "a" * 64, "bytes": 8},
        {"member": "L034/E161/w2", "sha256": "b" * 64, "bytes": 8},
    ]
    assert _ordered_sha(rows) != _ordered_sha(list(reversed(rows)))
    changed = [dict(row) for row in rows]
    changed[0]["bytes"] = 9
    assert _ordered_sha(rows) != _ordered_sha(changed)
