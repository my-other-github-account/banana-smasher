import json

import pytest

from banana_smasher.qtip4_option_fanin import fanin_qtip4_option_ledgers


BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


def _raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_fanin_selects_only_rows_bound_to_authenticated_physical_cells():
    physical = {
        "cell_id": "L000/E000_down",
        "basis_sha256": BASIS,
        "public_receipt_sha256": "a" * 64,
        "cell_receipt_sha256": "b" * 64,
        "codes_sha256": "c" * 64,
        "fallback_calls": 0,
        "errors": [],
    }
    selected = {
        "schema": "qtip4-v7-option-ledger-row-v1",
        "cell": physical["cell_id"],
        "tier": "qtip4",
        "bpw": 4.0,
        "public_receipt_sha256": physical["public_receipt_sha256"],
        "api_receipt_sha256": physical["cell_receipt_sha256"],
        "codes_sha256": physical["codes_sha256"],
        "fallback_calls": 0,
    }
    superseded = {**selected, "public_receipt_sha256": "d" * 64}

    ledger = fanin_qtip4_option_ledgers(
        _raw(physical), [_raw(superseded), _raw(selected)], expected_cells=1
    )
    assert ledger == _raw(selected)


def test_fanin_refuses_missing_selected_row():
    physical = {
        "cell_id": "L000/E000_down",
        "basis_sha256": BASIS,
        "public_receipt_sha256": "a" * 64,
        "cell_receipt_sha256": "b" * 64,
        "codes_sha256": "c" * 64,
        "fallback_calls": 0,
        "errors": [],
    }
    with pytest.raises(ValueError, match="missing option rows"):
        fanin_qtip4_option_ledgers(_raw(physical), [], expected_cells=1)
