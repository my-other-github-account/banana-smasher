from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import pytest

from evaluations.tools.receipts import (
    BALANCED64_V1_LOCK_SHA256,
    ReceiptError,
    aggregate_windows,
    main,
    verify_result_receipt,
    verify_suite_lock,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "notes/evaluations/results/deepseek-v4-flash-0731-balanced64-v1.json"
SUITE_LOCK = ROOT / "evaluations/configs/balanced64-v1.json"


def _load_result() -> dict[str, Any]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _load_lock() -> dict[str, Any]:
    return json.loads(SUITE_LOCK.read_text(encoding="utf-8"))


def _canonical_lock_digest(lock: dict[str, Any]) -> str:
    payload = copy.deepcopy(lock)
    payload.pop("suite_lock_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_rows(kld_value: str = "1.0") -> list[dict[str, Any]]:
    lock = _load_lock()
    rows = []
    for window in lock["windows"]:
        rows.append(
            {
                "schema": "banana-smasher.balanced64-window.v1",
                "suite_lock_sha256": lock["suite_lock_sha256"],
                "teacher_source_model_index_sha256": lock[
                    "teacher_source_model_index_sha256"
                ],
                "candidate_artifact_sha256": "c" * 64,
                "ordinal": window["ordinal"],
                "window_id": window["window_id"],
                "source_class": window["source_class"],
                "positions": lock["positions_per_window"],
                "kld_values": [kld_value] * lock["positions_per_window"],
                "top1_matches": 512,
            }
        )
    return rows


def test_published_balanced64_receipt_is_bound_to_suite_lock() -> None:
    summary = verify_result_receipt(_load_result(), _load_lock())

    assert summary["suite_lock_sha256"] == BALANCED64_V1_LOCK_SHA256
    assert summary["models"] == 4
    assert summary["positions"] == 65_536
    assert summary["full_gpu_replay"] == "blocked"
    assert summary["kld_ranking"] == [
        "UD-IQ4_XS",
        "UD-IQ3_XXS",
        "UD-IQ2_XXS",
        "DwarfStar-Q2-0731",
    ]
    assert summary["top1_ranking"] == [
        "UD-IQ4_XS",
        "UD-IQ3_XXS",
        "UD-IQ2_XXS",
        "DwarfStar-Q2-0731",
    ]


def test_suite_lock_recomputes_population_and_corrected_class_map() -> None:
    lock = _load_lock()
    verify_suite_lock(lock)
    windows = lock["windows"]

    assert lock["suite_lock_sha256"] == _canonical_lock_digest(lock)
    assert [row["ordinal"] for row in windows] == list(range(64))
    assert len({row["window_id"] for row in windows}) == 64
    assert dict(Counter(row["source_class"] for row in windows)) == {
        "agentic": 19,
        "chat": 7,
        "code": 9,
        "multilingual": 10,
        "prose": 10,
        "reasoning": 9,
    }
    assert lock["retired_class_map"]["status"] == "invalid-for-subgroup-reporting"


def test_verifier_cli_requires_and_reports_suite_lock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(["verify", str(RESULT), "--suite-lock", str(SUITE_LOCK)])
        == 0
    )
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["suite_lock_sha256"] == BALANCED64_V1_LOCK_SHA256
    assert rendered["kld_ranking"][0] == "UD-IQ4_XS"
    assert rendered["top1_ranking"][-1] == "DwarfStar-Q2-0731"


def test_aggregate_cli_binds_all_64_windows_and_per_position_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows_dir = tmp_path / "windows"
    rows_dir.mkdir()
    for row in _window_rows("1.0"):
        (rows_dir / f"{row['ordinal']:02d}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )

    assert (
        main(
            [
                "aggregate",
                str(rows_dir),
                "--suite-lock",
                str(SUITE_LOCK),
            ]
        )
        == 0
    )
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["windows"] == 64
    assert rendered["positions"] == 65_536
    assert rendered["top1_matches"] == 32_768
    assert rendered["kld_mean"] == "1.0"


def test_verifier_rejects_changed_global_bpw_denominator() -> None:
    receipt = _load_result()
    receipt["wire_parameter_denominator"] += 1
    for row in receipt["results"]:
        row["wire"]["parameter_denominator"] += 1

    with pytest.raises(ReceiptError, match="BPW denominator differs from suite lock"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_top1_rate_drift() -> None:
    receipt = _load_result()
    receipt["results"][0]["top1"]["rate"] = "0.5"

    with pytest.raises(ReceiptError, match="Top-1 rate"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_noncanonical_decimal_spelling() -> None:
    receipt = _load_result()
    receipt["results"][0]["top1"]["rate"] = "00.5"

    with pytest.raises(ReceiptError, match="canonical nonnegative decimal string"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_low_precision_bpw() -> None:
    receipt = _load_result()
    receipt["results"][0]["wire"]["normalized_bpw"] = "3"

    with pytest.raises(ReceiptError, match="at least 30 significant digits"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_malformed_nested_artifact_digest() -> None:
    receipt = _load_result()
    receipt["results"][0]["artifact"]["artifact_manifest_sha256"] = "not-a-digest"

    with pytest.raises(ReceiptError, match="artifact_manifest_sha256"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_missing_artifact_identity_status() -> None:
    receipt = _load_result()
    del receipt["results"][0]["artifact"]["identity_status"]

    with pytest.raises(ReceiptError, match="artifact key drift"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_extra_artifact_fields() -> None:
    receipt = _load_result()
    receipt["results"][0]["artifact"]["untracked"] = "value"

    with pytest.raises(ReceiptError, match="artifact key drift"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_verification_scope_drift() -> None:
    receipt = _load_result()
    receipt["verification_scope"]["full_gpu_replay"] = "available"

    with pytest.raises(ReceiptError, match="published limitations"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_duplicate_source_receipts() -> None:
    receipt = _load_result()
    source = copy.deepcopy(receipt["results"][0]["source_receipts"][0])
    source["label"] = "duplicate digest under another label"
    receipt["results"][0]["source_receipts"].append(source)

    with pytest.raises(ReceiptError, match="duplicate source digest"):
        verify_result_receipt(receipt, _load_lock())


def test_verifier_rejects_unavailable_source_claim_drift() -> None:
    receipt = _load_result()
    receipt["results"][0]["source_receipts"][0]["availability"] = "public"

    with pytest.raises(ReceiptError, match="source availability"):
        verify_result_receipt(receipt, _load_lock())


def test_suite_lock_rejects_content_drift_even_with_recomputed_digest() -> None:
    lock = _load_lock()
    lock["support"] = 1
    lock["suite_lock_sha256"] = _canonical_lock_digest(lock)

    with pytest.raises(ReceiptError, match="not the published BALANCED64_V1 authority"):
        verify_suite_lock(lock)


def test_suite_lock_rejects_regrouped_population_drift() -> None:
    lock = _load_lock()
    lock["windows"][0]["source_class"] = "agentic"
    lock["class_windows"]["agentic"] += 1
    lock["class_windows"]["prose"] -= 1
    lock["suite_lock_sha256"] = _canonical_lock_digest(lock)

    with pytest.raises(ReceiptError, match="not the published BALANCED64_V1 authority"):
        verify_suite_lock(lock)


def test_window_aggregation_is_independent_of_decimal_context() -> None:
    rows = _window_rows("0.1")
    lock = _load_lock()
    original_precision = getcontext().prec
    try:
        getcontext().prec = 7
        low_precision = aggregate_windows(rows, lock)
        getcontext().prec = 50
        high_precision = aggregate_windows(rows, lock)
    finally:
        getcontext().prec = original_precision

    assert low_precision == high_precision
    assert low_precision["kld_mean"] == "0.1"
    assert low_precision["top1_rate"] == Decimal("0.5")
    assert low_precision["classes"]["agentic"]["windows"] == 19


def test_window_aggregation_rejects_population_drift() -> None:
    rows = _window_rows()
    rows[0]["window_id"] += 1

    with pytest.raises(ReceiptError, match="frozen suite lock"):
        aggregate_windows(rows, _load_lock())


def test_window_aggregation_rejects_basis_drift() -> None:
    rows = _window_rows()
    rows[1]["candidate_artifact_sha256"] = "d" * 64

    with pytest.raises(ReceiptError, match="candidate_artifact_sha256 basis drift"):
        aggregate_windows(rows, _load_lock())


def test_window_aggregation_rejects_negative_kld_without_clamping() -> None:
    rows = _window_rows()
    rows[0]["kld_values"][0] = "-0.1"

    with pytest.raises(ReceiptError, match="no clamp"):
        aggregate_windows(rows, _load_lock())


@pytest.mark.parametrize("value", ["-0.0", "-0"])
def test_window_aggregation_rejects_negative_zero(value: str) -> None:
    rows = _window_rows()
    rows[0]["kld_values"][0] = value

    with pytest.raises(ReceiptError, match="no clamp"):
        aggregate_windows(rows, _load_lock())


def test_window_aggregation_requires_shortest_round_trip_binary64() -> None:
    rows = _window_rows()
    rows[0]["kld_values"][0] = "0.10000000000000001"

    with pytest.raises(ReceiptError, match="shortest round-trip"):
        aggregate_windows(rows, _load_lock())


def test_json_loader_rejects_duplicate_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "duplicate.json"
    receipt.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["verify", str(receipt), "--suite-lock", str(SUITE_LOCK)])
    assert exc_info.value.code == 1
    assert "duplicate JSON key: schema" in capsys.readouterr().err
