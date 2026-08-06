from __future__ import annotations

import hashlib

import numpy as np
import pytest

from banana_smasher.qtip_periodic_signal import (
    AVG_MEMBER_BASELINE_RECEIPT_SHA256,
    FF0731_MODEL_INDEX_SHA256,
    PERIODIC_SIGNAL_CANDIDATES,
    TEACHER_TOP8192_MANIFEST_SHA256,
    TRAIN64_BANK_MANIFEST_SHA256,
    TRAIN8_POSITION_CUTOFF,
    TRAIN8_ROW_IDS,
    TRAIN8_SUPPORT_WIDTH,
    score_periodic_train8_signal,
    write_periodic_train8_signal_receipt,
)


def _provenance() -> dict[str, object]:
    return {
        "avg_member_receipt_sha256": AVG_MEMBER_BASELINE_RECEIPT_SHA256,
        "bank_manifest_sha256": TRAIN64_BANK_MANIFEST_SHA256,
        "teacher_manifest_sha256": TEACHER_TOP8192_MANIFEST_SHA256,
        "candidate_artifact_sha256": {
            candidate: f"{index:x}" * 64
            for index, candidate in enumerate(PERIODIC_SIGNAL_CANDIDATES, 1)
        },
    }


def _authentic_shape_rows(row_ids: list[str]):
    support_row = np.arange(TRAIN8_SUPPORT_WIDTH, dtype=np.int32)
    support_ids = np.broadcast_to(
        support_row, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )
    teacher_row = np.zeros(TRAIN8_SUPPORT_WIDTH, dtype=np.float32)
    teacher_row[0] = 3.0
    teacher_logits = np.broadcast_to(
        teacher_row, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )
    wrong_row = teacher_row.copy()
    wrong_row[0] = 0.0
    wrong_row[1] = 4.0
    wrong_logits = np.broadcast_to(
        wrong_row, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )
    flat_logits = np.zeros(
        (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH), dtype=np.float32
    )
    candidates = {
        "qtip_k2": teacher_logits,
        "qtip_k3": flat_logits,
        "qtip25_avg_member": wrong_logits,
        "qtip25_periodic_23": teacher_logits,
    }
    full_vocab_argmax = {
        "qtip_k2": np.zeros(TRAIN8_POSITION_CUTOFF, dtype=np.int64),
        "qtip_k3": np.zeros(TRAIN8_POSITION_CUTOFF, dtype=np.int64),
        "qtip25_avg_member": np.ones(TRAIN8_POSITION_CUTOFF, dtype=np.int64),
        "qtip25_periodic_23": np.zeros(TRAIN8_POSITION_CUTOFF, dtype=np.int64),
    }
    for row_id in row_ids:
        yield {
            "row_id": row_id,
            "support_ids": support_ids,
            "teacher_logits": teacher_logits,
            "candidate_logits": candidates,
            "candidate_argmax": full_vocab_argmax,
        }


def test_periodic_train8_scores_authentic_shape_paired_signal(tmp_path) -> None:
    row_ids = list(TRAIN8_ROW_IDS)
    direct_error = {
        "qtip_k2": 2.0,
        "qtip_k3": 1.0,
        "qtip25_avg_member": 1.5,
        "qtip25_periodic_23": 1.25,
    }
    code_bits = {candidate: 10_000 for candidate in PERIODIC_SIGNAL_CANDIDATES}

    receipt = score_periodic_train8_signal(
        rows=_authentic_shape_rows(row_ids),
        expected_row_ids=row_ids,
        intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
        observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
        direct_error=direct_error,
        nominal_code_bits=code_bits,
        provenance=_provenance(),
        chunk_positions=8,
    )

    assert receipt["status"] == "PASS"
    assert receipt["quality_evidence"] is True
    assert receipt["row_ids"] == row_ids
    assert receipt["position_cutoff"] == 1024
    assert receipt["support_width"] == 8192
    assert receipt["paired_same_ids"] is True
    assert receipt["identical_total_nominal_code_bits"] is True
    assert (
        receipt["provenance"]["avg_member_receipt_sha256"]
        == AVG_MEMBER_BASELINE_RECEIPT_SHA256
    )
    assert receipt["candidates"]["qtip25_periodic_23"] == {
        "direct_error": 1.25,
        "nominal_code_bits": 10_000,
        "mean_support_renormalized_kld": 0.0,
        "top1_matches": 8192,
        "top1_positions": 8192,
        "top1_rate": 1.0,
    }
    assert receipt["candidates"]["qtip25_avg_member"]["top1_matches"] == 0
    assert receipt["candidates"]["qtip25_avg_member"]["top1_positions"] == 8192
    assert receipt["candidates"]["qtip25_avg_member"][
        "mean_support_renormalized_kld"
    ] > 0.0
    assert receipt["candidates"]["qtip_k3"][
        "mean_support_renormalized_kld"
    ] > 0.0

    path = tmp_path / "TRAIN8_SIGNAL.json"
    observed_sha = write_periodic_train8_signal_receipt(path, receipt)
    assert observed_sha == hashlib.sha256(path.read_bytes()).hexdigest()


def test_periodic_train8_refuses_non_ff0731_basis_before_rows() -> None:
    row_ids = list(TRAIN8_ROW_IDS)
    values = {candidate: 1 for candidate in PERIODIC_SIGNAL_CANDIDATES}

    with pytest.raises(ValueError, match="exact current FF0731 intended basis"):
        score_periodic_train8_signal(
            rows=[],
            expected_row_ids=row_ids,
            intended_basis_sha256="a" * 64,
            observed_basis_sha256="a" * 64,
            direct_error=values,
            nominal_code_bits=values,
            provenance=_provenance(),
        )


def test_periodic_train8_refuses_unpaired_ids_and_unequal_code_spend() -> None:
    row_ids = list(TRAIN8_ROW_IDS)
    errors = {candidate: 1.0 for candidate in PERIODIC_SIGNAL_CANDIDATES}
    unequal_bits = {
        candidate: (1 << 53) + 1 for candidate in PERIODIC_SIGNAL_CANDIDATES
    }
    unequal_bits["qtip25_periodic_23"] = (1 << 53) + 2

    with pytest.raises(ValueError, match="identical code bits"):
        score_periodic_train8_signal(
            rows=[],
            expected_row_ids=row_ids,
            intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            direct_error=errors,
            nominal_code_bits=unequal_bits,
            provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="row 0 id mismatch"):
        score_periodic_train8_signal(
            rows=[
                {
                    "row_id": "wrong-id",
                    "support_ids": np.empty((0, 0), dtype=np.int32),
                    "teacher_logits": np.empty((0, 0), dtype=np.float32),
                    "candidate_logits": {},
                }
            ],
            expected_row_ids=row_ids,
            intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            direct_error=errors,
            nominal_code_bits={candidate: 10_000 for candidate in PERIODIC_SIGNAL_CANDIDATES},
            provenance=_provenance(),
        )


def test_periodic_train8_refuses_duplicate_support_and_float64_logits() -> None:
    row_ids = list(TRAIN8_ROW_IDS)
    values = {candidate: 1 for candidate in PERIODIC_SIGNAL_CANDIDATES}
    duplicate_row = next(_authentic_shape_rows(row_ids))
    support_row = np.arange(TRAIN8_SUPPORT_WIDTH, dtype=np.int32)
    support_row[-1] = support_row[-2]
    duplicate_row["support_ids"] = np.broadcast_to(
        support_row, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )

    with pytest.raises(ValueError, match="8192 unique token ids"):
        score_periodic_train8_signal(
            rows=[duplicate_row],
            expected_row_ids=row_ids,
            intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            direct_error=values,
            nominal_code_bits=values,
            provenance=_provenance(),
        )

    extreme_row = next(_authentic_shape_rows(row_ids))
    extreme = np.zeros(TRAIN8_SUPPORT_WIDTH, dtype=np.float64)
    extreme[0] = 1e308
    extreme[1] = -1e308
    extreme_row["teacher_logits"] = np.broadcast_to(
        extreme, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )
    with pytest.raises(ValueError, match="finite floating-point logits"):
        score_periodic_train8_signal(
            rows=[extreme_row],
            expected_row_ids=row_ids,
            intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            direct_error=values,
            nominal_code_bits=values,
            provenance=_provenance(),
        )

    wrong_teacher_row = next(_authentic_shape_rows(row_ids))
    teacher_row = np.zeros(TRAIN8_SUPPORT_WIDTH, dtype=np.float32)
    teacher_row[1] = 1.0
    wrong_teacher_row["teacher_logits"] = np.broadcast_to(
        teacher_row, (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH)
    )
    with pytest.raises(ValueError, match="support index zero"):
        score_periodic_train8_signal(
            rows=[wrong_teacher_row],
            expected_row_ids=row_ids,
            intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
            direct_error=values,
            nominal_code_bits=values,
            provenance=_provenance(),
        )


def test_periodic_signal_receipt_refuses_symlink_overwrite_and_nan(tmp_path) -> None:
    row_ids = list(TRAIN8_ROW_IDS)
    values = {candidate: 1 for candidate in PERIODIC_SIGNAL_CANDIDATES}
    receipt = score_periodic_train8_signal(
        rows=_authentic_shape_rows(row_ids),
        expected_row_ids=row_ids,
        intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
        observed_basis_sha256=FF0731_MODEL_INDEX_SHA256,
        direct_error=values,
        nominal_code_bits=values,
        provenance=_provenance(),
        chunk_positions=8,
    )
    target = tmp_path / "target.json"
    target.write_text("sentinel\n")
    symlink = tmp_path / "receipt.json"
    symlink.symlink_to(target)

    with pytest.raises(FileExistsError, match="exists or is a symlink"):
        write_periodic_train8_signal_receipt(symlink, receipt)
    assert target.read_text() == "sentinel\n"

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(FileExistsError, match="ancestor is a symlink"):
        write_periodic_train8_signal_receipt(linked_parent / "receipt.json", receipt)
    assert not (outside / "receipt.json").exists()

    with pytest.raises(ValueError, match="completed evidence"):
        write_periodic_train8_signal_receipt(
            tmp_path / "minimal.json",
            {"schema": "banana-smasher-qtip25-periodic-train8-signal-v1"},
        )

    nan_path = tmp_path / "nan.json"
    receipt["candidates"]["qtip_k2"]["mean_support_renormalized_kld"] = float("nan")
    with pytest.raises(ValueError, match="is inconsistent"):
        write_periodic_train8_signal_receipt(nan_path, receipt)
    assert not nan_path.exists()
