from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

FF0731_MODEL_INDEX_SHA256 = (
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
)
AVG_MEMBER_BASELINE_RECEIPT_SHA256 = (
    "379391cbd84868b1ae4d8713148351ebff7aac02df24c053a257dd7fc66f1917"
)
TRAIN64_BANK_MANIFEST_SHA256 = (
    "b25d54cc55690d18844fb3339aaf9a07338c1b4f3e5a7292aebf7a39a19f833f"
)
TEACHER_TOP8192_MANIFEST_SHA256 = (
    "46e629f4cf273773068a4d79346095e0deffd56d1c886ae2c25216ae9072d58b"
)
PERIODIC_SIGNAL_CANDIDATES = (
    "qtip_k2",
    "qtip_k3",
    "qtip25_avg_member",
    "qtip25_periodic_23",
)
TRAIN8_WINDOW_COUNT = 8
TRAIN8_ROW_IDS = ("10", "12", "19", "24", "37", "45", "57", "60")
TRAIN8_POSITION_CUTOFF = 1024
TRAIN8_SUPPORT_WIDTH = 8192


def _require_basis(intended: str, observed: str) -> None:
    if intended != FF0731_MODEL_INDEX_SHA256:
        raise ValueError(
            "periodic quality signal requires the exact current FF0731 intended basis"
        )
    if observed != intended:
        raise ValueError(
            f"periodic quality signal basis mismatch: intended {intended}, observed {observed}"
        )


def _candidate_values(
    name: str, values: Mapping[str, float | int]
) -> dict[str, float]:
    if set(values) != set(PERIODIC_SIGNAL_CANDIDATES):
        raise ValueError(
            f"{name} must contain exactly {list(PERIODIC_SIGNAL_CANDIDATES)}"
        )
    normalized = {candidate: float(values[candidate]) for candidate in PERIODIC_SIGNAL_CANDIDATES}
    if not all(np.isfinite(value) and value >= 0.0 for value in normalized.values()):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return normalized


def _candidate_code_bits(values: Mapping[str, int]) -> dict[str, int]:
    if set(values) != set(PERIODIC_SIGNAL_CANDIDATES):
        raise ValueError(
            "nominal_code_bits must contain exactly "
            f"{list(PERIODIC_SIGNAL_CANDIDATES)}"
        )
    normalized: dict[str, int] = {}
    for candidate in PERIODIC_SIGNAL_CANDIDATES:
        value = values[candidate]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError("nominal_code_bits values must be positive exact integers")
        normalized[candidate] = int(value)
    if len(set(normalized.values())) != 1:
        raise ValueError("all periodic quality-signal candidates must spend identical code bits")
    return normalized


def _sha256_identity(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "avg_member_receipt_sha256",
        "bank_manifest_sha256",
        "teacher_manifest_sha256",
        "candidate_artifact_sha256",
    }
    if set(value) != required:
        raise ValueError(f"provenance must contain exactly {sorted(required)}")
    avg_member = _sha256_identity(
        "avg_member_receipt_sha256", value["avg_member_receipt_sha256"]
    )
    if avg_member != AVG_MEMBER_BASELINE_RECEIPT_SHA256:
        raise ValueError("periodic quality signal requires the immutable AVG-MEMBER receipt")
    bank_manifest = _sha256_identity(
        "bank_manifest_sha256", value["bank_manifest_sha256"]
    )
    if bank_manifest != TRAIN64_BANK_MANIFEST_SHA256:
        raise ValueError("periodic quality signal requires the frozen train_balanced64 manifest")
    teacher_manifest = _sha256_identity(
        "teacher_manifest_sha256", value["teacher_manifest_sha256"]
    )
    if teacher_manifest != TEACHER_TOP8192_MANIFEST_SHA256:
        raise ValueError("periodic quality signal requires the frozen FF0731 teacher manifest")
    candidate_artifacts = value["candidate_artifact_sha256"]
    if not isinstance(candidate_artifacts, Mapping) or set(candidate_artifacts) != set(
        PERIODIC_SIGNAL_CANDIDATES
    ):
        raise ValueError(
            "candidate_artifact_sha256 must contain exactly "
            f"{list(PERIODIC_SIGNAL_CANDIDATES)}"
        )
    return {
        "avg_member_receipt_sha256": avg_member,
        "bank_manifest_sha256": bank_manifest,
        "teacher_manifest_sha256": teacher_manifest,
        "candidate_artifact_sha256": {
            candidate: _sha256_identity(
                f"candidate_artifact_sha256[{candidate}]",
                candidate_artifacts[candidate],
            )
            for candidate in PERIODIC_SIGNAL_CANDIDATES
        },
    }


def _row_array(value: Any, *, name: str, integer: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH):
        raise ValueError(
            f"{name} must have shape "
            f"({TRAIN8_POSITION_CUTOFF}, {TRAIN8_SUPPORT_WIDTH}), got {result.shape}"
        )
    if integer:
        if result.dtype.kind not in "iu" or bool(np.any(result < 0)):
            raise ValueError(f"{name} must contain nonnegative integer token ids")
    elif (
        result.dtype.kind not in "f"
        or result.dtype.itemsize > 4
        or not bool(np.isfinite(result).all())
    ):
        raise ValueError(f"{name} must contain finite floating-point logits")
    return result


def _row_argmax(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.shape != (TRAIN8_POSITION_CUTOFF,)
        or result.dtype.kind not in "iu"
        or bool(np.any(result < 0))
    ):
        raise ValueError(
            f"{name} must contain {TRAIN8_POSITION_CUTOFF} nonnegative integer token ids"
        )
    return result


def _score_chunk(
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
    support_ids: np.ndarray,
    candidate_argmax: np.ndarray,
) -> tuple[float, int, int]:
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if bool(np.any(np.argmax(teacher, axis=1) != 0)):
        raise ValueError("teacher support index zero must be the full-vocabulary Top-1 token")

    teacher_max = np.max(teacher, axis=1, keepdims=True)
    teacher_shifted = teacher - teacher_max
    teacher_exp = np.exp(teacher_shifted)
    teacher_sum = np.sum(teacher_exp, axis=1, keepdims=True, dtype=np.float64)
    teacher_log_prob = teacher_shifted - np.log(teacher_sum)
    teacher_prob = teacher_exp / teacher_sum

    candidate_max = np.max(candidate, axis=1, keepdims=True)
    candidate_log_prob = candidate - candidate_max
    candidate_log_prob -= np.log(
        np.sum(np.exp(candidate_log_prob), axis=1, keepdims=True, dtype=np.float64)
    )
    kld_sum = float(
        np.sum(teacher_prob * (teacher_log_prob - candidate_log_prob), dtype=np.float64)
    )
    if not np.isfinite(kld_sum):
        raise ValueError("periodic quality-signal KLD is non-finite")

    positions = teacher.shape[0]
    teacher_tokens = support_ids[:, 0]
    matches = int(np.count_nonzero(teacher_tokens == candidate_argmax))
    return kld_sum, matches, positions


def _validate_support_chunk(support_ids: np.ndarray) -> None:
    ordered = np.sort(support_ids, axis=1)
    if bool(np.any(ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError("each teacher support row must contain 8192 unique token ids")


def _update_array_hash(
    digest: Any, label: str, row_id: str, array: np.ndarray
) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(row_id.encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(array).view(np.uint8))


def score_periodic_train8_signal(
    *,
    rows: Iterable[Mapping[str, Any]],
    expected_row_ids: Sequence[str],
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    direct_error: Mapping[str, float | int],
    nominal_code_bits: Mapping[str, int],
    provenance: Mapping[str, Any],
    chunk_positions: int = 16,
) -> dict[str, Any]:
    """Score the frozen paired train8 signal on one shared top-8192 support.

    Each row supplies ``row_id``, ``support_ids``, ``teacher_logits``, and
    ``candidate_logits``/``candidate_argmax`` mappings with the four declared
    controls/candidates. Support/logit arrays are exactly 1,024 positions by
    8,192 entries; each candidate argmax contains 1,024 full-vocabulary token
    ids. The first support id is the teacher's full-vocabulary argmax, matching
    the current anchor-sidecar contract. Processing is position-chunked so host
    runners can supply memmaps without dense copies.
    """
    _require_basis(intended_basis_sha256, observed_basis_sha256)
    row_ids = [str(value) for value in expected_row_ids]
    if tuple(row_ids) != TRAIN8_ROW_IDS:
        raise ValueError(
            f"periodic quality signal requires frozen row ids {list(TRAIN8_ROW_IDS)}"
        )
    errors = _candidate_values("direct_error", direct_error)
    code_bits = _candidate_code_bits(nominal_code_bits)
    identities = _provenance(provenance)
    chunk = int(chunk_positions)
    if not 1 <= chunk <= TRAIN8_POSITION_CUTOFF:
        raise ValueError(
            f"chunk_positions must be in [1, {TRAIN8_POSITION_CUTOFF}]"
        )

    totals = {
        candidate: {"kld_sum": 0.0, "top1_matches": 0, "positions": 0}
        for candidate in PERIODIC_SIGNAL_CANDIDATES
    }
    payload_hashes = {
        "teacher_support": hashlib.sha256(),
        **{candidate: hashlib.sha256() for candidate in PERIODIC_SIGNAL_CANDIDATES},
    }
    observed_row_ids: list[str] = []
    for row_index, row in enumerate(rows):
        if row_index >= TRAIN8_WINDOW_COUNT:
            raise ValueError("periodic quality signal received more than eight rows")
        row_id = str(row.get("row_id", ""))
        if row_id != row_ids[row_index]:
            raise ValueError(
                f"periodic quality signal row {row_index} id mismatch: "
                f"expected {row_ids[row_index]!r}, observed {row_id!r}"
            )
        observed_row_ids.append(row_id)
        support = _row_array(row.get("support_ids"), name="support_ids", integer=True)
        teacher = _row_array(row.get("teacher_logits"), name="teacher_logits")
        candidates = row.get("candidate_logits")
        if not isinstance(candidates, Mapping) or set(candidates) != set(
            PERIODIC_SIGNAL_CANDIDATES
        ):
            raise ValueError(
                "candidate_logits must contain exactly "
                f"{list(PERIODIC_SIGNAL_CANDIDATES)}"
            )
        candidate_argmax = row.get("candidate_argmax")
        if not isinstance(candidate_argmax, Mapping) or set(candidate_argmax) != set(
            PERIODIC_SIGNAL_CANDIDATES
        ):
            raise ValueError(
                "candidate_argmax must contain exactly "
                f"{list(PERIODIC_SIGNAL_CANDIDATES)}"
            )
        candidate_arrays = {
            candidate: _row_array(
                candidates[candidate], name=f"candidate_logits[{candidate}]"
            )
            for candidate in PERIODIC_SIGNAL_CANDIDATES
        }
        candidate_argmax_arrays = {
            candidate: _row_argmax(
                candidate_argmax[candidate], name=f"candidate_argmax[{candidate}]"
            )
            for candidate in PERIODIC_SIGNAL_CANDIDATES
        }
        _update_array_hash(
            payload_hashes["teacher_support"], "support_ids", row_id, support
        )
        _update_array_hash(
            payload_hashes["teacher_support"], "teacher_logits", row_id, teacher
        )
        for candidate in PERIODIC_SIGNAL_CANDIDATES:
            _update_array_hash(
                payload_hashes[candidate],
                "candidate_logits",
                row_id,
                candidate_arrays[candidate],
            )
            _update_array_hash(
                payload_hashes[candidate],
                "candidate_argmax",
                row_id,
                candidate_argmax_arrays[candidate],
            )
        for start in range(0, TRAIN8_POSITION_CUTOFF, chunk):
            stop = min(start + chunk, TRAIN8_POSITION_CUTOFF)
            support_chunk = support[start:stop]
            _validate_support_chunk(support_chunk)
            teacher_chunk = teacher[start:stop]
            for candidate, values in candidate_arrays.items():
                kld_sum, matches, positions = _score_chunk(
                    teacher_chunk,
                    values[start:stop],
                    support_chunk,
                    candidate_argmax_arrays[candidate][start:stop],
                )
                totals[candidate]["kld_sum"] += kld_sum
                totals[candidate]["top1_matches"] += matches
                totals[candidate]["positions"] += positions
    if observed_row_ids != row_ids:
        raise ValueError(
            f"periodic quality signal requires {TRAIN8_WINDOW_COUNT} rows; "
            f"observed {len(observed_row_ids)}"
        )

    result_rows: dict[str, dict[str, Any]] = {}
    for candidate in PERIODIC_SIGNAL_CANDIDATES:
        total = totals[candidate]
        positions = int(total["positions"])
        matches = int(total["top1_matches"])
        result_rows[candidate] = {
            "direct_error": errors[candidate],
            "nominal_code_bits": int(code_bits[candidate]),
            "mean_support_renormalized_kld": float(total["kld_sum"] / positions),
            "top1_matches": matches,
            "top1_positions": positions,
            "top1_rate": float(matches / positions),
        }

    return {
        "schema": "banana-smasher-qtip25-periodic-train8-signal-v1",
        "status": "PASS",
        "quality_evidence": True,
        "basis": {
            "intended_model_index_sha256": intended_basis_sha256,
            "observed_model_index_sha256": observed_basis_sha256,
        },
        "provenance": {
            **identities,
            "row_ids_sha256": hashlib.sha256(
                json.dumps(row_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "scored_payload_sha256": {
                name: digest.hexdigest() for name, digest in payload_hashes.items()
            },
        },
        "bank": "train_balanced64",
        "row_ids": row_ids,
        "position_cutoff": TRAIN8_POSITION_CUTOFF,
        "support_width": TRAIN8_SUPPORT_WIDTH,
        "top1_semantics": "full-vocabulary candidate argmax equals teacher support index zero",
        "kld_semantics": "teacher-to-candidate support-renormalized natural-log KLD",
        "paired_same_ids": True,
        "identical_total_nominal_code_bits": True,
        "candidates": result_rows,
        "safety": {
            "holdout_used": False,
            "repair_used": False,
            "update12_used": False,
            "historical_basis_used": False,
        },
    }


def _validate_receipt_for_write(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != "banana-smasher-qtip25-periodic-train8-signal-v1":
        raise ValueError("unexpected periodic quality-signal receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("quality_evidence") is not True:
        raise ValueError("periodic quality-signal receipt must contain completed evidence")
    basis = receipt.get("basis")
    if not isinstance(basis, Mapping):
        raise ValueError("periodic quality-signal receipt is missing basis identity")
    _require_basis(
        str(basis.get("intended_model_index_sha256", "")),
        str(basis.get("observed_model_index_sha256", "")),
    )
    if tuple(str(value) for value in receipt.get("row_ids", ())) != TRAIN8_ROW_IDS:
        raise ValueError("periodic quality-signal receipt has the wrong frozen row ids")
    if (
        receipt.get("position_cutoff") != TRAIN8_POSITION_CUTOFF
        or receipt.get("support_width") != TRAIN8_SUPPORT_WIDTH
        or receipt.get("paired_same_ids") is not True
        or receipt.get("identical_total_nominal_code_bits") is not True
    ):
        raise ValueError("periodic quality-signal receipt has incompatible scoring semantics")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("periodic quality-signal receipt is missing provenance")
    _provenance(
        {
            key: provenance.get(key)
            for key in (
                "avg_member_receipt_sha256",
                "bank_manifest_sha256",
                "teacher_manifest_sha256",
                "candidate_artifact_sha256",
            )
        }
    )
    payload_hashes = provenance.get("scored_payload_sha256")
    payload_names = {"teacher_support", *PERIODIC_SIGNAL_CANDIDATES}
    if not isinstance(payload_hashes, Mapping) or set(payload_hashes) != payload_names:
        raise ValueError("periodic quality-signal receipt is missing scored payload hashes")
    for name in sorted(payload_names):
        _sha256_identity(f"scored_payload_sha256[{name}]", payload_hashes[name])

    candidates = receipt.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(
        PERIODIC_SIGNAL_CANDIDATES
    ):
        raise ValueError("periodic quality-signal receipt has the wrong candidate cohort")
    code_bits: dict[str, int] = {}
    for candidate in PERIODIC_SIGNAL_CANDIDATES:
        result = candidates[candidate]
        if not isinstance(result, Mapping):
            raise ValueError(f"periodic quality-signal result {candidate} is malformed")
        direct_error = float(result.get("direct_error", float("nan")))
        mean_kld = float(result.get("mean_support_renormalized_kld", float("nan")))
        matches = result.get("top1_matches")
        positions = result.get("top1_positions")
        rate = float(result.get("top1_rate", float("nan")))
        bits = result.get("nominal_code_bits")
        if (
            not np.isfinite(direct_error)
            or direct_error < 0.0
            or not np.isfinite(mean_kld)
            or mean_kld < 0.0
            or isinstance(matches, bool)
            or not isinstance(matches, Integral)
            or not 0 <= int(matches) <= TRAIN8_WINDOW_COUNT * TRAIN8_POSITION_CUTOFF
            or positions != TRAIN8_WINDOW_COUNT * TRAIN8_POSITION_CUTOFF
            or not np.isfinite(rate)
            or rate != int(matches) / int(positions)
        ):
            raise ValueError(f"periodic quality-signal result {candidate} is inconsistent")
        code_bits[candidate] = bits
    _candidate_code_bits(code_bits)


def write_periodic_train8_signal_receipt(
    path: str | Path, receipt: Mapping[str, Any]
) -> str:
    """Write one deterministic quality-signal receipt and return its SHA-256."""
    _validate_receipt_for_write(receipt)
    destination = Path(path).expanduser().absolute()
    if any(parent.is_symlink() for parent in destination.parents):
        raise FileExistsError(
            f"periodic quality-signal receipt ancestor is a symlink: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        any(parent.is_symlink() for parent in destination.parents)
        or destination.exists()
        or destination.is_symlink()
    ):
        raise FileExistsError(
            f"periodic quality-signal receipt path exists or is a symlink: {destination}"
        )
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, destination, follow_symlinks=False)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload.encode()).hexdigest()
