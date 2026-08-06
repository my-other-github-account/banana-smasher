from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .backpack_contextual import (
    ContextualValuationError,
    _atomic_json,
    _json_input,
)


def _window_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validated_score(score: Mapping[str, Any], *, role: str) -> list[dict[str, Any]]:
    if (
        score.get("schema") != "banana-smasher-anchor-sidecar-score-v1"
        or score.get("status") != "PASS"
        or score.get("claimable") is not True
    ):
        raise ContextualValuationError(f"{role} score must be claimable sidecar PASS")
    rows = score.get("per_window")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ContextualValuationError(f"{role} score requires at least two windows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContextualValuationError(f"{role} per-window score is invalid")
        window_id = row.get("window_id")
        key = _window_key(window_id)
        positions = row.get("positions")
        mean_kld = row.get("mean_kld")
        top1_matches = row.get("top1_matches")
        if key in seen:
            raise ContextualValuationError(f"{role} window ids must be unique")
        seen.add(key)
        if not isinstance(positions, int) or isinstance(positions, bool) or positions < 1:
            raise ContextualValuationError(f"{role} window positions are invalid")
        if (
            not isinstance(mean_kld, (int, float))
            or isinstance(mean_kld, bool)
            or not math.isfinite(mean_kld)
            or mean_kld < 0
        ):
            raise ContextualValuationError(f"{role} window KLD is invalid")
        if (
            not isinstance(top1_matches, int)
            or isinstance(top1_matches, bool)
            or not 0 <= top1_matches <= positions
        ):
            raise ContextualValuationError(f"{role} window Top-1 count is invalid")
        normalized.append(
            {
                "window_id": window_id,
                "positions": positions,
                "mean_kld": float(mean_kld),
                "top1_matches": top1_matches,
            }
        )
    positions = sum(row["positions"] for row in normalized)
    mean_kld = math.fsum(
        row["mean_kld"] * row["positions"] for row in normalized
    ) / positions
    top1_matches = sum(row["top1_matches"] for row in normalized)
    if score.get("windows") != len(normalized) or score.get("positions") != positions:
        raise ContextualValuationError(f"{role} aggregate coverage mismatch")
    if not math.isclose(float(score.get("mean_kld")), mean_kld, abs_tol=1e-12):
        raise ContextualValuationError(f"{role} aggregate KLD mismatch")
    if score.get("top1_matches") != top1_matches:
        raise ContextualValuationError(f"{role} aggregate Top-1 mismatch")
    return normalized


def record_contextual_swap_measurement(
    anchor_path: str | Path,
    change_path: str | Path,
    anchor_score_path: str | Path,
    candidate_score_path: str | Path,
    *,
    measurement_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive one paired physical swap delta and append it to the anchor manifest."""

    anchor, anchor_binding = _json_input(anchor_path, role="anchor")
    change, change_binding = _json_input(change_path, role="change")
    anchor_score, anchor_score_binding = _json_input(
        anchor_score_path, role="anchor_score"
    )
    candidate_score, candidate_score_binding = _json_input(
        candidate_score_path, role="candidate_score"
    )
    manifest, _ = _json_input(
        measurement_manifest_path, role="measurement_manifest"
    )
    if (
        anchor.get("schema") != "banana-smasher-contextual-anchor-v1"
        or anchor.get("status") != "PASS"
    ):
        raise ContextualValuationError("contextual anchor must be v1 PASS")
    if anchor.get("physical_score_receipt_sha256") != anchor_score_binding["sha256"]:
        raise ContextualValuationError("anchor physical score SHA mismatch")
    if (
        change.get("schema") != "banana-smasher-contextual-change-v1"
        or change.get("status") != "READY"
        or change.get("anchor_assignment_sha256")
        != anchor.get("assignment_sha256")
    ):
        raise ContextualValuationError("contextual change must be anchor-bound READY v1")
    changed = change.get("change")
    if not isinstance(changed, Mapping):
        raise ContextualValuationError("contextual physical change is missing")
    cell = changed.get("cell")
    physical_identity = changed.get("physical_identity")
    if not isinstance(cell, str) or not cell:
        raise ContextualValuationError("contextual change cell is invalid")
    if not isinstance(physical_identity, str) or not physical_identity:
        raise ContextualValuationError("contextual change identity is invalid")
    if cell not in {row.get("cell") for row in anchor.get("cells", [])}:
        raise ContextualValuationError("contextual change references unknown cell")
    scope = change.get("scope")
    if not isinstance(scope, str) or not scope:
        raise ContextualValuationError("contextual change scope is invalid")

    anchor_rows = _validated_score(anchor_score, role="anchor")
    candidate_rows = _validated_score(candidate_score, role="candidate")
    anchor_identities = anchor_score.get("identities")
    candidate_identities = candidate_score.get("identities")
    if not isinstance(anchor_identities, Mapping) or not isinstance(
        candidate_identities, Mapping
    ):
        raise ContextualValuationError("physical score identities are missing")
    common_identity_fields = set(anchor_identities) - {"pack_sha256"}
    if (
        set(candidate_identities) - {"pack_sha256"} != common_identity_fields
        or any(
            anchor_identities[field] != candidate_identities[field]
            for field in common_identity_fields
        )
    ):
        raise ContextualValuationError("physical score source identities differ")
    candidate_pack_sha256 = change.get("candidate_pack_sha256")
    if (
        not isinstance(candidate_pack_sha256, str)
        or candidate_identities.get("pack_sha256") != candidate_pack_sha256
    ):
        raise ContextualValuationError("candidate score pack binding mismatch")
    if len(anchor_rows) != len(candidate_rows):
        raise ContextualValuationError("paired score window counts differ")

    window_deltas: list[dict[str, Any]] = []
    for anchor_row, candidate_row in zip(anchor_rows, candidate_rows, strict=True):
        if (
            _window_key(anchor_row["window_id"])
            != _window_key(candidate_row["window_id"])
            or anchor_row["positions"] != candidate_row["positions"]
        ):
            raise ContextualValuationError("paired score window identity/shape differs")
        window_deltas.append(
            {
                "window_id": anchor_row["window_id"],
                "positions": anchor_row["positions"],
                "delta_mean_kld": (
                    candidate_row["mean_kld"] - anchor_row["mean_kld"]
                ),
                "delta_top1_matches": (
                    candidate_row["top1_matches"]
                    - anchor_row["top1_matches"]
                ),
            }
        )
    total_weight = sum(row["positions"] for row in window_deltas)
    delta_mean_kld = math.fsum(
        row["delta_mean_kld"] * row["positions"] for row in window_deltas
    ) / total_weight
    sum_weight_squared = sum(row["positions"] ** 2 for row in window_deltas)
    variance_denominator = total_weight - (sum_weight_squared / total_weight)
    variance = math.fsum(
        row["positions"] * (row["delta_mean_kld"] - delta_mean_kld) ** 2
        for row in window_deltas
    ) / variance_denominator
    effective_windows = total_weight**2 / sum_weight_squared
    stderr_mean_kld = math.sqrt(variance / effective_windows)

    receipt: dict[str, Any] = {
        "schema": "banana-smasher-contextual-swap-measurement-v1",
        "status": "PASS",
        "anchor_assignment_sha256": anchor.get("assignment_sha256"),
        "candidate_assignment_sha256": change.get("candidate_assignment_sha256"),
        "anchor_score_sha256": anchor_score_binding["sha256"],
        "candidate_score_sha256": candidate_score_binding["sha256"],
        "candidate_pack_sha256": candidate_pack_sha256,
        "scope": scope,
        "windows": len(window_deltas),
        "positions": total_weight,
        "support_width": anchor_score.get("support_width"),
        "change": {"cell": cell, "physical_identity": physical_identity},
        "delta_mean_kld": delta_mean_kld,
        "delta_top1_matches": sum(
            row["delta_top1_matches"] for row in window_deltas
        ),
        "stderr_mean_kld": stderr_mean_kld,
        "per_window": window_deltas,
        "input_bindings": [
            anchor_binding,
            change_binding,
            anchor_score_binding,
            candidate_score_binding,
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    if (
        manifest.get("schema")
        != "banana-smasher-contextual-measurement-manifest-v1"
        or manifest.get("status") != "READY"
        or manifest.get("anchor_assignment_sha256")
        != anchor.get("assignment_sha256")
        or not isinstance(manifest.get("measurements"), list)
    ):
        raise ContextualValuationError("measurement manifest must be anchor-bound READY v1")
    key = (cell, physical_identity)
    existing = [
        row
        for row in manifest["measurements"]
        if isinstance(row, Mapping)
        and (row.get("change", {}).get("cell"), row.get("change", {}).get("physical_identity"))
        == key
    ]
    if existing and existing != [receipt]:
        raise ContextualValuationError("conflicting contextual measurement already exists")

    output = Path(output_path).expanduser().resolve()
    manifest_path = Path(measurement_manifest_path).expanduser().resolve()
    _atomic_json(output, receipt)
    if not existing:
        manifest["measurements"].append(receipt)
        _atomic_json(manifest_path, manifest)
    payload = output.read_bytes()
    return {
        "status": "PASS",
        "output": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": receipt["receipt_sha256"],
        "delta_mean_kld": delta_mean_kld,
        "delta_top1_matches": receipt["delta_top1_matches"],
        "stderr_mean_kld": stderr_mean_kld,
    }
