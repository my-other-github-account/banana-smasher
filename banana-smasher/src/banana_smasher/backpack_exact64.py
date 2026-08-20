from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .anchor import _score_bank, validate_bank_manifest
from .backpack_contextual import ContextualValuationError, _atomic_json
from .backpack_virtual import _canonical, verify_virtual_backpack


EXACT64_TERMINAL_SCHEMA = "banana-smasher-backpack-exact64-terminal-v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_backpack_exact64(
    virtual_root: str | Path,
    bank_manifest: Mapping[str, Any] | str | Path,
    teacher_producer: str | Path,
    candidate_producer: str | Path,
    *,
    raw_output_path: str | Path,
    output_path: str | Path,
    candidate_id: str,
    teacher_identity: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit the exact64 terminal schema from canonical Anchor producer scoring."""

    virtual = verify_virtual_backpack(virtual_root)
    if isinstance(bank_manifest, Mapping):
        manifest = dict(bank_manifest)
    else:
        manifest = json.loads(Path(bank_manifest).expanduser().read_text())
    validate_bank_manifest(manifest)
    raw_path = Path(raw_output_path).expanduser()
    anchor_receipt = _score_bank(
        manifest,
        teacher_producer,
        candidate_producer,
        raw_path,
        candidate_id=candidate_id,
        candidate_identity=candidate_identity,
        teacher_identity=teacher_identity,
        basis_sha256=str(virtual["basis_sha256"]),
    )
    rows = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != 64:
        raise ContextualValuationError(
            "canonical Anchor score must contain exactly 64 windows"
        )
    per_window = [
        {
            "window_id": row["window_id"],
            "positions": row["position_count"],
            "mean_kld": row["kld"],
            "top1_matches": row["top1_matches"],
        }
        for row in rows
    ]
    positions = sum(int(row["positions"]) for row in per_window)
    mean_kld = (
        math.fsum(float(row["mean_kld"]) * int(row["positions"]) for row in per_window)
        / positions
    )
    score = {
        "schema": "banana-smasher-anchor-sidecar-score-v1",
        "status": "PASS",
        "claimable": True,
        "candidate_id": candidate_id,
        "coverage": "64/64",
        "windows": 64,
        "support_width": 8192,
        "positions": positions,
        "mean_kld": mean_kld,
        "top1_matches": sum(int(row["top1_matches"]) for row in per_window),
        "identities": {
            "basis_sha256": virtual["basis_sha256"],
            "bank_sha256": manifest["content_hashes"]["manifest_payload_sha256"],
            "pack_manifest_sha256": virtual["manifest"]["sha256"],
            "teacher_sha256": anchor_receipt["bindings"]["teacher_sha256"],
            "candidate_sha256": anchor_receipt["bindings"]["candidate_sha256"],
            "scorer_sha256": anchor_receipt["bindings"]["scorer_sha256"],
            "raw_score_sha256": anchor_receipt["raw_sha256"],
        },
        "per_window": per_window,
    }
    destination = Path(output_path).expanduser()
    if destination.is_symlink():
        raise ContextualValuationError("exact64 score output must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical(score))
    return score


def bind_backpack_exact64(
    virtual_manifest_path: str | Path,
    score_receipt_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind canonical Anchor64 sidecar output to one virtual Backpack artifact.

    Transformer execution remains in the existing Anchor producer/scorer API. This
    adapter only validates the full 64-window/8192-support score and emits the
    terminal receipt consumed by contextual preparation.
    """

    virtual_path = Path(virtual_manifest_path).expanduser().resolve()
    score_path = Path(score_receipt_path).expanduser().resolve()
    if virtual_path.name != "BACKPACK_VIRTUAL_MANIFEST.json":
        raise ContextualValuationError("virtual manifest filename is not canonical")
    virtual = json.loads(virtual_path.read_text())
    if not isinstance(virtual, dict):
        raise ContextualValuationError("virtual manifest must be a JSON object")
    verified = verify_virtual_backpack(virtual_path.parent)

    score = json.loads(score_path.read_text())
    if not isinstance(score, dict):
        raise ContextualValuationError("exact64 score must be a JSON object")
    if (
        score.get("schema") != "banana-smasher-anchor-sidecar-score-v1"
        or score.get("status") != "PASS"
        or score.get("claimable") is not True
    ):
        raise ContextualValuationError(
            "exact64 score must be claimable Anchor sidecar PASS v1"
        )
    if score.get("windows") != 64 or score.get("support_width") != 8192:
        raise ContextualValuationError(
            "exact64 score requires 64 windows at support width 8192"
        )
    per_window = score.get("per_window")
    if not isinstance(per_window, list) or len(per_window) != 64:
        raise ContextualValuationError("exact64 score must bind 64 per-window rows")
    window_ids: list[object] = []
    weighted_kld: list[float] = []
    per_window_top1 = 0
    for row in per_window:
        if not isinstance(row, Mapping):
            raise ContextualValuationError("exact64 per-window rows must be objects")
        window_ids.append(row.get("window_id"))
        row_positions = row.get("positions")
        row_kld = row.get("mean_kld")
        row_top1 = row.get("top1_matches")
        if row_positions != 1024:
            raise ContextualValuationError("exact64 requires 1024 positions per window")
        if (
            not isinstance(row_kld, (int, float))
            or isinstance(row_kld, bool)
            or not math.isfinite(row_kld)
            or row_kld < 0
        ):
            raise ContextualValuationError("exact64 per-window KLD is invalid")
        if (
            not isinstance(row_top1, int)
            or isinstance(row_top1, bool)
            or not 0 <= row_top1 <= row_positions
        ):
            raise ContextualValuationError("exact64 per-window Top-1 count is invalid")
        weighted_kld.append(float(row_kld) * row_positions)
        per_window_top1 += row_top1
    window_keys = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in window_ids
    ]
    if len(set(window_keys)) != 64:
        raise ContextualValuationError("exact64 score window identities must be unique")
    positions = score.get("positions")
    top1_matches = score.get("top1_matches")
    mean_kld = score.get("mean_kld")
    if positions != 65536:
        raise ContextualValuationError("exact64 score requires exactly 65536 positions")
    if (
        not isinstance(top1_matches, int)
        or isinstance(top1_matches, bool)
        or top1_matches < 0
        or top1_matches > positions
    ):
        raise ContextualValuationError("exact64 score top1 matches are invalid")
    if (
        not isinstance(mean_kld, (int, float))
        or isinstance(mean_kld, bool)
        or not math.isfinite(mean_kld)
        or mean_kld < 0
    ):
        raise ContextualValuationError("exact64 score mean KLD is invalid")
    if sum(row.get("positions", -1) for row in per_window) != positions:
        raise ContextualValuationError(
            "exact64 per-window positions do not match total"
        )
    if per_window_top1 != top1_matches:
        raise ContextualValuationError(
            "exact64 per-window Top-1 matches do not match total"
        )
    if not math.isclose(
        math.fsum(weighted_kld) / positions,
        float(mean_kld),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualValuationError(
            "exact64 per-window KLD does not match aggregate"
        )
    identities = score.get("identities")
    if not isinstance(identities, Mapping):
        raise ContextualValuationError("exact64 score identities are missing")
    if identities.get("basis_sha256") != virtual.get("basis_sha256"):
        raise ContextualValuationError(
            "exact64 score basis does not match virtual Backpack"
        )
    if identities.get("pack_sha256") != verified.get("artifact_sha256"):
        raise ContextualValuationError(
            "exact64 score pack does not match virtual Backpack"
        )

    terminal = {
        "schema": EXACT64_TERMINAL_SCHEMA,
        "status": "PASS",
        "basis_sha256": virtual["basis_sha256"],
        "assignment_map_sha256": verified["assignment_map_sha256"],
        "pack_sha256": verified["artifact_sha256"],
        "score_receipt_sha256": _sha256_file(score_path),
        "score_receipt_bytes": score_path.stat().st_size,
        "score_receipt_path": str(score_path),
        "windows": 64,
        "positions": positions,
        "support_width": 8192,
        "mean_kld": float(mean_kld),
        "top1_matches": top1_matches,
        "top1_agreement": top1_matches / positions,
    }
    destination = Path(output_path).expanduser().resolve()
    _atomic_json(destination, terminal)
    return {
        **terminal,
        "receipt": str(destination),
        "receipt_sha256": _sha256_file(destination),
    }
