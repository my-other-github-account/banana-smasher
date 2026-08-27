from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .backpack_dimensions import DynamicDimensionsError

BASIS_RE = re.compile(r"[0-9a-f]{64}")
EXPERT_RE = re.compile(r"L(\d{3})\.E(\d{3})")
Q3_CELL_RE = re.compile(r"L(\d{3})\.E(\d{3})\.(down|fused13)")
Q2_PROJECTION_RE = re.compile(r"(?:^|[_/.])(w[123])(?:[._]|$)")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or BASIS_RE.fullmatch(value) is None:
        raise DynamicDimensionsError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DynamicDimensionsError(f"{label} must be an object")
    return value, raw


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(f"cannot read sensitivity ledger: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise DynamicDimensionsError("sensitivity ledger rows must be objects")
    return rows, raw


def _publish(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return raw


def prepare_mixed_backpack_transaction(
    solve_root: str | Path,
    sensitivity_ledger: str | Path,
    *,
    output: str | Path,
    destination_root: str | Path,
    canonical_commit_sha: str,
) -> dict[str, Any]:
    """Seal a claim-free physical-member transaction for one mixed solve."""

    if (
        not isinstance(canonical_commit_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", canonical_commit_sha) is None
    ):
        raise DynamicDimensionsError("canonical_commit_sha must be a lowercase Git SHA")
    commit = canonical_commit_sha
    root = Path(solve_root).expanduser().resolve()
    ledger_path = Path(sensitivity_ledger).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    destination = Path(destination_root)
    if not destination.is_absolute():
        raise DynamicDimensionsError("destination_root must be absolute")

    identity, identity_raw = _json(root / "identity.json", "mixed identity")
    assignment, assignment_raw = _json(root / "ASSIGNMENT.json", "mixed assignment")
    receipt, receipt_raw = _json(root / "RECEIPT.json", "mixed solve receipt")
    selected, selected_raw = _json(
        root / "SELECTED_PHYSICAL_MEMBERS.json", "selected physical roster"
    )
    ledger, ledger_raw = _jsonl(ledger_path)
    expected_schemas = (
        (identity, "banana-smasher-mixed-backpack-identity-v1"),
        (assignment, "banana-smasher-mixed-backpack-assignment-v1"),
        (receipt, "banana-smasher-mixed-backpack-solve-receipt-v1"),
        (selected, "banana-smasher-mixed-backpack-selected-members-v1"),
    )
    if any(value.get("schema") != schema for value, schema in expected_schemas):
        raise DynamicDimensionsError("mixed transaction inputs have invalid schemas")
    basis = _digest(identity.get("basis_sha256"), "basis_sha256")
    assignment_sha = _digest(identity.get("assignment_sha256"), "assignment_sha256")
    if any(value.get("basis_sha256") != basis for value in (assignment, receipt, selected)):
        raise DynamicDimensionsError("mixed transaction basis mismatch")
    if assignment.get("assignment_sha256") != assignment_sha or selected.get("assignment_sha256") != assignment_sha:
        raise DynamicDimensionsError("mixed transaction assignment mismatch")
    selected_descriptor = identity.get("selected_physical_members")
    if not isinstance(selected_descriptor, dict) or selected_descriptor.get("sha256") != _sha(selected_raw):
        raise DynamicDimensionsError("selected physical roster descriptor mismatch")

    expert_assignment: dict[tuple[int, int], str] = {}
    for key, tier in identity.get("assignment", {}).items():
        match = EXPERT_RE.fullmatch(str(key))
        if match is None or tier not in {"qtip2", "qtip3"}:
            raise DynamicDimensionsError(f"invalid expert assignment {key!r}")
        expert_assignment[(int(match.group(1)), int(match.group(2)))] = tier

    sensitivity: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index, row in enumerate(ledger):
        raw_tier = row.get("tier")
        tier = {"Q2": "qtip2", "QTIP3_V7": "qtip3"}.get(
            raw_tier if isinstance(raw_tier, str) else ""
        )
        if row.get("schema") != "banana-smasher-sensitivity-row-v1" or tier is None:
            raise DynamicDimensionsError(f"sensitivity row {index} is invalid")
        key = (int(row["layer"]), int(row["expert"]), tier)
        if key in sensitivity:
            raise DynamicDimensionsError(f"duplicate sensitivity row {key!r}")
        sensitivity[key] = row

    members: list[dict[str, Any]] = []
    payload_bytes = 0
    q2_members = 0
    for (layer, expert), tier in sorted(expert_assignment.items()):
        row = sensitivity.get((layer, expert, tier))
        if row is None:
            raise DynamicDimensionsError(f"missing sensitivity row L{layer:03d}.E{expert:03d}.{tier}")
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DynamicDimensionsError("selected sensitivity row has invalid bytes")
        payload_bytes += size
        if tier != "qtip2":
            continue
        sources = row.get("source_receipts")
        if not isinstance(sources, list) or len(sources) != 3 or size % 3:
            raise DynamicDimensionsError(f"incomplete Q2 roster L{layer:03d}.E{expert:03d}")
        projections: dict[str, dict[str, Any]] = {}
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise DynamicDimensionsError("Q2 source receipt must contain path and sha256")
            path = source["path"]
            match = Q2_PROJECTION_RE.search(path) if isinstance(path, str) else None
            if match is None or match.group(1) in projections:
                raise DynamicDimensionsError(f"cannot identify unique Q2 projection in {path!r}")
            projection = match.group(1)
            projections[projection] = {
                "cell_id": f"L{layer:03d}.E{expert:03d}.{projection}",
                "tier": "qtip2",
                "locator": path,
                "sha256": _digest(source["sha256"], "Q2 source sha256"),
                "bytes": size // 3,
            }
        if set(projections) != {"w1", "w2", "w3"}:
            raise DynamicDimensionsError("Q2 projection set is incomplete")
        members.extend(projections[name] for name in ("w1", "w2", "w3"))
        q2_members += 3

    selected_rows = selected.get("members")
    if not isinstance(selected_rows, list) or selected.get("members_expected") != len(selected_rows):
        raise DynamicDimensionsError("selected Q3 roster is incomplete")
    q3_totals: dict[tuple[int, int], int] = {}
    q3_seen: set[str] = set()
    for index, member in enumerate(selected_rows):
        artifact = member.get("artifact") if isinstance(member, dict) else None
        raw_cell_id = member.get("cell_id") if isinstance(member, dict) else None
        cell_id = raw_cell_id if isinstance(raw_cell_id, str) else ""
        match = Q3_CELL_RE.fullmatch(cell_id)
        if match is None or member.get("tier") != "qtip3" or cell_id in q3_seen or not isinstance(artifact, dict):
            raise DynamicDimensionsError(f"selected Q3 member {index} is invalid")
        q3_seen.add(cell_id)
        key = (int(match.group(1)), int(match.group(2)))
        if expert_assignment.get(key) != "qtip3":
            raise DynamicDimensionsError(f"unselected Q3 member {cell_id}")
        size = artifact.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DynamicDimensionsError(f"selected Q3 member {index} has invalid bytes")
        q3_totals[key] = q3_totals.get(key, 0) + size
        members.append({
            "cell_id": cell_id,
            "tier": "qtip3",
            "host": artifact.get("host"),
            "locator": artifact.get("path"),
            "sha256": _digest(artifact.get("sha256"), "Q3 source sha256"),
            "bytes": size,
        })
    q3_logical_bytes = 0
    q3_selected_code_bytes = 0
    for key, tier in expert_assignment.items():
        if tier == "qtip3":
            logical = sensitivity[(key[0], key[1], tier)]["bytes"]
            selected_codes = q3_totals.get(key, 0)
            if selected_codes <= 0 or selected_codes > logical:
                raise DynamicDimensionsError(f"Q3 physical code bytes disagree for {key!r}")
            q3_logical_bytes += logical
            q3_selected_code_bytes += selected_codes

    accounting = receipt.get("byte_accounting")
    if not isinstance(accounting, dict) or accounting.get("candidate_payload_bytes") != payload_bytes:
        raise DynamicDimensionsError("payload bytes disagree with solve receipt")
    if accounting.get("fixed_nonexpert_bytes", 0) + payload_bytes != accounting.get("whole_model_bytes"):
        raise DynamicDimensionsError("whole-model byte accounting is inconsistent")

    members.sort(key=lambda row: (row["cell_id"], row["tier"]))
    manifest = {
        "schema": "banana-smasher-mixed-backpack-transaction-v1",
        "status": "READY_TO_CAS",
        "basis_sha256": basis,
        "canonical_commit_sha": commit,
        "assignment_sha256": assignment_sha,
        "destination_root": str(destination),
        "inputs": {
            "identity": {"path": str(root / "identity.json"), "sha256": _sha(identity_raw), "bytes": len(identity_raw)},
            "assignment": {"path": str(root / "ASSIGNMENT.json"), "sha256": _sha(assignment_raw), "bytes": len(assignment_raw)},
            "solve_receipt": {"path": str(root / "RECEIPT.json"), "sha256": _sha(receipt_raw), "bytes": len(receipt_raw)},
            "selected_physical_members": {"path": str(root / "SELECTED_PHYSICAL_MEMBERS.json"), "sha256": _sha(selected_raw), "bytes": len(selected_raw)},
            "sensitivity_ledger": {"path": str(ledger_path), "sha256": _sha(ledger_raw), "bytes": len(ledger_raw)},
        },
        "byte_accounting": {
            **accounting,
            "qtip3_logical_bytes": q3_logical_bytes,
            "qtip3_selected_code_bytes": q3_selected_code_bytes,
            "qtip3_shared_and_transform_bytes": q3_logical_bytes - q3_selected_code_bytes,
        },
        "member_counts": {
            "experts": len(expert_assignment),
            "qtip2_wire_members": q2_members,
            "qtip3_projection_members": len(selected_rows),
            "total": len(members),
        },
        "members": members,
        "launch": {
            "requires_host_claim": True,
            "requires_empty_rank0": True,
            "argv": [
                "smash", "backpack", "prepare-mixed-transaction",
                "--solve-root", str(root),
                "--sensitivity-ledger", str(ledger_path),
                "--output", str(output_path),
                "--destination-root", str(destination),
                "--canonical-commit-sha", commit,
            ],
            "then": "canonical loader/contract materialization, checkpoint+SHA admission, W28, full64",
        },
    }
    manifest_raw = _publish(output_path, manifest)
    return {
        "schema": "banana-smasher-mixed-backpack-transaction-receipt-v1",
        "status": "READY_TO_CAS",
        "basis_sha256": basis,
        "assignment_sha256": assignment_sha,
        "manifest": {"path": str(output_path), "sha256": _sha(manifest_raw), "bytes": len(manifest_raw)},
        "byte_accounting": accounting,
        "member_counts": manifest["member_counts"],
        "launch_argv": manifest["launch"]["argv"],
    }
