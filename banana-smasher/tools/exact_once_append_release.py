#!/usr/bin/env python3
"""Locked exact-once JSONL append and exact host release primitives."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


APPEND_SCHEMA = "banana-smasher-exact-once-append-v3"
RELEASE_SCHEMA = "banana-smasher-exact-release-v3"


class ExactOnceError(RuntimeError):
    """Raised when a precondition fails before a canonical mutation."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _atomic_json(path: Path, value: object, *, preserve_from: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = preserve_from.stat() if preserve_from is not None and preserve_from.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if metadata is not None:
            os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
            os.fchown(descriptor, metadata.st_uid, metadata.st_gid)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_jsonl(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExactOnceError(f"ROW_NOT_OBJECT:{line_number}")
        rows.append(value)
    return rows


def _unique_ids(rows: Sequence[dict[str, Any]], id_key: str, label: str) -> list[str]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(id_key)
        if not isinstance(value, str) or not value:
            raise ExactOnceError(f"{label}_ID_INVALID:{index}")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ExactOnceError(f"{label}_IDS_NOT_UNIQUE")
    return ids


def exact_once_append(
    *,
    ledger_path: Path,
    rows: Sequence[dict[str, Any]],
    receipt_path: Path,
    expected_pre_sha256: str,
    expected_pre_count: int,
    id_key: str,
    transaction_id: str,
) -> dict[str, Any]:
    """Append unique rows exactly once while holding an exclusive ledger lock."""
    if not rows:
        raise ExactOnceError("NO_ROWS")
    added_ids = _unique_ids(rows, id_key, "ADDED")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("r+b") as ledger:
        fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
        ledger.seek(0)
        pre = ledger.read()
        pre_sha = _sha(pre)
        existing_rows = _parse_jsonl(pre)
        existing_ids = _unique_ids(existing_rows, id_key, "LEDGER")
        if pre_sha != expected_pre_sha256:
            raise ExactOnceError(f"PREIMAGE_SHA_MISMATCH:{pre_sha}")
        if len(existing_rows) != expected_pre_count:
            raise ExactOnceError(f"PREIMAGE_COUNT_MISMATCH:{len(existing_rows)}")
        overlap = sorted(set(existing_ids).intersection(added_ids))
        if overlap:
            raise ExactOnceError("OVERLAP:" + ",".join(overlap))
        append_raw = b"".join(_json_bytes(row) for row in rows)
        ledger.seek(0, os.SEEK_END)
        ledger.write(append_raw)
        ledger.flush()
        os.fsync(ledger.fileno())
        ledger.seek(0)
        post = ledger.read()
        post_rows = _parse_jsonl(post)
        if len(post_rows) != len(existing_rows) + len(rows):
            raise ExactOnceError("POST_COUNT_MISMATCH")
        if post[: len(pre)] != pre:
            raise ExactOnceError("PRIOR_BYTES_CHANGED")
        if post[len(pre) :] != append_raw:
            raise ExactOnceError("APPEND_BYTES_MISMATCH")
        result: dict[str, Any] = {
            "schema": APPEND_SCHEMA,
            "status": "PASS",
            "transaction_id": transaction_id,
            "ledger_path": str(ledger_path),
            "lock": {"mechanism": "fcntl.flock.LOCK_EX", "acquired": True},
            "id_key": id_key,
            "pre_count": len(existing_rows),
            "post_count": len(post_rows),
            "added_count": len(rows),
            "added_ids": added_ids,
            "pre_bytes": len(pre),
            "post_bytes": len(post),
            "pre_sha256": pre_sha,
            "post_sha256": _sha(post),
            "prior_bytes_preserved": True,
            "exact_once_overlap_count": 0,
            "created_unix": time.time(),
        }
        _atomic_json(receipt_path, result)
        fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)
    return result


def _pid_matches_startticks(pid: int, startticks: int | None) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    if startticks is None:
        return True
    try:
        observed = int(stat_path.read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return True
    return observed == startticks


def exact_release(
    *,
    claim_path: Path,
    receipt_path: Path,
    expected_pre_sha256: str,
    expected_task_id: str,
    transaction_id: str,
) -> dict[str, Any]:
    """CAS-release a claimed host only after its bound workload is dead."""
    lock_path = claim_path.with_name(f".{claim_path.name}.lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pre = claim_path.read_bytes()
        pre_sha = _sha(pre)
        if pre_sha != expected_pre_sha256:
            raise ExactOnceError(f"CLAIM_PREIMAGE_SHA_MISMATCH:{pre_sha}")
        claim = json.loads(pre)
        if claim.get("task_id") != expected_task_id:
            raise ExactOnceError(f"CLAIM_TASK_MISMATCH:{claim.get('task_id')}")
        if claim.get("state") != "CLAIMED":
            raise ExactOnceError(f"CLAIM_STATE_NOT_CLAIMED:{claim.get('state')}")
        pid = claim.get("workload_pid")
        startticks = claim.get("workload_startticks")
        if isinstance(pid, int) and _pid_matches_startticks(pid, startticks):
            raise ExactOnceError(f"WORKLOAD_ALIVE:{pid}")
        released = dict(claim)
        released.update(
            {
                "state": "RELEASED",
                "status": "RELEASED",
                "workload_pid": None,
                "workload_startticks": None,
                "holder_pid": None,
                "holder_startticks": None,
                "do_not_preempt": False,
                "previous_claim_sha256": pre_sha,
                "release_transaction_id": transaction_id,
                "released_unix": time.time(),
                "updated_unix": time.time(),
            }
        )
        _atomic_json(claim_path, released, preserve_from=claim_path)
        post = claim_path.read_bytes()
        result: dict[str, Any] = {
            "schema": RELEASE_SCHEMA,
            "status": "PASS",
            "transaction_id": transaction_id,
            "release_action": "CLAIMED_TO_RELEASED",
            "claim_path": str(claim_path),
            "claim_task_id": expected_task_id,
            "claim_pre_sha256": pre_sha,
            "claim_post_sha256": _sha(post),
            "workload_pid": pid,
            "workload_startticks": startticks,
            "workload_dead": True,
            "lock": {"path": str(lock_path), "mechanism": "fcntl.flock.LOCK_EX", "acquired": True},
            "created_unix": time.time(),
        }
        _atomic_json(receipt_path, result)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return result


def verify_released(*, claim_path: Path, receipt_path: Path, transaction_id: str) -> dict[str, Any]:
    """Emit an immutable receipt proving a claim was already released."""
    raw = claim_path.read_bytes()
    claim = json.loads(raw)
    if claim.get("state") != "RELEASED" or claim.get("status") != "RELEASED":
        raise ExactOnceError(f"CLAIM_NOT_RELEASED:{claim.get('state')}:{claim.get('status')}")
    pid = claim.get("workload_pid")
    if isinstance(pid, int) and _pid_matches_startticks(pid, claim.get("workload_startticks")):
        raise ExactOnceError(f"WORKLOAD_ALIVE:{pid}")
    result: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "status": "PASS",
        "transaction_id": transaction_id,
        "release_action": "ALREADY_RELEASED_VERIFIED",
        "claim_path": str(claim_path),
        "claim_sha256": _sha(raw),
        "claim_task_id": claim.get("task_id"),
        "workload_pid": pid,
        "workload_dead": True,
        "created_unix": time.time(),
    }
    _atomic_json(receipt_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--ledger", type=Path, required=True)
    merge.add_argument("--rows-jsonl", type=Path, required=True)
    merge.add_argument("--receipt", type=Path, required=True)
    merge.add_argument("--expected-pre-sha256", required=True)
    merge.add_argument("--expected-pre-count", type=int, required=True)
    merge.add_argument("--id-key", default="probe_id")
    merge.add_argument("--transaction-id", required=True)
    release = subparsers.add_parser("release")
    release.add_argument("--claim", type=Path, required=True)
    release.add_argument("--receipt", type=Path, required=True)
    release.add_argument("--expected-pre-sha256", required=True)
    release.add_argument("--expected-task-id", required=True)
    release.add_argument("--transaction-id", required=True)
    verify = subparsers.add_parser("verify-released")
    verify.add_argument("--claim", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--transaction-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.operation == "merge":
        result = exact_once_append(
            ledger_path=args.ledger,
            rows=_parse_jsonl(args.rows_jsonl.read_bytes()),
            receipt_path=args.receipt,
            expected_pre_sha256=args.expected_pre_sha256,
            expected_pre_count=args.expected_pre_count,
            id_key=args.id_key,
            transaction_id=args.transaction_id,
        )
    elif args.operation == "release":
        result = exact_release(
            claim_path=args.claim,
            receipt_path=args.receipt,
            expected_pre_sha256=args.expected_pre_sha256,
            expected_task_id=args.expected_task_id,
            transaction_id=args.transaction_id,
        )
    else:
        result = verify_released(
            claim_path=args.claim,
            receipt_path=args.receipt,
            transaction_id=args.transaction_id,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
