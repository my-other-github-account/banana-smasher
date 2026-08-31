import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "exact_once_append_release.py"
SPEC = importlib.util.spec_from_file_location("exact_once_append_release", TOOL)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_exact_once_append_preserves_prior_rows_and_rejects_replay(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    prior = [{"probe_id": "old-0", "status": "PASS"}, {"probe_id": "old-1", "status": "PASS"}]
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prior))
    pre_sha = _sha(ledger)
    receipt = tmp_path / "merge.json"
    added = [{"probe_id": "new-0", "status": "PASS"}, {"probe_id": "new-1", "status": "PASS"}]

    result = MODULE.exact_once_append(
        ledger_path=ledger,
        rows=added,
        receipt_path=receipt,
        expected_pre_sha256=pre_sha,
        expected_pre_count=2,
        id_key="probe_id",
        transaction_id="txn-1",
    )

    assert _rows(ledger) == prior + added
    assert result["schema"] == "banana-smasher-exact-once-append-v3"
    assert result["pre_count"] == 2
    assert result["post_count"] == 4
    assert result["added_count"] == 2
    assert result["added_ids"] == ["new-0", "new-1"]
    assert result["pre_sha256"] == pre_sha
    assert result["post_sha256"] == _sha(ledger)
    assert json.loads(receipt.read_text()) == result

    post = ledger.read_bytes()
    with pytest.raises(MODULE.ExactOnceError, match="OVERLAP"):
        MODULE.exact_once_append(
            ledger_path=ledger,
            rows=added,
            receipt_path=tmp_path / "replay.json",
            expected_pre_sha256=_sha(ledger),
            expected_pre_count=4,
            id_key="probe_id",
            transaction_id="txn-2",
        )
    assert ledger.read_bytes() == post


def test_exact_once_append_fails_closed_on_preimage_mismatch(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"probe_id":"old"}\n')
    with pytest.raises(MODULE.ExactOnceError, match="PREIMAGE_SHA"):
        MODULE.exact_once_append(
            ledger_path=ledger,
            rows=[{"probe_id": "new"}],
            receipt_path=tmp_path / "receipt.json",
            expected_pre_sha256="0" * 64,
            expected_pre_count=1,
            id_key="probe_id",
            transaction_id="txn",
        )


def test_exact_release_is_cas_bound_and_preserves_file_metadata(tmp_path: Path) -> None:
    claim = tmp_path / "HOST_CLAIM.json"
    claim.write_text(json.dumps({
        "schema": "claim-v1",
        "state": "CLAIMED",
        "status": "CLAIMED",
        "task_id": "task-a",
        "workload_pid": 99999999,
        "workload_startticks": 123,
    }) + "\n")
    os.chmod(claim, 0o640)
    pre_sha = _sha(claim)
    receipt = tmp_path / "release.json"

    result = MODULE.exact_release(
        claim_path=claim,
        receipt_path=receipt,
        expected_pre_sha256=pre_sha,
        expected_task_id="task-a",
        transaction_id="release-1",
    )

    released = json.loads(claim.read_text())
    assert released["state"] == "RELEASED"
    assert released["status"] == "RELEASED"
    assert released["workload_pid"] is None
    assert released["workload_startticks"] is None
    assert released["previous_claim_sha256"] == pre_sha
    assert result["schema"] == "banana-smasher-exact-release-v3"
    assert result["claim_pre_sha256"] == pre_sha
    assert result["claim_post_sha256"] == _sha(claim)
    assert json.loads(receipt.read_text()) == result
    assert claim.stat().st_mode & 0o777 == 0o640


def test_verify_released_emits_receipt_without_mutating_claim(tmp_path: Path) -> None:
    claim = tmp_path / "HOST_CLAIM.json"
    claim.write_text(json.dumps({"state": "RELEASED", "status": "RELEASED", "workload_pid": None}) + "\n")
    pre = claim.read_bytes()
    result = MODULE.verify_released(
        claim_path=claim,
        receipt_path=tmp_path / "verified.json",
        transaction_id="verify-1",
    )
    assert claim.read_bytes() == pre
    assert result["release_action"] == "ALREADY_RELEASED_VERIFIED"
    assert result["claim_sha256"] == hashlib.sha256(pre).hexdigest()
