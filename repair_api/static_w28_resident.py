"""Bounded static-W28 acceptance through the existing resident scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .api import ResidentRepairAPI
from . import sealed_pre_forward

BASIS_SHA256 = sealed_pre_forward.BASIS_SHA256
CHECKPOINT_SHA256 = sealed_pre_forward.CHECKPOINT_SHA256
W28_KLD = sealed_pre_forward.W28_KLD
W28_TOP1 = sealed_pre_forward.W28_TOP1
W28_WINDOW = 28
RESIDENT_BUDGET_SECONDS = 300.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_truth(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SEALED_W28_REFERENCE_SHA_MISMATCH:{observed}")
    value = json.loads(path.read_text())
    checkpoint_sha = value.get("loaded_sha", value.get("checkpoint_sha256"))
    if (
        value.get("status") != "PASS"
        or value.get("basis_sha256") != BASIS_SHA256
        or checkpoint_sha != CHECKPOINT_SHA256
        or value.get("kld_mean") != W28_KLD
        or value.get("top1") != W28_TOP1
    ):
        raise RuntimeError("SEALED_W28_REFERENCE_CONTRACT_MISMATCH")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "basis_sha256": BASIS_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "kld_mean": float(value["kld_mean"]),
        "top1": int(value["top1"]),
        "window": W28_WINDOW,
    }


def _identity_field(identity: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = identity.get(name)
        if value not in (None, ""):
            return value
    return None


def run_static_w28_resident_acceptance(
    *,
    task: str,
    root: Path,
    artifact_root: Path,
    checkpoint: str,
    canonical_pin: str,
    reference_receipt: Path,
    reference_sha256: str,
    rank_seat: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call ``ResidentRepairAPI.score`` once for sealed W28 and gate its receipt."""
    root = root.resolve()
    truth = _sealed_truth(reference_receipt.resolve(), reference_sha256)
    source_binding = sealed_pre_forward.source_binding()
    api = ResidentRepairAPI.open(
        artifact_root.resolve(), official_rank_seat=rank_seat
    )
    result = api.score(checkpoint, windows=(W28_WINDOW,))
    measurement = result.as_dict()
    counters = dict(result.runtime_counters)
    identity = dict(result.identity)
    checkpoint_identity = _identity_field(
        identity, "checkpoint_sha256", "loaded_checkpoint_sha256"
    )
    basis_identity = _identity_field(
        identity, "model_index_sha256", "basis_sha256", "source_model_index_sha256"
    )
    timed_reads = counters.get(
        "timed_score_file_reads", counters.get("file_reads_during_timed_score")
    )
    resident_ready = counters.get("resident_ready")
    resident_state_persisted = (
        result.execution_mode == "resident_in_memory"
        and int(counters.get("resident_engine_loads", 0)) == 1
        and int(counters.get("resident_checkpoint_rebinds", 0)) == 0
        and timed_reads == 0
        and isinstance(resident_ready, list)
        and bool(resident_ready)
    )
    passed = (
        tuple(result.windows) == (W28_WINDOW,)
        and result.positions == 1024
        and result.support == 8192
        and result.kld == truth["kld_mean"]
        and result.top1 == truth["top1"]
        and result.timed_wall_seconds is not None
        and result.timed_wall_seconds <= RESIDENT_BUDGET_SECONDS
        and checkpoint_identity == truth["checkpoint_sha256"]
        and basis_identity == truth["basis_sha256"]
        and resident_state_persisted
    )
    receipt = {
        "schema": "banana-smasher-static-w28-resident-acceptance-v1",
        "status": "PASS" if passed else "RED",
        "task_id": task,
        "canonical_code_commit": canonical_pin,
        "basis_sha256": BASIS_SHA256,
        "checkpoint_sha256": checkpoint_identity,
        "checkpoint_identity": identity,
        "source_binding": source_binding,
        "sealed_truth": truth,
        "sealed_truth_receipt_sha256": truth["sha256"],
        "measurement": measurement,
        "resident_state_persisted": resident_state_persisted,
        "resident_budget_seconds": RESIDENT_BUDGET_SECONDS,
        "full64_launched": False,
        "public_api": "ResidentRepairAPI.score",
        "one_variable": "resident packed-state scorer replaces per-layer full-BF16 static builder expansion",
        "runtime_rank_seat": dict(rank_seat) if rank_seat is not None else None,
    }
    path = root / "receipts" / f"STATIC_W28_RESIDENT_ACCEPTANCE.{task}.json"
    receipt["receipt_sha256"] = sealed_pre_forward.atomic_json(path, receipt)
    if not passed:
        raise RuntimeError(f"STATIC_W28_RESIDENT_RED:{measurement}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["canonical_pin"]:
        raise SystemExit(f"CANONICAL_PIN_REFUSED:{head}")
    receipt = run_static_w28_resident_acceptance(
        task=config["task_id"],
        root=args.root,
        artifact_root=Path(config["artifact_root"]),
        checkpoint=config["checkpoint"],
        canonical_pin=head,
        reference_receipt=Path(config["reference_receipt"]),
        reference_sha256=config["reference_sha256"],
        rank_seat=config.get("rank_seat"),
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
