#!/usr/bin/env python3
"""Run a resumable shard of W28 full-model sensitivity probes (PROBE_MANIFEST_V2).

Differences from ``sensitivity_probe_worker.py`` (v1), all forced by the v2
manifest and by the run6989 post-mortem:

* binds PROBE_MANIFEST_V2 (254 probes) instead of v1's 200;
* supports ``null_control`` (same-tier), ``replicate``, ``downgrade_control``
  and 2-cell ``additivity_joint`` roles;
* authenticates the *target payload bytes* against the ledger producer BEFORE
  spending GPU time, instead of discovering the mismatch inside the decoder
  after the 441 s instrument gate (this is the exact run6989 failure);
* runs a shard by explicit probe-id list, so nulls and replicates can be
  measured first as the campaign's power gate;
* records role, per-cell target unit proofs, and both root maps in each receipt.

The instrument gate, identity gates, canonical JSON and atomic-append receipt
discipline are unchanged from v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from banana_smasher.backpack_runtime_exact64 import _run_backpack_exact64
from banana_smasher.sensitivity_probe_v2 import (
    authenticate_target_units,
    materialize_sensitivity_candidate_v2,
)

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
PROBE_MANIFEST_V2_SHA = "ddcdd9421215dcfffe5c2c3dad374e3bd114a1072956e294d7724620e818090b"
PROBE_MANIFEST_V2_COUNT = 254
LEDGER_SHA = "45b124e40a0f41a10e25949efdf32cc11a4271f24cf1331c6dbab6deacd813ee"
BASELINE_MANIFEST_SHA = "bbed6a44c690f89555b44ccfd4c8b0a0c5ed5dda0aca7bb3d0de2b60fd30d07a"
BASELINE_W28_KLD = 0.09936928004026413
BASELINE_RECEIPT_SHA = "85ee77d256fc45e93a35211898a7701c0ba301961b17574107933cab15904d7d"
CHECKPOINT_SHA = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
BANK_SHA = "d16580470bbba0e93729c3dfedb618a12c41633bb2c66094884c01dd9431156c"

VERIFY_LADDER = Path("/home/dnola/missions/VERIFY_LADDER_t_ff8ce60a_s8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: object) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)


def resolve_target_root_map(candidate: dict, q2_map: Path, q3_map: Path) -> tuple[Path, str, Path, Path]:
    """Pick the root map the decoder must bind for this candidate's target tier."""

    target_tier = candidate.get("target_tier")
    target_q2, target_q3 = q2_map, q3_map
    maps = candidate.get("target_root_maps") or []
    if target_tier not in {"qtip2", "qtip3"}:
        return q3_map, "", target_q2, target_q3
    if len(maps) != 1:
        raise RuntimeError(f"TARGET_ROOT_MAP_AMBIGUOUS:{candidate.get('probe_id')}:{len(maps)}")
    declared = Path(maps[0]["path"])
    declared_sha = str(maps[0]["sha256"])
    # A null control re-materializes the incumbent tier, so the sealed baseline
    # root map IS the authority and the ledger's producer path is only the
    # provenance record. For any real swap the ledger's map is the authority.
    if candidate.get("is_null_control"):
        bound = q2_map if target_tier == "qtip2" else q3_map
        bound_sha = sha(bound)
    else:
        bound, bound_sha = declared, declared_sha
    if target_tier == "qtip2":
        target_q2 = bound
    else:
        target_q3 = bound
    return bound, bound_sha, target_q2, target_q3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--probe-ids", type=Path, help="JSON list of probe_ids to run, in order")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--reproduce-baseline", action="store_true")
    parser.add_argument("--skip-instrument-if-proven", type=Path,
                        help="path to a PASS INSTRUMENT.json from this same host/pin")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if sha(args.probe_manifest) != PROBE_MANIFEST_V2_SHA:
        raise RuntimeError("PROBE_MANIFEST_V2_SHA_RED")
    if sha(args.ledger) != LEDGER_SHA:
        raise RuntimeError("LEDGER_SHA_RED")
    manifest = json.loads(args.probe_manifest.read_text())
    if manifest.get("basis_sha256") != BASIS or manifest.get("probe_count") != PROBE_MANIFEST_V2_COUNT:
        raise RuntimeError("PROBE_MANIFEST_BASIS_RED")
    probes = manifest["probes"]
    by_id = {p["probe_id"]: p for p in probes}
    if len(by_id) != len(probes):
        raise RuntimeError("PROBE_ID_COLLISION_RED")

    if args.probe_ids:
        order = json.loads(args.probe_ids.read_text())
        if not isinstance(order, list) or not order:
            raise RuntimeError("PROBE_ID_LIST_RED")
        unknown = [pid for pid in order if pid not in by_id]
        if unknown:
            raise RuntimeError(f"PROBE_ID_UNKNOWN_RED:{unknown[:5]}")
        selected = [by_id[pid] for pid in order]
        shard_label = ["ids", args.probe_ids.name, len(order)]
    else:
        if args.start is None or args.end is None or not (0 <= args.start < args.end <= len(probes)):
            raise RuntimeError("PROBE_SHARD_RANGE_RED")
        selected = probes[args.start:args.end]
        shard_label = ["range", args.start, args.end]

    baseline_manifest = args.baseline_root / "BACKPACK_VIRTUAL_MANIFEST.json"
    baseline_index = args.baseline_root / "MATERIALIZATION_INDEX.jsonl"
    baseline_terminal = VERIFY_LADDER / "EXACT102_VIRTUAL_TERMINAL.json"
    baseline_receipt = VERIFY_LADDER / "receipts/RUNG3_W28_TERMINAL.json"
    model_root = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
    checkpoint = Path(
        "/home/dnola/missions/FULL64_REPAIR_t_686041f5/relocation_run6058/rank1/home/dnola/missions/"
        "RESIDENT_VALIDATE_t_6031426c_attempt17/artifact/checkpoints/PRE.pt"
    )
    bank = Path("/home/dnola/missions/MIXED_BACKPACK_t_91ccf93f_spark-8/run6457-backpack-w28/inputs/w28.bank.jsonl")
    teacher = Path("/home/dnola/missions/MIXED_BACKPACK_t_91ccf93f_spark-8/run6457-backpack-w28/inputs/teacher_w28.v2.json")
    q2_map = VERIFY_LADDER / "QTIP2_ROOT_MAP.json"
    q3_map = VERIFY_LADDER / "QTIP3_ROOT_MAP.json"
    gates = {
        model_root / "model.safetensors.index.json": BASIS,
        baseline_manifest: BASELINE_MANIFEST_SHA,
        baseline_receipt: BASELINE_RECEIPT_SHA,
        checkpoint: CHECKPOINT_SHA,
        bank: BANK_SHA,
    }
    for path, expected in gates.items():
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"IDENTITY_GATE_RED:{path}")
    if json.loads(baseline_receipt.read_text()).get("mean_kld") != BASELINE_W28_KLD:
        raise RuntimeError("BASELINE_W28_VALUE_RED")
    available = int(Path("/proc/meminfo").read_text().split("MemAvailable:", 1)[1].split()[0]) * 1024
    if available < 24_000_000_000:
        raise RuntimeError(f"MEMORY_PREFLIGHT_RED:{available}")

    instrument_path = root / "INSTRUMENT.json"
    if args.skip_instrument_if_proven and not instrument_path.exists():
        proven = json.loads(args.skip_instrument_if_proven.read_text())
        if (
            proven.get("status") != "PASS"
            or proven.get("observed_mean_kld") != BASELINE_W28_KLD
            or proven.get("absolute_delta") != 0.0
            or proven.get("basis_sha256") != BASIS
        ):
            raise RuntimeError("IMPORTED_INSTRUMENT_RED")
        imported = dict(proven)
        imported["imported_from"] = str(args.skip_instrument_if_proven)
        imported["imported_sha256"] = sha(args.skip_instrument_if_proven)
        imported["imported_unix"] = time.time()
        imported["import_justification"] = (
            "same host, same basis, same checkpoint/bank/model identity gates re-proven above; "
            "the 441 s baseline forward is byte-deterministic and was already PASS at delta 0.0"
        )
        atomic(instrument_path, imported)
    elif args.reproduce_baseline and not instrument_path.exists():
        instrument_root = root / "instrument"
        instrument_started = time.time()
        instrument = _run_backpack_exact64(
            model_root=model_root,
            bank_path=bank,
            teacher_manifest_path=teacher,
            virtual_manifest_path=baseline_manifest,
            virtual_manifest_sha256=BASELINE_MANIFEST_SHA,
            virtual_terminal_path=baseline_terminal,
            virtual_terminal_sha256=sha(baseline_terminal),
            materialization_index_path=baseline_index,
            qtip2_root_map_path=q2_map,
            qtip3_root_map_path=q3_map,
            checkpoint_path=checkpoint,
            checkpoint_sha256=CHECKPOINT_SHA,
            output_root=instrument_root,
            basis_sha256=BASIS,
            expected_windows=1,
            slice_id="sensitivity-instrument-baseline",
        )
        observed = float(instrument["mean_kld"])
        if not math.isclose(observed, BASELINE_W28_KLD, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"INSTRUMENT_BASELINE_RED:{observed}:{BASELINE_W28_KLD}")
        atomic(instrument_path, {
            "schema": "banana-smasher-sensitivity-instrument-gate-v1",
            "status": "PASS",
            "task_id": os.environ.get("BANANA_SMASHER_TASK", "t_605f25e1"),
            "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
            "canonical_git_pin": os.environ.get("BANANA_SMASHER_PIN"),
            "basis_sha256": BASIS,
            "known_mean_kld": BASELINE_W28_KLD,
            "observed_mean_kld": observed,
            "absolute_delta": abs(observed - BASELINE_W28_KLD),
            "runtime_terminal_path": str(instrument_root / "receipts/TERMINAL.json"),
            "runtime_terminal_sha256": sha(instrument_root / "receipts/TERMINAL.json"),
            "elapsed_seconds": time.time() - instrument_started,
            "created_unix": time.time(),
        })
        shutil.rmtree(instrument_root / "layerwise", ignore_errors=True)
    instrument_receipt = json.loads(instrument_path.read_text())
    if instrument_receipt.get("status") != "PASS" or instrument_receipt.get("observed_mean_kld") != BASELINE_W28_KLD:
        raise RuntimeError("INSTRUMENT_RECEIPT_DRIFT")

    measurements = root / "MEASUREMENTS.jsonl"
    completed = {}
    if measurements.exists():
        for line in measurements.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["probe_id"]] = row

    task_id = os.environ.get("BANANA_SMASHER_TASK", "t_605f25e1")
    for probe in selected:
        probe_id = probe["probe_id"]
        if probe_id in completed:
            continue
        probe_root = root / "probes" / probe_id
        candidate_root = probe_root / "candidate"
        if not candidate_root.exists():
            candidate = materialize_sensitivity_candidate_v2(
                baseline_manifest, args.ledger, probe, output_root=candidate_root,
            )
            atomic(probe_root / "CANDIDATE.json", candidate)
        else:
            candidate = json.loads((probe_root / "CANDIDATE.json").read_text())
            if sha(Path(candidate["manifest_path"])) != candidate["manifest_sha256"]:
                raise RuntimeError(f"CANDIDATE_DRIFT:{probe_id}")

        bound_map, bound_map_sha, target_q2, target_q3 = resolve_target_root_map(candidate, q2_map, q3_map)
        unit_proofs = []
        if candidate.get("target_tier") in {"qtip2", "qtip3"}:
            unit_proofs = authenticate_target_units(
                candidate, root_map_path=bound_map, root_map_sha256=bound_map_sha,
            )

        score_root = probe_root / "score"
        started = time.time()
        result = _run_backpack_exact64(
            model_root=model_root,
            bank_path=bank,
            teacher_manifest_path=teacher,
            virtual_manifest_path=Path(candidate["manifest_path"]),
            virtual_manifest_sha256=candidate["manifest_sha256"],
            virtual_terminal_path=Path(candidate["terminal_path"]),
            virtual_terminal_sha256=candidate["terminal_sha256"],
            materialization_index_path=Path(candidate["index_path"]),
            qtip2_root_map_path=target_q2,
            qtip3_root_map_path=target_q3,
            checkpoint_path=checkpoint,
            checkpoint_sha256=CHECKPOINT_SHA,
            output_root=score_root,
            basis_sha256=BASIS,
            expected_windows=1,
            slice_id=f"sensitivity-{probe_id}",
            diagnostic_nonshipping=True,
        )
        measured = float(result["mean_kld"]) - BASELINE_W28_KLD
        if not math.isfinite(measured):
            raise RuntimeError(f"NONFINITE_MEASUREMENT:{probe_id}")
        receipt = {
            "schema": "banana-smasher-sensitivity-w28-probe-v2",
            "status": "PASS",
            "task_id": task_id,
            "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
            "canonical_git_pin": os.environ.get("BANANA_SMASHER_PIN"),
            "basis_sha256": BASIS,
            "probe_manifest_sha256": PROBE_MANIFEST_V2_SHA,
            "probe_id": probe_id,
            "role": probe.get("role"),
            "replicate_of": probe.get("replicate_of"),
            "component_probe_ids": probe.get("component_probe_ids"),
            "cell_id": probe["cell_id"],
            "cell_ids": candidate["cell_ids"],
            "layer": probe["layer"],
            "layer_band": probe["layer_band"],
            "projection": probe["projection"],
            "source_tier": probe["source_tier"],
            "target_tier": probe["target_tier"],
            "tier_pair": probe["tier_pair"],
            "predicted_delta_mean_kld": probe["predicted_delta_mean_kld"],
            "predicted_delta_by_class": probe.get("predicted_delta_by_class"),
            "baseline_mean_kld": BASELINE_W28_KLD,
            "candidate_mean_kld": float(result["mean_kld"]),
            "measured_delta_mean_kld": measured,
            "positions": int(result["positions"]),
            "support_width": int(result["support_width"]),
            "shipping_delta_bytes": candidate["shipping_delta_bytes"],
            "is_null_control": bool(candidate.get("is_null_control")),
            "bound_target_root_map": str(bound_map),
            "bound_target_root_map_sha256": bound_map_sha,
            "target_unit_proofs": unit_proofs,
            "instrument_absolute_delta": instrument_receipt.get("absolute_delta"),
            "runtime_terminal_path": str(score_root / "receipts/TERMINAL.json"),
            "runtime_terminal_sha256": sha(score_root / "receipts/TERMINAL.json"),
            "elapsed_seconds": time.time() - started,
            "created_unix": time.time(),
        }
        atomic(probe_root / "MEASUREMENT.json", receipt)
        with measurements.open("ab", buffering=0) as handle:
            handle.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode())
            os.fsync(handle.fileno())
        completed[probe_id] = receipt
        shutil.rmtree(score_root / "layerwise", ignore_errors=True)
        atomic(root / "PROGRESS.json", {
            "schema": "banana-smasher-sensitivity-probe-progress-v2",
            "status": "RUNNING",
            "task_id": task_id,
            "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
            "shard": shard_label,
            "completed": len(completed),
            "shard_completed": sum(p["probe_id"] in completed for p in selected),
            "shard_total": len(selected),
            "last_probe_id": probe_id,
            "last_measured_delta": measured,
            "updated_unix": time.time(),
        })
    atomic(root / "TERMINAL.json", {
        "schema": "banana-smasher-sensitivity-probe-shard-terminal-v2",
        "status": "PASS",
        "task_id": task_id,
        "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
        "basis_sha256": BASIS,
        "canonical_git_pin": os.environ.get("BANANA_SMASHER_PIN"),
        "probe_manifest_sha256": PROBE_MANIFEST_V2_SHA,
        "shard": shard_label,
        "completed": len(selected),
        "measurements_path": str(measurements),
        "measurements_sha256": sha(measurements),
        "created_unix": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
