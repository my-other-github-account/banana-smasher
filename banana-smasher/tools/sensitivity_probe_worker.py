#!/usr/bin/env python3
"""Run a resumable shard of W28 full-model sensitivity probes."""

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
from banana_smasher.sensitivity_probe import materialize_sensitivity_candidate

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
PROBE_MANIFEST_SHA = "1cd60b0dd9553a3cae953f04bcd124e8ac88ff352397bafef8c41fe6933ab94b"
LEDGER_SHA = "45b124e40a0f41a10e25949efdf32cc11a4271f24cf1331c6dbab6deacd813ee"
BASELINE_MANIFEST_SHA = "bbed6a44c690f89555b44ccfd4c8b0a0c5ed5dda0aca7bb3d0de2b60fd30d07a"
BASELINE_W28_KLD = 0.09936928004026413
BASELINE_RECEIPT_SHA = "85ee77d256fc45e93a35211898a7701c0ba301961b17574107933cab15904d7d"
CHECKPOINT_SHA = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
BANK_SHA = "d16580470bbba0e93729c3dfedb618a12c41633bb2c66094884c01dd9431156c"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--reproduce-baseline", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if sha(args.probe_manifest) != PROBE_MANIFEST_SHA or sha(args.ledger) != LEDGER_SHA:
        raise RuntimeError("PROBE_INPUT_SHA_RED")
    manifest = json.loads(args.probe_manifest.read_text())
    if manifest.get("basis_sha256") != BASIS or manifest.get("probe_count") != 200:
        raise RuntimeError("PROBE_MANIFEST_BASIS_RED")
    probes = manifest["probes"]
    if not (0 <= args.start < args.end <= len(probes)):
        raise RuntimeError("PROBE_SHARD_RANGE_RED")

    baseline_manifest = args.baseline_root / "BACKPACK_VIRTUAL_MANIFEST.json"
    baseline_index = args.baseline_root / "MATERIALIZATION_INDEX.jsonl"
    baseline_terminal = Path("/home/dnola/missions/VERIFY_LADDER_t_ff8ce60a_s8/EXACT102_VIRTUAL_TERMINAL.json")
    baseline_receipt = Path("/home/dnola/missions/VERIFY_LADDER_t_ff8ce60a_s8/receipts/RUNG3_W28_TERMINAL.json")
    model_root = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
    checkpoint = Path("/home/dnola/missions/FULL64_REPAIR_t_686041f5/relocation_run6058/rank1/home/dnola/missions/RESIDENT_VALIDATE_t_6031426c_attempt17/artifact/checkpoints/PRE.pt")
    bank = Path("/home/dnola/missions/MIXED_BACKPACK_t_91ccf93f_spark-8/run6457-backpack-w28/inputs/w28.bank.jsonl")
    teacher = Path("/home/dnola/missions/MIXED_BACKPACK_t_91ccf93f_spark-8/run6457-backpack-w28/inputs/teacher_w28.v2.json")
    q2_map = Path("/home/dnola/missions/VERIFY_LADDER_t_ff8ce60a_s8/QTIP2_ROOT_MAP.json")
    q3_map = Path("/home/dnola/missions/VERIFY_LADDER_t_ff8ce60a_s8/QTIP3_ROOT_MAP.json")
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
    baseline_value = json.loads(baseline_receipt.read_text())
    if baseline_value.get("mean_kld") != BASELINE_W28_KLD:
        raise RuntimeError("BASELINE_W28_VALUE_RED")
    available = int(Path("/proc/meminfo").read_text().split("MemAvailable:", 1)[1].split()[0]) * 1024
    if available < 24_000_000_000:
        raise RuntimeError(f"MEMORY_PREFLIGHT_RED:{available}")

    instrument_path = root / "INSTRUMENT.json"
    if args.reproduce_baseline and not instrument_path.exists():
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
            "task_id": "t_4d50f501",
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
    elif args.reproduce_baseline:
        instrument = json.loads(instrument_path.read_text())
        if instrument.get("status") != "PASS" or instrument.get("observed_mean_kld") != BASELINE_W28_KLD:
            raise RuntimeError("INSTRUMENT_RECEIPT_DRIFT")

    measurements = root / "MEASUREMENTS.jsonl"
    completed = {}
    if measurements.exists():
        for line in measurements.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["probe_id"]] = row
    for index in range(args.start, args.end):
        probe = probes[index]
        probe_id = probe["probe_id"]
        if probe_id in completed:
            continue
        probe_root = root / "probes" / probe_id
        candidate_root = probe_root / "candidate"
        if not candidate_root.exists():
            candidate = materialize_sensitivity_candidate(
                baseline_manifest,
                args.ledger,
                probe,
                output_root=candidate_root,
            )
            atomic(probe_root / "CANDIDATE.json", candidate)
        else:
            candidate = json.loads((probe_root / "CANDIDATE.json").read_text())
            if sha(Path(candidate["manifest_path"])) != candidate["manifest_sha256"]:
                raise RuntimeError(f"CANDIDATE_DRIFT:{probe_id}")
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
            qtip2_root_map_path=q2_map,
            qtip3_root_map_path=q3_map,
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
            "schema": "banana-smasher-sensitivity-w28-probe-v1",
            "status": "PASS",
            "task_id": "t_4d50f501",
            "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
            "canonical_git_pin": os.environ.get("BANANA_SMASHER_PIN"),
            "basis_sha256": BASIS,
            "probe_manifest_sha256": PROBE_MANIFEST_SHA,
            "probe_id": probe_id,
            "cell_id": probe["cell_id"],
            "layer_band": probe["layer_band"],
            "projection": probe["projection"],
            "source_tier": probe["source_tier"],
            "target_tier": probe["target_tier"],
            "tier_pair": probe["tier_pair"],
            "predicted_delta_mean_kld": probe["predicted_delta_mean_kld"],
            "baseline_mean_kld": BASELINE_W28_KLD,
            "candidate_mean_kld": float(result["mean_kld"]),
            "measured_delta_mean_kld": measured,
            "positions": int(result["positions"]),
            "support_width": int(result["support_width"]),
            "shipping_delta_bytes": candidate["shipping_delta_bytes"],
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
            "schema": "banana-smasher-sensitivity-probe-progress-v1",
            "status": "RUNNING",
            "task_id": "t_4d50f501",
            "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
            "shard": [args.start, args.end],
            "completed": len(completed),
            "shard_completed": sum(probes[i]["probe_id"] in completed for i in range(args.start, args.end)),
            "shard_total": args.end - args.start,
            "last_probe_id": probe_id,
            "updated_unix": time.time(),
        })
    atomic(root / "TERMINAL.json", {
        "schema": "banana-smasher-sensitivity-probe-shard-terminal-v1",
        "status": "PASS",
        "task_id": "t_4d50f501",
        "board_run_id": int(os.environ.get("BANANA_SMASHER_RUN_ID", "0")),
        "basis_sha256": BASIS,
        "canonical_git_pin": os.environ.get("BANANA_SMASHER_PIN"),
        "shard": [args.start, args.end],
        "completed": args.end - args.start,
        "measurements_path": str(measurements),
        "measurements_sha256": sha(measurements),
        "created_unix": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
