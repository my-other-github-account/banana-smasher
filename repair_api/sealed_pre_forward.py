from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

BUILDER_SHA256 = "11ead706db562197e76cdc320d5d13044bb254a411b6412326667f524ddf29ed"
PLANESOURCE_SHA256 = "167603b5662437a2f9fc4b3ead1561d777a7a831a898133993b9e1c0c26c9f87"
BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT_SHA256 = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
CANDIDATE_IDENTITY = "51074d5fedfc922b8442cb6cf988773f32991c16e6cf34ca21131c4f7b1726f8"
TRAINER_SHA256 = "72b4c43018126d04eda43025f32f4a8b0cb5fe1b9cc57807ac162170f3a60d60"
STATIC_W28_WRAPPER_SHA256 = "ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
STATIC_W28_EXPERT_SHA256 = "4ba1411601b186dd0d6a3a89c829320f1b50e3112a40db40034e9fbadfb5d552"
SEALED_MODEL_ROOT = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
W28_KLD = 0.1364830042977786
W28_TOP1 = 880
POSITIONS = 1024
SUPPORT = 8192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(payload).hexdigest()


def source_binding(root: Path | None = None) -> dict[str, Any]:
    assets = (root or Path(__file__).parent) / "assets"
    builder = assets / "builder_B2_PUBLISHED_PRE.py"
    planesource = assets / "official_local_planesource.py"
    observed = {"builder": sha256(builder), "planesource": sha256(planesource)}
    expected = {"builder": BUILDER_SHA256, "planesource": PLANESOURCE_SHA256}
    if observed != expected:
        raise RuntimeError(f"SEALED_PRE_SOURCE_HASH_MISMATCH:{observed}")
    builder_text = builder.read_text()
    planesource_text = planesource.read_text()
    required = {
        "single_window_microbatch": "mbs = [slice(i, min(i + a.mb, len(wins)))" in builder_text,
        "official_forward": "hidden[mi] = lay(" in builder_text,
        "planesource_layer": "class PlaneSource:" in planesource_text and "def layer(self, layer: int):" in planesource_text,
    }
    if not all(required.values()):
        raise RuntimeError(f"SEALED_PRE_SOURCE_SURFACE_MISMATCH:{required}")
    return {
        "status": "PASS",
        "builder_path": str(builder),
        "builder_sha256": observed["builder"],
        "builder_forward_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:593-643",
        "planesource_path": str(planesource),
        "planesource_sha256": observed["planesource"],
        "planesource_forward_source": "repair_api/assets/official_local_planesource.py:592-624",
        "known_value_fixture": {"window": 28, "kld_mean": W28_KLD, "top1": W28_TOP1},
        "surface": required,
    }


def bind_sealed_pre_resident_config(config: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact sealed source identity to the existing resident provider."""
    config["sealed_pre_source_binding"] = source_binding()
    # W28=0.136483 was produced by the immutable static grouped provider bytes.
    # The full-weight reconstruction is a diagnostic comparator, not that
    # accepted provider, so never let the sealed binding silently select it.
    config["provider_resolution_mode"] = "STATIC_W28_GROUPED"
    config.setdefault("resident_validation_expert_implementation", "accepted_static_w28")
    # Default to the accepted two-window physical geometry, but preserve an
    # explicit singleton request used by the public parity-tap comparator.
    config.setdefault("score_window_batch_size", 2)
    config.setdefault("sealed_builder_window_microbatch", 2)
    assets = Path(__file__).resolve().parent / "assets"
    config["trainer_source"] = str(assets / "static_w28_modern_green_clean_u0.py")
    config["trainer_source_sha256"] = TRAINER_SHA256
    # OfficialK2ResidentRankEngine imports these historical top-level modules
    # before the trainer. Bind the same commit-owned provider bytes here so the
    # accepted trainer cannot inherit the mutable full64 provider from the
    # artifact manifest.
    config["fast_k2_wrapper_source"] = str(assets / "static_w28_fast_k2_grouped.py")
    config["fast_k2_wrapper_source_sha256"] = STATIC_W28_WRAPPER_SHA256
    config["resident_expert_source"] = str(assets / "static_w28_fast_v7_expert_base.py")
    config["resident_expert_source_sha256"] = STATIC_W28_EXPERT_SHA256
    return config["sealed_pre_source_binding"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"SEALED_PRE_IMPORT_SPEC_REFUSED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bind_l034(planesource: Any, roster_path: Path) -> None:
    import torch

    roster_sha = sha256(roster_path)
    roster = json.loads(roster_path.read_text())
    if roster.get("basis_sha256") != BASIS_SHA256 or roster.get("member_count") != 768:
        raise RuntimeError("SEALED_PRE_L034_ROSTER_REFUSED")
    base = roster_path.parent
    rows = {(int(row["expert"]), str(row["projection"])): row for row in roster["members"]}
    expected = {(expert, projection) for expert in range(256) for projection in ("w1", "w2", "w3")}
    if set(rows) != expected:
        raise RuntimeError("SEALED_PRE_L034_COVERAGE_REFUSED")

    def load_complete34(self):
        self.active_lut = planesource.candidate_lut(34)
        self.counters["compact_layers_touched"].append(34)
        self.counters["local_staged_layers"].append(34)
        self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
        self._write_progress(status="RUNNING_PRESERVED_L034", active_layer=34)

        def read(expert: int, projection: str):
            row = rows[(expert, projection)]
            path = base / row["path"]
            if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                raise RuntimeError(f"L034 selected-wire member identity refused E{expert:03d}/{projection}")
            return self._decode(path, projection)

        return read

    planesource.PlaneSource._load_complete34 = load_complete34
    planesource.COMPLETE34_BINDING = roster_path
    planesource.COMPLETE34_BINDING_SHA = roster_sha


def _prepare_exact_modules(*, task: str, rank: int, root: Path, config: dict[str, Any], checkpoint: Path):
    binding = source_binding()
    assets = Path(binding["builder_path"]).parent
    hardcoded_checkpoint = Path("/home/dnola/missions/PRE_SCORE_t_9e5a36e1/track-b-s2/inputs/PRE_f9bffe04.pt")
    hardcoded_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not hardcoded_checkpoint.exists():
        shutil.copyfile(checkpoint, hardcoded_checkpoint)
    if sha256(hardcoded_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("SEALED_PRE_HARDCODED_CHECKPOINT_REFUSED")
    os.environ["BANANA_SMASHER_CHECKPOINT"] = str(hardcoded_checkpoint)
    os.environ["BANANA_SMASHER_CHECKPOINT_SHA256"] = CHECKPOINT_SHA256
    os.environ["BANANA_SMASHER_CANDIDATE_IDENTITY"] = CANDIDATE_IDENTITY
    builder = _load_module(f"sealed_pre_builder_rank{rank}", assets / "builder_B2_PUBLISHED_PRE.py")
    planesource = _load_module(f"sealed_pre_planesource_rank{rank}", assets / "official_local_planesource.py")
    planesource.BUILDER = builder
    planesource.TASK = task
    planesource.RUN = int(os.environ.get("HERMES_KANBAN_RUN_ID", "0") or 0)
    planesource.ROOT = root
    planesource.MISSION = root
    planesource.PROGRESS = root / "receipts" / f"SEALED_PRE_FORWARD_PROGRESS.rank{rank}.json"
    planesource.STAGE_ROOT = Path(f"/dev/shm/FULL64_SEALED_PRE_{task}_rank{rank}/runtime_layer")
    planesource.CACHE_ROOT = Path(f"/dev/shm/FULL64_SEALED_PRE_{task}_rank{rank}/cache")
    planesource.CLAIM = Path("/home/dnola/HOST_CLAIM.json")
    planesource.MODEL = SEALED_MODEL_ROOT
    planesource.CHECKPOINT = hardcoded_checkpoint
    planesource.CHECKPOINT_SHA = CHECKPOINT_SHA256
    planesource.CANDIDATE_IDENTITY = CANDIDATE_IDENTITY
    roster = Path(config["l034_roster"])
    _bind_l034(planesource, roster)
    builder.PlaneSource = planesource.PlaneSource
    return builder, planesource, binding


def _score_outputs(out: Path, teacher_root: Path, windows: tuple[int, ...]) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    rows: list[dict[str, Any]] = []
    for window in windows:
        teacher_path = teacher_root / f"t8192_win{window}.pt"
        candidate_path = out / f"q8192_win{window}.pt"
        teacher = torch.load(teacher_path, map_location="cpu", mmap=True, weights_only=True)
        candidate = torch.load(candidate_path, map_location="cpu", mmap=True, weights_only=True)
        reference = teacher["logprob"][:POSITIONS, :SUPPORT].numpy().astype(np.float64, copy=False)
        observed = candidate["q_lp_at_ref"][:POSITIONS, :SUPPORT].numpy().astype(np.float64, copy=False)
        reference_max = np.max(reference, axis=1, keepdims=True)
        observed_max = np.max(observed, axis=1, keepdims=True)
        reference_norm = reference - (reference_max + np.log(np.exp(reference - reference_max).sum(axis=1, keepdims=True)))
        observed_norm = observed - (observed_max + np.log(np.exp(observed - observed_max).sum(axis=1, keepdims=True)))
        terms = np.sum(np.exp(reference_norm) * (reference_norm - observed_norm), axis=1, dtype=np.float64)
        kld_sum = math.fsum(float(value) for value in terms.tolist())
        top1 = int((candidate["q_argmax"][:POSITIONS].numpy() == teacher["idx"][:POSITIONS, 0].numpy()).sum())
        rows.append({
            "window": window, "positions": POSITIONS, "support": SUPPORT,
            "kld_sum_binary64": kld_sum, "kld_mean": kld_sum / POSITIONS, "top1": top1,
            "teacher_sha256": sha256(teacher_path), "candidate_logits_sha256": sha256(candidate_path),
        })
    return rows


def _run_builder(builder: Any, *, root: Path, config: dict[str, Any], windows: tuple[int, ...], label: str) -> tuple[list[dict[str, Any]], float]:
    out = root / "sealed_pre_outputs" / label
    if out.exists():
        raise RuntimeError(f"SEALED_PRE_OUTPUT_ALREADY_EXISTS:{out}")
    out.mkdir(parents=True)
    original_argv = sys.argv
    started = time.perf_counter()
    source_args = ["--local-dir", str(SEALED_MODEL_ROOT)]
    if int(config["rank"]) == 0:
        source_args = [
            "--remote", "dnola@192.168.200.4:/home/dnola/models/hf/DeepSeek-V4-Flash-0731",
            "--shard-buf", str(root / "shard_buf"), "--keep-shards", "3",
        ]
    try:
        sys.argv = [
            str(Path(builder.__file__)), "--mode", "planes", "--planes-dir", str(root / "SEALED_PRE_CONTRACT.json"),
            "--ref-dir", str(config["validation_teacher_root"]), "--corpus", str(config["validation_corpus"]),
            "--meta-dir", str(SEALED_MODEL_ROOT), *source_args,
            "--out", str(out), "--cand-pos-limit", str(POSITIONS), "--count", str(len(windows)),
            "--chunk", str(len(windows)), "--mb", "1", "--windows", ",".join(map(str, windows)),
            "--tag", f"PUBLIC_API_SEALED_PRE_{label}",
        ]
        result = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
    if result:
        raise RuntimeError(f"SEALED_PRE_BUILDER_RC:{result}")
    wall = time.perf_counter() - started
    return _score_outputs(out, Path(config["validation_teacher_root"]), windows), wall


def run_sealed_pre_forward(*, task: str, rank: int, root: Path, config: dict[str, Any], checkpoint: Path,
                           reference: dict[str, Any], canonical_pin: str) -> None:
    import torch

    os.environ["NCCL_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    os.environ["GLOO_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
    builder, planesource, binding = _prepare_exact_modules(
        task=task, rank=rank, root=root, config=config, checkpoint=checkpoint
    )
    if rank == 0:
        admission_rows, admission_wall = _run_builder(builder, root=root, config=config, windows=(28,), label="W28_ADMISSION")
        admission = admission_rows[0]
        admission_status = admission["kld_mean"] == W28_KLD and admission["top1"] == W28_TOP1
        admission_payload = {"pass": admission_status, "wall": admission_wall, "row": admission}
    else:
        admission_payload = None
    broadcast = [admission_payload]
    torch.distributed.broadcast_object_list(broadcast, src=0)
    admission_payload = broadcast[0]
    admission_receipt = {
        "schema": "banana-smasher-sealed-pre-w28-admission-v1",
        "status": "PASS" if admission_payload["pass"] else "RED",
        "task_id": task, "rank": rank, "canonical_code_commit": canonical_pin,
        "basis_sha256": BASIS_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_binding": binding, "measurement": admission_payload["row"],
        "admission_wall_seconds": admission_payload["wall"],
    }
    admission_path = root / "receipts" / f"SEALED_PRE_W28_ADMISSION.{task}.rank{rank}.json"
    admission_receipt["receipt_sha256"] = atomic_json(admission_path, admission_receipt)
    if not admission_payload["pass"]:
        raise RuntimeError(f"SEALED_PRE_W28_RED:{admission_payload['row']}")

    windows = tuple(int(value) for value in reference["coverage"]["expected_windows"])
    assigned = windows[rank::2]
    torch.distributed.barrier()
    full_started = time.perf_counter()
    local_rows, local_wall = _run_builder(
        builder, root=root, config=config, windows=assigned, label=f"FULL64_RANK{rank}"
    )
    gathered: list[Any] = [None, None]
    torch.distributed.all_gather_object(
        gathered,
        {"rank": rank, "rows": local_rows, "wall": local_wall},
    )
    post_load_wall = time.perf_counter() - full_started
    rows = sorted((row for payload in gathered for row in payload["rows"]), key=lambda row: windows.index(row["window"]))
    aggregate = {
        "positions": sum(row["positions"] for row in rows),
        "kld_sum": math.fsum(row["kld_sum_binary64"] for row in rows),
        "top1": sum(row["top1"] for row in rows),
    }
    aggregate["kld_mean"] = aggregate["kld_sum"] / aggregate["positions"]
    status = "PASS" if post_load_wall < 300 and abs(aggregate["kld_mean"] - 0.22920699467439512) <= 5e-4 and aggregate["top1"] == 56534 else "RATE_LOW" if post_load_wall >= 300 else "AGGREGATE_RED"
    terminal = {
        "schema": "banana-smasher-sealed-pre-full64-terminal-v1", "status": status,
        "task_id": task, "rank": rank, "canonical_code_commit": canonical_pin,
        "basis_sha256": BASIS_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_binding": binding, "admission_receipt": str(admission_path),
        "post_load_wall_seconds": post_load_wall, "rank_phase_profiles": gathered,
        "aggregate": aggregate, "per_window": rows,
    }
    terminal_path = root / "receipts" / f"SEALED_PRE_FULL64_{status}.{task}.rank{rank}.json"
    terminal["receipt_sha256"] = atomic_json(terminal_path, terminal)
    print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
    torch.distributed.destroy_process_group()
    if status != "PASS":
        raise RuntimeError(f"{status}:{post_load_wall}:{aggregate}")
