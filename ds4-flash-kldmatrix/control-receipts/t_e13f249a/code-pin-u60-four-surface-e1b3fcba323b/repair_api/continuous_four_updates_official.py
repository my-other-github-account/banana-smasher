"""Official two-Spark continuous-only U0->U4 public API adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .balanced64 import ArtifactError
from .modern_green_resident import ModernGreenResidentEngine
from .resume_equivalence_official import (
    _Holder,
    _ModelProxy,
    _OptimizerProxy,
    _SchedulerProxy,
    _resident_engine_config,
    _write_progress,
)

PUBLIC_METHOD = "ResidentRepairAPI.continuous_four_updates"
PUBLIC_VERSION = "resident-api-continuous-four-updates-v1-official-two-spark"
BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
U0_SHA256 = "7978d1002d7e4ecfa280f646f70cc76638c0e7bd833cc3cc13a2de999050133f"
CLEAN_U0_LOCK_SHA256 = "7eb5edeb8583abba450a6f94de3cfe4fee0ab053c962bfcc1d035bd2d0c30fc2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_official_continuous_four_updates(
    api,
    *,
    config: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Execute exact N=4 continuous-only training and U0/U2/U4 scores."""
    import torch

    if config.get("authorized_api") is not True or config.get("world_size") != 2:
        raise ArtifactError("official continuous arm requires authorized_api=True/world_size=2")
    rank = int(config["rank"])
    if rank not in (0, 1):
        raise ArtifactError("official continuous arm rank must be 0 or 1")
    if config.get("basis_sha256") != BASIS_SHA256:
        raise ArtifactError("official continuous arm basis/model-index SHA drift")
    if config.get("checkpoint_sha256") != U0_SHA256:
        raise ArtifactError("official continuous arm U0 declaration drift")
    if config.get("clean_u0_lock_sha256") != CLEAN_U0_LOCK_SHA256:
        raise ArtifactError("official continuous arm CLEAN_U0_LOCK declaration drift")

    dist = torch.distributed
    if not dist.is_initialized():
        dist.init_process_group(
            backend=str(config.get("distributed_backend", "nccl")), init_method="env://"
        )
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise ArtifactError("torchrun process group does not match exact two-Spark identity")
    ranges = {int(key): tuple(int(value) for value in row) for key, row in config["layer_split"].items()}
    if ranges != {0: (0, 20), 1: (21, 42)}:
        raise ArtifactError("official continuous arm requires canonical 0:20/21:42 split")

    start_key = api.artifact.checkpoint_key("UPDATE_000")
    start_path = api.artifact.checkpoint_path(start_key)
    if _sha256(start_path) != U0_SHA256:
        raise ArtifactError("official continuous arm U0 byte SHA gate failed")
    lock_path_value = config.get("clean_u0_lock_path")
    if not isinstance(lock_path_value, str) or not lock_path_value:
        raise ArtifactError("official continuous arm requires clean_u0_lock_path")
    if _sha256(Path(lock_path_value)) != CLEAN_U0_LOCK_SHA256:
        raise ArtifactError("official continuous arm CLEAN_U0_LOCK byte SHA gate failed")

    extension = config.get("fast_k2_extension")
    if extension:
        extension_sha = config.get("fast_k2_extension_sha256")
        if not isinstance(extension_sha, str) or not extension_sha:
            raise ArtifactError("official continuous arm requires a pinned fast-K2 extension SHA")
        if _sha256(Path(str(extension))) != extension_sha:
            raise ArtifactError("official continuous arm fast-K2 extension SHA gate failed")
        os.environ["FAST_K2_EXTENSION"] = str(extension)
        os.environ["FAST_K2_EXTENSION_SHA256"] = extension_sha
        os.environ["FAST_K2_MODULE_NAME"] = str(
            config.get("fast_k2_module_name", "banana_fast_k2_grouped_0c3cc723fe66")
        )
    required_environment = {
        "BR_MANIFEST": config.get("binrepair_manifest"),
        "BR_DELTA_DIR": config.get("binrepair_delta_dir"),
        "BR_VQ3B_DIR": config.get("binrepair_vq3b_dir"),
        "BR_CORPUS": config.get("corpus"),
        "BR_TEACH": config.get("teacher_root"),
        "BR_TRAIN": config.get("train_windows", ",".join(str(value) for value in range(20, 84))),
        "BR_PROBE": config.get("probe_windows", ",".join(str(value) for value in range(20, 84))),
        "BR_FAST_STACK": "1",
        "BR_ATTN_IMPL": "sdpa",
    }
    missing = [key for key, value in required_environment.items() if value is None]
    if missing:
        raise ArtifactError("official continuous environment is incomplete: " + ", ".join(missing))
    os.environ.update({key: str(value) for key, value in required_environment.items()})

    holders: list[_Holder] = []
    progress_path = Path(receipt_path).parent / "PROGRESS.json"

    def progress_callback(**fields: Any) -> None:
        details = dict(fields)
        phase = str(details.pop("phase", "cold_load"))
        _write_progress(
            progress_path,
            rank=rank,
            arm="continuous",
            update=0,
            phase=phase,
            details=details,
        )

    def model_factory(start_payload):
        if holders:
            raise ArtifactError("continuous arm attempted to instantiate a second resident model")
        engine_config = _resident_engine_config(config)
        engine_config["progress_callback"] = progress_callback
        engine = ModernGreenResidentEngine(
            payload=start_payload,
            config=engine_config,
            rank=rank,
            layer_ranges=ranges,
        )
        engine.global_step = 0
        holder = _Holder(engine, "continuous")
        holders.append(holder)
        return _ModelProxy(holder)

    def optimizer_factory(model):
        return _OptimizerProxy(model.holder)

    def scheduler_factory(optimizer):
        return _SchedulerProxy(optimizer.holder)

    def update_fn(model, optimizer, scheduler, update):
        engine = model.holder.engine
        if engine is None:
            raise ArtifactError("continuous update attempted after release")
        progress_path = Path(receipt_path).parent / "PROGRESS.json"
        _write_progress(progress_path, rank=rank, arm="continuous", update=int(update), phase="before_step")
        report = engine._step(int(update) - 1)
        engine.global_step = int(update)
        model.holder.update = int(update)
        _write_progress(progress_path, rank=rank, arm="continuous", update=int(update), phase="after_step")
        return {
            "loss": report["loss"],
            "optimizer_steps": 1,
            "scheduler_steps": 1,
            "checkpoint_loaded": False,
            "resident_optimizer_step": True,
            "timings": dict(report["timings"]),
            "rank_reports": report["rank_reports"],
        }

    def resident_score_fn(model, update, windows):
        engine = model.holder.engine
        if engine is None:
            raise ArtifactError("continuous score attempted after release")
        progress_path = Path(receipt_path).parent / "PROGRESS.json"
        _write_progress(progress_path, rank=rank, arm="continuous", update=int(update), phase="before_score")
        measured = engine.score_resident(tuple(int(value) for value in windows))
        _write_progress(progress_path, rank=rank, arm="continuous", update=int(update), phase="after_score")
        return measured

    def state_fingerprint_fn(model, optimizer, scheduler, update):
        engine = model.holder.engine
        if engine is None:
            raise ArtifactError("continuous fingerprint attempted after release")
        merged, merged_optimizer, gathered = engine._gather_state()
        row = None
        if rank == 0:
            scheduler_state = (gathered or {}).get("scheduler")
            if not isinstance(merged, Mapping) or not isinstance(merged_optimizer, Mapping) or not isinstance(scheduler_state, Mapping):
                raise ArtifactError("continuous global two-rank state gather is incomplete")
            row = {
                "model": hashlib.sha256(api._canonical_state_bytes(merged)).hexdigest(),
                "optimizer": hashlib.sha256(api._canonical_state_bytes(merged_optimizer)).hexdigest(),
                "scheduler": hashlib.sha256(api._canonical_state_bytes(scheduler_state)).hexdigest(),
                "scope": "global_two_rank",
            }
        rows = [row]
        dist.broadcast_object_list(rows, src=0)
        if not isinstance(rows[0], Mapping):
            raise ArtifactError("continuous global fingerprint fan-in produced no result")
        return dict(rows[0])

    replay = {
        "model_factory": model_factory,
        "optimizer_factory": optimizer_factory,
        "scheduler_factory": scheduler_factory,
        "update_fn": update_fn,
        "resident_score_fn": resident_score_fn,
        "state_fingerprint_fn": state_fingerprint_fn,
        "release_fn": lambda model, optimizer, scheduler: model.holder.release(),
        "geometry": {"layers": 43, "ranks": 2, "windows_per_update": 4},
        "basis_sha256": config["basis_sha256"],
        "corpus_sha256": config["corpus_sha256"],
        "seed": int(config.get("seed", 1701)),
    }
    result = api.continuous_four_updates(
        "UPDATE_000", replay=replay, receipt_path=receipt_path
    )
    result["public_api"].update({"method": PUBLIC_METHOD, "version": PUBLIC_VERSION})
    result["canonical_identity"] = {
        "basis_sha256": BASIS_SHA256,
        "start_checkpoint_sha256": U0_SHA256,
        "clean_u0_lock_sha256": CLEAN_U0_LOCK_SHA256,
        "corpus_sha256": config["corpus_sha256"],
        "rank": rank,
        "world_size": 2,
    }
    return result
