"""Official two-Spark published-PRE->U4 resume-equivalence replay adapter.

This module only builds the concrete callbacks consumed by the public
``ResidentRepairAPI.resume_equivalence`` method.  Training remains in the
accepted ``ModernGreenResidentEngine``; no command/runner fallback exists.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .balanced64 import ArtifactError
from .modern_green_resident import ModernGreenResidentEngine, _cpu_tree

PUBLIC_METHOD = "ResidentRepairAPI.resume_equivalence"
PUBLIC_VERSION = "resident-api-resume-equivalence-v5-clean-u0-disk-midpoint-static-w28"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resident_engine_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map the sealed resume-config field names to the public engine schema."""
    mapped = dict(config)
    mapped["manifest"] = config["binrepair_manifest"]
    mapped["delta_dir"] = config["binrepair_delta_dir"]
    mapped["vq3b_dir"] = config["binrepair_vq3b_dir"]
    return mapped


def _write_progress(
    destination: Path,
    *,
    rank: int,
    arm: str,
    update: int,
    phase: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema": "banana-smasher-resume-equivalence-progress-v1",
        "rank": rank,
        "arm": arm,
        "update": update,
        "phase": phase,
    }
    if details:
        payload["details"] = dict(details)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _canonical_checkpoint_bytes(
    payload: Mapping[str, Any], *, rank: int, dist, torch_module
) -> bytes:
    """Serialize once on rank 0 and broadcast the exact checkpoint bytes."""
    checkpoint_bytes = None
    if rank == 0:
        stream = io.BytesIO()
        torch_module.save(payload, stream)
        checkpoint_bytes = stream.getvalue()
    row = [checkpoint_bytes]
    dist.broadcast_object_list(row, src=0)
    if not isinstance(row[0], bytes) or not row[0]:
        raise ArtifactError("rank0 checkpoint serialization broadcast returned invalid bytes")
    return row[0]


class _Holder:
    def __init__(self, engine: ModernGreenResidentEngine, label: str) -> None:
        self.engine: ModernGreenResidentEngine | None = engine
        self.label = label
        self.update = 0

    def release(self) -> None:
        engine, self.engine = self.engine, None
        if engine is None:
            return
        try:
            engine.close()
        finally:
            # The API still retains proxy objects for earlier arms.  Clearing
            # the single holder severs every heavy model/optimizer reference.
            engine.student = None
            engine.optimizer = None
            engine.scheduler = None
            import torch
            torch.cuda.empty_cache()


class _ModelProxy:
    checkpoint_loaded = False

    def __init__(self, holder: _Holder) -> None:
        self.holder = holder

    def _engine(self) -> ModernGreenResidentEngine:
        if self.holder.engine is None:
            raise ArtifactError("released official resume arm was accessed")
        return self.holder.engine

    def resident_ready(self) -> bool:
        engine = self._engine()
        return engine.student is not None and engine.dist.is_initialized()

    def state_dict(self) -> dict[str, dict[str, Any]]:
        engine = self._engine()
        return {
            surface: {name: value.detach().cpu().clone() for name, value in rows}
            for surface, rows in (
                ("luts", engine.luts), ("norms", engine.norms), ("outputs", engine.outputs)
            )
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        engine = self._engine()
        if set(state) != {"luts", "norms", "outputs"}:
            raise ArtifactError("resume model state must contain exact official trainable surfaces")
        for surface, rows in (
            ("luts", engine.luts), ("norms", engine.norms), ("outputs", engine.outputs)
        ):
            engine.trainer.load_local_state(rows, state[surface], engine.student.device)


def _load_rank_local_optimizer_state(engine, value: Mapping[str, Any]) -> None:
    """Load a local midpoint state or project canonical global U0 state."""
    groups = value.get("param_groups")
    global_state = value.get("state")
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(global_state, Mapping):
        raise ArtifactError("resume optimizer state has no canonical three-surface lineage")
    local_state = engine.optimizer.state_dict()
    local_groups = local_state.get("param_groups")
    if not isinstance(local_groups, list) or len(local_groups) != 3:
        raise ArtifactError("resident optimizer has no canonical three-surface lineage")
    if all(len(source.get("params", [])) == len(local.get("params", []))
           for source, local in zip(groups, local_groups)):
        engine.optimizer.load_state_dict(value)
        return

    local_rows = {"luts": engine.luts, "norms": engine.norms, "outputs": engine.outputs}
    for index, surface in enumerate(("luts", "norms", "outputs")):
        names = [name for name, _param in local_rows[surface]]
        global_names = list(engine.state[surface])
        source_group = groups[index]
        source_ids = list(source_group.get("params", []))
        if len(source_ids) != len(global_names):
            raise ArtifactError(f"resume optimizer {surface} names/IDs do not match")
        global_id_by_name = dict(zip(global_names, source_ids))
        local_ids = local_groups[index]["params"]
        if len(local_ids) != len(names):
            raise ArtifactError(f"resident optimizer {surface} names/IDs do not match")
        for name, local_id in zip(names, local_ids):
            global_id = global_id_by_name.get(name)
            if global_id is None:
                raise ArtifactError(f"resume optimizer state missing official parameter {name}")
            state = global_state.get(global_id, global_state.get(str(global_id)))
            if state is not None:
                local_state["state"][local_id] = copy.deepcopy(state)
        local_groups[index].update({
            key: copy.deepcopy(item) for key, item in source_group.items() if key != "params"
        })
        local_groups[index]["params"] = local_ids
    engine.optimizer.load_state_dict(local_state)


class _OptimizerProxy:
    checkpoint_loaded = False
    def __init__(self, holder: _Holder) -> None: self.holder = holder
    def _value(self):
        if self.holder.engine is None or self.holder.engine.optimizer is None:
            raise ArtifactError("released official optimizer was accessed")
        return self.holder.engine.optimizer
    def state_dict(self): return _cpu_tree(__import__("torch"), self._value().state_dict())
    def load_state_dict(self, value):
        if self.holder.engine is None:
            raise ArtifactError("released official optimizer was accessed")
        return _load_rank_local_optimizer_state(self.holder.engine, value)


class _SchedulerProxy:
    checkpoint_loaded = False
    def __init__(self, holder: _Holder) -> None: self.holder = holder
    def _value(self):
        if self.holder.engine is None or self.holder.engine.scheduler is None:
            raise ArtifactError("released official scheduler was accessed")
        return self.holder.engine.scheduler
    def state_dict(self): return copy.deepcopy(self._value().state_dict())
    def load_state_dict(self, value): return self._value().load_state_dict(value)


def run_official_resume_equivalence(
    api, *, config: Mapping[str, Any], receipt_path: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Execute the exact two-arm N=4/mid-U2 experiment through the public API."""
    import torch

    if config.get("authorized_api") is not True or config.get("world_size") != 2:
        raise ArtifactError("official resume equivalence requires authorized_api=True/world_size=2")
    rank = int(config["rank"])
    if rank not in (0, 1):
        raise ArtifactError("official resume equivalence rank must be 0 or 1")
    # torchrun owns the rendezvous store.  Initialize the one shared process
    # group through its env:// contract before constructing a heavy engine;
    # otherwise TORCHELASTIC_USE_AGENT_STORE makes a second tcp:// group treat
    # both ranks as clients and no rank ever hosts that second store.
    # Pin bootstrap traffic to the exact paired QSFP rail.  Both hosts also
    # expose Wi-Fi and a second QSFP subnet; unconstrained NCCL selection can
    # leave rank 0 accepting forever while rank 1 polls another interface.
    socket_ifname = str(config.get("nccl_socket_ifname", ""))
    if not socket_ifname or not (Path("/sys/class/net") / socket_ifname).is_dir():
        raise ArtifactError("official resume equivalence requires a live NCCL socket interface")
    os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname
    os.environ["GLOO_SOCKET_IFNAME"] = socket_ifname
    dist = torch.distributed
    progress_path = Path(receipt_path).parent / "PROGRESS.json"
    _write_progress(
        progress_path,
        rank=rank,
        arm="bootstrap",
        update=0,
        phase="dist_init_start",
        details={
            "backend": str(config.get("distributed_backend", "nccl")),
            "init_method": "env://",
            "timeout_seconds": 600,
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
        },
    )
    if not dist.is_initialized():
        dist.init_process_group(
            backend=str(config.get("distributed_backend", "nccl")),
            init_method="env://",
            timeout=timedelta(seconds=600),
        )
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise ArtifactError("torchrun process group does not match the exact two-Spark rank")
    _write_progress(
        progress_path,
        rank=rank,
        arm="bootstrap",
        update=0,
        phase="dist_init_complete",
        details={"backend": str(config.get("distributed_backend", "nccl"))},
    )
    ranges = {int(k): tuple(int(x) for x in v) for k, v in config["layer_split"].items()}
    if ranges != {0: (0, 20), 1: (21, 42)}:
        raise ArtifactError("official resume equivalence requires the canonical 0:20/21:42 split")
    start_key = api.artifact.checkpoint_key("UPDATE_000")
    start_path = api.artifact.checkpoint_path(start_key)
    if _sha256(start_path) != config.get("checkpoint_sha256"):
        raise ArtifactError("official resume equivalence clean U0 SHA gate failed")
    start_payload = torch.load(start_path, map_location="cpu", weights_only=False)
    if int(start_payload.get("next_update", -1)) != 0:
        raise ArtifactError("official resume equivalence clean U0 is not update zero")
    optimizer_state = start_payload.get("optimizer", start_payload.get("optimizer_state"))
    if not isinstance(optimizer_state, Mapping) or optimizer_state.get("state"):
        raise ArtifactError("clean U0 must contain empty Adam state")
    extension = config.get("fast_k2_extension")
    if extension:
        os.environ["FAST_K2_EXTENSION"] = str(extension)
        if config.get("fast_k2_extension_sha256"):
            os.environ["FAST_K2_EXTENSION_SHA256"] = str(config["fast_k2_extension_sha256"])
        os.environ["FAST_K2_MODULE_NAME"] = str(config.get("fast_k2_module_name", "banana_fast_k2_grouped_0c3cc723fe66"))

    labels = ("continuous", "resume_pre", "resume_post")
    required_environment = {
        "BR_MANIFEST": config.get("binrepair_manifest"),
        "BR_DELTA_DIR": config.get("binrepair_delta_dir"),
        "BR_VQ3B_DIR": config.get("binrepair_vq3b_dir"),
        "BR_CORPUS": config.get("corpus"),
        "BR_TEACH": config.get("teacher_root"),
        "BR_TRAIN": config.get("train_windows", ",".join(str(x) for x in range(20, 84))),
        "BR_PROBE": config.get("probe_windows", ",".join(str(x) for x in range(20, 84))),
        "BR_FAST_STACK": "1",
        "BR_ATTN_IMPL": "sdpa",
    }
    missing_environment = [key for key, value in required_environment.items() if value is None]
    if missing_environment:
        raise ArtifactError("official resume environment is incomplete: " + ", ".join(missing_environment))
    os.environ.update({key: str(value) for key, value in required_environment.items()})
    holders: list[_Holder] = []
    snapshots: dict[tuple[str, int], Mapping[str, Any]] = {}
    resident_scores: dict[str, Mapping[str, Any]] = {}
    live_parent_sha = {
        "continuous": str(config["checkpoint_sha256"]),
        "resume": str(config["checkpoint_sha256"]),
    }

    def model_factory():
        index = len(holders)
        if index >= len(labels):
            raise ArtifactError("resume equivalence instantiated an unexpected extra arm")
        if index == 2:
            for old in holders:
                old.release()
        logical_arm = "resume" if labels[index].startswith("resume") else "continuous"
        progress_path = Path(receipt_path).parent / "PROGRESS.json"
        _write_progress(
            progress_path, rank=rank, arm=logical_arm, update=0, phase="engine_init_start"
        )
        engine_config = _resident_engine_config(config)

        def progress_callback(**fields: Any) -> None:
            phase = str(fields.get("phase", "unknown"))
            _write_progress(
                progress_path,
                rank=rank,
                arm=logical_arm,
                update=0,
                phase=f"engine_{phase}",
                details=fields,
            )

        engine_config["progress_callback"] = progress_callback
        engine = ModernGreenResidentEngine(
            payload=start_payload,
            config=engine_config,
            rank=rank,
            layer_ranges=ranges,
        )
        _write_progress(
            progress_path,
            rank=rank,
            arm=logical_arm,
            update=0,
            phase="engine_init_complete",
            details={"load_seconds": engine.student.load_seconds},
        )
        engine.global_step = 0
        holder = _Holder(engine, labels[index])
        holders.append(holder)
        return _ModelProxy(holder)

    def optimizer_factory(model):
        return _OptimizerProxy(model.holder)

    def scheduler_factory(optimizer):
        return _SchedulerProxy(optimizer.holder)

    def update_fn(model, optimizer, scheduler, update):
        holder = model.holder
        engine = holder.engine
        if engine is None:
            raise ArtifactError("official resume update attempted after release")
        logical_arm = "resume" if holder.label.startswith("resume") else "continuous"
        progress_path = Path(receipt_path).parent / "PROGRESS.json"
        _write_progress(
            progress_path, rank=rank, arm=logical_arm, update=int(update), phase="before_step"
        )
        report = engine._step(int(update) - 1)
        _write_progress(
            progress_path, rank=rank, arm=logical_arm, update=int(update), phase="after_step"
        )
        engine.global_step = int(update)
        holder.update = int(update)
        merged, merged_optimizer, gathered = engine._gather_state()
        _write_progress(
            progress_path, rank=rank, arm=logical_arm, update=int(update), phase="after_gather"
        )
        payload = None
        if rank == 0:
            payload = {
                "state": merged,
                "optimizer": merged_optimizer,
                "scheduler": (gathered or {}).get("scheduler"),
                "loss": report["loss"],
            }
        row = [payload]
        engine.dist.broadcast_object_list(row, src=0)
        snapshot = _cpu_tree(torch, row[0])
        snapshots[(logical_arm, int(update))] = snapshot
        checkpoint_root = Path(checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema": "banana-smasher-resume-equivalence-checkpoint-identity-v1",
            "task_id": config.get("task_id"),
            "arm": logical_arm,
            "update": int(update),
            "basis_sha256": config["basis_sha256"],
            "parent_checkpoint_sha256": live_parent_sha[logical_arm],
            "public_api_method": PUBLIC_METHOD,
            "public_api_version": PUBLIC_VERSION,
        }
        identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        identity_sha = hashlib.sha256(identity_bytes).hexdigest()
        persisted_payload = {
            "format": "banana-smasher-resume-equivalence-checkpoint-v1",
            "next_update": int(update),
            "identity": identity,
            "identity_sha256": identity_sha,
            "state": snapshot["state"],
            "optimizer": snapshot["optimizer"],
            "scheduler": snapshot["scheduler"],
            "objective": {"update": int(update) - 1, "after": snapshot.get("loss")},
        }
        checkpoint_bytes = _canonical_checkpoint_bytes(
            persisted_payload, rank=rank, dist=engine.dist, torch_module=torch
        )
        path = checkpoint_root / f"{logical_arm.upper()}_UPDATE_{int(update):03d}.pt"
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(checkpoint_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(directory_fd)
        os.close(directory_fd)
        observed_sha = _sha256(path)
        paired_shas: list[Any] = [None, None]
        engine.dist.all_gather_object(paired_shas, observed_sha)
        if len(set(paired_shas)) != 1:
            raise ArtifactError(
                f"paired every-update checkpoint byte SHA mismatch for {logical_arm} U{update}: {paired_shas}"
            )
        live_parent_sha[logical_arm] = observed_sha
        _write_progress(
            progress_path,
            rank=rank,
            arm=logical_arm,
            update=int(update),
            phase="checkpoint_persisted",
            details={
                "path": str(path),
                "sha256": observed_sha,
                "identity_sha256": identity_sha,
            },
        )
        return {
            "loss": report["loss"],
            "optimizer_steps": 1,
            "scheduler_steps": 1,
            "checkpoint_loaded": False,
            "resident_optimizer_step": True,
        }

    replay = {
        "model_factory": model_factory,
        "optimizer_factory": optimizer_factory,
        "scheduler_factory": scheduler_factory,
        "update_fn": update_fn,
        "release_fn": lambda model: model.holder.release(),
        "geometry": {"layers": 43, "ranks": 2, "windows_per_update": 4},
        "basis_sha256": config["basis_sha256"],
        "corpus_sha256": config["corpus_sha256"],
        "seed": int(config.get("seed", 1701)),
    }
    try:
        result = api.resume_equivalence(
            "UPDATE_000", replay=replay, total_updates=4, midpoint_update=2,
            midpoint_checkpoint_path=Path(checkpoint_dir) / "RESUME_MIDPOINT_UPDATE_002.pt",
            receipt_path=receipt_path,
        )
    finally:
        for holder in holders:
            holder.release()

    required = {(arm, update) for arm in ("continuous", "resume") for update in (2, 4)}
    if not required <= set(snapshots):
        raise ArtifactError(f"official resume snapshots missing: {sorted(required - set(snapshots))}")

    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_rows: dict[str, Any] = {}
    parent_by_arm = {"continuous": config["checkpoint_sha256"], "resume": config["checkpoint_sha256"]}
    for arm in ("continuous", "resume"):
        for update in (2, 4):
            snapshot = snapshots[(arm, update)]
            identity = {
                "schema": "banana-smasher-resume-equivalence-checkpoint-identity-v1",
                "task_id": config.get("task_id"),
                "arm": arm,
                "update": update,
                "basis_sha256": config["basis_sha256"],
                "parent_checkpoint_sha256": parent_by_arm[arm],
                "public_api_method": PUBLIC_METHOD,
                "public_api_version": PUBLIC_VERSION,
            }
            identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            identity_sha = hashlib.sha256(identity_bytes).hexdigest()
            payload = {
                "format": "banana-smasher-resume-equivalence-checkpoint-v1",
                "next_update": update,
                "identity": identity,
                "identity_sha256": identity_sha,
                "state": snapshot["state"],
                "optimizer": snapshot["optimizer"],
                "scheduler": snapshot["scheduler"],
                "objective": {"update": update - 1, "after": snapshot.get("loss")},
            }
            path = destination / f"{arm.upper()}_UPDATE_{update:03d}.pt"
            temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            engine_dist = holders[-1].engine.dist if holders[-1].engine is not None else None
            if engine_dist is None:
                # The public API releases heavy engines before persistence; use
                # the still-initialized torch.distributed process group.
                import torch.distributed as dist
                engine_dist = dist
            checkpoint_bytes = _canonical_checkpoint_bytes(
                payload, rank=rank, dist=engine_dist, torch_module=torch
            )
            with temporary.open("wb") as stream:
                stream.write(checkpoint_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            observed = _sha256(path)
            hashes: list[Any] = [None, None]
            engine_dist.all_gather_object(hashes, observed)
            if len(set(hashes)) != 1:
                raise ArtifactError(f"paired checkpoint byte SHA mismatch for {arm} U{update}: {hashes}")
            checkpoint_rows[f"{arm}_u{update}"] = {
                "path": str(path), "sha256": observed, "identity_sha256": identity_sha,
                "state_sha256": api._state_fingerprint({"state": snapshot["state"]}),
                "parent_checkpoint_sha256": parent_by_arm[arm], "next_update": update,
            }
            parent_by_arm[arm] = observed

    manifest_path = api.artifact.root / "ARTIFACT.json"
    manifest = copy.deepcopy(api.artifact.manifest)
    key_by_label = {"pre": "PRE"}
    for arm in ("continuous", "resume"):
        for update in (2, 4):
            label = f"{arm}_u{update}"
            row = checkpoint_rows[label]
            key = f"{arm.upper()}_UPDATE_{update:03d}"
            key_by_label[label] = key
            path = Path(row["path"])
            manifest["checkpoints"][key] = {
                "path": str(path.relative_to(api.artifact.root)),
                "sha256": row["sha256"],
                "identity_sha256": row["identity_sha256"],
                "parent_sha256": row["parent_checkpoint_sha256"],
                "next_update": update,
            }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    api.artifact.manifest.clear()
    api.artifact.manifest.update(manifest)

    # Scoring is intentionally deferred to the one SHA-pinned static W28
    # builder after training seals U0/U2/U4. This process may not invoke a
    # full64 or substitute scorer.
    scores: dict[str, Any] = {}

    result["public_api"] = {"method": PUBLIC_METHOD, "version": PUBLIC_VERSION}
    result["canonical_identity"] = {
        "basis_sha256": config["basis_sha256"],
        "start_checkpoint_sha256": config["checkpoint_sha256"],
        "clean_u0_lock_sha256": config["clean_u0_lock_sha256"],
        "corpus_sha256": config["corpus_sha256"],
    }
    result["checkpoints"] = checkpoint_rows
    result["scores"] = scores
    result["comparison"] = {
        "u2_state_fingerprint_equal": checkpoint_rows["continuous_u2"]["state_sha256"] == checkpoint_rows["resume_u2"]["state_sha256"],
        "u4_state_fingerprint_equal": checkpoint_rows["continuous_u4"]["state_sha256"] == checkpoint_rows["resume_u4"]["state_sha256"],
        "u4_first_divergence_update": result.get("first_divergence_update"),
    }
    return result
