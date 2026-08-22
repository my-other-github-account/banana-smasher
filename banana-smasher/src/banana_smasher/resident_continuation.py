"""Official Modern Green grouped-K2 resident continuation engine.

This module is deliberately coupled to the accepted clean-U0 trainer.  It does
not manufacture a loss from checkpoint tensors: it constructs the resident
ShardStudent, routes the real model through both layer partitions, evaluates
the teacher KL objective, and runs the trainer's legal LUT/RMS/gain surface
through Adam and LambdaLR.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from .resident_balanced64 import ArtifactError
from .resident_terminal_scorer import ResidentScoreAccumulator, score_terminal_hidden

MODEL_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
ADMISSION_SHA256 = "76d0674eb0cd37fc9022bac5e048c2b77c721826182222ae0a0609e29607a2c5"
CORPUS_SHA256 = "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5"
TRAINER_SHA256 = "a55c2f5104b8d9dd06d845684d168be6f6e9dae637bac08443bd6ddbaf94201a"
OFFICIAL_PHYSICAL_LAYER_SHA256 = "5d4ca4ac7d25e96fd428e55b2a7e18e074bac9d8aa23004bddbb6bde15d020d5"
WINDOWS_PER_STEP = 4
PIPELINE_MICROBATCH = 4
# Keep score-only attention at the proven per-step memory envelope. Balanced64
# still covers all 64 ordered 1024-token windows; only the transient batch changes.
SCORE_MICROBATCH = 1
SCORE_LOGIT_MICROBATCH = 1
BASE_LRS = {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2}
HISTORICAL_BASE_LRS = {"luts": 2.5e-4, "norms": 2.5e-5, "outputs": 2.5e-4}
HISTORICAL_SAMPLING_MODE = "historical_category_stratified_v1"
HISTORICAL_TRAIN_BANK_SHA256 = "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3"
HISTORICAL_CATEGORIES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _construct_shard_student(
    trainer: Any,
    *,
    torch: Any,
    np: Any,
    base: Any,
    official_k2: Any,
    model_root: Path,
    admission: Mapping[str, Any],
    parent_root: Path,
    member_roster_path: Path,
    member_roster_sha256: str,
    payload: Mapping[str, Any],
    rank: int,
    first: int,
    last: int,
    status_cb: Any,
) -> Any:
    """Construct the authenticated trainer using its declared roster ABI."""
    loader = getattr(trainer, "load_member_roster", None)
    common = {
        "torch": torch,
        "np": np,
        "base": base,
        "official_k2": official_k2,
        "model_root": model_root,
        "admission": admission,
        "input_state": payload,
        "rank": rank,
        "first": first,
        "last": last,
        "status_cb": status_cb,
    }
    if callable(loader):
        members = loader(member_roster_path, member_roster_sha256)
        return trainer.ShardStudent(member_roster=members, **common)
    parameters = inspect.signature(trainer.ShardStudent.__init__).parameters
    if "parent_root" not in parameters:
        raise RuntimeError("resident trainer lacks a recognized provider roster ABI")
    legacy = {"parent_root": parent_root}
    if "l034_roster" in parameters:
        legacy["l034_roster"] = member_roster_path
    return trainer.ShardStudent(**legacy, **common)


def _select_trainer_fwht(trainer: Any) -> None:
    """Select Quack on the authenticated trainer module that owns grouped-K2."""
    selector = getattr(trainer, "set_fwht_backend", None)
    if not callable(selector):
        raise ArtifactError("official trainer lacks the required Quack FWHT selector")
    selector("quack")


def _historical_mode(config: Mapping[str, Any]) -> bool:
    return (
        config.get("sampling_mode") == HISTORICAL_SAMPLING_MODE
        or config.get("training_recipe") == "historical_true_u16_stratified_v1"
    )


def _historical_schedule(config: Mapping[str, Any]) -> Mapping[str, Any]:
    path_value = config.get("window_schedule") or config.get("stratified_schedule_path")
    expected_value = config.get("window_schedule_sha256") or config.get("stratified_schedule_sha256")
    if not path_value or not expected_value:
        raise ArtifactError("historical window schedule path and SHA are required")
    path = Path(str(path_value)).expanduser().resolve()
    expected_sha = str(expected_value)
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise ArtifactError("historical window schedule SHA mismatch")
    try:
        schedule = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"historical window schedule is unreadable: {exc}") from exc
    bank = schedule.get("train_bank", {}) if isinstance(schedule, Mapping) else {}
    if (
        schedule.get("schema") != "banana-smasher.category-stratified-continuation-schedule.v1"
        or schedule.get("global_updates") != [17, 64]
        or bank.get("membership_sha256") != HISTORICAL_TRAIN_BANK_SHA256
        or config.get("train_bank_membership_sha256", HISTORICAL_TRAIN_BANK_SHA256) != HISTORICAL_TRAIN_BANK_SHA256
        or (
            bank.get("counts") is not None
            and tuple(bank.get("counts", {}).get(category) for category in HISTORICAL_CATEGORIES)
            != (12, 10, 11, 10, 10, 11)
        )
    ):
        raise ArtifactError("historical window schedule identity drift")
    return schedule


def _training_window_ids(config: Mapping[str, Any]) -> list[int]:
    if not _historical_mode(config):
        return list(range(20, 84))
    bank = _historical_schedule(config)["train_bank"]["windows_by_category"]
    windows = [int(window) for category in HISTORICAL_CATEGORIES for window in bank[category]]
    if len(windows) != 64 or len(set(windows)) != 64:
        raise ArtifactError("historical train bank must contain exactly 64 unique windows")
    return windows


def _validated_base_lrs(config: Mapping[str, Any]) -> dict[str, float]:
    if not _historical_mode(config):
        return dict(BASE_LRS)
    value = config.get("base_learning_rates")
    try:
        observed = {name: float(value[name]) for name in ("luts", "norms", "outputs")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("historical base learning rates are required") from exc
    if observed != HISTORICAL_BASE_LRS:
        raise ArtifactError("historical base learning rates must remain exact")
    return observed


def _window_microbatches(config: Mapping[str, Any], update: int) -> list[list[int]]:
    """Return the preregistered resident windows for one optimizer update."""
    if _historical_mode(config):
        if config.get("windows_per_update") != 6 or config.get("pipeline_microbatch") != 2:
            raise ArtifactError("historical category sampler requires six windows in three 2-window groups")
        schedule = _historical_schedule(config)
        global_update = int(update) + 1
        row = next(
            (item for item in schedule.get("updates", []) if item.get("global_update") == global_update),
            None,
        )
        if not isinstance(row, Mapping):
            raise ArtifactError(f"historical window schedule has no U{global_update} row")
        order = row.get("microbatch_category_order")
        by_category = row.get("windows_by_category", {})
        if (
            not isinstance(order, list)
            or len(order) != 6
            or set(order) != set(HISTORICAL_CATEGORIES)
            or row.get("gradient_accumulation", 6) != 6
            or row.get("optimizer_steps", 1) != 1
            or any(
                "loss_weight" in by_category[category]
                and float(by_category[category]["loss_weight"]) != 1.0 / 6.0
                for category in order
            )
        ):
            raise ArtifactError(f"historical window schedule contract drift at U{global_update}")
        windows = [int(by_category[category]["window_id"]) for category in order]
        return [windows[index:index + 2] for index in range(0, 6, 2)]
    windows_per_update = config.get("windows_per_update", WINDOWS_PER_STEP)
    pipeline_microbatch = config.get("pipeline_microbatch", PIPELINE_MICROBATCH)
    if isinstance(windows_per_update, bool) or isinstance(pipeline_microbatch, bool):
        raise ArtifactError("official resident window geometry must use integer counts")
    try:
        windows_per_update = int(windows_per_update)
        pipeline_microbatch = int(pipeline_microbatch)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("official resident window geometry must use integer counts") from exc
    if pipeline_microbatch != PIPELINE_MICROBATCH or windows_per_update not in {WINDOWS_PER_STEP, 16}:
        raise ArtifactError("official resident window geometry must be 4 or preregistered 16 as 4-window microbatches")
    if windows_per_update == WINDOWS_PER_STEP:
        first = 20 + WINDOWS_PER_STEP * (int(update) % 16)
    else:
        # Four U16..U19 updates traverse all 64 corpus windows exactly once.
        first = 20 + 16 * ((int(update) - 16) % 4)
    windows = [20 + ((first - 20 + offset) % 64) for offset in range(windows_per_update)]
    return [windows[index:index + PIPELINE_MICROBATCH] for index in range(0, len(windows), PIPELINE_MICROBATCH)]


def _sampling_plan(config: Mapping[str, Any], start_update: int, target_update: int) -> dict[str, Any]:
    if target_update <= start_update:
        raise ArtifactError("sampling plan target must advance beyond its start")
    updates = []
    exposure_counts: dict[str, int] = {}
    for update in range(int(start_update), int(target_update)):
        groups = _window_microbatches(config, update)
        windows = [window for group in groups for window in group]
        for window in windows:
            key = str(window)
            exposure_counts[key] = exposure_counts.get(key, 0) + 1
        updates.append(
            {
                "global_update": update + 1,
                "windows": windows,
                "window_microbatches": groups,
            }
        )
    return {
        "schema": "resident-window-exposure-plan-v1",
        "sampling_mode": config.get("sampling_mode", "modern_green_sequential"),
        "schedule_sha256": config.get("window_schedule_sha256"),
        "train_bank_membership_sha256": config.get("train_bank_membership_sha256"),
        "updates": updates,
        "exposure_counts": exposure_counts,
        "total_exposures": sum(exposure_counts.values()),
    }


def _validated_lr_scale(config: Mapping[str, Any]) -> float:
    """Return a declared LR scale while preserving the loaded cosine position."""
    value = config.get("lr_scale", 1.0)
    if isinstance(value, bool):
        raise ArtifactError("official resident lr_scale must be a finite positive number")
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("official resident lr_scale must be a finite positive number") from exc
    if not (0.0 < scale <= 1.0) or scale != scale or scale == float("inf"):
        raise ArtifactError("official resident lr_scale must be in (0, 1]")
    if _historical_mode(config) and scale != 1.0:
        raise ArtifactError("historical sampling intervention requires lr_scale=1.0")
    return scale


def _schedule_multiplier(config: Mapping[str, Any], update: int, historical: Any) -> float:
    """Select the immutable historical schedule or a declared fresh warm restart."""
    mode = config.get("schedule_mode", "loaded_global_cosine")
    if mode == "loaded_historical_warmup_constant":
        multiplier = (int(update) + 1) / 16.0 if int(update) < 16 else 1.0
        if int(update) >= 16 and multiplier != 1.0:
            raise ArtifactError("historical completed warmup must remain at multiplier 1.0")
        return multiplier
    if mode == "loaded_global_cosine":
        return float(historical(int(update)))
    if mode != "fresh_cosine_warmup":
        raise ArtifactError(f"unsupported official resident schedule_mode: {mode}")
    try:
        restart = int(config["schedule_restart_update"])
        warmup = int(config["schedule_warmup_updates"])
        cosine_steps = int(config["schedule_cosine_steps"])
        minimum = float(config["schedule_min_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("fresh cosine warmup schedule contract is incomplete") from exc
    if restart != 16 or warmup < 1 or cosine_steps < warmup or not (0.0 < minimum <= 1.0):
        raise ArtifactError("fresh cosine warmup schedule contract is invalid")
    relative = max(0, int(update) - restart)
    if relative < warmup:
        return float(relative + 1) / float(warmup)
    decay_step = min(relative - warmup, cosine_steps - warmup)
    decay_span = cosine_steps - warmup
    if decay_span == 0:
        return minimum
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_span))
    return minimum + (1.0 - minimum) * cosine


def _scheduler_state_action(config: Mapping[str, Any], next_update: int) -> str:
    """Declare the one permitted scheduler-only reset without touching Adam state."""
    if config.get("schedule_mode") != "fresh_cosine_warmup":
        return "LOAD_INHERITED_SCHEDULE"
    if int(config.get("schedule_restart_update", -1)) != 16:
        raise ArtifactError("fresh cosine scheduler reset must bind to the U16 boundary")
    if int(next_update) == 16:
        if config.get("schedule_state_reset") != "fresh_cosine_from_u16_only":
            raise ArtifactError("fresh cosine at U16 requires explicit scheduler-only reset authorization")
        return "RESET_INHERITED_U16_SCHEDULE_ONLY"
    return "LOAD_FRESH_CONTINUATION_SCHEDULE"


def _checkpoint_cursor(payload: Mapping[str, Any]) -> int:
    top_level = payload.get("next_update")
    identity = payload.get("identity")
    identity_value = identity.get("next_update") if isinstance(identity, Mapping) else None
    if top_level is None and identity_value is None:
        raise ArtifactError("official resident checkpoint cursor is missing")
    if (
        top_level is not None
        and identity_value is not None
        and int(top_level) != int(identity_value)
    ):
        raise ArtifactError("official resident checkpoint cursor identity drift")
    cursor = int(top_level if top_level is not None else identity_value)
    if not 0 <= cursor < 64:
        raise ArtifactError("official resident checkpoint cursor must be within U0..U63")
    return cursor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_lut_admission(
    admission: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    materialization_root: Path | None = None,
    manifest_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Admit only original or exact loaded-checkpoint float16 LUT wire bytes."""
    import numpy as np

    document = copy.deepcopy(dict(admission))
    roster = document.get("trainable_roster", {}).get("luts", [])
    checkpoint_luts = state.get("luts")
    rebound: list[dict[str, Any]] = []
    for row in roster:
        layer = int(row["layer"])
        name = str(row["name"])
        source_manifest = row.get("source_manifest")
        if isinstance(source_manifest, Mapping):
            manifest_path = Path(str(source_manifest["path"])).expanduser().resolve()
            if not manifest_path.is_file() and manifest_root is not None:
                candidate = (manifest_root / f"L{layer:03d}" / "parent" / "QTIP_V7_MANIFEST.json").resolve()
                if candidate.is_file() and _sha256_file(candidate) == str(source_manifest["sha256"]):
                    source_manifest["path"] = str(candidate)
        wire = row["wire"]
        path = Path(str(wire["source_path"])).expanduser().resolve()
        observed = _sha256_file(path) if path.is_file() else None
        if observed == str(wire["sha256"]):
            continue
        value = checkpoint_luts.get(name) if isinstance(checkpoint_luts, Mapping) else None
        if value is None:
            raise ArtifactError(f"L{layer:03d} checkpoint-derived LUT has no loaded state")
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        expected_bytes = array.astype("<f2").tobytes()
        expected = hashlib.sha256(expected_bytes).hexdigest()
        if array.shape != (1024,) or len(expected_bytes) != 2048:
            raise ArtifactError(f"L{layer:03d} checkpoint-derived LUT SHA mismatch")
        if observed is None:
            if materialization_root is None:
                raise ArtifactError(f"L{layer:03d} provider LUT is missing and no checkpoint materialization root was declared")
            materialization_root.mkdir(parents=True, exist_ok=True)
            path = (materialization_root / f"L{layer:03d}.{expected}.tlut.f16").resolve()
            if path.exists():
                if not path.is_file() or _sha256_file(path) != expected:
                    raise ArtifactError(f"L{layer:03d} checkpoint LUT materialization collision")
            else:
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                try:
                    with temporary.open("xb") as stream:
                        stream.write(expected_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    temporary.unlink(missing_ok=True)
            observed = expected
            wire["source_path"] = str(path)
        if observed != expected:
            raise ArtifactError(f"L{layer:03d} checkpoint-derived LUT SHA mismatch")
        wire["sha256"] = observed
        rebound.append(
            {
                "layer": layer,
                "name": name,
                "source": "checkpoint_state_float16_wire",
                "sha256": observed,
            }
        )
    return document, rebound


def _cpu_tree(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, item) for item in value)
    return value


def _load_source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"cannot import official resident source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_expert_source_path() -> Path:
    path = Path(__file__).resolve().parents[3] / "runtime" / "v7" / "runner" / "fast_v7_expert_base.py"
    _require_file(path, OFFICIAL_PHYSICAL_LAYER_SHA256, "sealed parity expert source")
    return path


def _bind_official_expert_source() -> Any:
    """Bind the accepted clamp-free, ordered-reduction expert implementation."""
    runner = _official_expert_source_path().parent
    previous = sys.modules.get("fast_k2_grouped")
    try:
        _load_source_module("fast_k2_grouped", runner / "fast_k2_grouped.py")
        return _load_source_module("fast_v7_expert_base", _official_expert_source_path())
    finally:
        if previous is None:
            sys.modules.pop("fast_k2_grouped", None)
        else:
            sys.modules["fast_k2_grouped"] = previous


def _require_file(path: Path, expected_sha: str | None, label: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"official resident {label} is missing: {path}")
    if expected_sha:
        observed = _sha256_file(path)
        if observed != expected_sha:
            raise ArtifactError(f"official resident {label} SHA mismatch: {observed} != {expected_sha}")


def _cuda_sync(torch: Any) -> None:
    torch.cuda.synchronize()


def _score_group_logits(
    lm_head: Any, final: Any, torch: Any, *, offset: int
) -> Any:
    """Project one bounded slice of a large pipeline score group."""
    stop = offset + SCORE_LOGIT_MICROBATCH
    return lm_head(final[offset:stop].to(torch.bfloat16))


def _score_window_groups(windows: tuple[int, ...]) -> list[list[int]]:
    """Use hot batch-one groups for a W28 canary or exact full64 score."""
    if len(windows) not in (1, 64):
        raise ArtifactError("resident physical score requires W28 canary or exactly 64 windows")
    return [
        list(windows[offset : offset + SCORE_MICROBATCH])
        for offset in range(0, len(windows), SCORE_MICROBATCH)
    ]


def _enqueue_rank_send(dist: Any, pending: list[tuple[Any, Any]], tensor: Any) -> None:
    """Keep one activation send in flight while rank0 computes its successor."""
    work = dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, tensor, 1, group=None)]
    )[0]
    pending.append((work, tensor))
    if len(pending) >= 2:
        work, _keepalive = pending.pop(0)
        work.wait()


def _flush_rank_sends(pending: list[tuple[Any, Any]]) -> None:
    for work, _keepalive in pending:
        work.wait()
    pending.clear()


def _recv_rank_activation(dist: Any, activation: Any) -> None:
    """Receive through the same batched P2P path used by rank0.

    NCCL lazy communicator setup requires both peers to use the batched API;
    a blocking ``recv`` on rank1 can wait for a different communicator key
    until the TCPStore timeout while rank0 computes its first score group.
    """
    work = dist.batch_isend_irecv(
        [dist.P2POp(dist.irecv, activation, 0, group=None)]
    )[0]
    work.wait()


class ModernGreenResidentEngine:
    """One rank of the accepted two-Spark resident grouped-K2 trainer."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        config: Mapping[str, Any],
        rank: int,
        layer_ranges: Mapping[int, tuple[int, int]],
    ) -> None:
        try:
            import torch
            import torch.distributed as dist
            from torch.utils.checkpoint import checkpoint
        except ImportError as exc:
            raise ArtifactError("official resident continuation requires PyTorch") from exc
        self.torch = torch
        self.dist = dist
        self.checkpoint = checkpoint
        self.rank = rank
        self.config = config
        self.score_only = config.get("score_only") is True
        self.lr_scale = _validated_lr_scale(config)
        self.base_lrs = {
            name: value * self.lr_scale for name, value in _validated_base_lrs(config).items()
        }
        self.pipeline_microbatch = int(config.get("pipeline_microbatch", PIPELINE_MICROBATCH))
        if self.pipeline_microbatch != PIPELINE_MICROBATCH:
            raise ArtifactError("official resident engine requires PIPELINE_MICROBATCH=4")
        self.device = torch.device(str(config.get("device", "cuda")))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ArtifactError("official resident student requires a CUDA device")
        torch.cuda.set_device(int(config.get("cuda_device", 0)))
        # Establish the NCCL communicator before rank-specific shard construction.
        # Rank0's parent materialization is much slower than rank1's and must not
        # leave rank1 lazily opening a peer connection that expires first.
        self._init_distributed()
        self.layer_ranges = layer_ranges
        self.first, self.last = layer_ranges[rank]
        self.payload = payload
        self.state = payload.get("state")
        if not isinstance(self.state, Mapping):
            raise ArtifactError("U16 checkpoint state must contain official trainable surfaces")
        if set(self.state) != {"luts", "norms", "outputs"}:
            raise ArtifactError("U16 state must contain exactly luts, norms, and outputs")
        self.trainer_path = Path(str(config["trainer_source"])).expanduser().resolve()
        _require_file(self.trainer_path, str(config.get("trainer_source_sha256", TRAINER_SHA256)), "trainer source")
        self.model_root = Path(str(config["model_root"])).expanduser().resolve()
        self.asset_root = Path(str(config["asset_root"])).expanduser().resolve()
        self.member_roster = Path(str(config["member_roster"])).expanduser().resolve()
        self.member_roster_sha256 = str(config["member_roster_sha256"])
        self.teacher_root = Path(str(config["teacher_root"])).expanduser().resolve()
        self.corpus_path = Path(str(config["corpus"])).expanduser().resolve()
        self.manifest_path = Path(str(config["manifest"])).expanduser().resolve()
        self.delta_dir = Path(str(config["delta_dir"])).expanduser().resolve()
        self.vq3b_dir = Path(str(config["vq3b_dir"])).expanduser().resolve()
        self._configure_import_environment()
        self._prepare_import_paths()
        _bind_official_expert_source()
        self.trainer = _load_source_module(
            f"banana_smasher_modern_green_api_{os.getpid()}_{rank}", self.trainer_path
        )
        if getattr(self.trainer, "MODEL_INDEX_SHA256", None) != MODEL_INDEX_SHA256:
            raise ArtifactError("official trainer model-index identity drift")
        self._prepare_import_paths()
        self.base = self._load_base()
        try:
            from banana_smasher import qtip_k2 as official_k2
        except Exception as exc:
            raise ArtifactError(f"official grouped-K2 backend is unavailable: {exc}") from exc
        self.official_k2 = official_k2
        self.model_root = Path(str(config["model_root"])).expanduser().resolve()
        self.asset_root = Path(str(config["asset_root"])).expanduser().resolve()
        self.member_roster = Path(str(config["member_roster"])).expanduser().resolve()
        self.member_roster_sha256 = str(config["member_roster_sha256"])
        self.teacher_root = Path(str(config["teacher_root"])).expanduser().resolve()
        self.corpus_path = Path(str(config["corpus"])).expanduser().resolve()
        _require_file(self.model_root / "model.safetensors.index.json", MODEL_INDEX_SHA256, "model index")
        admission_path = self.asset_root / "code" / "JOINT_REPAIR_ADMISSION.json"
        _require_file(admission_path, str(config.get("admission_sha256", ADMISSION_SHA256)), "joint admission")
        _require_file(self.corpus_path, str(config.get("corpus_sha256", CORPUS_SHA256)), "training corpus")
        if not self.teacher_root.is_dir():
            raise ArtifactError(f"official resident teacher root is missing: {self.teacher_root}")
        admission = json.loads(admission_path.read_text())
        if admission.get("framework") != "banana-smasher":
            raise ArtifactError("official resident admission framework drift")
        if len(admission.get("trainable_roster", {}).get("luts", [])) != 43:
            raise ArtifactError("official resident LUT roster drift")
        admission, self.checkpoint_lut_provider_bindings = _checkpoint_lut_admission(
            admission,
            self.state,
            materialization_root=(
                Path(str(config["checkpoint_lut_root"])).expanduser().resolve()
                if config.get("checkpoint_lut_root")
                else None
            ),
            manifest_root=(
                Path(str(config["provider_manifest_root"])).expanduser().resolve()
                if config.get("provider_manifest_root")
                else None
            ),
        )
        self._configure_base()
        self.status: dict[str, Any] = {}
        parent_root = Path(str(config.get("parent_root", ""))).expanduser().resolve()
        if not parent_root.is_dir():
            raise ArtifactError(f"official resident parent root is missing: {parent_root}")
        self.student = _construct_shard_student(
            self.trainer,
            torch=torch,
            np=__import__("numpy"),
            base=self.base,
            official_k2=official_k2,
            model_root=self.model_root,
            admission=admission,
            parent_root=parent_root,
            member_roster_path=self.member_roster,
            member_roster_sha256=self.member_roster_sha256,
            payload=payload,
            rank=rank,
            first=self.first,
            last=self.last,
            status_cb=self._status,
        )
        self.luts, self.norms, self.outputs = self.trainer.expose_local_dense(torch, self.student, admission)
        self._load_local_trainable_state()
        # Construction has its own immutable admission path and can transiently
        # exceed the 112 GiB rail if the score backend is activated early. Select
        # Quack only after the resident payload is complete, before any forward.
        _select_trainer_fwht(self.trainer)
        self.optimizer = torch.optim.Adam(
            [
                {"params": [p for _name, p in self.luts], "lr": self.base_lrs["luts"], "group_name": "luts"},
                {"params": [p for _name, p in self.norms], "lr": self.base_lrs["norms"], "group_name": "norms"},
                {"params": [p for _name, p in self.outputs], "lr": self.base_lrs["outputs"], "group_name": "outputs"},
            ],
            foreach=False,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[lambda step: _schedule_multiplier(self.config, step, self.trainer.current_multiplier)] * 3,
        )
        self._load_optimizer_scheduler_state()
        self._load_training_data()
        self.global_step = _checkpoint_cursor(payload)

    def _prepare_import_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        paths = [
            self.trainer_path.parent,
            self.asset_root / "source",
            self.asset_root / "source" / "site",
            repository_root / "runtime" / "v7" / "vendor" / "src_lp4",
            repository_root / "runtime" / "v7" / "vendor" / "src",
        ]
        dependency_value = getattr(self, "config", {}).get("trainer_dependency_root")
        if dependency_value is not None:
            dependency_root = Path(str(dependency_value)).expanduser().resolve()
            expected = self.config.get("trainer_dependency_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ArtifactError("trainer dependency SHA is required")
            _require_file(
                dependency_root / "fast_k2_grouped.py",
                expected,
                "trainer grouped-K2 dependency",
            )
            paths.append(dependency_root)
        for path in paths:
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def _configure_import_environment(self) -> None:
        """Bind the immutable base loader before importing its source module."""
        os.environ["BR_MANIFEST"] = str(self.manifest_path)
        os.environ["BR_DELTA_DIR"] = str(self.delta_dir)
        os.environ["BR_VQ3B_DIR"] = str(self.vq3b_dir)
        os.environ["BR_CORPUS"] = str(self.corpus_path)
        os.environ["BR_TEACH"] = str(self.teacher_root)
        os.environ.setdefault("BR_TRAIN", "20,21,22,23")
        os.environ.setdefault("BR_PROBE", "20,21,22,23")

    def _load_base(self) -> Any:
        path = self.asset_root / "source" / "base_binrepair_e2e.py"
        expected = self.config.get("base_source_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ArtifactError("base resident model source SHA is required")
        _require_file(path, expected, "base resident model source")
        module = _load_source_module(f"banana_smasher_modern_green_base_{os.getpid()}_{self.rank}", path)
        module.T.CKPT = str(self.model_root)
        module.T.DEV = "cuda"
        return module

    def _configure_base(self) -> None:
        os.environ["BR_CORPUS"] = str(self.corpus_path)
        os.environ["BR_TEACH"] = str(self.teacher_root)
        attention = str(self.config.get("attention_implementation", "eager"))
        if attention != "eager":
            raise ArtifactError(
                "canonical raw-U0 resident scoring requires A1-equivalent eager attention"
            )
        os.environ["BR_ATTN_IMPL"] = attention
        os.environ.setdefault("BR_FAST_STACK", "1")
        self.base.T.CKPT = str(self.model_root)
        self.base.T.DEV = "cuda"
        import random
        random.seed(1701)
        self.torch.manual_seed(1701)
        self.torch.cuda.manual_seed_all(1701)

    def _status(self, **fields: Any) -> None:
        self.status.update(fields)

    def _load_local_trainable_state(self) -> None:
        loader = self.trainer.load_local_state
        for surface, rows in (("luts", self.luts), ("norms", self.norms), ("outputs", self.outputs)):
            saved = self.state.get(surface)
            if not isinstance(saved, Mapping):
                raise ArtifactError(f"U16 checkpoint missing official {surface} state")
            try:
                loader(rows, saved, self.student.device)
            except Exception as exc:
                raise ArtifactError(f"U16 official {surface} state cannot load: {exc}") from exc

    def _load_optimizer_scheduler_state(self) -> None:
        if self.score_only:
            self.scheduler_state_action = "SCORE_ONLY_NO_TRAINING_LINEAGE"
            return
        optimizer_payload = self.payload.get("optimizer", self.payload.get("optimizer_state"))
        if not isinstance(optimizer_payload, Mapping):
            raise ArtifactError("U16 checkpoint is missing the shared Adam optimizer state")
        groups = optimizer_payload.get("param_groups")
        global_state = optimizer_payload.get("state")
        if not isinstance(groups, list) or len(groups) != 3 or not isinstance(global_state, Mapping):
            raise ArtifactError("U16 Adam state has no canonical three-surface lineage")
        local_state = self.optimizer.state_dict()
        local_groups = local_state["param_groups"]
        local_rows = {"luts": self.luts, "norms": self.norms, "outputs": self.outputs}
        for index, surface in enumerate(("luts", "norms", "outputs")):
            names = [name for name, _param in local_rows[surface]]
            global_names = list(self.state[surface])
            source_group = groups[index]
            source_ids = list(source_group.get("params", []))
            if len(source_ids) != len(global_names):
                raise ArtifactError(f"U16 optimizer {surface} names/IDs do not match")
            global_id_by_name = dict(zip(global_names, source_ids))
            local_ids = local_groups[index]["params"]
            for name, local_id in zip(names, local_ids):
                global_id = global_id_by_name.get(name)
                if global_id is None:
                    raise ArtifactError(f"U16 optimizer state missing official parameter {name}")
                value = global_state.get(global_id, global_state.get(str(global_id)))
                if value is not None:
                    local_state["state"][local_id] = _cpu_tree(self.torch, value)
            local_groups[index].update({key: value for key, value in source_group.items() if key != "params"})
            local_groups[index]["params"] = local_ids
        self.optimizer.load_state_dict(local_state)
        scheduler_payload = self.payload.get("scheduler", self.payload.get("scheduler_state"))
        if not isinstance(scheduler_payload, Mapping):
            raise ArtifactError("U16 checkpoint is missing the shared LambdaLR scheduler state")
        self.scheduler_state_action = _scheduler_state_action(
            self.config, int(self.payload.get("next_update", 16))
        )
        if self.scheduler_state_action == "RESET_INHERITED_U16_SCHEDULE_ONLY":
            return
        try:
            self.scheduler.load_state_dict(dict(scheduler_payload))
        except Exception as exc:
            raise ArtifactError(f"U16 LambdaLR state cannot load: {exc}") from exc

    def _load_training_data(self) -> None:
        training = [] if getattr(self, "score_only", False) else _training_window_ids(self.config)
        score_windows = [int(value) for value in self.config.get("score_windows", ())]
        train_corpus = str(
            self.config["train_corpus"]
            if "train_corpus" in self.config
            else self.corpus_path
        )
        train_teachers = str(
            self.config["train_teacher_root"]
            if "train_teacher_root" in self.config
            else self.teacher_root
        )
        score_corpus = str(self.config.get("score_corpus", train_corpus))
        score_teachers = str(self.config.get("score_teacher_root", train_teachers))
        original_corpus = self.base.T.CORPUS
        original_teachers = self.base.T.TEACH
        try:
            self.base.T.CORPUS = train_corpus
            self.base.T.TEACH = train_teachers
            corpus = self.base.T.load_corpus()
            self.ids_cache = {
                window: self.base.T.window_ids(corpus, window)[0]
                .unsqueeze(0)
                .to(self.student.device)
                for window in training
            }
            self.real_lengths = {
                window: self.base.T.window_ids(corpus, window)[1]
                for window in training
            }
            self.teacher_cache = {}
            if self.rank == 1:
                self.teacher_cache = {
                    window: self.base.T.teacher_rows(window) for window in training
                }

            self.base.T.CORPUS = score_corpus
            self.base.T.TEACH = score_teachers
            corpus = self.base.T.load_corpus()
            self.score_ids_cache = {
                window: self.base.T.window_ids(corpus, window)[0]
                .unsqueeze(0)
                .to(self.student.device)
                for window in score_windows
            }
            self.score_real_lengths = {
                window: self.base.T.window_ids(corpus, window)[1]
                for window in score_windows
            }
            self.score_teacher_cache = {}
            if self.rank == 1:
                self.score_teacher_cache = {
                    window: self.base.T.teacher_rows(window) for window in score_windows
                }
        finally:
            self.base.T.CORPUS = original_corpus
            self.base.T.TEACH = original_teachers

    def _teacher_support(
        self,
        window: int,
        length: int,
        *,
        exact_rows: bool = False,
        score: bool = False,
    ) -> tuple[Any, Any, Any]:
        cache = self.score_teacher_cache if score else self.teacher_cache
        idx, lp_n, p_n = cache[window]
        shapes = [tuple(value.shape) for value in (idx, lp_n, p_n)]
        if any(
            len(shape) != 2
            or shape[1] != 8192
            or (shape[0] != length if exact_rows else shape[0] < length)
            for shape in shapes
        ):
            raise ArtifactError(
                f"resident teacher window {window} must have exact {length}x8192 support"
            )
        return idx, lp_n, p_n

    def score_balanced64(self, windows: Any) -> dict[str, Any]:
        """Score a W28 canary or full64 through one hot batch-one resident rail."""
        selected = tuple(int(value) for value in windows)
        if len(selected) not in (1, 64) or len(set(selected)) != len(selected):
            raise ArtifactError(
                "resident physical score requires W28 canary or 64 unique ordered windows"
            )
        missing = [window for window in selected if window not in self.score_ids_cache]
        if missing:
            raise ArtifactError(f"resident physical score windows were not preloaded: {missing}")
        started = time.perf_counter()
        pending_sends: list[tuple[Any, Any]] = []
        forward_seconds = 0.0
        readout_seconds = 0.0
        glue_seconds = 0.0
        torch = self.torch
        accumulator = ResidentScoreAccumulator(torch) if self.rank == 1 else None
        local_score: dict[str, Any] | None = None
        with torch.inference_mode():
            for group in _score_window_groups(selected):
                ids = torch.cat(
                    [self.score_ids_cache[window] for window in group], dim=0
                )
                shape = (
                    len(group),
                    self.base.T.T_TRAIN,
                    int(self.student.config.hc_mult),
                    int(self.student.config.hidden_size),
                )
                if self.rank == 0:
                    compute_started = time.perf_counter()
                    embeds = self.student.model.model.embed_tokens(ids)
                    hidden = embeds.unsqueeze(2).expand(
                        -1, -1, self.student.config.hc_mult, -1
                    ).contiguous()
                    hidden = self._run_layers(hidden, ids, False)
                    if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                        raise ArtifactError(
                            f"official score activation geometry drift: {tuple(hidden.shape)} {hidden.dtype}"
                        )
                    forward_seconds += time.perf_counter() - compute_started
                    glue_started = time.perf_counter()
                    _enqueue_rank_send(self.dist, pending_sends, hidden.contiguous())
                    glue_seconds += time.perf_counter() - glue_started
                else:
                    activation = torch.empty(
                        shape, dtype=torch.bfloat16, device=self.student.device
                    )
                    glue_started = time.perf_counter()
                    _recv_rank_activation(self.dist, activation)
                    glue_seconds += time.perf_counter() - glue_started
                    compute_started = time.perf_counter()
                    hidden = self._run_layers(activation, ids, False)
                    final = self.student.model.model.norm(
                        self.student.model.model.hc_head(hidden)
                    )
                    forward_seconds += time.perf_counter() - compute_started
                    lengths = [int(self.score_real_lengths[window]) for window in group]
                    if any(length != 1024 for length in lengths):
                        raise ArtifactError(
                            f"resident Balanced64 group has non-1024 lengths: {lengths}"
                        )
                    for row, window in enumerate(group):
                        length = int(self.score_real_lengths[window])
                        idx, lp_n, _p_n = self._teacher_support(
                            window, length, exact_rows=True, score=True
                        )
                        readout_started = time.perf_counter()
                        q_lp, q_argmax = score_terminal_hidden(
                            final[row, :length],
                            idx[:length],
                            self.student.model.lm_head,
                            chunk_size=128,
                            compute_dtype=torch.bfloat16,
                        )
                        readout_seconds += time.perf_counter() - readout_started
                        glue_started = time.perf_counter()
                        assert accumulator is not None
                        accumulator.add(
                            window, idx[:length], lp_n[:length], q_lp, q_argmax
                        )
                        glue_seconds += time.perf_counter() - glue_started
                        del q_lp, q_argmax
            if self.rank == 0:
                glue_started = time.perf_counter()
                _flush_rank_sends(pending_sends)
                glue_seconds += time.perf_counter() - glue_started
            else:
                glue_started = time.perf_counter()
                assert accumulator is not None
                local_score = accumulator.finalize()
                glue_seconds += time.perf_counter() - glue_started
        gathered: list[Any] = [None, None]
        glue_started = time.perf_counter()
        self.dist.all_gather_object(gathered, local_score if self.rank == 1 else None)
        glue_seconds += time.perf_counter() - glue_started
        score = gathered[1]
        if not isinstance(score, dict) or len(score.get("per_window", [])) != len(selected):
            raise ArtifactError("rank1 resident score did not publish complete rows")
        positions = int(score["positions"])
        if positions != len(selected) * 1024:
            raise ArtifactError("resident physical score position count drift")
        _cuda_sync(torch)
        elapsed = time.perf_counter() - started
        return {
            "mean_kld": float(score["mean_kld"]),
            "kld_sum": float(score["kld_sum"]),
            "top1_matches": int(score["top1_matches"]),
            "positions": positions,
            "per_window": list(score["per_window"]),
            "checkpoint": f"UPDATE_{self.global_step:03d}",
            "timed_wall_seconds": elapsed,
            "execution_mode": "resident_model_in_memory",
            "runtime_counters": {
                "model_constructions": 1,
                "checkpoint_loads_during_score": 0,
                "candidate_file_reads_during_score": 0,
                "windows": len(selected),
                "forward_seconds": forward_seconds,
                "readout_seconds": readout_seconds,
                "glue_seconds": glue_seconds,
            },
        }

    def _init_distributed(self) -> None:
        if self.dist.is_initialized():
            if self.dist.get_world_size() != 2 or self.dist.get_rank() != self.rank:
                raise ArtifactError("existing process group does not match the exact two-Spark rank")
            self.dist.barrier()
            return
        master_addr = str(self.config.get("master_addr", "127.0.0.1"))
        master_port = int(self.config.get("master_port", 29598))
        init_method = str(self.config.get("init_method", f"tcp://{master_addr}:{master_port}"))
        socket_interface = self.config.get("distributed_socket_interface")
        if socket_interface is not None:
            socket_interface = str(socket_interface).strip()
            if not socket_interface:
                raise ArtifactError("distributed_socket_interface must be non-empty")
            os.environ["NCCL_SOCKET_IFNAME"] = socket_interface
        try:
            self.dist.init_process_group(
                backend=str(self.config.get("distributed_backend", "nccl")),
                init_method=init_method,
                rank=self.rank,
                world_size=2,
            )
            self.dist.barrier()
        except Exception as exc:
            raise ArtifactError(f"official two-Spark process-group initialization failed: {exc}") from exc

    def _positional(self, ids: Any, template: Any, cache: Any) -> tuple[Any, Any, Any]:
        pos = self.torch.arange(ids.shape[1], device=self.student.device).unsqueeze(0)
        from transformers.masking_utils import create_sliding_window_causal_mask
        embeddings = self.student.model.model.rotary_emb
        pe = {
            "main": embeddings(template, position_ids=pos, layer_type="main"),
            "compress": embeddings(template, position_ids=pos, layer_type="compress"),
        }
        mask = create_sliding_window_causal_mask(
            config=self.student.config,
            inputs_embeds=template,
            attention_mask=None,
            past_key_values=cache,
            position_ids=pos,
        )
        return pos, pe, mask

    def _run_layers(self, hidden: Any, ids: Any, train: bool) -> Any:
        from transformers.cache_utils import DynamicCache
        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        mask_cache = DynamicCache(config=self.student.config)
        pos, pe, mask = self._positional(ids, template, mask_cache)
        for index in range(self.first, self.last + 1):
            layer = self.student.model.model.layers[index]
            def layer_fn(current: Any, layer: Any = layer) -> Any:
                return layer(
                    current,
                    position_embeddings=pe,
                    position_ids=pos,
                    attention_mask=mask,
                    input_ids=ids,
                    # The sealed A1 builder uses one fresh cache per layer for a
                    # full-sequence prefill. Reusing a layer-indexed cache here
                    # changes the physical activation and was rejected by the
                    # public-path parity gate.
                    past_key_values=DynamicCache(config=self.student.config),
                )
            if train:
                hidden = self.checkpoint(layer_fn, hidden, use_reentrant=False)
            else:
                hidden = layer_fn(hidden)
        return hidden

    def _loss_group(self, hidden: Any, group: list[int]) -> Any:
        final = self.student.model.model.norm(self.student.model.model.hc_head(hidden))
        losses = []
        for row, window in enumerate(group):
            length = self.real_lengths[window]
            idx, lp_n, p_n = self._teacher_support(window, length)
            logits = self.student.model.lm_head(final[row, :length].to(self.torch.bfloat16))
            q = logits.gather(1, idx[:length]).float()
            qn = q - q.logsumexp(-1, keepdim=True)
            losses.append((p_n[:length] * (lp_n[:length] - qn)).sum(-1).mean())
        return self.torch.stack(losses).mean()

    def _pipeline_pass(self, group: list[int], *, gradient_scale: float = 1.0) -> tuple[float | None, dict[str, float]]:
        if len(group) != self.pipeline_microbatch:
            raise ArtifactError(f"official pipeline group must contain {self.pipeline_microbatch} windows")
        torch = self.torch
        ids = torch.cat([self.ids_cache[window] for window in group], dim=0)
        shape = (self.pipeline_microbatch, self.base.T.T_TRAIN, int(self.student.config.hc_mult), int(self.student.config.hidden_size))
        if self.rank == 0:
            started = time.perf_counter()
            embeds = self.student.model.model.embed_tokens(ids)
            hidden = embeds.unsqueeze(2).expand(-1, -1, self.student.config.hc_mult, -1).contiguous()
            hidden = self._run_layers(hidden, ids, True)
            _cuda_sync(torch)
            forward_seconds = time.perf_counter() - started
            if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                raise ArtifactError(f"official pipeline activation geometry drift: {tuple(hidden.shape)} {hidden.dtype}")
            torch.distributed.send(hidden.detach().contiguous(), dst=1)
            grad = torch.empty_like(hidden)
            torch.distributed.recv(grad, src=1)
            backward_started = time.perf_counter()
            hidden.backward(grad)
            _cuda_sync(torch)
            backward_seconds = time.perf_counter() - backward_started
            return None, {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds}
        activation = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
        receive_started = time.perf_counter()
        torch.distributed.recv(activation, src=0)
        activation.requires_grad_(True)
        hidden = self._run_layers(activation, ids, True)
        loss = self._loss_group(hidden, group)
        _cuda_sync(torch)
        forward_seconds = time.perf_counter() - receive_started
        backward_started = time.perf_counter()
        (loss * gradient_scale).backward()
        _cuda_sync(torch)
        backward_seconds = time.perf_counter() - backward_started
        if activation.grad is None:
            raise ArtifactError("official pipeline boundary gradient is missing")
        torch.distributed.send(activation.grad.contiguous(), dst=0)
        return float(loss.detach().cpu()), {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds}

    def _local_params(self) -> list[tuple[str, Any]]:
        return [*self.luts, *self.norms, *self.outputs]

    def _local_norm(self, values: list[Any]) -> float:
        return sum(float(value.detach().float().pow(2).sum().cpu()) for value in values) ** 0.5

    def _step(self, global_step: int) -> dict[str, Any]:
        torch = self.torch
        params = self._local_params()
        before = [parameter.detach().clone() for _name, parameter in params]
        self.optimizer.zero_grad(set_to_none=True)
        multiplier = _schedule_multiplier(self.config, global_step, self.trainer.current_multiplier)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lrs[group["group_name"]] * multiplier
        applied_learning_rates = {
            str(group["group_name"]): float(group["lr"])
            for group in self.optimizer.param_groups
        }
        microbatches = _window_microbatches(self.config, global_step)
        group_windows = [window for group in microbatches for window in group]
        dist_started = time.perf_counter()
        losses: list[float] = []
        timing_rows: list[dict[str, float]] = []
        gradient_scale = 1.0 / len(microbatches)
        for group in microbatches:
            loss_value, timing = self._pipeline_pass(group, gradient_scale=gradient_scale)
            if loss_value is not None:
                losses.append(loss_value)
            timing_rows.append(timing)
        loss = sum(losses) / len(losses) if losses else None
        timing = {
            key: sum(float(row[key]) for row in timing_rows)
            for key in ("forward_seconds", "backward_seconds")
        }
        forward_backward_seconds = time.perf_counter() - dist_started
        optimizer_started = time.perf_counter()
        self.optimizer.step()
        self.scheduler.step()
        _cuda_sync(torch)
        optimizer_seconds = time.perf_counter() - optimizer_started
        gradients = [parameter.grad for _name, parameter in params if parameter.grad is not None]
        gradient_norm = self._local_norm(gradients)
        delta_norm = self._local_norm([parameter.detach() - old for (_name, parameter), old in zip(params, before)])
        local = {
            "rank": self.rank,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "parameter_delta_norm": delta_norm,
            "timings": {
                "forward_seconds": timing["forward_seconds"],
                "backward_seconds": timing["backward_seconds"],
                "optimizer_seconds": optimizer_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "wall_seconds": forward_backward_seconds + optimizer_seconds,
            },
            "windows": group_windows,
            "window_microbatches": microbatches,
            "windows_per_update": len(group_windows),
            "gradient_average_divisor": len(microbatches),
            "lr_scale": self.lr_scale,
            "schedule_mode": self.config.get("schedule_mode", "loaded_global_cosine"),
            "schedule_multiplier": multiplier,
            "scheduler_cursor_before_update": global_step,
            "configured_learning_rates": {
                name: float(value * multiplier) for name, value in self.base_lrs.items()
            },
            "realized_learning_rates": applied_learning_rates,
            "base_learning_rates": dict(self.base_lrs),
            "nonzero_gradients": sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients),
            "trainable_tensors": len(params),
            "process_gpu_evidence": {
                "pid": os.getpid(),
                "hostname": os.uname().nodename,
                "rank": self.rank,
                "device": str(self.student.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            },
        }
        rows: list[Any] = [None, None]
        self.dist.all_gather_object(rows, local)
        global_gradient = sum(float(row["gradient_norm"]) ** 2 for row in rows) ** 0.5
        global_delta = sum(float(row["parameter_delta_norm"]) ** 2 for row in rows) ** 0.5
        losses = [row["loss"] for row in rows if row["loss"] is not None]
        local["gradient_norm"] = global_gradient
        local["parameter_delta_norm"] = global_delta
        local["loss"] = losses[0] if losses else None
        local["timings"] = {key: max(float(row["timings"][key]) for row in rows) for key in local["timings"]}
        local["rank_reports"] = rows
        if global_gradient <= 0.0 or global_delta <= 0.0 or not losses:
            raise ArtifactError(f"official resident U{global_step + 1} produced no real gradient/delta")
        return local

    def _gather_state(self) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any]]:
        torch = self.torch
        rows: list[Any] = [None, None]
        local_params = {"luts": self.luts, "norms": self.norms, "outputs": self.outputs}
        local_state = {
            "rank": self.rank,
            **{surface: {name: parameter.detach().cpu().clone() for name, parameter in values} for surface, values in local_params.items()},
            "param_names": {surface: [name for name, _parameter in values] for surface, values in local_params.items()},
            "optimizer": _cpu_tree(torch, self.optimizer.state_dict()),
        }
        self.dist.all_gather_object(rows, local_state)
        if self.rank != 0:
            self.dist.barrier()
            return None, None, {"rank_rows": rows}
        merged = {"luts": {}, "norms": {}, "outputs": {}}
        for row in rows:
            for surface in merged:
                overlap = set(merged[surface]) & set(row[surface])
                if overlap:
                    raise ArtifactError(f"official resident state overlap: {surface} {sorted(overlap)[:3]}")
                merged[surface].update(row[surface])
        if {surface: len(values) for surface, values in merged.items()} != {"luts": 43, "norms": 235, "outputs": 43}:
            raise ArtifactError("official resident merged trainable surface coverage drift")
        optimizer = self.trainer.merge_optimizer_state(rows, merged)
        scheduler = _cpu_tree(torch, self.scheduler.state_dict())
        report = {"rank_rows": rows, "optimizer": optimizer, "scheduler": scheduler}
        self.dist.barrier()
        return merged, optimizer, report

    def advance_to(self, target_update: int) -> tuple[Mapping[str, Any] | None, dict[str, Any], Mapping[str, Any] | None]:
        start = self.global_step
        if target_update <= start:
            raise ArtifactError("official resident target must advance beyond current update")
        last: dict[str, Any] | None = None
        merged_state: Mapping[str, Any] | None = None
        optimizer_state: Mapping[str, Any] | None = None
        report_state: Mapping[str, Any] | None = None
        update_reports: list[dict[str, Any]] = []
        for update in range(start, target_update):
            last = self._step(update)
            update_reports.append(
                {
                    "global_update": update + 1,
                    "windows": list(last["windows"]),
                    "window_microbatches": [list(group) for group in last["window_microbatches"]],
                    "windows_per_update": int(last["windows_per_update"]),
                    "gradient_average_divisor": int(last["gradient_average_divisor"]),
                    "base_learning_rates": dict(last["base_learning_rates"]),
                    "schedule_mode": last["schedule_mode"],
                    "schedule_multiplier": float(last["schedule_multiplier"]),
                    "scheduler_cursor_before_update": int(last["scheduler_cursor_before_update"]),
                    "configured_learning_rates": dict(last["configured_learning_rates"]),
                    "realized_learning_rates": dict(last["realized_learning_rates"]),
                }
            )
            self.global_step = update + 1
        merged_state, optimizer_state, report_state = self._gather_state()
        if last is None:
            raise ArtifactError("official resident continuation performed no steps")
        step_report = {
            "resident_optimizer_step": True,
            "optimizer_steps": target_update - start,
            "scheduler_steps": target_update - start,
            "checkpoint_loaded": True,
            "gradient_norm": last["gradient_norm"],
            "parameter_delta_norm": last["parameter_delta_norm"],
            "loss": last["loss"],
            "timings": last["timings"],
            "windows": last["windows"],
            "process_gpu_evidence": last["process_gpu_evidence"],
            "rank_reports": last["rank_reports"],
            "rank_provenance": [int(row["rank"]) for row in last["rank_reports"]],
            "sampling_plan": _sampling_plan(self.config, start, target_update),
            "actual_update_reports": update_reports,
            "scheduler_state_action": self.scheduler_state_action,
            "optimizer_state": optimizer_state,
            "scheduler_state": (report_state or {}).get("scheduler") if isinstance(report_state, Mapping) else None,
            "model_engine": "official-ShardStudent-grouped-K2-FWHT-resident",
            "frozen_surfaces": ["packed_codes", "assignments", "scales"],
            "trainable_surfaces": ["luts", "rmsnorms", "output_gains"],
        }
        return merged_state, step_report, report_state

    def broadcast_persisted(self, value: Any) -> Any:
        row = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(row, src=0)
        return row[0]

    def close(self) -> None:
        if self.dist.is_initialized() and self.config.get("destroy_process_group", False):
            self.dist.destroy_process_group()
