"""Official Modern Green grouped-K2 resident continuation engine.

This module is deliberately coupled to the accepted clean-U0 trainer.  It does
not manufacture a loss from checkpoint tensors: it constructs the resident
ShardStudent, routes the real model through both layer partitions, evaluates
the teacher KL objective, and runs the trainer's legal LUT/RMS/gain surface
through Adam and LambdaLR.
"""
from __future__ import annotations

import copy
import gc
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

MODEL_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
ADMISSION_SHA256 = "76d0674eb0cd37fc9022bac5e048c2b77c721826182222ae0a0609e29607a2c5"
CORPUS_SHA256 = "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5"
TRAINER_SHA256 = "126c11f306a12ed35c1234bd12952a32662c3bd81fc2e74361f0a55ebdc21fc0"
OFFICIAL_PHYSICAL_LAYER_SHA256 = "fc612f7863ad9d09a9faf11e203a9d20739b7dbb273b982fc3d36ee01d15a9b4"
WINDOWS_PER_STEP = 4
PIPELINE_MICROBATCH = 4
# Keep score-only attention at the proven per-step memory envelope. Balanced64
# still covers all 64 ordered 1024-token windows; only the transient batch changes.
SCORE_MICROBATCH = 4
SCORE_LOGIT_MICROBATCH = 4
BASE_LRS = {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2}
EXPERT_PLANE_SURFACE = "expert_planes_l028_su_sv"
HISTORICAL_BASE_LRS = {"luts": 2.5e-4, "norms": 2.5e-5, "outputs": 2.5e-4}
HISTORICAL_SAMPLING_MODE = "historical_category_stratified_v1"
HISTORICAL_TRAIN_BANK_SHA256 = "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3"
HISTORICAL_CATEGORIES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _record_cold_start_phase(
    config: Mapping[str, Any],
    *,
    rank: int,
    phase: str,
    boundary: str,
    elapsed_seconds: float | None = None,
) -> None:
    """Append one durable boundary without changing construction behavior."""
    configured = config.get("cold_start_phase_receipt")
    if not configured:
        return
    path = Path(str(configured).format(rank=rank)).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "schema": "banana-smasher-cold-start-phase-v1",
        "rank": int(rank),
        "pid": os.getpid(),
        "phase": phase,
        "boundary": boundary,
        "unix_time": time.time(),
    }
    if elapsed_seconds is not None:
        row["elapsed_seconds"] = float(elapsed_seconds)
    payload = (json.dumps(row, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cold_start_phase(config: Mapping[str, Any], rank: int, phase: str, action: Any) -> Any:
    _record_cold_start_phase(config, rank=rank, phase=phase, boundary="start")
    started = time.perf_counter()
    result = action()
    _record_cold_start_phase(
        config,
        rank=rank,
        phase=phase,
        boundary="complete",
        elapsed_seconds=time.perf_counter() - started,
    )
    return result


def _validated_expert_plane_expansion(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("expert_plane_expansion")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ArtifactError("expert-plane expansion contract must be a mapping")
    if value.get("surface") != EXPERT_PLANE_SURFACE or value.get("layer") != 28:
        raise ArtifactError("expert-plane expansion is restricted to the L028 SU/SV surface")
    if value.get("components") != ["SU", "SV"]:
        raise ArtifactError("expert-plane expansion components must be exactly SU/SV")
    if value.get("projections") != ["w1", "w2", "w3"]:
        raise ArtifactError("expert-plane expansion projections must be exactly w1/w2/w3")
    if value.get("static_w28") is not True or value.get("immutable_wire") is not True:
        raise ArtifactError("expert-plane expansion requires static W28 and immutable wire bytes")
    roster_sha = value.get("roster_sha256")
    if not isinstance(roster_sha, str) or len(roster_sha) != 64 or any(
        character not in "0123456789abcdef" for character in roster_sha
    ):
        raise ArtifactError("expert-plane expansion requires a lowercase SHA-256 roster identity")
    try:
        learning_rate = float(value.get("learning_rate"))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("expert-plane expansion requires a finite positive learning rate") from exc
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ArtifactError("expert-plane expansion requires a finite positive learning rate")
    return dict(value)


def _activate_expert_plane_surface(
    student: Any,
    checkpoint_state: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    checkpoint_cursor: int,
) -> list[tuple[str, Any]]:
    """Promote the exact L028 module and seed PRE or restore persisted masters."""
    experts = getattr(student, "experts", None)
    module = experts.get(28) if isinstance(experts, Mapping) else None
    if module is None:
        return []
    try:
        rows = list(module.promote_l028_su_sv())
    except Exception as exc:
        raise ArtifactError(f"L028 SU/SV promotion failed: {exc}") from exc
    saved = checkpoint_state.get(EXPERT_PLANE_SURFACE)
    if saved is None:
        if checkpoint_cursor != 0:
            raise ArtifactError("continuation checkpoint is missing L028 SU/SV state")
        return rows
    try:
        module.load_expert_plane_state(saved)
    except Exception as exc:
        raise ArtifactError(f"L028 SU/SV checkpoint state cannot load: {exc}") from exc
    return rows


def _classify_expert_plane_update(
    torch: Any,
    rows: list[tuple[str, Any]],
    before: list[Any],
) -> dict[str, Any]:
    """Prove the full promoted roster is trainable and optimizer-consumable.

    Routed experts that receive no token in a finite physical batch legitimately
    have finite zero gradients and deltas.  They remain bound to Adam and are
    persisted; nonzero scientific motion is enforced separately at update scope.
    """
    if len(rows) != len(before):
        raise ArtifactError("L028 SU/SV before-image coverage drift")
    missing_trainable = [
        name for name, parameter in rows if not bool(parameter.requires_grad)
    ]
    missing_gradient = [
        name
        for name, parameter in rows
        if parameter.grad is None
        or not bool(torch.isfinite(parameter.grad).all().item())
    ]
    missing_delta = [
        name
        for (name, parameter), old in zip(rows, before)
        if not bool(torch.isfinite(parameter).all().item())
        or not bool(torch.isfinite(parameter.detach() - old).all().item())
    ]
    gradient_nonzero = sum(
        int(bool(torch.count_nonzero(parameter.grad).item()))
        for _name, parameter in rows
        if parameter.grad is not None
    )
    delta_nonzero = sum(
        int(bool(torch.count_nonzero(parameter.detach() - old).item()))
        for (_name, parameter), old in zip(rows, before)
    )
    return {
        "missing_trainable": missing_trainable,
        "missing_gradient": missing_gradient,
        "missing_delta": missing_delta,
        "gradient_present": f"{len(rows) - len(missing_gradient)}/{len(rows)}",
        "gradient_nonzero": f"{gradient_nonzero}/{len(rows)}",
        "delta_nonzero": f"{delta_nonzero}/{len(rows)}",
        "gradient": f"{len(rows) - len(missing_gradient)}/{len(rows)}",
        "delta": f"{len(rows) - len(missing_delta)}/{len(rows)}",
    }


def _merge_expanded_optimizer_state(
    state_rows: list[Mapping[str, Any]],
    ordered_state: Mapping[str, Mapping[str, Any]],
    surfaces: tuple[str, ...],
    dormant_names: set[str],
) -> dict[str, Any]:
    ordered_names = {surface: list(ordered_state[surface]) for surface in surfaces}
    global_ids: dict[str, int] = {}
    for surface in surfaces:
        for name in ordered_names[surface]:
            if name in global_ids:
                raise ArtifactError(f"optimizer parameter name overlap: {name}")
            global_ids[name] = len(global_ids)
    merged_state: dict[int, Any] = {}
    seen: set[int] = set()
    templates: dict[str, dict[str, Any]] = {}
    for row in state_rows:
        local = row["optimizer"]
        local_names = row["param_names"]
        groups = local.get("param_groups")
        if not isinstance(groups, list) or len(groups) != len(surfaces):
            raise ArtifactError("expanded optimizer parameter-group count drift")
        local_ids_seen: set[int] = set()
        for surface, group in zip(surfaces, groups):
            names = list(local_names[surface])
            ids = list(group.get("params", []))
            if len(names) != len(ids):
                raise ArtifactError(f"expanded optimizer name/id drift: {surface}")
            template = {key: value for key, value in group.items() if key != "params"}
            if surface in templates and templates[surface] != template:
                raise ArtifactError(f"expanded optimizer group setting drift: {surface}")
            templates[surface] = template
            for name, local_id in zip(names, ids):
                if name not in global_ids or local_id in local_ids_seen:
                    raise ArtifactError(f"expanded optimizer parameter identity drift: {name}")
                local_ids_seen.add(local_id)
                global_id = global_ids[name]
                if global_id in seen:
                    raise ArtifactError(f"expanded optimizer parameter overlap: {name}")
                seen.add(global_id)
                value = local["state"].get(local_id, local["state"].get(str(local_id)))
                if value is not None:
                    merged_state[global_id] = value
        dangling = set(local["state"]) - local_ids_seen
        dangling -= {str(value) for value in local_ids_seen}
        if dangling:
            raise ArtifactError("expanded optimizer state has unbound local ids")
    if seen != set(range(len(global_ids))):
        raise ArtifactError("expanded optimizer global parameter coverage drift")
    missing = {name for name, global_id in global_ids.items() if global_id not in merged_state}
    if not missing.issubset(set(global_ids) & dormant_names):
        raise ArtifactError("expanded optimizer sparse-state coverage drift")
    return {
        "state": merged_state,
        "param_groups": [
            {
                **templates[surface],
                "params": [global_ids[name] for name in ordered_names[surface]],
            }
            for surface in surfaces
        ],
    }


def _build_fp64_adam(torch: Any, param_groups: list[dict[str, Any]]) -> Any:
    """Adam with FP64 moments and FP32-or-better update arithmetic.

    PyTorch's stock Adam stores moments in the parameter dtype and casts them
    back to that dtype during ``load_state_dict``. The validated U45 recipe
    requires FP64 moments both before and after resume, so this small optimizer
    keeps the state contract explicit while preserving the standard Adam rule.
    """

    class FP64MomentAdam(torch.optim.Optimizer):
        def __init__(self, groups: list[dict[str, Any]]) -> None:
            super().__init__(
                groups,
                {"lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8},
            )

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            super().load_state_dict(state_dict)
            for state in self.state.values():
                for name in ("exp_avg", "exp_avg_sq"):
                    value = state.get(name)
                    if value is not None:
                        state[name] = value.to(dtype=torch.float64)

        def step(self, closure: Any = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            with torch.no_grad():
                for group in self.param_groups:
                    beta1, beta2 = group["betas"]
                    lr = float(group["lr"])
                    eps = float(group["eps"])
                    for parameter in group["params"]:
                        gradient = parameter.grad
                        if gradient is None:
                            continue
                        if gradient.is_sparse:
                            raise RuntimeError("FP64MomentAdam does not support sparse gradients")
                        state = self.state[parameter]
                        if not state:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(
                                parameter, dtype=torch.float64
                            )
                            state["exp_avg_sq"] = torch.zeros_like(
                                parameter, dtype=torch.float64
                            )
                        state["step"] = int(state["step"]) + 1
                        step = state["step"]
                        grad64 = gradient.detach().to(dtype=torch.float64)
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        exp_avg.mul_(beta1).add_(grad64, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(
                            grad64, grad64, value=1.0 - beta2
                        )
                        step_size = lr / (1.0 - beta1**step)
                        denominator = exp_avg_sq.sqrt().div_(
                            math.sqrt(1.0 - beta2**step)
                        ).add_(eps)
                        update = (exp_avg / denominator).to(dtype=parameter.dtype)
                        parameter.add_(update, alpha=-step_size)
            return loss

    return FP64MomentAdam(param_groups)


def _enforce_update_loss_guard(
    *,
    loss: float,
    baseline: float,
    previous_loss: float | None,
    global_update: int,
) -> dict[str, float | bool | None]:
    current = float(loss)
    initial = float(baseline)
    prior = None if previous_loss is None else float(previous_loss)
    limit = 2.0 * initial
    if not math.isfinite(current) or not math.isfinite(initial) or initial <= 0.0:
        raise ArtifactError(f"resident loss guard received non-finite loss at U{global_update}")
    if current > limit:
        raise ArtifactError(
            f"resident loss explosion at U{global_update}: loss={current:.17g} "
            f"> 2x baseline limit={limit:.17g}"
        )
    return {
        "baseline_loss": initial,
        "loss": current,
        "previous_loss": prior,
        "nonincreasing": prior is None or current <= prior,
        "explosion_limit": limit,
    }


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


def _official_expert_source_path(config: Mapping[str, Any] | None = None) -> Path:
    configured = config.get("resident_expert_source") if config is not None else None
    path = (
        Path(str(configured)).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3]
        / "runtime"
        / "v7"
        / "runner"
        / "fast_v7_expert_base.py"
    )
    expected = (
        config.get("resident_expert_source_sha256", OFFICIAL_PHYSICAL_LAYER_SHA256)
        if config is not None
        else OFFICIAL_PHYSICAL_LAYER_SHA256
    )
    _require_file(path, str(expected), "sealed parity expert source")
    return path


def _bind_official_expert_source(config: Mapping[str, Any] | None = None) -> Any:
    """Bind the accepted clamp-free, ordered-reduction expert implementation."""
    source = (
        _official_expert_source_path()
        if config is None
        else _official_expert_source_path(config)
    )
    if config is not None and isinstance(config.get("mixed_backpack_runtime"), Mapping):
        module = _load_source_module("fast_v7_expert_base", source)
        configure = getattr(module, "configure_mixed_backpack", None)
        if not callable(configure):
            raise ArtifactError(
                "mixed resident expert source lacks configure_mixed_backpack()"
            )
        configure(config)
        return module
    runner = source.parent
    grouped_source = runner / "fast_k2_grouped.py"
    if config is not None and config.get("fast_k2_wrapper_source"):
        grouped_source = Path(str(config["fast_k2_wrapper_source"])).expanduser().resolve()
        grouped_sha = config.get("fast_k2_wrapper_source_sha256")
        if not isinstance(grouped_sha, str) or len(grouped_sha) != 64:
            raise ArtifactError("grouped-K2 wrapper source SHA is required")
        _require_file(grouped_source, grouped_sha, "grouped-K2 wrapper source")
    if config is not None and config.get("fast_k2_extension"):
        extension = Path(str(config["fast_k2_extension"])).expanduser().resolve()
        extension_sha = config.get("fast_k2_extension_sha256")
        if not isinstance(extension_sha, str) or len(extension_sha) != 64:
            raise ArtifactError("grouped-K2 prebuilt extension SHA is required")
        _require_file(extension, extension_sha, "grouped-K2 prebuilt extension")
        module_name = config.get("fast_k2_module_name")
        if not isinstance(module_name, str) or not module_name.isidentifier():
            raise ArtifactError("grouped-K2 prebuilt extension module name is required")
        os.environ["FAST_K2_EXTENSION"] = str(extension)
        os.environ["FAST_K2_EXTENSION_SHA256"] = extension_sha
        os.environ["FAST_K2_MODULE_NAME"] = module_name
    previous = sys.modules.get("fast_k2_grouped")
    try:
        grouped = _load_source_module("fast_k2_grouped", grouped_source)
        bind_stream_sync = getattr(grouped, "bind_backward_stream_sync", None)
        if callable(bind_stream_sync):
            bind_stream_sync(_cuda_default_stream_wait_for_current)
        module = _load_source_module("fast_v7_expert_base", source)
        provider = getattr(module, "FullyResidentGroupedV7Experts", None)
        if (
            config is not None
            and config.get("fast_v7_expert_source_sha256")
            == "0b673aaa31dedaaf604488bb71543e92560167cdef7e6bade50b65b4568b9f81"
            and provider is not None
            and "swiglu_limit" not in inspect.signature(provider).parameters
        ):
            class LegacySwiGLUProvider(provider):
                def __init__(self, *args: Any, swiglu_limit: float = 10.0, **kwargs: Any) -> None:
                    if float(swiglu_limit) != 10.0:
                        raise ArtifactError("legacy grouped provider requires swiglu_limit=10")
                    super().__init__(*args, **kwargs)

            module.FullyResidentGroupedV7Experts = LegacySwiGLUProvider
        return module
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


def _cuda_default_stream_wait_for_current(torch: Any) -> None:
    """Order default-stream gradient consumption after the grouped kernel."""
    producer = torch.cuda.current_stream()
    completed = producer.record_event()
    torch.cuda.default_stream().wait_event(completed)


def _score_group_logits(
    lm_head: Any, final: Any, torch: Any, *, offset: int
) -> Any:
    """Project one bounded slice of a large pipeline score group."""
    stop = offset + SCORE_LOGIT_MICROBATCH
    return lm_head(final[offset:stop].to(torch.bfloat16))


def _score_window_groups(windows: tuple[int, ...]) -> list[list[int]]:
    """Use bounded score-only groups without changing training microbatches."""
    if len(windows) != 64:
        raise ArtifactError("resident physical score requires exactly 64 windows")
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
        self.expert_plane_contract = _validated_expert_plane_expansion(config)
        if self.expert_plane_contract is not None and _historical_mode(config):
            raise ArtifactError("L028 expert-plane expansion requires fresh static-W28 lineage")
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
        _cold_start_phase(config, rank, "init_distributed", self._init_distributed)
        self.layer_ranges = layer_ranges
        self.first, self.last = layer_ranges[rank]
        self.payload = payload
        self.state = payload.get("state")
        if not isinstance(self.state, Mapping):
            raise ArtifactError("U16 checkpoint state must contain official trainable surfaces")
        base_surfaces = {"luts", "norms", "outputs"}
        scale_surface = (
            {"scales"} if config.get("trainable_quantization_scales") is True else set()
        )
        allowed_surfaces = base_surfaces | scale_surface | (
            {EXPERT_PLANE_SURFACE} if self.expert_plane_contract is not None else set()
        )
        admissible_state_keys = (
            (base_surfaces, allowed_surfaces)
            if self.expert_plane_contract is not None
            else (base_surfaces, allowed_surfaces)
        )
        if set(self.state) not in admissible_state_keys:
            raise ArtifactError("resident state trainable-surface schema drift")
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
        _cold_start_phase(config, rank, "configure_import_environment", self._configure_import_environment)
        _cold_start_phase(config, rank, "prepare_import_paths", self._prepare_import_paths)
        _cold_start_phase(config, rank, "bind_official_expert_source", lambda: _bind_official_expert_source(config))
        self.trainer = _cold_start_phase(
            config,
            rank,
            "load_trainer_source",
            lambda: _load_source_module(
                f"banana_smasher_modern_green_api_{os.getpid()}_{rank}", self.trainer_path
            ),
        )
        if getattr(self.trainer, "MODEL_INDEX_SHA256", None) != MODEL_INDEX_SHA256:
            raise ArtifactError("official trainer model-index identity drift")
        _cold_start_phase(config, rank, "prepare_import_paths_after_trainer", self._prepare_import_paths)
        self.base = _cold_start_phase(config, rank, "load_base", self._load_base)
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
        admission, self.checkpoint_lut_provider_bindings = _cold_start_phase(
            config,
            rank,
            "checkpoint_lut_admission",
            lambda: _checkpoint_lut_admission(
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
        self.expert_planes = (
            _activate_expert_plane_surface(
                self.student,
                self.state,
                self.expert_plane_contract,
                checkpoint_cursor=_checkpoint_cursor(payload),
            )
            if self.expert_plane_contract is not None
            else []
        )
        if self.expert_plane_contract is not None:
            expected_local = 1536 if self.first <= 28 <= self.last else 0
            if len(self.expert_planes) != expected_local:
                raise ArtifactError("rank-local L028 SU/SV promoted coverage drift")
            self.base_lrs[EXPERT_PLANE_SURFACE] = float(
                self.expert_plane_contract["learning_rate"]
            ) * self.lr_scale
            for surface in ("luts", "norms", "outputs"):
                self.base_lrs[surface] = 0.0
        self._load_local_trainable_state()
        self.scales = self._configure_trainable_quantization_scales()
        if self.scales:
            self.base_lrs["scales"] = self.base_lrs["luts"]
        # Construction has its own immutable admission path and can transiently
        # exceed the 112 GiB rail if the score backend is activated early. Select
        # Quack only after the resident payload is complete, before any forward.
        _select_trainer_fwht(self.trainer)
        self.optimizer: Any = None
        self.scheduler: Any = None
        if not self.score_only:
            self.optimizer = _build_fp64_adam(
                torch,
                [
                    {"params": [p for _name, p in self.luts], "lr": self.base_lrs["luts"], "group_name": "luts"},
                    {"params": [p for _name, p in self.norms], "lr": self.base_lrs["norms"], "group_name": "norms"},
                    {"params": [p for _name, p in self.outputs], "lr": self.base_lrs["outputs"], "group_name": "outputs"},
                    *(
                        [{"params": [p for _name, p in self.scales], "lr": self.base_lrs["scales"], "group_name": "scales"}]
                        if self.scales else []
                    ),
                    *(
                        [{"params": [p for _name, p in self.expert_planes], "lr": self.base_lrs[EXPERT_PLANE_SURFACE], "group_name": EXPERT_PLANE_SURFACE}]
                        if self.expert_plane_contract is not None
                        else []
                    ),
                ],
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=[lambda step: _schedule_multiplier(self.config, step, self.trainer.current_multiplier)]
                * (3 + int(bool(self.scales)) + int(self.expert_plane_contract is not None)),
            )
            self._load_optimizer_scheduler_state()
        self._load_training_data()
        # CUDA unified-memory allocations are not effectively bounded by the
        # systemd MemoryMax cgroup on DGX Spark.  Construction and checkpoint
        # restoration can leave reclaimable allocator pages resident; return
        # them before the first backward so the validated U45 step starts from
        # the measured live-tensor envelope rather than a stale cache peak.
        self.torch.cuda.empty_cache()
        self.global_step = _checkpoint_cursor(payload)

    def memory_ledger(self) -> dict[str, Any]:
        """Account unique CUDA parameter/buffer storage by immediate module."""
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        student = getattr(self, "student", None)
        model = getattr(student, "model", None)
        if model is not None:
            for name, module in model.named_modules():
                total = 0
                for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
                    if tensor is None or tensor.device.type != "cuda" or tensor.is_meta:
                        continue
                    storage = tensor.untyped_storage()
                    key = (int(storage.data_ptr()), int(storage.nbytes()))
                    if key in seen:
                        continue
                    seen.add(key)
                    total += int(storage.nbytes())
                if total:
                    rows.append({"module": name or "<root>", "bytes": total})
        rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
        return {
            "module_rows": rows,
            "module_bytes": sum(int(row["bytes"]) for row in rows),
            "torch_allocated_bytes": int(self.torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(self.torch.cuda.memory_reserved()),
        }

    def close(self, *, phase: str) -> dict[str, Any]:
        """Destroy one score/train phase and fail closed above 10 GiB residue."""
        before = self.memory_ledger()
        if self.dist.is_initialized():
            self.dist.barrier()
            self.dist.destroy_process_group()
        for name in (
            "optimizer", "scheduler", "student", "luts", "norms", "outputs", "scales", "expert_planes",
            "ids_cache", "real_lengths", "teacher_cache", "score_ids_cache",
            "score_real_lengths", "score_teacher_cache",
        ):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        collect = getattr(self.torch.cuda, "ipc_collect", None)
        if callable(collect):
            collect()
        allocated = int(self.torch.cuda.memory_allocated())
        reserved = int(self.torch.cuda.memory_reserved())
        limit = 10 * 1024**3
        if allocated >= limit:
            raise ArtifactError(
                f"resident phase teardown retained {allocated} CUDA bytes (limit {limit})"
            )
        return {
            "schema": "banana-smasher-resident-phase-release-v1",
            "status": "PASS",
            "phase": phase,
            "pre_release": before,
            "post_release_allocated_bytes": allocated,
            "post_release_reserved_bytes": reserved,
            "limit_bytes": limit,
        }

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
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            runtime_root = os.environ.get("BANANA_SMASHER_RUN_ROOT")
            if runtime_root:
                cache_root = Path(runtime_root).expanduser().resolve() / "torch_extensions"
            else:
                cache_root = (
                    Path(os.environ.get("TMPDIR", "/tmp")).expanduser().resolve()
                    / f"banana-smasher-{os.getuid()}"
                    / "torch_extensions"
                )
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_root)

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
        if fields.get("phase") == "loading":
            # Shard construction reports after each layer has dropped its temporary
            # state dict. Return those unused CUDA allocator pages before loading
            # the next layer; on unified-memory hosts they otherwise accumulate
            # outside the useful resident set and can starve the NVIDIA driver.
            self.torch.cuda.empty_cache()

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

    def _configure_trainable_quantization_scales(self) -> list[tuple[str, Any]]:
        if self.config.get("trainable_quantization_scales") is not True:
            return []
        experts = getattr(self.student, "experts", None)
        if not isinstance(experts, Mapping) or not experts:
            raise ArtifactError("trainable quantization scales require resident grouped experts")
        saved = self.state.get("scales")
        rows: list[tuple[str, Any]] = []
        for layer, module in sorted(experts.items(), key=lambda item: int(item[0])):
            for projection in ("w1", "w2", "w3"):
                for axis in ("su", "sv"):
                    attribute = f"{axis}_{projection}"
                    value = getattr(module, attribute, None)
                    if value is None or not hasattr(value, "shape") or not value.is_floating_point():
                        raise ArtifactError(f"resident grouped scale seam missing: L{int(layer):03d}/{attribute}")
                    name = f"layers.{int(layer)}.scales.{attribute}"
                    control = value.detach().float().clone()
                    initial = control.clone()
                    if saved is not None:
                        if not isinstance(saved, Mapping) or name not in saved:
                            raise ArtifactError(f"checkpoint missing trainable quantization scale: {name}")
                        checkpoint_value = saved[name]
                        if tuple(checkpoint_value.shape) != tuple(initial.shape):
                            raise ArtifactError(f"checkpoint trainable quantization scale shape drift: {name}")
                        initial.copy_(checkpoint_value.to(device=initial.device, dtype=initial.dtype))
                    if attribute in getattr(module, "_buffers", {}):
                        del module._buffers[attribute]
                    if attribute in getattr(module, "_parameters", {}):
                        del module._parameters[attribute]
                    parameter = self.torch.nn.Parameter(initial, requires_grad=True)
                    parameter._banana_scale_control = control
                    module.register_parameter(attribute, parameter)
                    rows.append((name, parameter))
        return rows

    def _project_scale_trust_region(self) -> dict[str, Any]:
        requested = self.config.get("quantization_scale_relative_trust_region")
        if requested is None:
            return {"enabled": False, "clipped_elements": 0}
        bound = float(requested)
        if not math.isfinite(bound) or not 0.0 < bound < 1.0:
            raise ArtifactError("quantization scale relative trust region must be finite in (0, 1)")
        clipped = 0
        maximum = 0.0
        with self.torch.no_grad():
            for name, parameter in self.scales:
                control = getattr(parameter, "_banana_scale_control", None)
                if control is None:
                    raise ArtifactError(f"scale trust-region control missing: {name}")
                low = self.torch.minimum(control * (1.0 - bound), control * (1.0 + bound))
                high = self.torch.maximum(control * (1.0 - bound), control * (1.0 + bound))
                before = parameter.detach().clone()
                parameter.copy_(self.torch.maximum(low, self.torch.minimum(high, parameter)))
                clipped += int(self.torch.count_nonzero(parameter != before).item())
                nonzero = control != 0
                if self.torch.any(nonzero):
                    relative = ((parameter[nonzero] - control[nonzero]) / control[nonzero]).abs()
                    maximum = max(maximum, float(relative.max().detach().cpu()))
        if maximum > bound + 1.0e-6:
            raise ArtifactError("quantization scale relative trust region projection failed")
        return {"enabled": True, "bound": bound, "clipped_elements": clipped, "maximum_relative_delta": maximum}

    def _load_optimizer_scheduler_state(self) -> None:
        if self.score_only:
            self.scheduler_state_action = "SCORE_ONLY_NO_TRAINING_LINEAGE"
            return
        optimizer_payload = self.payload.get("optimizer", self.payload.get("optimizer_state"))
        scheduler_payload = self.payload.get("scheduler", self.payload.get("scheduler_state"))
        if _checkpoint_cursor(self.payload) == 0:
            if isinstance(optimizer_payload, Mapping) or isinstance(scheduler_payload, Mapping):
                raise ArtifactError("published PRE must start with fresh optimizer and scheduler state")
            self.scheduler_state_action = "FRESH_PRE_OPTIMIZER_AND_SCHEDULE"
            return
        if not isinstance(optimizer_payload, Mapping):
            raise ArtifactError("continuation checkpoint is missing the shared Adam optimizer state")
        groups = optimizer_payload.get("param_groups")
        global_state = optimizer_payload.get("state")
        source_surfaces = (
            ("luts", "norms", "outputs")
            + (("scales",) if "scales" in self.state else ())
            + ((EXPERT_PLANE_SURFACE,) if self.expert_plane_contract is not None else ())
        )
        if not isinstance(groups, list) or len(groups) != len(source_surfaces) or not isinstance(global_state, Mapping):
            raise ArtifactError("resident Adam state has no canonical trainable-surface lineage")
        local_state = self.optimizer.state_dict()
        local_groups = local_state["param_groups"]
        local_rows = {
            "luts": self.luts,
            "norms": self.norms,
            "outputs": self.outputs,
            "scales": self.scales,
            **({EXPERT_PLANE_SURFACE: self.expert_planes} if self.expert_plane_contract is not None else {}),
        }
        for index, surface in enumerate(source_surfaces):
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
            # Checkpoint optimizer metadata may carry legacy roster labels such
            # as ``all43_luts``.  Preserve numeric Adam state, but keep the
            # canonical live surface identity used by the per-step LR gate.
            local_groups[index]["group_name"] = surface
        if self.scales and "scales" not in self.state:
            for key in ("lr", "initial_lr", "betas", "eps", "weight_decay", "amsgrad"):
                if key in local_groups[0]:
                    local_groups[3][key] = local_groups[0][key]
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
            scheduler_state = dict(scheduler_payload)
            if self.scales and "scales" not in self.state:
                for key in ("base_lrs", "_last_lr"):
                    values = scheduler_state.get(key)
                    if isinstance(values, list) and len(values) + 1 == len(self.optimizer.param_groups):
                        scheduler_state[key] = [*values, values[0]]
            self.scheduler.load_state_dict(scheduler_state)
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
        """Score the live ShardStudent without loading checkpoint/candidate files."""
        selected = tuple(int(value) for value in windows)
        if len(selected) != 64 or len(set(selected)) != 64:
            raise ArtifactError("resident physical score requires 64 unique ordered windows")
        missing = [window for window in selected if window not in self.score_ids_cache]
        if missing:
            raise ArtifactError(f"resident physical score windows were not preloaded: {missing}")
        started = time.perf_counter()
        local_rows: list[dict[str, Any]] = []
        pending_sends: list[tuple[Any, Any]] = []
        pipeline_compute_seconds = 0.0
        pipeline_wait_seconds = 0.0
        torch = self.torch
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
                    pipeline_compute_seconds += time.perf_counter() - compute_started
                    wait_started = time.perf_counter()
                    _enqueue_rank_send(self.dist, pending_sends, hidden.contiguous())
                    pipeline_wait_seconds += time.perf_counter() - wait_started
                else:
                    activation = torch.empty(
                        shape, dtype=torch.bfloat16, device=self.student.device
                    )
                    wait_started = time.perf_counter()
                    _recv_rank_activation(self.dist, activation)
                    pipeline_wait_seconds += time.perf_counter() - wait_started
                    compute_started = time.perf_counter()
                    hidden = self._run_layers(activation, ids, False)
                    final = self.student.model.model.norm(
                        self.student.model.model.hc_head(hidden)
                    )
                    lengths = [int(self.score_real_lengths[window]) for window in group]
                    if any(length != 1024 for length in lengths):
                        raise ArtifactError(
                            f"resident Balanced64 group has non-1024 lengths: {lengths}"
                        )
                    for offset in range(0, len(group), SCORE_LOGIT_MICROBATCH):
                        group_logits = _score_group_logits(
                            self.student.model.lm_head,
                            final[:, :1024],
                            torch,
                            offset=offset,
                        )
                        for row, window in enumerate(
                            group[offset : offset + SCORE_LOGIT_MICROBATCH]
                        ):
                            length = int(self.score_real_lengths[window])
                            idx, lp_n, p_n = self._teacher_support(
                                window, length, exact_rows=True, score=True
                            )
                            logits = group_logits[row, :length]
                            q = logits.gather(1, idx[:length]).float()
                            qn = q - q.logsumexp(-1, keepdim=True)
                            terms = (p_n[:length] * (lp_n[:length] - qn)).sum(-1)
                            local_rows.append(
                                {
                                    "window": window,
                                    "positions": length,
                                    "kld_sum": float(terms.double().sum().cpu()),
                                    "top1": int(
                                        (logits.argmax(-1) == idx[:length, 0]).sum().cpu()
                                    ),
                                }
                            )
                            del logits, q, qn, terms
                        del group_logits
                    pipeline_compute_seconds += time.perf_counter() - compute_started
            if self.rank == 0:
                wait_started = time.perf_counter()
                _flush_rank_sends(pending_sends)
                pipeline_wait_seconds += time.perf_counter() - wait_started
        gathered: list[Any] = [None, None]
        self.dist.all_gather_object(
            gathered, local_rows if self.rank == 1 else None
        )
        rows = gathered[1]
        if not isinstance(rows, list) or len(rows) != 64:
            raise ArtifactError("rank1 resident score did not publish 64 complete rows")
        positions = sum(int(row["positions"]) for row in rows)
        if positions != 64 * 1024:
            raise ArtifactError("resident physical score position count drift")
        _cuda_sync(torch)
        elapsed = time.perf_counter() - started
        return {
            "mean_kld": math.fsum(float(row["kld_sum"]) for row in rows) / positions,
            "top1_matches": sum(int(row["top1"]) for row in rows),
            "positions": positions,
            "checkpoint": f"UPDATE_{self.global_step:03d}",
            "timed_wall_seconds": elapsed,
            "execution_mode": "resident_model_in_memory",
            "runtime_counters": {
                "model_constructions": 1,
                "checkpoint_loads_during_score": 0,
                "candidate_file_reads_during_score": 0,
                "windows": 64,
                "pipeline_compute_seconds": pipeline_compute_seconds,
                "pipeline_wait_seconds": pipeline_wait_seconds,
            },
        }

    def _init_distributed(self) -> None:
        if self.dist.is_initialized():
            if self.dist.get_world_size() != 2 or self.dist.get_rank() != self.rank:
                raise ArtifactError("existing process group does not match the exact two-Spark rank")
            self.dist.barrier()
            return
        master_addr = os.environ.get(
            "MASTER_ADDR", str(self.config.get("master_addr", "127.0.0.1"))
        ).strip()
        if not master_addr:
            raise ArtifactError("MASTER_ADDR/master_addr must be non-empty")
        master_port = int(
            os.environ.get("MASTER_PORT", str(self.config.get("master_port", 29598)))
        )
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
        activation_checkpointing = train and bool(
            self.config.get("activation_checkpointing", True)
        )
        if activation_checkpointing and not hidden.requires_grad:
            # Reentrant checkpointing needs at least one grad-bearing input or
            # it returns a detached output. Rank 0 starts from frozen token
            # embeddings, so arm only the pipeline activation leaf; model
            # weights remain frozen while layer-local repair parameters train.
            hidden.requires_grad_(True)
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
            if activation_checkpointing:
                # Reentrant checkpointing executes the layer forward under
                # no_grad and reconstructs it during backward.  The
                # non-reentrant variant records the full rank-local autograd
                # graph and exceeded DGX Spark unified memory before the first
                # pipeline send, even after allocator-cache trimming.
                hidden = self.checkpoint(layer_fn, hidden, use_reentrant=True)
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
        return [*self.luts, *self.norms, *self.outputs, *self.scales, *self.expert_planes]

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
        scale_trust_region = self._project_scale_trust_region()
        self.scheduler.step()
        _cuda_sync(torch)
        optimizer_seconds = time.perf_counter() - optimizer_started
        gradients = [parameter.grad for _name, parameter in params if parameter.grad is not None]
        gradient_norm = self._local_norm(gradients)
        delta_norm = self._local_norm([parameter.detach() - old for (_name, parameter), old in zip(params, before)])
        expert_coverage = None
        if self.expert_plane_contract is not None:
            expert_before = before[-len(self.expert_planes):] if self.expert_planes else []
            expert_coverage = _classify_expert_plane_update(
                torch, self.expert_planes, expert_before
            )
            frozen_count = len(params) - len(self.expert_planes)
            mutated_frozen = [
                name
                for (name, parameter), old in zip(
                    params[:frozen_count], before[:frozen_count]
                )
                if bool(torch.count_nonzero(parameter.detach() - old).item())
            ]
            if (
                expert_coverage["missing_trainable"]
                or expert_coverage["missing_gradient"]
                or expert_coverage["missing_delta"]
                or mutated_frozen
            ):
                raise ArtifactError(
                    "L028 SU/SV coverage gate failed: "
                    f"missing_trainable={expert_coverage['missing_trainable'][:3]} "
                    f"missing_gradient={expert_coverage['missing_gradient'][:3]} "
                    f"missing_delta={expert_coverage['missing_delta'][:3]} "
                    f"mutated_frozen={mutated_frozen[:3]}"
                )
            expert_coverage["frozen_mutations"] = 0
        local = {
            "rank": self.rank,
            "loss": loss,
            "scale_trust_region": scale_trust_region,
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
            "expert_plane_coverage": expert_coverage,
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
        local_params = {
            "luts": self.luts,
            "norms": self.norms,
            "outputs": self.outputs,
            **({"scales": self.scales} if self.scales else {}),
            **({EXPERT_PLANE_SURFACE: self.expert_planes} if self.expert_plane_contract is not None else {}),
        }
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
        merged = {surface: {} for surface in local_params}
        for row in rows:
            for surface in merged:
                overlap = set(merged[surface]) & set(row[surface])
                if overlap:
                    raise ArtifactError(f"official resident state overlap: {surface} {sorted(overlap)[:3]}")
                merged[surface].update(row[surface])
        expected_coverage = {"luts": 43, "norms": 235, "outputs": 43}
        if self.expert_plane_contract is not None:
            expected_coverage[EXPERT_PLANE_SURFACE] = 1536
        if self.scales:
            expected_coverage["scales"] = 258
        if {surface: len(values) for surface, values in merged.items()} != expected_coverage:
            raise ArtifactError("official resident merged trainable surface coverage drift")
        optimizer = (
            _merge_expanded_optimizer_state(
                rows,
                merged,
                tuple(merged),
                set(getattr(self.trainer, "DORMANT_NORMS", set())),
            )
            if self.expert_plane_contract is not None or self.scales
            else self.trainer.merge_optimizer_state(rows, merged)
        )
        scheduler = _cpu_tree(torch, self.scheduler.state_dict())
        report = {"rank_rows": rows, "optimizer": optimizer, "scheduler": scheduler}
        self.dist.barrier()
        return merged, optimizer, report

    def advance_to(
        self,
        target_update: int,
        *,
        loss_guard_baseline: float | None = None,
        loss_guard_receipt_path: str | Path | None = None,
    ) -> tuple[Mapping[str, Any] | None, dict[str, Any], Mapping[str, Any] | None]:
        start = self.global_step
        if target_update <= start:
            raise ArtifactError("official resident target must advance beyond current update")
        last: dict[str, Any] | None = None
        merged_state: Mapping[str, Any] | None = None
        optimizer_state: Mapping[str, Any] | None = None
        report_state: Mapping[str, Any] | None = None
        update_reports: list[dict[str, Any]] = []
        previous_loss: float | None = None
        for update in range(start, target_update):
            last = self._step(update)
            loss_guard = None
            if loss_guard_baseline is not None:
                if loss_guard_receipt_path is None:
                    raise ArtifactError("resident loss guard requires a receipt path")
                try:
                    loss_guard = _enforce_update_loss_guard(
                        loss=last["loss"],
                        baseline=loss_guard_baseline,
                        previous_loss=previous_loss,
                        global_update=update + 1,
                    )
                except ArtifactError:
                    receipt_path = Path(loss_guard_receipt_path).expanduser().resolve()
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    receipt = {
                        "schema": "banana-smasher-resident-loss-guard-v1",
                        "status": "HALTED_LOSS_EXPLOSION",
                        "global_update": update + 1,
                        "baseline_loss": float(loss_guard_baseline),
                        "loss": float(last["loss"]),
                        "explosion_limit": 2.0 * float(loss_guard_baseline),
                        "accepted_checkpoint_written": False,
                    }
                    temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
                    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
                    with temporary.open("wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, receipt_path)
                    raise
                previous_loss = float(last["loss"])
            update_reports.append(
                {
                    "global_update": update + 1,
                    "loss": last["loss"],
                    "loss_guard": loss_guard,
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
            "frozen_surfaces": ["packed_codes", "assignments"],
            "trainable_surfaces": [
                "luts", "rmsnorms", "output_gains",
                *(["scales"] if self.scales else []),
                *([EXPERT_PLANE_SURFACE] if self.expert_plane_contract is not None else []),
            ],
        }
        return merged_state, step_report, report_state

    def broadcast_persisted(self, value: Any) -> Any:
        row = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(row, src=0)
        return row[0]
