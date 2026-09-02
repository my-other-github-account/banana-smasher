"""Official Modern Green grouped-K2 resident continuation engine.

This module is deliberately coupled to the accepted clean-U0 trainer.  It does
not manufacture a loss from checkpoint tensors: it constructs the resident
ShardStudent, routes the real model through both layer partitions, evaluates
the teacher KL objective, and runs the trainer's legal LUT/RMS/gain surface
through Adam and LambdaLR.
"""
from __future__ import annotations

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

from .balanced64 import ArtifactError

MODEL_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
ADMISSION_SHA256 = "76d0674eb0cd37fc9022bac5e048c2b77c721826182222ae0a0609e29607a2c5"
CORPUS_SHA256 = "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5"
TRAINER_SHA256 = "a55c2f5104b8d9dd06d845684d168be6f6e9dae637bac08443bd6ddbaf94201a"
HISTORICAL_TRAINER_SHA256 = "c8df3ab6a815fd69e401db7047afee53e9b0ce5652bf7fbcb9116d308c1b8e24"
WINDOWS_PER_STEP = 4
PIPELINE_MICROBATCH = 4
BASE_LRS = {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2}
EXPERT_PLANE_SURFACE = "expert_planes_l028_su_sv"
AUTHENTICATED_U60_CHECKPOINT_SHA256 = "1962213d88ee6b6df62311c58258dc72c38c5aca761e4dd00d3a375588c95c95"
AUTHENTICATED_U60_IDENTITY_SHA256 = "f0f5479111be56a34c32d21dba214372097350c8072cf7da4aa9fdeb37fdf4bd"
AUTHENTICATED_U64_IDENTITY_SHA256 = "2bbb634af51cbc3e3c0c8c575fec12844a9f59c4a2b4c7cbf8d2a737efb16f7b"
AUTHENTICATED_U64_STATE_SHA256 = "dc5b702f7c90ebe1161980ea648425a8f9103588eb24d8af1a32597e33547e19"
U60_EXPERT_PLANE_BASE_LR = 7.5e-5
HISTORICAL_BASE_LRS = {"luts": 2.5e-4, "norms": 2.5e-5, "outputs": 2.5e-4}
PUBLISHED_PRE_RECIPE_ID = "published_pre_lower_lr_warmup16_cosine64_v1"
PUBLISHED_PRE_SHA256 = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
PUBLISHED_PRE_BASE_LRS = {"luts": 1.0e-3, "norms": 1.0e-4, "outputs": 1.0e-3}
STATIC_W28_VALIDATION_CORPUS_SHA256 = "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd"
STATIC_W28_TEACHER_SHA256 = "561753481a1e08aee88e28f5fa0c6e727f4af679494c39679e87ed5189e2653d"
SEALED_GROUPED_WRAPPER_SHA256 = "37b919ae6adb34987e0e20ba4318352d9bf07b5183008d023a1822b3daf75126"
SEALED_GROUPED_EXPERT_SHA256 = "8080d1e6ef6752c7823a4db0426c6ea048b830a1ece173b30a7b12f716d1685b"
STATIC_W28_GROUPED_WRAPPER_SHA256 = "0d4ece20b602fc59ffef349183db2bea0861b4a7f7c0ef93e50fd728310e7371"
STATIC_W28_GROUPED_EXPERT_SHA256 = "fc612f7863ad9d09a9faf11e203a9d20739b7dbb273b982fc3d36ee01d15a9b4"
CONTROLLED_ARM_IDS = {
    "lr_scale_only",
    "cosine_restart_only",
    "warmup_restart_only",
    "window_dose_only",
    "from_u0_historical_control",
    "from_u0_lr_scale_only",
    "from_u0_cosine_only",
    "from_u0_window_dose6_only",
}
FROM_U0_ARM_IDS = {
    "from_u0_historical_control",
    "from_u0_lr_scale_only",
    "from_u0_cosine_only",
    "from_u0_window_dose6_only",
}


def _canonical_l28_expert_plane_spec() -> list[tuple[str, tuple[int]]]:
    """Return the authenticated UPDATE_060 plane names and geometries in order."""
    shapes = {
        ("w1", "SU"): 4096, ("w1", "SV"): 2048,
        ("w2", "SU"): 2048, ("w2", "SV"): 4096,
        ("w3", "SU"): 4096, ("w3", "SV"): 2048,
    }
    return [
        (f"model.layers.28.mlp.experts.E{expert:03d}.{projection}.{plane}", (shapes[(projection, plane)],))
        for projection in ("w1", "w2", "w3")
        for plane in ("SU", "SV")
        for expert in range(256)
    ]


def _validate_trainable_state_schema(torch: Any, payload: Mapping[str, Any]) -> bool:
    """Validate legacy U0, authenticated U60, or its exact persisted U64 successor."""
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ArtifactError("checkpoint state must contain official trainable surfaces")
    identity_value = payload.get("identity")
    identity = identity_value if isinstance(identity_value, Mapping) else {}
    step = int(payload.get("next_update", identity.get("next_update", -1)))
    keys = list(state)
    three = ["luts", "norms", "outputs"]
    if keys == three:
        if step != 0:
            raise ArtifactError("three-surface state is retained only for PRE/U0")
        return False
    four = [*three, EXPERT_PLANE_SURFACE]
    if keys != four:
        raise ArtifactError("checkpoint trainable surface schema/order drift")
    checkpoint_sha = payload.get("checkpoint_sha256", identity.get("checkpoint_sha256"))
    authenticated_u60 = (
        step == 60 and checkpoint_sha == AUTHENTICATED_U60_CHECKPOINT_SHA256
    )
    authenticated_u64 = (
        step == 64
        and identity.get("schema") == "resident-continuation-checkpoint-identity-v1"
        and identity.get("checkpoint") == "UPDATE_064"
        and identity.get("next_update") == 64
        and identity.get("parent_checkpoint_sha256") == AUTHENTICATED_U60_CHECKPOINT_SHA256
        and identity.get("parent_identity_sha256") == AUTHENTICATED_U60_IDENTITY_SHA256
        and identity.get("state_sha256") == AUTHENTICATED_U64_STATE_SHA256
        and identity.get("optimizer_scheduler_lineage") == "fresh-published-pre-adam-lambdalr"
        and identity.get("identity_sha256") == AUTHENTICATED_U64_IDENTITY_SHA256
    )
    if not (authenticated_u60 or authenticated_u64):
        raise ArtifactError(
            "four-surface state requires authenticated UPDATE_060 or exact UPDATE_064 successor"
        )
    planes = state.get(EXPERT_PLANE_SURFACE)
    if not isinstance(planes, Mapping):
        raise ArtifactError("UPDATE_060 expert-plane surface must be a mapping")
    expected = _canonical_l28_expert_plane_spec()
    if list(planes) != [name for name, _shape in expected]:
        raise ArtifactError("UPDATE_060 expert-plane names are not in canonical order")
    for name, shape in expected:
        value = planes[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ArtifactError(f"UPDATE_060 expert plane {name} must be float32")
        if tuple(value.shape) != shape:
            raise ArtifactError(f"UPDATE_060 expert plane {name} geometry drift")
    return True


def _bind_l28_expert_plane_state(
    torch: Any, student: Any, saved: Mapping[str, Any]
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Install 1,536 FP32 masters and make the resident L28 provider read them."""
    provider = student.model.model.layers[28].mlp.experts
    rows: list[tuple[str, Any]] = []
    by_slot: dict[tuple[str, str], list[Any]] = {}
    for index, (name, shape) in enumerate(_canonical_l28_expert_plane_spec()):
        value = saved[name]
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ArtifactError(f"UPDATE_060 expert plane {name} runtime geometry drift")
        parameter = torch.nn.Parameter(value.detach().to(student.device).clone(), requires_grad=True)
        provider.register_parameter(f"u60_plane_{index:04d}", parameter)
        rows.append((name, parameter))
        projection, plane = name.rsplit(".", 2)[-2:]
        by_slot.setdefault((projection, plane), []).append(parameter)

    def bind_runtime(module: Any, _inputs: Any) -> None:
        for (projection, plane), parameters in by_slot.items():
            attr = f"{plane.lower()}_{projection}"
            target = module._buffers.get(attr, getattr(module, attr, None))
            if target is None:
                raise ArtifactError(f"resident L28 provider is missing {attr}")
            module._buffers[attr] = torch.stack(parameters, dim=0).to(dtype=target.dtype)

    provider.register_forward_pre_hook(bind_runtime)
    candidate_name, candidate = rows[0]
    proof = {
        "candidate_name": candidate_name,
        "candidate_read": bool(torch.equal(candidate.detach().cpu(), saved[candidate_name].detach().cpu())),
        "candidate_dtype": str(candidate.dtype),
        "candidate_shape": list(candidate.shape),
    }
    if not proof["candidate_read"]:
        raise ArtifactError("UPDATE_060 expert-plane candidate readback mismatch")
    return rows, proof


def _merge_sharded_optimizer_state(
    state_rows: list[Mapping[str, Any]],
    ordered_state: Mapping[str, Mapping[str, Any]],
    surfaces: tuple[str, ...],
) -> dict[str, Any]:
    """Merge rank-local optimizer IDs into checkpoint surface/name order."""
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
        if len(local["param_groups"]) != len(surfaces):
            raise ArtifactError("local optimizer parameter-group count drift")
        local_ids_seen: set[int] = set()
        for surface, group in zip(surfaces, local["param_groups"]):
            names = list(local_names[surface])
            ids = list(group["params"])
            if len(names) != len(ids):
                raise ArtifactError(f"local optimizer name/id drift: {surface}")
            template = {key: value for key, value in group.items() if key != "params"}
            previous = templates.setdefault(surface, template)
            if previous != template:
                raise ArtifactError(f"optimizer group setting drift across ranks: {surface}")
            for name, local_id in zip(names, ids):
                if name not in global_ids or local_id in local_ids_seen:
                    raise ArtifactError(f"optimizer parameter binding drift: {name}")
                local_ids_seen.add(local_id)
                global_id = global_ids[name]
                if global_id in seen:
                    raise ArtifactError(f"optimizer parameter overlap: {name}")
                seen.add(global_id)
                value = local["state"].get(local_id, local["state"].get(str(local_id)))
                if value is not None:
                    merged_state[global_id] = value
        dangling = {int(value) for value in local["state"]} - local_ids_seen
        if dangling:
            raise ArtifactError(f"optimizer state has unbound local ids: {sorted(dangling)[:3]}")
    if seen != set(range(len(global_ids))):
        raise ArtifactError("global optimizer parameter coverage drift")
    groups = []
    for surface in surfaces:
        group = dict(templates[surface])
        group["params"] = [global_ids[name] for name in ordered_names[surface]]
        groups.append(group)
    return {"state": merged_state, "param_groups": groups}


def _fp64_state_adam(
    torch: Any, param_groups: list[dict[str, Any]], *, gradient_scale: float = 1.0
):
    """Adam with FP64 moments and update arithmetic over FP32 masters.

    The Adam rule is unchanged: there is no clipping, rescaling, or LR change.
    Only gradient squaring and update formation use wider arithmetic.
    """

    if gradient_scale <= 0.0 or not math.isfinite(gradient_scale):
        raise ArtifactError("FP64 Adam gradient scale must be finite and positive")

    class FP64StateAdam(torch.optim.Optimizer):
        def __init__(self, groups):
            super().__init__(groups, {
                "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8,
                "weight_decay": 0.0, "amsgrad": False,
            })

        def load_state_dict(self, state_dict):
            source_groups = state_dict["param_groups"]
            if len(source_groups) != len(self.param_groups):
                raise ArtifactError("FP64 Adam checkpoint param-group count mismatch")
            source_states_by_group = []
            for source_group, target_group in zip(source_groups, self.param_groups):
                source_ids = source_group["params"]
                if len(source_ids) != len(target_group["params"]):
                    raise ArtifactError("FP64 Adam checkpoint parameter count mismatch")
                source_states_by_group.append(
                    [state_dict["state"].get(source_id, {}) for source_id in source_ids]
                )

            result = super().load_state_dict(state_dict)
            for group in self.param_groups:
                group.setdefault("weight_decay", 0.0)
                group.setdefault("amsgrad", False)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1.0e-8)
            for group, source_states in zip(self.param_groups, source_states_by_group):
                for parameter, source_state in zip(group["params"], source_states):
                    if not source_state:
                        continue
                    state = self.state[parameter]
                    step = source_state.get("step", state.get("step", 0))
                    if isinstance(step, torch.Tensor):
                        step = float(step.detach().cpu().item())
                    state["step"] = torch.tensor(
                        step, dtype=torch.float64, device=parameter.device
                    )
                    source_scale = float(source_state.get("gradient_scale", 1.0))
                    scale_ratio = gradient_scale / source_scale
                    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        value = source_state.get(name)
                        if value is not None:
                            power = 1 if name == "exp_avg" else 2
                            state[name] = value.detach().to(
                                device=parameter.device, dtype=torch.float64
                            ).mul_(scale_ratio ** power)
                    state["gradient_scale"] = gradient_scale
            return result

        @torch.no_grad()
        def step(self, closure=None):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for parameter_index, parameter in enumerate(group["params"]):
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("FP64-state Adam does not support sparse gradients")
                    if not bool(torch.isfinite(gradient).all().item()):
                        raise ArtifactError(
                            "nonfinite gradient before FP64 Adam update: "
                            f"group={group.get('group_name', 'unnamed')} "
                            f"parameter_index={parameter_index}"
                        )
                    state = self.state[parameter]
                    if not state:
                        state["step"] = torch.zeros((), dtype=torch.float64, device=parameter.device)
                        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float64)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float64)
                        state["gradient_scale"] = gradient_scale
                    state["step"].add_(1)
                    gradient64 = gradient.detach().to(dtype=torch.float64)
                    if group["weight_decay"]:
                        gradient64.add_(
                            parameter.detach().to(dtype=torch.float64),
                            alpha=group["weight_decay"] * gradient_scale,
                        )
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(gradient64, alpha=1.0 - beta1)
                    if not bool(torch.isfinite(exp_avg).all().item()):
                        raise ArtifactError(
                            "nonfinite FP64 Adam exp_avg during mutation: "
                            f"group={group.get('group_name', 'unnamed')} "
                            f"parameter_index={parameter_index} "
                            f"gradient_abs_max={float(gradient64.abs().max().item())} "
                            f"gradient_scale={gradient_scale}"
                        )
                    exp_avg_sq.mul_(beta2).addcmul_(gradient64, gradient64, value=1.0 - beta2)
                    if not bool(torch.isfinite(exp_avg_sq).all().item()):
                        raise ArtifactError(
                            "nonfinite FP64 Adam exp_avg_sq during mutation: "
                            f"group={group.get('group_name', 'unnamed')} "
                            f"parameter_index={parameter_index} "
                            f"gradient_abs_max={float(gradient64.abs().max().item())} "
                            f"gradient_scale={gradient_scale}"
                        )
                    step = float(state["step"].item())
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(
                        group["eps"] * gradient_scale
                    )
                    delta64 = exp_avg.div(denominator).mul_(-group["lr"] / bias_correction1)
                    parameter.add_(delta64.to(dtype=parameter.dtype))
            return loss

    return FP64StateAdam(param_groups)


def _controlled_arm_origin(arm_id: str) -> int:
    if arm_id not in CONTROLLED_ARM_IDS:
        raise ArtifactError(f"controlled arm id is not registered: {arm_id!r}")
    return 0 if arm_id in FROM_U0_ARM_IDS else 16


def _controlled_scheduler_step(arm_id: str, local_step: int) -> int:
    step = _controlled_arm_origin(arm_id) + int(local_step)
    if not 0 <= step < 64:
        raise ArtifactError("controlled scheduler cursor must be within U0..U63")
    return step


def _controlled_arm_policy(config: Mapping[str, Any], global_step: int) -> tuple[dict[str, float], float, int]:
    """Resolve a registered config-only arm policy at a zero-based update cursor."""
    arm = config.get("controlled_arm_id")
    if arm not in CONTROLLED_ARM_IDS:
        raise ArtifactError(f"controlled arm id is not registered: {arm!r}")
    step = int(global_step)
    if arm in FROM_U0_ARM_IDS:
        if not 0 <= step < 64:
            raise ArtifactError("from-U0 controlled arm cursor must be within U0..U63")
        if arm == "from_u0_cosine_only":
            multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / 64.0))
        else:
            multiplier = min(1.0, (step + 1) / 16.0)
            if arm == "from_u0_lr_scale_only":
                multiplier *= 0.125
        windows_per_update = 6 if arm == "from_u0_window_dose6_only" else 2
        return dict(HISTORICAL_BASE_LRS), multiplier, windows_per_update
    if not 16 <= step < 64:
        raise ArtifactError("from-U16 controlled arm cursor must be within U16..U63")
    relative = step - 16
    if arm == "lr_scale_only":
        multiplier = 0.125
    elif arm == "cosine_restart_only":
        multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * relative / 48.0))
    elif arm == "warmup_restart_only":
        multiplier = min(1.0, (relative + 1) / 8.0)
    else:
        multiplier = 1.0
    windows_per_update = 16 if arm == "window_dose_only" else 6
    return dict(HISTORICAL_BASE_LRS), multiplier, windows_per_update


def _published_pre_recipe_policy(
    config: Mapping[str, Any],
    global_step: int,
    *,
    authenticated_u60_continuation: bool = False,
    terminal_scheduler_cursor: bool = False,
) -> tuple[dict[str, float], float, list[int]]:
    """Resolve David's sealed PRE recipe: lower LR, warmup, then true cosine."""
    if config.get("recipe_id") != PUBLISHED_PRE_RECIPE_ID:
        raise ArtifactError("published PRE recipe id drift")
    declared_pre = config.get("published_pre_checkpoint_sha256") == PUBLISHED_PRE_SHA256
    authenticated_u60 = (
        authenticated_u60_continuation
        and config.get("checkpoint_sha256") == AUTHENTICATED_U60_CHECKPOINT_SHA256
    )
    if not (declared_pre or authenticated_u60):
        raise ArtifactError("published PRE checkpoint declaration drift")
    lr_scalar = float(config.get("lr_scale", 1.0))
    if not math.isfinite(lr_scalar) or lr_scalar <= 0.0:
        raise ArtifactError("published PRE lr_scale must be finite and positive")
    step = int(global_step)
    if not (0 <= step < 64 or terminal_scheduler_cursor and step == 64):
        raise ArtifactError("published PRE recipe cursor must be within U0..U63")
    if step < 16:
        multiplier = (step + 1) / 16.0
    else:
        relative = step - 16
        multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * relative / 48.0))
    base_lrs = dict(PUBLISHED_PRE_BASE_LRS)
    if authenticated_u60:
        base_lrs[EXPERT_PLANE_SURFACE] = U60_EXPERT_PLANE_BASE_LR
    return base_lrs, multiplier * lr_scalar, [28, 56]


def _published_pre_controlled_schedule(
    schedule: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[int], int, str]:
    """Bind sealed source rows 21..24 to fresh-PRE U1..U4 in file order."""
    raw_labels = config.get("controlled_window_schedule_source_rows")
    if raw_labels != [21, 22, 23, 24]:
        raise ArtifactError("fresh PRE controlled schedule requires source rows [21, 22, 23, 24]")
    labels = [21, 22, 23, 24]
    boundary = schedule.get("expected_first_four_update_boundary")
    rows = boundary.get("updates") if isinstance(boundary, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(labels) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        raise ArtifactError("fresh PRE controlled schedule requires exactly four immutable source rows")
    observed_labels = [int(row.get("global_update", -1)) for row in rows]
    if observed_labels != labels:
        raise ArtifactError("fresh PRE controlled schedule source-row labels drift")
    try:
        windows_per_update = int(config["controlled_windows_per_update"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("fresh PRE controlled_windows_per_update is required") from exc
    source_delta = schedule.get("exact_parameter_delta", {})
    source_to = source_delta.get("to", {}) if isinstance(source_delta, Mapping) else {}
    if source_to.get("windows_per_optimizer_update") != windows_per_update:
        raise ArtifactError("fresh PRE controlled windows-per-update source drift")
    pipeline_microbatch = int(config.get("pipeline_microbatch", 2))
    if windows_per_update <= 0 or pipeline_microbatch <= 0 or windows_per_update % pipeline_microbatch:
        raise ArtifactError("fresh PRE controlled window dose is not pipeline divisible")
    pipeline_groups = windows_per_update // pipeline_microbatch
    expected_weights = {
        "category_loss_weight": 1.0 / windows_per_update,
        "pipeline_microbatch": pipeline_microbatch,
        "pipeline_groups_per_update": pipeline_groups,
        "group_gradient_scale": 1.0 / pipeline_groups,
    }
    if any(source_to.get(key) != value for key, value in expected_weights.items()):
        raise ArtifactError("fresh PRE controlled schedule weighting drift")
    membership = schedule.get("unchanged_fields", {}).get("train_bank_membership_sha256")
    if membership != "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3":
        raise ArtifactError("controlled window schedule membership drift")
    return rows, list(labels), windows_per_update, str(membership)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _require_file(path: Path, expected_sha: str | None, label: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"official resident {label} is missing: {path}")
    if expected_sha:
        observed = _sha256_file(path)
        if observed != expected_sha:
            raise ArtifactError(f"official resident {label} SHA mismatch: {observed} != {expected_sha}")


def _has_static_w28_binding(config: Mapping[str, Any]) -> bool:
    return (
        config.get("resident_validation_proof") is True
        or config.get("static_w28_gate") is not None
    )


def _uses_static_w28_provider(config: Mapping[str, Any]) -> bool:
    """Return whether the public static-W28 rail owns provider and fixture binding."""
    return (
        config.get("recipe_id") == PUBLISHED_PRE_RECIPE_ID
        and _has_static_w28_binding(config)
    )


def _resolve_validation_corpus(
    config: Mapping[str, Any],
    *,
    teacher_root: Path,
    training_corpus: Path,
    published_pre_proof: bool,
) -> tuple[Path, str | None]:
    """Bind static PRE W28 to the accepted evaluation corpus, not the train bank."""
    if not published_pre_proof:
        return (
            Path(str(config.get("validation_corpus", training_corpus))).expanduser().resolve(),
            config.get("validation_corpus_sha256"),
        )
    path = Path(
        str(
            config.get(
                "validation_corpus",
                Path("/home/dnola/missions/DS4_TEACHER/static/windows_ds4_eval.json"),
            )
        )
    ).expanduser().resolve()
    expected = str(
        config.get(
            "validation_corpus_sha256", STATIC_W28_VALIDATION_CORPUS_SHA256
        )
    )
    if expected != STATIC_W28_VALIDATION_CORPUS_SHA256:
        raise ArtifactError("static W28 validation corpus declaration drift")
    return path, expected


def _resolve_scorer_aligned_training_corpus(config: Mapping[str, Any]) -> Path:
    """Select canonical static-W28 tokens for the explicitly aligned training arm."""
    training = Path(str(config["corpus"])).expanduser().resolve()
    source = config.get("w28_only_training_corpus_source")
    if source is None:
        return training
    if source != "canonical_eval":
        raise ArtifactError("W28-only training corpus source must equal canonical_eval")
    if not _uses_static_w28_provider(config):
        raise ArtifactError(
            "W28-only canonical-eval training corpus requires the static W28 provider"
        )
    path, expected = _resolve_validation_corpus(
        config,
        teacher_root=Path(str(config.get("validation_teacher_root", "."))),
        training_corpus=training,
        published_pre_proof=True,
    )
    if expected != STATIC_W28_VALIDATION_CORPUS_SHA256:
        raise ArtifactError("W28-only canonical-eval training corpus identity drift")
    return path


def _require_static_w28_teacher(teacher_root: Path) -> str:
    """Bind the public static-W28 rail to its accepted teacher tensor bytes."""
    path = teacher_root / "t8192_win28.pt"
    observed = _sha256_file(path)
    if observed != STATIC_W28_TEACHER_SHA256:
        raise ArtifactError(
            "static W28 validation teacher SHA mismatch: "
            f"{observed} != {STATIC_W28_TEACHER_SHA256}"
        )
    return observed


def _resolve_scorer_aligned_training_teacher_root(config: Mapping[str, Any]) -> Path:
    """Select one teacher source for both training loss and held-out scoring."""
    configured = config.get("scorer_aligned_training_teacher_root")
    root = Path(str(configured if configured is not None else config["teacher_root"]))
    root = root.expanduser().resolve()
    if configured is None:
        return root
    validation = config.get("validation_teacher_root")
    if not isinstance(validation, str) or not validation:
        raise ArtifactError(
            "scorer-aligned training teacher requires validation_teacher_root"
        )
    validation_root = Path(validation).expanduser().resolve()
    if validation_root != root:
        raise ArtifactError(
            "training and scorer teacher roots diverge under scorer-aligned support"
        )
    return root


def _resolve_trainer_source(config: Mapping[str, Any]) -> tuple[Path, str]:
    """Bind static PRE/U1 validation to the accepted trainer bytes."""
    if _uses_static_w28_provider(config):
        return (
            Path(__file__).resolve().parent / "assets" / "static_w28_modern_green_clean_u0.py",
            TRAINER_SHA256,
        )
    return (
        Path(str(config["trainer_source"])).expanduser().resolve(),
        str(config.get("trainer_source_sha256", TRAINER_SHA256)),
    )


def _install_runtime_modules(config: Mapping[str, Any]) -> None:
    """Install explicitly hashed wrapper/expert modules under trainer names."""
    extension_value = config.get("fast_k2_extension")
    wrapper_value = config.get("fast_k2_wrapper_source")
    expert_value = config.get("fast_v7_expert_source")
    if extension_value is None and wrapper_value is None and expert_value is None:
        return
    if not all(isinstance(value, str) for value in (extension_value, wrapper_value, expert_value)):
        raise ArtifactError("resident runtime module paths must be supplied together")
    extension = Path(str(extension_value)).expanduser().resolve()
    sealed_published_pre = _uses_static_w28_provider(config)
    if sealed_published_pre:
        # The deployed commit pin, not a warm parent config path, owns the exact
        # runtime arithmetic used to reproduce the imported sealed builder.
        # Attempt19 proved the inherited external wrapper/expert hashes remained
        # unchanged across canonical commits and silently bypassed both fixes.
        assets = Path(__file__).resolve().parent / "assets"
        # Static PRE/U1 identity comparisons must use the exact public provider
        # that produced the accepted PRE W28 value.  The mutable training assets
        # later acquired candidate-arithmetic experiments; rebinding validation
        # to those files made byte-identical model states score differently.
        wrapper = assets / "static_w28_fast_k2_grouped.py"
        expert = assets / "static_w28_fast_v7_expert_base.py"
        wrapper_sha = STATIC_W28_GROUPED_WRAPPER_SHA256
        expert_sha = STATIC_W28_GROUPED_EXPERT_SHA256
    else:
        wrapper = Path(str(wrapper_value)).expanduser().resolve()
        expert = Path(str(expert_value)).expanduser().resolve()
        wrapper_sha = str(config.get("fast_k2_wrapper_source_sha256", ""))
        expert_sha = str(config.get("fast_v7_expert_source_sha256", ""))
    extension_sha = str(config.get("fast_k2_extension_sha256", ""))
    _require_file(extension, extension_sha, "fast K2 extension")
    _require_file(wrapper, wrapper_sha, "fast K2 wrapper")
    _require_file(expert, expert_sha, "fast V7 expert source")
    os.environ["FAST_K2_EXTENSION"] = str(extension)
    os.environ["FAST_K2_EXTENSION_SHA256"] = extension_sha
    os.environ["FAST_K2_MODULE_NAME"] = str(config.get("fast_k2_module_name", extension.stem))
    _load_source_module("fast_k2_grouped", wrapper)
    expert_module = _load_source_module("fast_v7_expert_base", expert)
    expert_class = getattr(expert_module, "FullyResidentGroupedV7Experts", None)
    if expert_class is None:
        return
    swiglu_parameter = inspect.signature(expert_class).parameters.get("swiglu_limit")
    if swiglu_parameter is None:
        # The accepted PRE provider predates the trainer's constructor-only
        # SwiGLU field.  Adapt the public ABI outside the hash-bound provider so
        # its exact route arithmetic and source identity remain unchanged.
        class HistoricalNoLimitCompatibleExpert(expert_class):
            def __init__(
                self, *args: Any, swiglu_limit: float | None = None, **kwargs: Any
            ) -> None:
                del swiglu_limit
                super().__init__(*args, **kwargs)

        HistoricalNoLimitCompatibleExpert.__name__ = expert_class.__name__
        HistoricalNoLimitCompatibleExpert.__qualname__ = expert_class.__qualname__
        HistoricalNoLimitCompatibleExpert.__module__ = expert_class.__module__
        expert_module.FullyResidentGroupedV7Experts = HistoricalNoLimitCompatibleExpert
        return
    if swiglu_parameter.default is not inspect.Parameter.empty:
        return
    model_config_path = Path(str(config["model_root"])).expanduser().resolve() / "config.json"
    _require_file(model_config_path, None, "model config")
    model_config = json.loads(model_config_path.read_text())
    sealed_limit = float(model_config.get("swiglu_limit", float("nan")))
    if not math.isfinite(sealed_limit) or sealed_limit <= 0:
        raise ArtifactError("official resident model SwiGLU limit is invalid")

    class HistoricalConstructorCompatibleExpert(expert_class):
        def __init__(self, *args: Any, swiglu_limit: float = sealed_limit, **kwargs: Any) -> None:
            super().__init__(*args, swiglu_limit=swiglu_limit, **kwargs)

    HistoricalConstructorCompatibleExpert.__name__ = expert_class.__name__
    HistoricalConstructorCompatibleExpert.__qualname__ = expert_class.__qualname__
    HistoricalConstructorCompatibleExpert.__module__ = expert_class.__module__
    expert_module.FullyResidentGroupedV7Experts = HistoricalConstructorCompatibleExpert


def _cuda_sync(torch: Any) -> None:
    torch.cuda.synchronize()


def _checkpoint_route_replay_supported(gate: Any) -> bool:
    """Only learned top-k routers may replay a captured discrete route.

    Hash routers consume ``input_ids`` and must execute their original forward in
    both checkpoint passes.  Replacing that stateful lookup only during
    recomputation changes the saved-tensor path (the observed 343-vs-342 fault).
    """
    return (
        gate is not None
        and hasattr(gate, "score_fn")
        and hasattr(gate, "e_score_correction_bias")
        and not hasattr(gate, "tid2eid")
    )


def _checkpoint_topk_route(
    torch: Any,
    gate: Any,
    hidden_states: Any,
    *,
    fixed_indices: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Run one canonical learned top-k gate path with optional route binding.

    Non-reentrant checkpointing validates the tensors saved by the original and
    recomputed forwards.  The canonical top-k selection must therefore execute
    in both passes even though recomputation binds the differentiable weights to
    the discrete indices captured by the original forward.
    """
    flat = hidden_states.reshape(-1, int(gate.hidden_dim))
    logits = torch.nn.functional.linear(flat, gate.weight)
    scores = gate.score_fn(logits)
    fresh_indices = torch.topk(
        scores + gate.e_score_correction_bias,
        int(gate.top_k),
        dim=-1,
        sorted=False,
    ).indices
    indices = fresh_indices if fixed_indices is None else fixed_indices
    if tuple(indices.shape[:-1]) != tuple(scores.shape[:-1]):
        raise ArtifactError("checkpoint route replay geometry drift")
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * gate.routed_scaling_factor, indices


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
        self.activation_checkpointing = bool(config.get("activation_checkpointing", True))
        self.activation_checkpoint_interval = int(config.get("activation_checkpoint_interval", 1))
        self.checkpoint_use_reentrant = bool(config.get("checkpoint_use_reentrant", False))
        if self.activation_checkpoint_interval < 1:
            raise ArtifactError("activation checkpoint interval must be positive")
        self.controlled_arm_id = config.get("controlled_arm_id")
        self.controlled_arm = self.controlled_arm_id is not None
        self.published_pre_recipe = config.get("recipe_id") == PUBLISHED_PRE_RECIPE_ID
        if self.published_pre_recipe and self.controlled_arm:
            raise ArtifactError("published PRE recipe cannot also declare a controlled arm")
        self.published_pre_controlled_windows = (
            self.published_pre_recipe and "controlled_window_schedule" in config
        )
        default_pipeline_microbatch = 2 if (self.controlled_arm or self.published_pre_recipe) else PIPELINE_MICROBATCH
        self.pipeline_microbatch = int(config.get("pipeline_microbatch", default_pipeline_microbatch))
        if self.pipeline_microbatch < 1:
            raise ArtifactError("pipeline microbatch must be positive")
        self.score_pipeline_microbatch = int(
            config.get("score_pipeline_microbatch", PIPELINE_MICROBATCH)
        )
        if self.score_pipeline_microbatch < 1:
            raise ArtifactError("score pipeline microbatch must be positive")
        self.score_head_window_microbatch = int(
            config.get("score_head_window_microbatch", 1)
        )
        if self.score_head_window_microbatch < 1:
            raise ArtifactError("score head window microbatch must be positive")
        self.device = torch.device(str(config.get("device", "cuda")))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ArtifactError("official resident student requires a CUDA device")
        torch.cuda.set_device(int(config.get("cuda_device", 0)))
        self.layer_ranges = layer_ranges
        self.first, self.last = layer_ranges[rank]
        self.payload = payload
        self.state = payload.get("state")
        self.has_expert_plane_surface = _validate_trainable_state_schema(torch, payload)
        self.surface_names = (
            ("luts", "norms", "outputs", EXPERT_PLANE_SURFACE)
            if self.has_expert_plane_surface else ("luts", "norms", "outputs")
        )
        identity_value = payload.get("identity")
        identity = identity_value if isinstance(identity_value, Mapping) else {}
        self.global_step = int(payload.get("next_update", identity.get("next_update", 16)))
        if self.controlled_arm:
            if self.controlled_arm_id in FROM_U0_ARM_IDS:
                if not 0 <= self.global_step < 64:
                    raise ArtifactError("from-U0 controlled arm checkpoint cursor must be within U0..U63")
            elif not 16 <= self.global_step < 64:
                raise ArtifactError("from-U16 controlled arm checkpoint cursor must be within U16..U63")
        if self.controlled_arm:
            self.trainer_path = Path(str(config["trainer_source"])).expanduser().resolve()
            expected_trainer_sha = str(
                config.get("trainer_source_sha256", HISTORICAL_TRAINER_SHA256)
            )
        else:
            self.trainer_path, expected_trainer_sha = _resolve_trainer_source(config)
        _require_file(self.trainer_path, expected_trainer_sha, "trainer source")
        if self.controlled_arm and str(config.get("trainer_source_sha256", expected_trainer_sha)) != HISTORICAL_TRAINER_SHA256:
            raise ArtifactError("controlled arms require the sealed historical trainer")
        self.model_root = Path(str(config["model_root"])).expanduser().resolve()
        self.asset_root = Path(str(config["asset_root"])).expanduser().resolve()
        self.parent_root = Path(str(config["parent_root"])).expanduser().resolve()
        self.l034_roster = Path(str(config["l034_roster"])).expanduser().resolve()
        self.teacher_root = _resolve_scorer_aligned_training_teacher_root(config)
        self.corpus_path = _resolve_scorer_aligned_training_corpus(config)
        self.manifest_path = Path(str(config["manifest"])).expanduser().resolve()
        self.delta_dir = Path(str(config["delta_dir"])).expanduser().resolve()
        self.vq3b_dir = Path(str(config["vq3b_dir"])).expanduser().resolve()
        self._configure_import_environment()
        self._prepare_import_paths()
        _install_runtime_modules(config)
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
        self.parent_root = Path(str(config["parent_root"])).expanduser().resolve()
        self.l034_roster = Path(str(config["l034_roster"])).expanduser().resolve()
        self.corpus_path = _resolve_scorer_aligned_training_corpus(config)
        _require_file(self.model_root / "model.safetensors.index.json", MODEL_INDEX_SHA256, "model index")
        admission_path = self.asset_root / "code" / "JOINT_REPAIR_ADMISSION.json"
        _require_file(admission_path, str(config.get("admission_sha256", ADMISSION_SHA256)), "joint admission")
        training_corpus_sha = (
            STATIC_W28_VALIDATION_CORPUS_SHA256
            if config.get("w28_only_training_corpus_source") == "canonical_eval"
            else str(config.get("corpus_sha256", CORPUS_SHA256))
        )
        _require_file(self.corpus_path, training_corpus_sha, "training corpus")
        if not self.teacher_root.is_dir():
            raise ArtifactError(f"official resident teacher root is missing: {self.teacher_root}")
        admission = json.loads(admission_path.read_text())
        if admission.get("framework") != "banana-smasher":
            raise ArtifactError("official resident admission framework drift")
        if len(admission.get("trainable_roster", {}).get("luts", [])) != 43:
            raise ArtifactError("official resident LUT roster drift")
        self._configure_base()
        self.status: dict[str, Any] = {}
        self.student = self.trainer.ShardStudent(
            torch=torch,
            np=__import__("numpy"),
            base=self.base,
            official_k2=official_k2,
            model_root=self.model_root,
            admission=admission,
            parent_root=self.parent_root,
            l034_roster=self.l034_roster,
            input_state=payload,
            rank=rank,
            first=self.first,
            last=self.last,
            status_cb=self._status,
            defer_dense_l034=False,
        )
        self.luts, self.norms, self.outputs = self.trainer.expose_local_dense(torch, self.student, admission)
        self._load_local_trainable_state()
        self.expert_planes_l028_su_sv: list[tuple[str, Any]] = []
        self.expert_plane_candidate_read: dict[str, Any] | None = None
        if self.has_expert_plane_surface and self.first <= 28 <= self.last:
            self.expert_planes_l028_su_sv, self.expert_plane_candidate_read = (
                _bind_l28_expert_plane_state(
                    torch, self.student, self.state[EXPERT_PLANE_SURFACE]
                )
            )
        optimizer_lrs = (
            PUBLISHED_PRE_BASE_LRS if self.published_pre_recipe
            else HISTORICAL_BASE_LRS if self.controlled_arm else BASE_LRS
        )
        lr_calibration_divisors = {"luts": 4.6, "norms": 4.3, "outputs": 5.2}
        if config.get("authorized_gradient_calibration") == "measured-pre-gradient-max-ratio-v1":
            if config.get("gradient_lr_divisors") != lr_calibration_divisors:
                raise ArtifactError("measured PRE gradient LR calibration divisors drift")
            optimizer_lrs = {
                name: optimizer_lrs[name] / lr_calibration_divisors[name]
                for name in optimizer_lrs
            }
        self.equivalent_gradient_scale = (
            2.0 ** -192
            if config.get("authorized_optimizer_arithmetic_repair")
                == "adam-fp64-state-and-update-arithmetic"
            else 1.0
        )
        optimizer_groups = [
                {"params": [p for _name, p in self.luts], "lr": optimizer_lrs["luts"], "group_name": "luts"},
                {"params": [p for _name, p in self.norms], "lr": optimizer_lrs["norms"], "group_name": "norms"},
                {"params": [p for _name, p in self.outputs], "lr": optimizer_lrs["outputs"], "group_name": "outputs"},
        ]
        if self.has_expert_plane_surface:
            optimizer_groups.append({
                "params": [p for _name, p in self.expert_planes_l028_su_sv],
                "lr": U60_EXPERT_PLANE_BASE_LR,
                "group_name": EXPERT_PLANE_SURFACE,
            })
        self.optimizer = _fp64_state_adam(
            torch, optimizer_groups,
            gradient_scale=self.equivalent_gradient_scale,
        )
        if self.published_pre_recipe:
            lr_lambda = lambda local_step: _published_pre_recipe_policy(
                self.config,
                int(local_step),
                authenticated_u60_continuation=self.has_expert_plane_surface,
                terminal_scheduler_cursor=True,
            )[1]
        elif self.controlled_arm:
            lr_lambda = lambda local_step: _controlled_arm_policy(
                self.config, _controlled_scheduler_step(str(self.controlled_arm_id), int(local_step))
            )[1]
        else:
            lr_lambda = self.trainer.current_multiplier
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lr_lambda] * len(optimizer_groups)
        )
        self._load_optimizer_scheduler_state()
        self._load_training_data()
        self._load_controlled_window_schedule()
        self._init_distributed()

    def _prepare_import_paths(self) -> None:
        for path in (
            self.trainer_path.parent,
            self.asset_root / "source",
            self.asset_root / "source" / "site",
            Path("/home/dnola/missions/LP4_REPAIR/src_lp4"),
            Path("/home/dnola/missions/LP4_REPAIR/src"),
        ):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def _configure_import_environment(self) -> None:
        """Bind immutable base-loader inputs before importing its source."""
        os.environ["BR_MANIFEST"] = str(self.manifest_path)
        os.environ["BR_DELTA_DIR"] = str(self.delta_dir)
        os.environ["BR_VQ3B_DIR"] = str(self.vq3b_dir)
        os.environ["BR_CORPUS"] = str(self.corpus_path)
        os.environ["BR_TEACH"] = str(self.teacher_root)
        canonical_windows = list(range(20, 84))
        for key, config_key in (("BR_TRAIN", "train_windows"), ("BR_PROBE", "probe_windows")):
            configured = self.config.get(config_key, canonical_windows)
            if isinstance(configured, str):
                value = configured
            else:
                value = ",".join(str(int(window)) for window in configured)
            if not value:
                raise ArtifactError(f"official resident {config_key} cannot be empty")
            os.environ[key] = value

    def _load_base(self) -> Any:
        path = self.asset_root / "source" / "base_binrepair_e2e.py"
        _require_file(path, None, "base resident model source")
        module = _load_source_module(f"banana_smasher_modern_green_base_{os.getpid()}_{self.rank}", path)
        module.T.CKPT = str(self.model_root)
        module.T.DEV = "cuda"
        return module

    def _configure_base(self) -> None:
        os.environ["BR_CORPUS"] = str(self.corpus_path)
        os.environ["BR_TEACH"] = str(self.teacher_root)
        if self.published_pre_recipe and _has_static_w28_binding(self.config):
            from .official_k2_resident_score import (
                T8192_BUILDER_SOURCE_SHA256,
                _configured_attention_implementation,
            )

            sealed_attention = _configured_attention_implementation({})
            if sealed_attention != "eager":
                raise ArtifactError("sealed builder attention implementation drift")
            attention = str(
                self.config.get(
                    "resident_validation_attention_implementation", sealed_attention
                )
            ).lower()
            if attention not in {"eager", "sdpa"}:
                raise ArtifactError(
                    "resident validation attention implementation must be eager or sdpa"
                )
            # Bind both the attention implementation and routed-expert arithmetic
            # to the imported sealed builder.  The ordinary grouped kernel keeps
            # weights implicit and accumulates the projection in FP32; RUN1698
            # materialized each inverse-transformed weight at BF16 before its BF16
            # GEMM.  The hash-bound grouped wrapper already exposes that exact
            # arithmetic path without source reads or a second scorer.
            expert_implementation = str(
                self.config.get(
                    "resident_validation_expert_implementation", "sealed_bf16_full_weight"
                )
            ).lower()
            if expert_implementation not in {
                "sealed_bf16_full_weight", "packed_cuda_bf16_boundary",
            }:
                raise ArtifactError(
                    "resident validation expert implementation must be "
                    "sealed_bf16_full_weight or packed_cuda_bf16_boundary"
                )
            os.environ["BR_ATTN_IMPL"] = attention
            packed_validation = expert_implementation == "packed_cuda_bf16_boundary"
            os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "0" if packed_validation else "1"
            os.environ["FAST_K2_SEALED_PROJECTION_BF16"] = "1" if packed_validation else "0"
            os.environ["FAST_K2_SEALED_NO_SWIGLU_CLAMP"] = "1"
            self.sealed_builder_binding = {
                "builder_source_sha256": T8192_BUILDER_SOURCE_SHA256,
                "provider_wrapper_sha256": STATIC_W28_GROUPED_WRAPPER_SHA256,
                "provider_expert_sha256": STATIC_W28_GROUPED_EXPERT_SHA256,
                "attention_implementation": attention,
                "expert_arithmetic": "accepted-static-w28-grouped-k2-provider",
                "expert_swiglu_clamp": "accepted-static-w28-provider-boundary",
                "fixture": "canonical-eval-corpus-token_ids-padded-to-2048",
                "readout": "full-softmax-gather-at-teacher-idx-fp16",
            }
            if packed_validation:
                self.sealed_builder_binding["expert_implementation"] = expert_implementation
        else:
            os.environ.setdefault("BR_ATTN_IMPL", "sdpa")
        os.environ.setdefault("BR_FAST_STACK", "1")
        self.base.T.CKPT = str(self.model_root)
        self.base.T.DEV = "cuda"
        import random
        random.seed(1701)
        self.torch.manual_seed(1701)
        self.torch.cuda.manual_seed_all(1701)

    def _status(self, **fields: Any) -> None:
        self.status.update(fields)
        progress_callback = self.config.get("progress_callback")
        if callable(progress_callback):
            progress_callback(**fields)

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

    def _restore_measured_gradient_lr_calibration(self) -> None:
        """Apply the authorized LR calibration after loading legacy optimizer state.

        ``Optimizer.load_state_dict`` restores the checkpoint param-group LRs and
        therefore overwrites the calibrated constructor values.  Old published-PRE
        checkpoints do not carry a calibration marker; divide those loaded values
        exactly once and persist markers so later resumes remain idempotent.
        """
        calibration = "measured-pre-gradient-max-ratio-v1"
        if self.config.get("authorized_gradient_calibration") != calibration:
            return
        divisors = {"luts": 4.6, "norms": 4.3, "outputs": 5.2}
        if self.config.get("gradient_lr_divisors") != divisors:
            raise ArtifactError("measured PRE gradient LR calibration divisors drift")
        needs_scheduler_scale = False
        for surface, group in zip(("luts", "norms", "outputs"), self.optimizer.param_groups):
            divisor = divisors[surface]
            marker = group.get("gradient_lr_divisor")
            if marker is None:
                group["lr"] = float(group["lr"]) / divisor
                if "initial_lr" in group:
                    group["initial_lr"] = float(group["initial_lr"]) / divisor
                needs_scheduler_scale = True
            elif float(marker) != divisor:
                raise ArtifactError(f"loaded {surface} gradient LR calibration marker drift")
            group["gradient_lr_divisor"] = divisor
            group["authorized_gradient_calibration"] = calibration
        if needs_scheduler_scale:
            self.scheduler.base_lrs = [
                float(value) / divisors[surface] if surface in divisors else float(value)
                for surface, value in zip(self.surface_names, self.scheduler.base_lrs)
            ]
            self.scheduler._last_lr = [
                float(value) / divisors[surface] if surface in divisors else float(value)
                for surface, value in zip(self.surface_names, self.scheduler._last_lr)
            ]

    def _load_optimizer_scheduler_state(self) -> None:
        optimizer_payload = self.payload.get("optimizer", self.payload.get("optimizer_state"))
        scheduler_payload = self.payload.get("scheduler", self.payload.get("scheduler_state"))
        if self.published_pre_recipe and self.global_step == 0:
            if optimizer_payload is not None or scheduler_payload is not None:
                raise ArtifactError("published PRE must start with fresh Adam and scheduler state")
            return
        if not isinstance(optimizer_payload, Mapping):
            raise ArtifactError("U16 checkpoint is missing the shared Adam optimizer state")
        groups = optimizer_payload.get("param_groups")
        global_state = optimizer_payload.get("state")
        if (
            not isinstance(groups, list)
            or len(groups) != len(self.surface_names)
            or not isinstance(global_state, Mapping)
        ):
            raise ArtifactError("checkpoint Adam state has no canonical trainable-surface lineage")
        local_state = self.optimizer.state_dict()
        local_groups = local_state["param_groups"]
        local_rows = {
            "luts": self.luts,
            "norms": self.norms,
            "outputs": self.outputs,
            EXPERT_PLANE_SURFACE: self.expert_planes_l028_su_sv,
        }
        for index, surface in enumerate(self.surface_names):
            names = [name for name, _param in local_rows[surface]]
            global_names = list(self.state[surface])
            source_group = groups[index]
            source_ids = list(source_group.get("params", []))
            if len(source_ids) != len(global_names):
                raise ArtifactError(f"checkpoint optimizer {surface} names/IDs do not match")
            global_id_by_name = dict(zip(global_names, source_ids))
            local_ids = local_groups[index]["params"]
            for name, local_id in zip(names, local_ids):
                global_id = global_id_by_name.get(name)
                if global_id is None:
                    raise ArtifactError(f"checkpoint optimizer state missing official parameter {name}")
                value = global_state.get(global_id, global_state.get(str(global_id)))
                if value is not None:
                    local_state["state"][local_id] = _cpu_tree(self.torch, value)
            local_groups[index].update({key: value for key, value in source_group.items() if key != "params"})
            local_groups[index]["params"] = local_ids
        self.optimizer.load_state_dict(local_state)
        if self.controlled_arm:
            controlled_base_lrs, multiplier, _windows = _controlled_arm_policy(
                self.config, self.global_step
            )
            for surface, group in zip(("luts", "norms", "outputs"), self.optimizer.param_groups):
                group["initial_lr"] = controlled_base_lrs[surface]
                group["lr"] = controlled_base_lrs[surface] * multiplier
        scheduler_payload = self.payload.get("scheduler", self.payload.get("scheduler_state"))
        if not isinstance(scheduler_payload, Mapping):
            raise ArtifactError("checkpoint is missing the shared LambdaLR scheduler state")
        if self.controlled_arm and self.global_step == _controlled_arm_origin(str(self.controlled_arm_id)):
            # Optimizer moments are inherited, but every registered trajectory
            # starts its declared schedule at its own U0 or U16 origin.
            return
        try:
            self.scheduler.load_state_dict(dict(scheduler_payload))
        except Exception as exc:
            raise ArtifactError(f"U16 LambdaLR state cannot load: {exc}") from exc
        self._restore_measured_gradient_lr_calibration()

    def _load_training_data(self) -> None:
        if self.controlled_arm or getattr(self, "published_pre_controlled_windows", False):
            schedule_path = Path(str(self.config.get("controlled_window_schedule", ""))).expanduser().resolve()
            expected_sha = str(self.config.get("controlled_window_schedule_sha256", ""))
            _require_file(schedule_path, expected_sha, "controlled window schedule")
            self.controlled_schedule = json.loads(schedule_path.read_text())
            if self.controlled_arm:
                bank = self.controlled_schedule.get("train_bank", {}).get("windows_by_category", {})
                ordered = sorted({int(window) for values in bank.values() for window in values})
                if len(ordered) != 64:
                    raise ArtifactError("controlled train bank must contain exactly 64 unique windows")
            else:
                rows, _labels, _count, _membership = _published_pre_controlled_schedule(
                    self.controlled_schedule, self.config
                )
                ordered = sorted({
                    int(window) for row in rows for window in row.get("windows", [])
                })
        elif self.published_pre_recipe and self.config.get("resident_validation_proof") is True:
            # This rail is admitted by the public API for exactly one update.
            # Preloading all 64 training teachers made cold construction read
            # ~6.4 GiB although the sealed PRE recipe consumes only two rows.
            # Keep the proof resident while loading exactly its U1 dose.
            _base_lrs, _multiplier, ordered = _published_pre_recipe_policy(
                self.config,
                self.global_step,
                authenticated_u60_continuation=self.has_expert_plane_surface,
            )
        else:
            ordered = list(range(20, 84))
        corpus = self.base.T.load_corpus()
        self.ids_cache = {
            window: self.base.T.window_ids(corpus, window)[0].unsqueeze(0).to(self.student.device)
            for window in ordered
        }
        self.real_lengths = {window: self.base.T.window_ids(corpus, window)[1] for window in ordered}
        self.teacher_cache = {}
        if self.rank == 1:
            for window in ordered:
                self.teacher_cache[window] = self.base.T.teacher_rows(window)

    def _load_controlled_window_schedule(self) -> None:
        self.controlled_windows: dict[int, list[int]] = {}
        if not (self.controlled_arm or self.published_pre_controlled_windows):
            return
        if self.controlled_arm:
            _controlled_arm_policy(self.config, self.global_step)
        else:
            _published_pre_recipe_policy(
                self.config,
                self.global_step,
                authenticated_u60_continuation=self.has_expert_plane_surface,
            )
        schedule = self.controlled_schedule
        if self.controlled_arm:
            membership = schedule.get("train_bank", {}).get("membership_sha256")
            if membership != "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3":
                raise ArtifactError("controlled window schedule membership drift")
            rows = schedule.get("updates")
            if not isinstance(rows, list):
                raise ArtifactError("controlled window schedule has no updates")
            _base_lrs, _multiplier, expected_count = _controlled_arm_policy(
                self.config, self.global_step
            )
            origin = _controlled_arm_origin(str(self.controlled_arm_id))
            source_labels: list[int] = []
        else:
            rows, source_labels, expected_count, _membership = _published_pre_controlled_schedule(
                schedule, self.config
            )
            origin = 0
        schedule_mode = "controlled arm" if self.controlled_arm else "controlled window schedule"
        self.controlled_schedule_source_rows: dict[int, int] = {}
        for ordinal, row in enumerate(rows):
            update = int(row.get("global_update", -1)) - 1 if self.controlled_arm else ordinal
            if "windows" in row:
                windows = [int(value) for value in row["windows"]]
            else:
                order = row.get("microbatch_category_order", [])
                by_category = row.get("windows_by_category", {})
                windows = [int(by_category[name]["window_id"]) for name in order]
            if len(windows) != expected_count or len(set(windows)) != expected_count:
                raise ArtifactError(f"{schedule_mode} window dose drift at U{update + 1}")
            if any(window not in self.ids_cache for window in windows):
                raise ArtifactError(f"{schedule_mode} window membership drift at U{update + 1}")
            self.controlled_windows[update] = windows
            if not self.controlled_arm:
                self.controlled_schedule_source_rows[update] = source_labels[ordinal]
        required_updates = range(origin, 64) if self.controlled_arm else range(0, len(rows))
        if any(update not in self.controlled_windows for update in required_updates):
            end = 64 if self.controlled_arm else len(rows)
            raise ArtifactError(f"controlled window schedule must cover U{origin + 1}..U{end}")

    def _init_distributed(self) -> None:
        socket_ifname = str(self.config.get("nccl_socket_ifname", ""))
        if not socket_ifname or not (Path("/sys/class/net") / socket_ifname).is_dir():
            raise ArtifactError("official resident continuation requires a live NCCL socket interface")
        os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname
        os.environ["GLOO_SOCKET_IFNAME"] = socket_ifname
        if self.dist.is_initialized():
            if self.dist.get_world_size() != 2 or self.dist.get_rank() != self.rank:
                raise ArtifactError("existing process group does not match the exact two-Spark rank")
        else:
            master_addr = str(self.config.get("master_addr", "127.0.0.1"))
            master_port = int(self.config.get("master_port", 29598))
            init_method = str(self.config.get("init_method", f"tcp://{master_addr}:{master_port}"))
            try:
                self.dist.init_process_group(
                    backend=str(self.config.get("distributed_backend", "nccl")),
                    init_method=init_method,
                    rank=self.rank,
                    world_size=2,
                )
            except Exception as exc:
                raise ArtifactError(f"official two-Spark process-group initialization failed: {exc}") from exc
        self._warm_p2p_communicator()

    def _warm_p2p_communicator(self) -> None:
        """Collectively create the two-rank NCCL P2P communicator before a long forward."""
        peer = 1 - self.rank
        outgoing = self.torch.full((1,), self.rank, dtype=self.torch.int32, device=self.student.device)
        incoming = self.torch.empty_like(outgoing)
        try:
            if self.rank == 0:
                self._batch_p2p_send(outgoing, dst=peer)
                self._batch_p2p_recv(incoming, src=peer)
            else:
                self._batch_p2p_recv(incoming, src=peer)
                self._batch_p2p_send(outgoing, dst=peer)
            _cuda_sync(self.torch)
        except Exception as exc:
            raise ArtifactError(f"official two-Spark P2P communicator warmup failed: {exc}") from exc
        if int(incoming.item()) != peer:
            raise ArtifactError("official two-Spark P2P communicator warmup identity drift")

    def _batch_p2p_send(self, tensor: Any, *, dst: int) -> None:
        """Send through the same batched P2P API used by warmup."""
        print(f"BANANA_P2P rank={self.rank} op=send phase=enter peer={dst}", flush=True)
        requests = self.dist.batch_isend_irecv([
            self.dist.P2POp(self.dist.isend, tensor, dst)
        ])
        if len(requests) != 1:
            raise ArtifactError("official resident batched P2P send request drift")
        requests[0].wait()
        print(f"BANANA_P2P rank={self.rank} op=send phase=complete peer={dst}", flush=True)

    def _batch_p2p_recv(self, tensor: Any, *, src: int) -> None:
        """Receive through the same batched P2P API used by warmup."""
        print(f"BANANA_P2P rank={self.rank} op=recv phase=enter peer={src}", flush=True)
        requests = self.dist.batch_isend_irecv([
            self.dist.P2POp(self.dist.irecv, tensor, src)
        ])
        if len(requests) != 1:
            raise ArtifactError("official resident batched P2P receive request drift")
        requests[0].wait()
        print(f"BANANA_P2P rank={self.rank} op=recv phase=complete peer={src}", flush=True)

    def _batch_p2p_exchange(
        self,
        outgoing: Any,
        *,
        dst: int,
        incoming: Any,
        src: int,
    ) -> None:
        """Exchange one pair through the known-green one-way batched P2P path."""
        if self.rank == 0:
            self._batch_p2p_send(outgoing, dst=dst)
            self._batch_p2p_recv(incoming, src=src)
        else:
            self._batch_p2p_recv(incoming, src=src)
            self._batch_p2p_send(outgoing, dst=dst)

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

    @staticmethod
    def _sink_corrected_sdpa_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        **_kwargs: Any,
    ) -> tuple[Any, None]:
        """Run fused SDPA while retaining DeepseekV4 attention-sink semantics.

        The ordinary Transformers SDPA adapter drops ``module.sinks``.  Keep its
        fused output, then recover the small missing denominator with a bounded
        query-chunk log-sum-exp.  This repeats QK only (not softmax/value) and
        avoids backend-private padded/base conventions in the low-level LSE.
        """
        torch = __import__("torch")
        heads = int(query.shape[1])
        if int(key.shape[1]) != heads:
            if heads % int(key.shape[1]):
                raise ArtifactError("sink-corrected SDPA GQA head geometry drift")
            repeats = heads // int(key.shape[1])
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=float(dropout), is_causal=False, scale=float(scaling),
        )
        query_rows = int(query.shape[-2])
        chunk_size = int(_kwargs.get("sink_lse_query_chunk_size", 256))
        if chunk_size <= 0:
            raise ArtifactError("sink-corrected SDPA chunk must be positive")
        key_t = key.transpose(2, 3)
        sinks = module.sinks.reshape(1, -1, 1)
        for start in range(0, query_rows, chunk_size):
            end = min(start + chunk_size, query_rows)
            scores = torch.matmul(query[..., start:end, :], key_t) * float(scaling)
            if attention_mask is not None:
                scores = scores + attention_mask[..., start:end, :]
            logsumexp = torch.logsumexp(scores, dim=-1)
            sink_scale = torch.sigmoid(logsumexp - sinks.to(logsumexp.dtype)).to(output.dtype)
            output[..., start:end, :].mul_(sink_scale.unsqueeze(-1))
        return output.transpose(1, 2).contiguous(), None

    def _install_sink_corrected_sdpa(self) -> None:
        if getattr(self, "_sink_sdpa_installed", False):
            return
        layer = self.student.model.model.layers[self.first]
        attention = layer.self_attn
        forward = attention.forward
        function = getattr(forward, "__func__", forward)
        namespace = getattr(function, "__globals__", None)
        interface = namespace.get("ALL_ATTENTION_FUNCTIONS") if isinstance(namespace, dict) else None
        register = getattr(interface, "register", None)
        if not callable(register):
            raise ArtifactError("sink-corrected SDPA registration seam drift")
        implementation = "official_k2_sink_corrected_sdpa"
        register(implementation, self._sink_corrected_sdpa_forward)
        self.student.model.config._attn_implementation = implementation
        self.student.model.config._attn_implementation_internal = implementation
        self._sink_sdpa_installed = True

    @staticmethod
    def _snapshot_layer_cache(layer_cache: Any) -> tuple[Any, Any, bool, int | None]:
        """Capture an immutable copy of the exact pre-layer cache state."""
        cumulative_length = getattr(layer_cache, "cumulative_length", None)
        keys = layer_cache.keys
        values = layer_cache.values
        return (
            keys.detach().clone() if hasattr(keys, "detach") else keys,
            values.detach().clone() if hasattr(values, "detach") else values,
            bool(layer_cache.is_initialized),
            cumulative_length,
        )

    @staticmethod
    def _restore_layer_cache(
        layer_cache: Any,
        snapshot: tuple[Any, Any, bool, int | None],
    ) -> None:
        """Restore the exact pre-layer state before checkpoint recomputation."""
        keys, values, initialized, cumulative_length = snapshot
        layer_cache.keys = keys
        layer_cache.values = values
        layer_cache.is_initialized = initialized
        if cumulative_length is not None:
            layer_cache.cumulative_length = cumulative_length

    def _run_layers(
        self,
        hidden: Any,
        ids: Any,
        train: bool,
        *,
        layer_capture: Any | None = None,
    ) -> Any:
        from transformers.cache_utils import DynamicCache
        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        cache = DynamicCache(config=self.student.config)
        if str(self.config.get("resident_validation_attention_implementation", "eager")).lower() == "sdpa":
            self._install_sink_corrected_sdpa()
        pos, pe, mask = self._positional(ids, template, cache)
        if (
            train
            and self.activation_checkpointing
            and self.checkpoint_use_reentrant
            and not hidden.requires_grad
        ):
            # Reentrant checkpointing requires at least one grad-requiring input;
            # rank 0 starts from frozen embeddings, while later pipeline ranks
            # already receive a grad-enabled activation leaf.
            hidden = hidden.detach().requires_grad_(True)

        def layer_block(
            current: Any,
            start: int,
            stop: int,
            *,
            recompute: bool = False,
            snapshots: dict[int, tuple[Any, Any, bool, int | None]] | None = None,
            route_indices: dict[int, Any] | None = None,
        ) -> Any:
            for index in range(start, stop):
                if snapshots is not None:
                    if recompute:
                        self._restore_layer_cache(cache.layers[index], snapshots[index])
                    else:
                        snapshots[index] = self._snapshot_layer_cache(cache.layers[index])

                # DeepseekV4 top-k routing can cross a near-tie on an otherwise
                # deterministic CUDA recomputation.  A changed expert population
                # changes checkpointed tensor geometry and, more importantly,
                # would backpropagate through a different route than the forward.
                # Retain only the discrete forward indices; on recomputation,
                # rebuild logits and differentiable weights for those exact indices.
                layer = self.student.model.model.layers[index]
                gate = getattr(getattr(layer, "mlp", None), "gate", None)
                original_gate_forward = None
                if route_indices is not None and _checkpoint_route_replay_supported(gate):
                    assert gate is not None
                    original_gate_forward = gate.forward

                    def checkpoint_gate(
                        hidden_states: Any,
                        *_gate_args: Any,
                        _gate: Any = gate,
                        _index: int = index,
                        **_gate_kwargs: Any,
                    ):
                        fixed_indices = route_indices.get(_index) if recompute else None
                        result = _checkpoint_topk_route(
                            self.torch,
                            _gate,
                            hidden_states,
                            fixed_indices=fixed_indices,
                        )
                        if not recompute:
                            route_indices[_index] = result[2].detach()
                        return result

                    gate.forward = checkpoint_gate
                try:
                    # Decoder-layer KV state is token-time state, not an
                    # inter-layer activation frontier. Evaluation and checkpoint
                    # recomputation therefore require one isolated cache per layer;
                    # otherwise each layer appends another full 8192-token KV plane,
                    # producing quadratic growth, wrong logits, and eventual OOM.
                    active_cache = (
                        DynamicCache(config=self.student.config)
                        if not train or snapshots is not None
                        else cache
                    )
                    current = layer(
                        current,
                        position_embeddings=pe,
                        position_ids=pos,
                        attention_mask=mask,
                        input_ids=ids,
                        past_key_values=active_cache,
                    )
                    if layer_capture is not None and not recompute:
                        layer_capture(index, current)
                finally:
                    if original_gate_forward is not None:
                        assert gate is not None
                        gate.forward = original_gate_forward
                    if snapshots is not None and index in snapshots:
                        # No later layer consumes this layer's KV cache during a
                        # full-sequence training pass. Restore immediately so
                        # non-reentrant recomputation cannot observe the original
                        # forward's appended keys/values (1280-vs-1536 drift).
                        self._restore_layer_cache(cache.layers[index], snapshots[index])
            return current
        if train and self.activation_checkpointing:
            for start in range(self.first, self.last + 1, self.activation_checkpoint_interval):
                stop = min(start + self.activation_checkpoint_interval, self.last + 1)
                snapshots: dict[int, tuple[Any, Any, bool, int | None]] = {}
                route_indices: dict[int, Any] = {}
                invocation = {"count": 0}

                # Keep invocation state private to this checkpoint segment.  Both
                # reentrant and non-reentrant checkpoint implementations call the
                # function once for the original forward and again for backward
                # recomputation; a shared context flag is not reliable when several
                # checkpoint segments are live in the same autograd graph.
                def checkpointed_block(
                    current: Any,
                    _start: int = start,
                    _stop: int = stop,
                    _snapshots: dict[int, tuple[Any, Any, bool, int | None]] = snapshots,
                    _route_indices: dict[int, Any] = route_indices,
                    _invocation: dict[str, int] = invocation,
                ) -> Any:
                    recompute = _invocation["count"] > 0
                    _invocation["count"] += 1
                    return layer_block(
                        current,
                        _start,
                        _stop,
                        recompute=recompute,
                        snapshots=_snapshots,
                        route_indices=_route_indices,
                    )

                hidden = self.checkpoint(
                    checkpointed_block,
                    hidden,
                    use_reentrant=self.checkpoint_use_reentrant,
                )
        else:
            hidden = layer_block(hidden, self.first, self.last + 1)
        return hidden

    def _loss_group(self, hidden: Any, group: list[int]) -> Any:
        final = self.student.model.model.norm(self.student.model.model.hc_head(hidden))
        training_token_span_value = self.config.get("w28_only_training_token_span")
        training_token_span = None
        if training_token_span_value is not None:
            if group != [28]:
                raise ArtifactError("W28-only training token span requires window membership [28]")
            training_token_span = int(training_token_span_value)
            if training_token_span != 1024:
                raise ArtifactError("W28-only training token span must equal canonical scorer span 1024")
        training_readout_normalization = self.config.get("w28_only_training_readout_normalization")
        if training_readout_normalization is not None:
            if group != [28]:
                raise ArtifactError(
                    "W28-only training readout normalization requires window membership [28]"
                )
            if training_readout_normalization != "full_vocab_log_softmax_then_gather_fp16":
                raise ArtifactError(
                    "W28-only training readout normalization must match the static scorer"
                )
        losses = []
        for row, window in enumerate(group):
            idx, lp_n, p_n = self.teacher_cache[window]
            length = self.real_lengths[window]
            if training_token_span is not None:
                if length < training_token_span:
                    raise ArtifactError("W28 training fixture has fewer tokens than canonical scorer span")
                length = min(length, training_token_span)
            logits = self.student.model.lm_head(final[row, :length].to(self.torch.bfloat16))
            if training_readout_normalization is not None:
                logprob = self.torch.log_softmax(logits, dim=-1)
                q = logprob.gather(1, idx[:length]).to(self.torch.float16).float()
            else:
                q = logits.gather(1, idx[:length]).float()
            qn = q - q.logsumexp(-1, keepdim=True)
            losses.append((p_n[:length] * (lp_n[:length] - qn)).sum(-1).mean())
        return self.torch.stack(losses).mean()

    def diagnose_w28_forward_modes(self) -> dict[str, Any]:
        """Compare the frozen W28 train/static forwards without an optimizer step."""
        if self.config.get("w28_forward_mode_ab_probe") is not True:
            raise ArtifactError("W28 forward-mode A/B requires explicit probe authorization")
        if self.config.get("w28_only_training_token_span") != 1024:
            raise ArtifactError("W28 forward-mode A/B requires the canonical 1024-token span")

        torch = self.torch
        ids = self.ids_cache[28]
        shape = (
            1,
            self.base.T.T_TRAIN,
            int(self.student.config.hc_mult),
            int(self.student.config.hidden_size),
        )
        model = self.student.model
        original_mode = bool(model.training)

        def fingerprint(index: int, current: Any) -> dict[str, Any]:
            detached = current.detach().contiguous()
            raw = detached.view(torch.uint8).cpu().numpy().tobytes()
            return {
                "layer": index,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "finite": bool(torch.isfinite(detached).all().item()),
            }

        def run_branch(train: bool) -> dict[str, Any]:
            captures: list[dict[str, Any]] = []
            embeds = None

            def capture(index: int, current: Any) -> None:
                captures.append(fingerprint(index, current))

            model.train(train)
            started = time.perf_counter()
            if self.rank == 0:
                embeds = model.model.embed_tokens(ids)
                hidden = embeds.unsqueeze(2).expand(
                    -1, -1, self.student.config.hc_mult, -1
                ).contiguous()
                hidden = self._run_layers(hidden, ids, train, layer_capture=capture)
                _cuda_sync(torch)
                if tuple(hidden.shape) != shape:
                    raise ArtifactError("W28 forward-mode A/B rank0 activation geometry drift")
                self._batch_p2p_send(hidden.detach().contiguous(), dst=1)
                loss = None
            else:
                hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                self._batch_p2p_recv(hidden, src=0)
                if train:
                    hidden.requires_grad_(True)
                hidden = self._run_layers(hidden, ids, train, layer_capture=capture)
                loss = float(self._loss_group(hidden, [28]).detach().cpu())
                _cuda_sync(torch)
            elapsed = time.perf_counter() - started
            del hidden
            if embeds is not None:
                del embeds
            torch.cuda.empty_cache()
            return {
                "rank": self.rank,
                "forward_train_argument": train,
                "model_mode": "train" if train else "eval",
                "layer_fingerprints": captures,
                "support_normalized_kld": loss,
                "wall_seconds": elapsed,
            }

        try:
            training = run_branch(True)
            self.dist.barrier()
            static = run_branch(False)
            self.dist.barrier()
        finally:
            model.train(original_mode)

        first_local = None
        if len(training["layer_fingerprints"]) != len(static["layer_fingerprints"]):
            raise ArtifactError("W28 forward-mode A/B layer capture count drift")
        for left, right in zip(
            training["layer_fingerprints"], static["layer_fingerprints"]
        ):
            if left["layer"] != right["layer"]:
                raise ArtifactError("W28 forward-mode A/B layer ordering drift")
            if left["sha256"] != right["sha256"]:
                first_local = int(left["layer"])
                break

        local = {
            "rank": self.rank,
            "layer_range": [self.first, self.last],
            "first_divergent_layer": first_local,
            "training": training,
            "static": static,
        }
        gathered: list[Any] = [None for _ in range(self.dist.get_world_size())]
        self.dist.all_gather_object(gathered, local)
        divergent = [
            (int(row["first_divergent_layer"]), int(row["rank"]))
            for row in gathered
            if row["first_divergent_layer"] is not None
        ]
        result = {
            "schema": "banana-smasher-w28-forward-mode-layer-ab-v1",
            "status": "PASS",
            "first_divergent_layer": min(divergent)[0] if divergent else None,
            "first_divergent_rank": min(divergent)[1] if divergent else None,
            "all_layer_outputs_bitwise_equal": not divergent,
            "rank_rows": gathered,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_saves": 0,
            "model_mode_restored": bool(model.training) == original_mode,
        }
        rows: list[Any] = [result if self.rank == 1 else None]
        self.dist.broadcast_object_list(rows, src=1)
        if not isinstance(rows[0], Mapping):
            raise ArtifactError("W28 forward-mode A/B fan-in produced no result")
        return dict(rows[0])

    def diagnose_w28_batch_context(self) -> dict[str, Any]:
        """Compare row-0 W28 under singleton and sealed mb=2 execution context."""
        if self.config.get("w28_batch_context_ab_probe") is not True:
            raise ArtifactError("W28 batch-context A/B requires explicit probe authorization")
        if self.config.get("w28_only_training_corpus_source") != "canonical_eval":
            raise ArtifactError("W28 batch-context A/B requires canonical-eval training tokens")
        if self.config.get("w28_only_training_token_span") != 1024:
            raise ArtifactError("W28 batch-context A/B requires the canonical 1024-token span")
        validation_teacher_root = self.config.get("validation_teacher_root")
        if not isinstance(validation_teacher_root, str) or not validation_teacher_root:
            raise ArtifactError("W28 batch-context A/B requires validation_teacher_root")
        prepared = self.preload_validation([28], validation_teacher_root)
        validation_ids = prepared.get("ids")
        if not isinstance(validation_ids, Mapping) or any(
            window not in validation_ids for window in (28, 56)
        ):
            raise ArtifactError("W28 batch-context A/B requires preloaded W28 and W56 inputs")

        torch = self.torch

        def fingerprint(index: int, current: Any) -> dict[str, Any]:
            row0 = current[0:1].detach().contiguous()
            raw = row0.view(torch.uint8).cpu().numpy().tobytes()
            return {
                "layer": index,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(row0.shape),
                "dtype": str(row0.dtype),
                "finite": bool(torch.isfinite(row0).all().item()),
            }

        def run_branch(windows: tuple[int, ...]) -> dict[str, Any]:
            captures: list[dict[str, Any]] = []

            def capture(index: int, current: Any) -> None:
                captures.append(fingerprint(index, current))

            ids = torch.cat([validation_ids[window] for window in windows], dim=0)
            shape = (
                len(windows), ids.shape[1],
                int(self.student.config.hc_mult), int(self.student.config.hidden_size),
            )
            started = time.perf_counter()
            if self.rank == 0:
                embeds = self.student.model.model.embed_tokens(ids)
                hidden = embeds.unsqueeze(2).expand(
                    -1, -1, self.student.config.hc_mult, -1
                ).contiguous()
                hidden = self._run_layers(hidden, ids, True, layer_capture=capture)
                _cuda_sync(torch)
                if tuple(hidden.shape) != shape:
                    raise ArtifactError("W28 batch-context A/B rank0 activation geometry drift")
                self._batch_p2p_send(hidden.detach().contiguous(), dst=1)
                loss = None
                del embeds
            else:
                hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                self._batch_p2p_recv(hidden, src=0)
                hidden = self._run_layers(hidden, ids, True, layer_capture=capture)
                loss = float(self._loss_group(hidden[0:1], [28]).detach().cpu())
                _cuda_sync(torch)
            elapsed = time.perf_counter() - started
            del hidden
            torch.cuda.empty_cache()
            return {
                "rank": self.rank,
                "physical_windows": list(windows),
                "w28_layer_fingerprints": captures,
                "w28_scalar": loss,
                "wall_seconds": elapsed,
            }

        branches: dict[str, dict[str, Any]] = {}
        for windows in ((28,), (28, 56)):
            key = "singleton" if len(windows) == 1 else "sealed_mb2_context"
            branches[key] = run_branch(windows)
            self.dist.barrier()

        left = branches["singleton"]["w28_layer_fingerprints"]
        right = branches["sealed_mb2_context"]["w28_layer_fingerprints"]
        if len(left) != len(right):
            raise ArtifactError("W28 batch-context A/B layer capture count drift")
        first_local = None
        for a, b in zip(left, right):
            if a["layer"] != b["layer"]:
                raise ArtifactError("W28 batch-context A/B layer ordering drift")
            if a["sha256"] != b["sha256"]:
                first_local = int(a["layer"])
                break
        local = {
            "rank": self.rank,
            "layer_range": [self.first, self.last],
            "first_divergent_layer": first_local,
            "branches": branches,
        }
        gathered: list[Any] = [None for _ in range(self.dist.get_world_size())]
        self.dist.all_gather_object(gathered, local)
        divergent = [
            (int(row["first_divergent_layer"]), int(row["rank"]))
            for row in gathered if row["first_divergent_layer"] is not None
        ]
        result = {
            "schema": "banana-smasher-w28-batch-context-layer-ab-v1",
            "status": "PASS",
            "first_divergent_layer": min(divergent)[0] if divergent else None,
            "first_divergent_rank": min(divergent)[1] if divergent else None,
            "all_w28_layer_outputs_bitwise_equal": not divergent,
            "rank_rows": gathered,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_saves": 0,
        }
        rows: list[Any] = [result if self.rank == 1 else None]
        self.dist.broadcast_object_list(rows, src=1)
        if not isinstance(rows[0], Mapping):
            raise ArtifactError("W28 batch-context A/B fan-in produced no result")
        return dict(rows[0])

    def _pipeline_pass(self, group: list[int], *, loss_divisor: int = 1) -> tuple[float | None, dict[str, float]]:
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
            self._batch_p2p_send(hidden.detach().contiguous(), dst=1)
            grad = torch.empty_like(hidden)
            self._batch_p2p_recv(grad, src=1)
            backward_started = time.perf_counter()
            hidden.backward(grad)
            _cuda_sync(torch)
            backward_seconds = time.perf_counter() - backward_started
            return None, {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds}
        activation = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
        receive_started = time.perf_counter()
        self._batch_p2p_recv(activation, src=0)
        activation.requires_grad_(True)
        hidden = self._run_layers(activation, ids, True)
        loss = self._loss_group(hidden, group)
        _cuda_sync(torch)
        forward_seconds = time.perf_counter() - receive_started
        expected_pre_backward = self.config.get("w28_only_expected_pre_backward_scalar")
        if expected_pre_backward is not None and self.global_step == 0:
            if group != [28] or loss_divisor != 1:
                raise ArtifactError(
                    "W28-only pre-backward gate requires one undivided W28 objective"
                )
            observed_pre_backward = float(loss.detach().cpu())
            expected_pre_backward = float(expected_pre_backward)
            delta = abs(observed_pre_backward - expected_pre_backward)
            tolerance = float(
                self.config.get("w28_only_pre_backward_tolerance", 1.0e-12)
            )
            if (
                not math.isfinite(observed_pre_backward)
                or not math.isfinite(expected_pre_backward)
                or not math.isfinite(tolerance)
                or tolerance < 0.0
                or delta > tolerance
            ):
                raise ArtifactError(
                    "W28-only pre-backward scalar equality failed: "
                    f"observed={observed_pre_backward!r} "
                    f"expected={expected_pre_backward!r} delta={delta!r} "
                    f"tolerance={tolerance!r}"
                )
            self.status["w28_only_pre_backward_gate"] = {
                "status": "PASS",
                "window_membership": [28],
                "observed_scalar": observed_pre_backward,
                "expected_scalar": expected_pre_backward,
                "delta": delta,
                "tolerance": tolerance,
                "direction_vs_mixed_w28_w56": "lower",
                "optimizer_steps_before_gate": 0,
            }
        backward_started = time.perf_counter()
        (loss * self.equivalent_gradient_scale / float(loss_divisor)).backward()
        _cuda_sync(torch)
        backward_seconds = time.perf_counter() - backward_started
        if activation.grad is None:
            raise ArtifactError("official pipeline boundary gradient is missing")
        self._batch_p2p_send(activation.grad.contiguous(), dst=0)
        return float(loss.detach().cpu()), {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds}

    def _pipeline_update_1f1b(
        self, groups: list[list[int]], *, loss_divisor: int
    ) -> tuple[float | None, dict[str, float]]:
        """Run multiple fixed-size groups as a two-stage 1F1B pipeline."""
        if (
            len(groups) < 2
            or loss_divisor != len(groups)
            or any(len(group) != self.pipeline_microbatch for group in groups)
        ):
            raise ArtifactError("official 1F1B pipeline grouping drift")
        torch = self.torch
        shape = (
            self.pipeline_microbatch,
            self.base.T.T_TRAIN,
            int(self.student.config.hc_mult),
            int(self.student.config.hidden_size),
        )
        forward_seconds = 0.0
        backward_seconds = 0.0
        losses: list[float] = []
        if self.rank == 0:
            def forward(group: list[int]) -> Any:
                nonlocal forward_seconds
                started = time.perf_counter()
                ids = torch.cat([self.ids_cache[window] for window in group], dim=0)
                embeds = self.student.model.model.embed_tokens(ids)
                hidden = embeds.unsqueeze(2).expand(
                    -1, -1, self.student.config.hc_mult, -1
                ).contiguous()
                hidden = self._run_layers(hidden, ids, True)
                _cuda_sync(torch)
                forward_seconds += time.perf_counter() - started
                if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                    raise ArtifactError(
                        f"official pipeline activation geometry drift: {tuple(hidden.shape)} {hidden.dtype}"
                    )
                return hidden

            pending = forward(groups[0])
            self._batch_p2p_send(pending.detach().contiguous(), dst=1)
            for index in range(1, len(groups)):
                current = forward(groups[index])
                gradient = torch.empty_like(pending)
                self._batch_p2p_exchange(
                    current.detach().contiguous(), dst=1, incoming=gradient, src=1
                )
                started = time.perf_counter()
                pending.backward(gradient)
                _cuda_sync(torch)
                backward_seconds += time.perf_counter() - started
                pending = current
            gradient = torch.empty_like(pending)
            self._batch_p2p_recv(gradient, src=1)
            started = time.perf_counter()
            pending.backward(gradient)
            _cuda_sync(torch)
            backward_seconds += time.perf_counter() - started
        else:
            activation = torch.empty(
                shape, dtype=torch.bfloat16, device=self.student.device
            )
            self._batch_p2p_recv(activation, src=0)
            for index, group in enumerate(groups):
                ids = torch.cat([self.ids_cache[window] for window in group], dim=0)
                activation.requires_grad_(True)
                started = time.perf_counter()
                hidden = self._run_layers(activation, ids, True)
                loss = self._loss_group(hidden, group)
                _cuda_sync(torch)
                forward_seconds += time.perf_counter() - started
                started = time.perf_counter()
                (loss * self.equivalent_gradient_scale / float(loss_divisor)).backward()
                _cuda_sync(torch)
                backward_seconds += time.perf_counter() - started
                if activation.grad is None:
                    raise ArtifactError("official pipeline boundary gradient is missing")
                losses.append(float(loss.detach().cpu()))
                if index + 1 < len(groups):
                    next_activation = torch.empty_like(activation)
                    self._batch_p2p_exchange(
                        activation.grad.contiguous(),
                        dst=0,
                        incoming=next_activation,
                        src=0,
                    )
                    activation = next_activation
                else:
                    self._batch_p2p_send(activation.grad.contiguous(), dst=0)
        return (
            sum(losses) / len(losses) if losses else None,
            {
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
            },
        )

    def _local_params(self) -> list[tuple[str, Any]]:
        return [
            *self.luts,
            *self.norms,
            *self.outputs,
            *self.expert_planes_l028_su_sv,
        ]

    def _local_norm(self, values: list[Any]) -> float:
        return sum(float(value.detach().float().pow(2).sum().cpu()) for value in values) ** 0.5

    def _optimizer_precision_report(self) -> dict[str, Any]:
        torch = self.torch
        maxima = {"exp_avg": 0.0, "exp_avg_sq": 0.0}
        dtypes: dict[str, set[str]] = {name: set() for name in maxima}
        finite = True
        entries = 0
        for state in self.optimizer.state.values():
            if state:
                entries += 1
            for name in maxima:
                value = state.get(name)
                if value is None:
                    continue
                dtypes[name].add(str(value.dtype))
                finite = finite and bool(torch.isfinite(value).all().item())
                maxima[name] = max(maxima[name], float(value.detach().abs().max().cpu()))
        return {
            "mechanism": "adam-fp64-state-and-update-arithmetic",
            "equivalent_gradient_scale": self.equivalent_gradient_scale,
            "state_entries": entries,
            "state_dtypes": {name: sorted(values) for name, values in dtypes.items()},
            "finite": finite,
            "max_abs": maxima,
        }

    def _step(self, global_step: int) -> dict[str, Any]:
        torch = self.torch
        params = self._local_params()
        before = [parameter.detach().clone() for _name, parameter in params]
        self.optimizer.zero_grad(set_to_none=True)
        if self.published_pre_recipe:
            base_lrs, multiplier, default_windows = _published_pre_recipe_policy(
                self.config,
                global_step,
                authenticated_u60_continuation=self.has_expert_plane_surface,
            )
            group_windows = (
                self.controlled_windows[global_step % len(self.controlled_windows)]
                if self.published_pre_controlled_windows
                else default_windows
            )
            if self.published_pre_controlled_windows and len(group_windows) != int(
                self.config["controlled_windows_per_update"]
            ):
                raise ArtifactError("fresh PRE controlled window schedule changed after admission")
        elif self.controlled_arm:
            base_lrs, multiplier, expected_windows = _controlled_arm_policy(self.config, global_step)
            group_windows = self.controlled_windows[global_step]
            if len(group_windows) != expected_windows:
                raise ArtifactError("controlled arm window schedule changed after admission")
        else:
            base_lrs = dict(BASE_LRS)
            if self.has_expert_plane_surface:
                base_lrs[EXPERT_PLANE_SURFACE] = U60_EXPERT_PLANE_BASE_LR
            multiplier = self.trainer.current_multiplier(global_step)
            group_windows = [20 + 4 * (global_step % 16) + offset for offset in range(4)]
        for group in self.optimizer.param_groups:
            group["lr"] = base_lrs[group["group_name"]] * multiplier
        groups = [group_windows[index:index + self.pipeline_microbatch] for index in range(0, len(group_windows), self.pipeline_microbatch)]
        if not groups or any(len(group) != self.pipeline_microbatch for group in groups):
            raise ArtifactError("controlled arm pipeline grouping drift")
        dist_started = time.perf_counter()
        if len(groups) > 1:
            loss, timing = self._pipeline_update_1f1b(
                groups, loss_divisor=len(groups)
            )
        else:
            loss, timing = self._pipeline_pass(groups[0], loss_divisor=len(groups))
        forward_backward_seconds = time.perf_counter() - dist_started
        optimizer_started = time.perf_counter()
        self.optimizer.step()
        self.scheduler.step()
        _cuda_sync(torch)
        optimizer_seconds = time.perf_counter() - optimizer_started
        optimizer_precision = self._optimizer_precision_report()
        if not optimizer_precision["finite"]:
            raise ArtifactError("FP64 Adam optimizer state became nonfinite")
        gradients = [parameter.grad for _name, parameter in params if parameter.grad is not None]
        gradient_norm = self._local_norm(gradients)
        named_deltas = [
            (name, parameter.detach() - old)
            for (name, parameter), old in zip(params, before)
        ]
        delta_norm = self._local_norm([delta for _name, delta in named_deltas])
        expert_plane_deltas = [
            (name, delta) for name, delta in named_deltas
            if name.startswith("model.layers.28.mlp.experts.")
        ]
        expert_plane_delta_norm = self._local_norm(
            [delta for _name, delta in expert_plane_deltas]
        )
        expert_plane_nonzero_delta_name = None
        if expert_plane_deltas:
            delta_rows = [
                (float(delta.detach().float().pow(2).sum().cpu()), name)
                for name, delta in expert_plane_deltas
            ]
            best_delta, best_name = max(delta_rows)
            if best_delta > 0.0:
                expert_plane_nonzero_delta_name = best_name
        local = {
            "rank": self.rank,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "parameter_delta_norm": delta_norm,
            "expert_plane_delta_norm": expert_plane_delta_norm,
            "expert_plane_nonzero_delta_name": expert_plane_nonzero_delta_name,
            "expert_plane_candidate_read": self.expert_plane_candidate_read,
            "optimizer_precision": optimizer_precision,
            "timings": {
                "forward_seconds": timing["forward_seconds"],
                "backward_seconds": timing["backward_seconds"],
                "optimizer_seconds": optimizer_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "wall_seconds": forward_backward_seconds + optimizer_seconds,
            },
            "windows": group_windows,
            "controlled_arm_id": self.controlled_arm_id,
            "recipe_id": self.config.get("recipe_id"),
            "applied_multiplier": multiplier,
            "applied_learning_rates": {name: value * multiplier for name, value in base_lrs.items()},
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
        global_expert_plane_delta = sum(
            float(row["expert_plane_delta_norm"]) ** 2 for row in rows
        ) ** 0.5
        expert_plane_delta_names = [
            row["expert_plane_nonzero_delta_name"] for row in rows
            if row["expert_plane_nonzero_delta_name"] is not None
        ]
        losses = [row["loss"] for row in rows if row["loss"] is not None]
        local["gradient_norm"] = global_gradient
        local["parameter_delta_norm"] = global_delta
        local["expert_plane_delta_norm"] = global_expert_plane_delta
        local["expert_plane_nonzero_delta_name"] = (
            expert_plane_delta_names[0] if expert_plane_delta_names else None
        )
        local["loss"] = losses[0] if losses else None
        local["timings"] = {key: max(float(row["timings"][key]) for row in rows) for key in local["timings"]}
        local["rank_reports"] = rows
        if global_gradient <= 0.0 or global_delta <= 0.0 or not losses:
            raise ArtifactError(f"official resident U{global_step + 1} produced no real gradient/delta")
        if self.has_expert_plane_surface and (
            global_expert_plane_delta <= 0.0 or not expert_plane_delta_names
        ):
            raise ArtifactError(
                f"official resident U{global_step + 1} produced no expert-plane delta"
            )
        return local

    def _gather_state(self) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any]]:
        torch = self.torch
        rows: list[Any] = [None, None]
        local_params = {
            "luts": self.luts,
            "norms": self.norms,
            "outputs": self.outputs,
            EXPERT_PLANE_SURFACE: self.expert_planes_l028_su_sv,
        }
        local_state = {
            "rank": self.rank,
            **{
                surface: {name: parameter.detach().cpu().clone() for name, parameter in local_params[surface]}
                for surface in self.surface_names
            },
            "param_names": {
                surface: [name for name, _parameter in local_params[surface]]
                for surface in self.surface_names
            },
            "optimizer": _cpu_tree(torch, self.optimizer.state_dict()),
        }
        self.dist.all_gather_object(rows, local_state)
        if self.rank != 0:
            self.dist.barrier()
            return None, None, {"rank_rows": rows}
        merged = {surface: {} for surface in self.surface_names}
        for row in rows:
            for surface in merged:
                overlap = set(merged[surface]) & set(row[surface])
                if overlap:
                    raise ArtifactError(f"official resident state overlap: {surface} {sorted(overlap)[:3]}")
                merged[surface].update(row[surface])
        expected_counts = {"luts": 43, "norms": 235, "outputs": 43}
        if self.has_expert_plane_surface:
            expected_counts[EXPERT_PLANE_SURFACE] = 1536
        if {surface: len(values) for surface, values in merged.items()} != expected_counts:
            raise ArtifactError("official resident merged trainable surface coverage drift")
        optimizer = _merge_sharded_optimizer_state(rows, merged, self.surface_names)
        scheduler = _cpu_tree(torch, self.scheduler.state_dict())
        report = {"rank_rows": rows, "optimizer": optimizer, "scheduler": scheduler}
        self.dist.barrier()
        return merged, optimizer, report

    def advance_to(
        self, target_update: int, *, gather_state: bool = True
    ) -> tuple[Mapping[str, Any] | None, dict[str, Any], Mapping[str, Any] | None]:
        start = self.global_step
        if target_update <= start:
            raise ArtifactError("official resident target must advance beyond current update")
        last: dict[str, Any] | None = None
        merged_state: Mapping[str, Any] | None = None
        optimizer_state: Mapping[str, Any] | None = None
        report_state: Mapping[str, Any] | None = None
        for update in range(start, target_update):
            last = self._step(update)
            self.global_step = update + 1
        merged_state = optimizer_state = report_state = None
        if gather_state:
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
            "expert_plane_delta_norm": last["expert_plane_delta_norm"],
            "expert_plane_nonzero_delta_name": last["expert_plane_nonzero_delta_name"],
            "expert_plane_candidate_read": next(
                (
                    row["expert_plane_candidate_read"]
                    for row in last["rank_reports"]
                    if row.get("expert_plane_candidate_read") is not None
                ),
                None,
            ),
            "loss": last["loss"],
            "timings": last["timings"],
            "windows": last["windows"],
            "process_gpu_evidence": last["process_gpu_evidence"],
            "rank_reports": last["rank_reports"],
            "rank_provenance": [int(row["rank"]) for row in last["rank_reports"]],
            "optimizer_state": optimizer_state,
            "scheduler_state": (report_state or {}).get("scheduler") if isinstance(report_state, Mapping) else None,
            "state_gathered": gather_state,
            "model_engine": "official-ShardStudent-grouped-K2-FWHT-resident",
            "frozen_surfaces": ["packed_codes", "assignments", "scales"],
            "trainable_surfaces": [
                "luts", "rmsnorms", "output_gains",
                *([EXPERT_PLANE_SURFACE] if self.has_expert_plane_surface else []),
            ],
        }
        return merged_state, step_report, report_state

    def preload_validation(self, windows: Any, teacher_root: str | Path) -> dict[str, Any]:
        """Materialize canonical validation inputs before the timed forward."""
        from .balanced64 import POSITIONS_PER_WINDOW, SUPPORT, _load_torch
        from .official_k2_resident_score import (
            SOURCE_CONTEXT_TOKENS,
            _canonical_causal_score_tokens,
            _physical_canary_batch_windows,
        )

        ordered = tuple(int(value) for value in windows)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ArtifactError("resident validation windows must be non-empty and unique")
        physical = ordered
        physical_batch_size = len(ordered)
        published_pre_proof = (
            self.published_pre_recipe and _has_static_w28_binding(self.config)
        )
        if published_pre_proof and ordered == (28,):
            # RUN1698 produced the trusted W28 candidate in the sealed builder's
            # first mb=2 group (W28, W56). W56 is execution context only: the
            # public validation contract still reports and reduces W28 alone.
            physical_batch_size = int(self.config.get("sealed_builder_window_microbatch", 2))
            physical = _physical_canary_batch_windows(ordered, physical_batch_size, (28, 56))
        elif published_pre_proof and len(ordered) == 64:
            # Full64 preserves the exact ordered windows and per-window scorer
            # arithmetic, but executes them in the same bounded physical groups
            # as the sealed W28 fixture. A single batch of 64 expands the rank-0
            # activation to tens of GiB and OOMs before the first P2P send.
            physical_batch_size = int(self.config.get("sealed_builder_window_microbatch", 2))
            if physical_batch_size < 1 or len(ordered) % physical_batch_size:
                raise ArtifactError("published PRE full64 physical batch must divide 64")
        root = Path(teacher_root).expanduser().resolve()
        if not root.is_dir():
            raise ArtifactError(f"resident validation teacher root is missing: {root}")
        teacher_sha256_by_window: dict[str, str] = {}
        if published_pre_proof and ordered == (28,):
            teacher_sha256_by_window["28"] = _require_static_w28_teacher(root)
        cache_key = (ordered, str(root))
        cache = getattr(self, "_validation_input_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        validation_corpus_path, expected_corpus_sha = _resolve_validation_corpus(
            self.config,
            teacher_root=root,
            training_corpus=self.corpus_path,
            published_pre_proof=published_pre_proof,
        )
        if expected_corpus_sha is not None:
            observed_corpus_sha = _sha256_file(validation_corpus_path)
            if observed_corpus_sha != expected_corpus_sha:
                raise ArtifactError(
                    f"resident validation corpus SHA mismatch: {observed_corpus_sha} != {expected_corpus_sha}"
                )
        try:
            corpus = json.loads(validation_corpus_path.read_text())
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot load resident validation corpus: {exc}") from exc
        if not isinstance(corpus, list):
            raise ArtifactError("resident validation corpus must be a list")
        pad_token = int(self.config.get("pad_token_id", 1))
        ids_cache: dict[int, Any] = {}
        teacher_cache: dict[int, tuple[Any, Any]] = {}
        for window in physical:
            if window < 0 or window >= len(corpus) or not isinstance(corpus[window], Mapping):
                raise ArtifactError(f"resident validation corpus is missing window {window}")
            row = corpus[window]
            tokens = row.get("token_ids")
            real_len = int(row.get("real_len", len(tokens) if isinstance(tokens, list) else 0))
            tokens = _canonical_causal_score_tokens(tokens, real_len=real_len, pad_token_id=pad_token)
            ids = self.torch.full(
                (1, SOURCE_CONTEXT_TOKENS), pad_token, dtype=self.torch.long,
                device=self.student.device,
            )
            ids[0, : len(tokens)] = self.torch.tensor(
                tokens, dtype=self.torch.long, device=self.student.device
            )
            ids_cache[window] = ids
            if self.rank == 1 and window in ordered:
                teacher = _load_torch(root / f"t8192_win{window}.pt")
                idx = teacher.get("idx") if isinstance(teacher, Mapping) else None
                logprob = teacher.get("logprob") if isinstance(teacher, Mapping) else None
                if not hasattr(idx, "shape") or not hasattr(logprob, "shape"):
                    raise ArtifactError(f"resident validation teacher window {window} is malformed")
                idx = idx[:POSITIONS_PER_WINDOW, :SUPPORT].to(
                    dtype=self.torch.int64, device="cpu"
                ).contiguous()
                logprob = logprob[:POSITIONS_PER_WINDOW, :SUPPORT].to(
                    dtype=self.torch.float16, device="cpu"
                ).contiguous()
                expected = (POSITIONS_PER_WINDOW, SUPPORT)
                if tuple(idx.shape) != expected or tuple(logprob.shape) != expected:
                    raise ArtifactError(f"resident validation teacher window {window} geometry drift")
                teacher_cache[window] = (idx, logprob)
        prepared = {
            "windows": ordered, "physical_windows": physical,
            "physical_batch_size": physical_batch_size, "teacher_root": root,
            "corpus_path": validation_corpus_path,
            "corpus_sha256": expected_corpus_sha or _sha256_file(validation_corpus_path),
            "teacher_sha256_by_window": teacher_sha256_by_window,
            "ids": ids_cache, "teachers": teacher_cache,
        }
        cache[cache_key] = prepared
        self._validation_input_cache = cache
        return prepared

    def _device_parameter_fingerprint(self) -> dict[str, Any]:
        """Hash one live trainable tensor per rank and bind the pair."""
        params = self._local_params()
        if not params:
            raise ArtifactError("resident validation has no live trainable parameter")
        name, parameter = params[0]
        tensor = parameter.detach().contiguous()
        raw = tensor.reshape(-1).view(self.torch.uint8).cpu().numpy().tobytes()
        local = {
            "rank": self.rank, "name": name, "device": str(tensor.device),
            "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        rows: list[Any] = [None, None]
        self.dist.all_gather_object(rows, local)
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "ranks": rows}

    def _local_resident_mechanism_snapshot(self) -> dict[str, Any]:
        """Snapshot counters from the already-resident routed expert objects."""
        totals: dict[str, int] = {}
        providers: list[str] = []
        for index in range(self.first, self.last + 1):
            layer = self.student.model.model.layers[index]
            mlp = getattr(layer, "mlp", None)
            provider = getattr(mlp, "experts", mlp)
            stats = getattr(provider, "mechanism_stats", None)
            if not callable(stats):
                continue
            providers.append(type(provider).__name__)
            observed = stats()
            if not isinstance(observed, Mapping):
                raise ArtifactError("resident expert mechanism stats must be a mapping")
            for key, value in observed.items():
                if isinstance(value, (int, bool)):
                    totals[key] = totals.get(key, 0) + int(value)
        return {
            "provider_classes": sorted(set(providers)),
            "provider_count": len(providers),
            "counters": totals,
        }

    @staticmethod
    def _mechanism_counter_delta(
        before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> dict[str, int]:
        before_counters = before.get("counters", {})
        after_counters = after.get("counters", {})
        keys = set(before_counters) | set(after_counters)
        return {
            key: int(after_counters.get(key, 0)) - int(before_counters.get(key, 0))
            for key in sorted(keys)
        }

    def _validate_preloaded(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        """Run the sealed readout on inputs already resident outside timing."""
        from .balanced64 import POSITIONS_PER_WINDOW, SUPPORT
        from .official_k2_resident_score import PayloadModelReadCounter
        import numpy as np

        ordered = tuple(int(value) for value in prepared["windows"])
        physical = tuple(int(value) for value in prepared.get("physical_windows", ordered))
        physical_batch_size = int(prepared.get("physical_batch_size", self.score_pipeline_microbatch))
        if physical_batch_size < 1 or len(physical) % physical_batch_size:
            raise ArtifactError("resident validation physical fixture batch drift")
        ids_cache = prepared["ids"]
        teacher_cache = prepared["teachers"]
        roots = (
            self.model_root, self.asset_root, self.parent_root,
            Path(prepared["teacher_root"]), Path(prepared["corpus_path"]), self.manifest_path,
            self.delta_dir, self.vq3b_dir,
        )
        reads = PayloadModelReadCounter(roots)
        ready_counter = reads.mark_resident_ready()
        torch = self.torch
        started = time.perf_counter()
        terms: list[float] = []
        top1 = 0
        per_window: list[dict[str, Any]] = []
        objective_ab: dict[str, Any] | None = None
        rank_phase_profiles: list[dict[str, Any]] = []
        try:
            for offset in range(0, len(physical), physical_batch_size):
                batch = physical[offset:offset + physical_batch_size]
                mechanism_before = self._local_resident_mechanism_snapshot()
                ids = torch.cat([ids_cache[window] for window in batch], dim=0)
                shape = (
                    len(batch), ids.shape[1], int(self.student.config.hc_mult),
                    int(self.student.config.hidden_size),
                )
                if self.rank == 0:
                    forward_started = time.perf_counter()
                    embeds = self.student.model.model.embed_tokens(ids)
                    hidden = embeds.unsqueeze(2).expand(
                        -1, -1, self.student.config.hc_mult, -1
                    ).contiguous()
                    hidden = self._run_layers(hidden, ids, False)
                    _cuda_sync(torch)
                    forward_ms = (time.perf_counter() - forward_started) * 1000.0
                    if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                        raise ArtifactError("resident validation activation geometry drift")
                    p2p_started = time.perf_counter()
                    self._batch_p2p_send(hidden.detach().contiguous(), dst=1)
                    _cuda_sync(torch)
                    p2p_ms = (time.perf_counter() - p2p_started) * 1000.0
                    mechanism_after = self._local_resident_mechanism_snapshot()
                    mechanism_delta = self._mechanism_counter_delta(
                        mechanism_before, mechanism_after
                    )
                    if mechanism_delta.get("reconstruction_calls", 0) != 0:
                        raise ArtifactError("resident validation reconstructed weights inside a batch")
                    rank_phase_profiles.append({
                        "rank": self.rank, "batch_windows": list(batch),
                        "weight_reconstruction_ms": 0.0,
                        "forward_ms": forward_ms, "p2p_ms": p2p_ms, "readout_ms": 0.0,
                        "mechanism_before": mechanism_before,
                        "mechanism_after": mechanism_after,
                        "mechanism_counter_delta": mechanism_delta,
                    })
                    del hidden, ids, embeds
                else:
                    p2p_started = time.perf_counter()
                    hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                    self._batch_p2p_recv(hidden, src=0)
                    _cuda_sync(torch)
                    p2p_ms = (time.perf_counter() - p2p_started) * 1000.0
                    forward_started = time.perf_counter()
                    hidden = self._run_layers(hidden, ids, False)
                    _cuda_sync(torch)
                    forward_ms = (time.perf_counter() - forward_started) * 1000.0
                    readout_started = time.perf_counter()
                    final = self.student.model.model.norm(self.student.model.model.hc_head(hidden))
                    for batch_index, window in enumerate(batch):
                        if window not in ordered:
                            continue
                        teacher_idx, teacher_logprob = teacher_cache[window]
                        logits = self.student.model.lm_head(
                            final[batch_index, :POSITIONS_PER_WINDOW].to(torch.bfloat16)
                        ).float()
                        if (
                            self.config.get("objective_ab_probe_w28") is True
                            or self.config.get("scorer_aligned_preupdate_gate") is True
                        ) and window == 28:
                            train_idx, train_lp_n, train_p_n = self.teacher_cache[window]
                            train_idx_device = train_idx[:POSITIONS_PER_WINDOW].to(
                                device=self.student.device, non_blocking=False
                            )
                            scorer_idx_device = teacher_idx.to(
                                device=self.student.device, non_blocking=False
                            )
                            support_index_matches = train_idx_device == scorer_idx_device
                            training_gathered = logits.gather(1, train_idx_device).float()
                            training_qn = training_gathered - training_gathered.logsumexp(
                                -1, keepdim=True
                            )
                            scorer_gathered = logits.gather(1, scorer_idx_device).float()
                            scorer_qn = scorer_gathered - scorer_gathered.logsumexp(
                                -1, keepdim=True
                            )
                            scorer_lp = teacher_logprob.to(
                                device=self.student.device, dtype=torch.float32,
                                non_blocking=False,
                            )
                            scorer_lp_n = scorer_lp - scorer_lp.logsumexp(-1, keepdim=True)
                            scorer_p_n = scorer_lp_n.exp()
                            old_tailfix_values = (
                                scorer_p_n * (scorer_lp_n - scorer_qn)
                            ).sum(-1)
                            new_loss_group_values = (
                                train_p_n[:POSITIONS_PER_WINDOW]
                                * (train_lp_n[:POSITIONS_PER_WINDOW] - training_qn)
                            ).sum(-1)
                            objective_ab = {
                                "window": 28,
                                "real_len": int(self.real_lengths[window]),
                                "positions_compared": POSITIONS_PER_WINDOW,
                                "old_tailfix_token_kld_mean": float(old_tailfix_values.mean()),
                                "new_loss_group_token_kld_mean": float(new_loss_group_values.mean()),
                                "old_new_bitwise_equal": bool(
                                    torch.equal(old_tailfix_values, new_loss_group_values)
                                ),
                                "support_index_equal_count": int(support_index_matches.sum()),
                                "support_index_total": int(support_index_matches.numel()),
                                "support_index_equal_fraction": float(
                                    support_index_matches.float().mean()
                                ),
                                "training_teacher_lp_logsumexp_mean": float(
                                    train_lp_n[:POSITIONS_PER_WINDOW].logsumexp(-1).mean()
                                ),
                                "training_teacher_lp_logsumexp_abs_max": float(
                                    train_lp_n[:POSITIONS_PER_WINDOW].logsumexp(-1).abs().max()
                                ),
                                "scorer_teacher_lp_logsumexp_mean": float(
                                    scorer_lp.logsumexp(-1).mean()
                                ),
                                "scorer_teacher_lp_logsumexp_abs_max": float(
                                    scorer_lp.logsumexp(-1).abs().max()
                                ),
                                "training_teacher_p_row_sum_mean": float(
                                    train_p_n[:POSITIONS_PER_WINDOW].sum(-1).mean()
                                ),
                                "training_teacher_p_row_sum_abs_error_max": float(
                                    (train_p_n[:POSITIONS_PER_WINDOW].sum(-1) - 1.0).abs().max()
                                ),
                                "training_teacher_probability_matches_exp_logprob_max_abs": float(
                                    (
                                        train_p_n[:POSITIONS_PER_WINDOW]
                                        - train_lp_n[:POSITIONS_PER_WINDOW].exp()
                                    ).abs().max()
                                ),
                                "hidden_shape": list(hidden.shape),
                                "final_shape": list(final.shape),
                                "logits_shape": list(logits.shape),
                            }
                        logprob = torch.log_softmax(logits, dim=-1)
                        idx_device = teacher_idx.to(device=self.student.device, non_blocking=False)
                        q_lp = logprob.gather(1, idx_device).to(torch.float16).cpu().numpy().astype(
                            np.float64, copy=False
                        )
                        q_argmax = logprob.argmax(-1).to(torch.int64).cpu().numpy()
                        ref_lp = teacher_logprob.numpy().astype(np.float64, copy=False)
                        idx0 = teacher_idx[:, 0].numpy()
                        ref_max = np.max(ref_lp, axis=1, keepdims=True)
                        cand_max = np.max(q_lp, axis=1, keepdims=True)
                        ref_norm = ref_lp - (
                            ref_max + np.log(np.sum(
                                np.exp(ref_lp - ref_max), axis=1,
                                dtype=np.float64, keepdims=True,
                            ))
                        )
                        cand_norm = q_lp - (
                            cand_max + np.log(np.sum(
                                np.exp(q_lp - cand_max), axis=1,
                                dtype=np.float64, keepdims=True,
                            ))
                        )
                        values = np.sum(
                            np.exp(ref_norm) * (ref_norm - cand_norm),
                            axis=1, dtype=np.float64,
                        )
                        if values.size != POSITIONS_PER_WINDOW or not np.isfinite(values).all():
                            raise ArtifactError(f"resident validation invalid KLD at window {window}")
                        row_terms = [float(value) for value in values.tolist()]
                        row_top1 = int(np.count_nonzero(q_argmax == idx0))
                        terms.extend(row_terms)
                        top1 += row_top1
                        per_window.append({
                            "ordinal": ordered.index(window), "window": window,
                            "positions": POSITIONS_PER_WINDOW, "support": SUPPORT,
                            "kld_sum_binary64": math.fsum(row_terms), "top1": row_top1,
                        })
                        del logits, logprob, idx_device
                    _cuda_sync(torch)
                    readout_ms = (time.perf_counter() - readout_started) * 1000.0
                    mechanism_after = self._local_resident_mechanism_snapshot()
                    mechanism_delta = self._mechanism_counter_delta(
                        mechanism_before, mechanism_after
                    )
                    if mechanism_delta.get("reconstruction_calls", 0) != 0:
                        raise ArtifactError("resident validation reconstructed weights inside a batch")
                    rank_phase_profiles.append({
                        "rank": self.rank, "batch_windows": list(batch),
                        "weight_reconstruction_ms": 0.0,
                        "forward_ms": forward_ms, "p2p_ms": p2p_ms,
                        "readout_ms": readout_ms,
                        "mechanism_before": mechanism_before,
                        "mechanism_after": mechanism_after,
                        "mechanism_counter_delta": mechanism_delta,
                    })
                    del hidden, ids, final
            _cuda_sync(torch)
        finally:
            reads.active = False
        read_delta = reads.delta(ready_counter)
        local_terminal = {
            "rank": self.rank,
            "timed_model_payload_reads": read_delta,
            "timed_score_file_reads": read_delta,
            "file_read_paths": list(reads.paths),
            "fallback_calls": int(self.status.get("fallback_calls", 0)),
            "reconstruction_calls": int(self.status.get("reconstruction_calls", 0)),
            "cpu_relay_bytes": int(self.status.get("cpu_relay_bytes", 0)),
            "model_object_id": id(self.student.model),
        }
        terminals: list[Any] = [None, None]
        self.dist.all_gather_object(terminals, local_terminal)
        phase_profiles_by_rank: list[Any] = [None, None]
        self.dist.all_gather_object(phase_profiles_by_rank, rank_phase_profiles)
        forbidden = (
            "timed_model_payload_reads", "timed_score_file_reads", "fallback_calls",
            "reconstruction_calls", "cpu_relay_bytes",
        )
        if any(any(int(row[key]) != 0 for key in forbidden) for row in terminals):
            raise ArtifactError(f"resident validation zero-read closure failed: {terminals}")
        result = None
        if self.rank == 1:
            expected_positions = len(ordered) * POSITIONS_PER_WINDOW
            if len(terms) != expected_positions or len(per_window) != len(ordered):
                raise ArtifactError("resident validation position/window closure drift")
            result = {
                "kld_mean": math.fsum(terms) / len(terms), "top1": top1,
                "positions": len(terms), "support": SUPPORT,
                "windows": list(ordered), "physical_fixture_windows": list(physical),
                "physical_fixture_batch_size": physical_batch_size, "per_window": per_window,
                "objective_ab_probe_w28": objective_ab,
                "execution_mode": "resident_in_memory",
                "timed_wall_seconds": time.perf_counter() - started,
                "phase_profiles_by_rank": phase_profiles_by_rank,
                "runtime_counters": {
                    "timed_model_payload_reads": 0, "timed_score_file_reads": 0,
                    "fallback_calls": 0, "reconstruction_calls": 0,
                    "cpu_relay_bytes": 0, "resident_ready": terminals,
                },
            }
        rows = [result]
        self.dist.broadcast_object_list(rows, src=1)
        if not isinstance(rows[0], Mapping):
            raise ArtifactError("resident validation fan-in produced no result")
        return dict(rows[0])

    def validate(self, windows: Any, teacher_root: str | Path) -> dict[str, Any]:
        """Validate the trainer's existing model without a weight reload."""
        prepared = self.preload_validation(windows, teacher_root)
        fingerprint = self._device_parameter_fingerprint()
        model = self.student.model
        was_training = bool(model.training)
        model.eval()
        try:
            with self.torch.no_grad():
                result = self._validate_preloaded(prepared)
        finally:
            model.train(was_training)
        counters = dict(result.get("runtime_counters", {}))
        counters.update({
            "model_mode_before": "train" if was_training else "eval",
            "model_mode_restored": bool(model.training) == was_training,
            "checkpoint_reloads": 0,
            "trainer_object_id": id(self), "model_object_id": id(model),
        })
        result.update({
            "schema": "banana-smasher-resident-trainer-validate-v1",
            "checkpoint_sha256": self.config.get("checkpoint_sha256"),
            "validation_corpus_sha256": prepared["corpus_sha256"],
            "validation_teacher_sha256_by_window": prepared.get(
                "teacher_sha256_by_window", {}
            ),
            "sealed_builder_binding": getattr(self, "sealed_builder_binding", None),
            "device_parameter_fingerprint": fingerprint,
            "runtime_counters": counters,
        })
        return result

    def score_resident(self, windows: tuple[int, ...]) -> dict[str, Any]:
        """Score the current resident state without checkpoint materialization."""
        ordered = tuple(int(value) for value in windows)
        if len(ordered) != 64 or ordered != tuple(range(20, 84)):
            raise ArtifactError("continuous resident score requires exact ordered windows 20..83")
        from .official_k2_resident_score import PayloadModelReadCounter
        import numpy as np

        roots = (
            self.model_root, self.asset_root, self.parent_root, self.teacher_root,
            self.corpus_path, self.manifest_path, self.delta_dir, self.vq3b_dir,
        )
        reads = PayloadModelReadCounter(roots)
        ready_counter = reads.mark_resident_ready()
        torch = self.torch
        started = time.perf_counter()
        terms: list[float] = []
        top1 = 0
        per_window: list[dict[str, Any]] = []
        try:
            for offset in range(0, len(ordered), self.score_pipeline_microbatch):
                batch = ordered[offset:offset + self.score_pipeline_microbatch]
                ids = torch.cat([self.ids_cache[window] for window in batch], dim=0)
                shape = (
                    len(batch), ids.shape[1], int(self.student.config.hc_mult),
                    int(self.student.config.hidden_size),
                )
                with torch.no_grad():
                    if self.rank == 0:
                        embeds = self.student.model.model.embed_tokens(ids)
                        hidden = embeds.unsqueeze(2).expand(
                            -1, -1, self.student.config.hc_mult, -1
                        ).contiguous()
                        hidden = self._run_layers(hidden, ids, False)
                        torch.cuda.synchronize()
                        if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                            raise ArtifactError("continuous resident score activation geometry drift")
                        self.dist.send(hidden.detach().contiguous(), dst=1)
                    else:
                        hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                        self.dist.recv(hidden, src=0)
                        hidden = self._run_layers(hidden, ids, False)
                        torch.cuda.synchronize()
                        final = self.student.model.model.norm(self.student.model.model.hc_head(hidden))
                        count = 1024
                        for head_offset in range(0, len(batch), self.score_head_window_microbatch):
                            head_windows = batch[
                                head_offset:head_offset + self.score_head_window_microbatch
                            ]
                            head_final = final[
                                head_offset:head_offset + len(head_windows), :count
                            ]
                            logits = self.student.model.lm_head(
                                head_final.reshape(len(head_windows) * count, *head_final.shape[2:]).to(
                                    torch.bfloat16
                                )
                            ).float()
                            logprob = torch.log_softmax(logits, dim=-1).reshape(
                                len(head_windows), count, -1
                            )
                            for head_index, window in enumerate(head_windows):
                                teacher_idx, teacher_logprob, _teacher_probability = self.teacher_cache[window]
                                window_logprob = logprob[head_index]
                                idx_device = teacher_idx.to(device=self.student.device, non_blocking=False)
                                q_lp = window_logprob.gather(1, idx_device).to(torch.float16).cpu().numpy().astype(
                                    np.float64, copy=False
                                )
                                q_argmax = window_logprob.argmax(-1).to(torch.int64).cpu().numpy()
                                ref_lp = teacher_logprob.detach().cpu().numpy().astype(
                                    np.float64, copy=False
                                )
                                idx0 = teacher_idx[:, 0].detach().cpu().numpy()
                                ref_max = np.max(ref_lp, axis=1, keepdims=True)
                                cand_max = np.max(q_lp, axis=1, keepdims=True)
                                ref_norm = ref_lp - (
                                    ref_max + np.log(np.sum(np.exp(ref_lp - ref_max), axis=1, dtype=np.float64, keepdims=True))
                                )
                                cand_norm = q_lp - (
                                    cand_max + np.log(np.sum(np.exp(q_lp - cand_max), axis=1, dtype=np.float64, keepdims=True))
                                )
                                values = np.sum(
                                    np.exp(ref_norm) * (ref_norm - cand_norm), axis=1, dtype=np.float64
                                )
                                if values.size != count or not np.isfinite(values).all():
                                    raise ArtifactError(f"continuous resident score invalid KLD at window {window}")
                                row_terms = [float(value) for value in values.tolist()]
                                row_top1 = int(np.count_nonzero(q_argmax == idx0))
                                terms.extend(row_terms)
                                top1 += row_top1
                                per_window.append({
                                    "ordinal": offset + head_offset + head_index,
                                    "window": window,
                                    "positions": count,
                                    "support": 8192,
                                    "kld_sum_binary64": math.fsum(row_terms),
                                    "top1": row_top1,
                                })
            torch.cuda.synchronize()
        finally:
            reads.active = False
        read_delta = reads.delta(ready_counter)
        local_terminal = {
            "rank": self.rank,
            "timed_model_payload_reads": read_delta,
            "timed_score_file_reads": read_delta,
            "file_read_paths": list(reads.paths),
            "fallback_calls": int(self.status.get("fallback_calls", 0)),
            "reconstruction_calls": int(self.status.get("reconstruction_calls", 0)),
            "cpu_relay_bytes": int(self.status.get("cpu_relay_bytes", 0)),
            "model_object_id": id(self.student),
            "optimizer_object_id": id(self.optimizer),
            "scheduler_object_id": id(self.scheduler),
        }
        terminals: list[Any] = [None, None]
        self.dist.all_gather_object(terminals, local_terminal)
        if any(
            any(int(row[key]) != 0 for key in (
                "timed_model_payload_reads", "timed_score_file_reads", "fallback_calls",
                "reconstruction_calls", "cpu_relay_bytes",
            ))
            for row in terminals
        ):
            raise ArtifactError(f"continuous resident zero-read closure failed: {terminals}")
        result = None
        if self.rank == 1:
            if len(terms) != 64 * 1024 or len(per_window) != 64:
                raise ArtifactError("continuous resident score position/window closure drift")
            result = {
                "kld_mean": math.fsum(terms) / len(terms),
                "top1": top1,
                "positions": len(terms),
                "support": 8192,
                "windows": list(ordered),
                "per_window": per_window,
                "execution_mode": "resident_in_memory",
                "timed_wall_seconds": time.perf_counter() - started,
                "runtime_counters": {
                    "timed_model_payload_reads": 0,
                    "timed_score_file_reads": 0,
                    "fallback_calls": 0,
                    "reconstruction_calls": 0,
                    "cpu_relay_bytes": 0,
                    "resident_ready": terminals,
                },
            }
        rows = [result]
        self.dist.broadcast_object_list(rows, src=1)
        if not isinstance(rows[0], Mapping):
            raise ArtifactError("continuous resident score fan-in produced no result")
        return dict(rows[0])

    def broadcast_persisted(self, value: Any) -> Any:
        row = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(row, src=0)
        return row[0]

    def close(self) -> None:
        if self.dist.is_initialized() and self.config.get("destroy_process_group", False):
            self.dist.destroy_process_group()
