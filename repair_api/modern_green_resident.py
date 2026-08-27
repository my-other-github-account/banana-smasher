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
from typing import Any, Callable, Mapping, cast

from .balanced64 import ArtifactError

MODEL_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
ADMISSION_SHA256 = "76d0674eb0cd37fc9022bac5e048c2b77c721826182222ae0a0609e29607a2c5"
CORPUS_SHA256 = "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5"
TRAINER_SHA256 = "a55c2f5104b8d9dd06d845684d168be6f6e9dae637bac08443bd6ddbaf94201a"
HISTORICAL_TRAINER_SHA256 = "c8df3ab6a815fd69e401db7047afee53e9b0ce5652bf7fbcb9116d308c1b8e24"
WINDOWS_PER_STEP = 4
PIPELINE_MICROBATCH = 4
BASE_LRS = {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2}
HISTORICAL_BASE_LRS = {"luts": 2.5e-4, "norms": 2.5e-5, "outputs": 2.5e-4}
PUBLISHED_PRE_RECIPE_ID = "published_pre_lower_lr_warmup16_cosine64_v1"
PUBLISHED_PRE_SHA256 = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
PUBLISHED_PRE_BASE_LRS = {"luts": 1.0e-3, "norms": 1.0e-4, "outputs": 1.0e-3}
STATIC_W28_VALIDATION_CORPUS_SHA256 = "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd"
STATIC_W28_TEACHER_SHA256 = "561753481a1e08aee88e28f5fa0c6e727f4af679494c39679e87ed5189e2653d"
SEALED_GROUPED_WRAPPER_SHA256 = "37b919ae6adb34987e0e20ba4318352d9bf07b5183008d023a1822b3daf75126"
SEALED_GROUPED_EXPERT_SHA256 = "8080d1e6ef6752c7823a4db0426c6ea048b830a1ece173b30a7b12f716d1685b"
ACCEPTED_W28_PRODUCER_COMMIT = "0eebc78245129bcdc47fbb08964f6c2145b7ff7b"
ACCEPTED_W28_EXTENSION_SHA256 = "dedb8798912f0ad31f9002f53407cde153ee50e1b8da272c2b4b976cb1a6922d"
ACCEPTED_W28_RECEIPT_SHA256_BY_RANK = {
    0: "e3ad2a26830d7b481d69af981121faa63935c174d23ad88e1c0bc80e1d1e1816",
    1: "7c3fbd8435cc2712933ce19b4cddd939d76f5ce36bab7d7b06fc00e52dbe95e7",
}
STATIC_W28_GROUPED_WRAPPER_SHA256 = "ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
STATIC_W28_GROUPED_EXPERT_SHA256 = "64403d3e9b9761c3fcc636ba24d4d65c635f57675c1f749af312d441d55407c4"
FAST_K2_EXTENSION_CPP_SHA256 = "de5e3f522fe3ef02d1b82edebd85569f5f1fe6d2b7c17261e010e40883063dee"
FAST_K2_EXTENSION_CUDA_SHA256 = "dbc226dc6bcf4b467f0193824c41adc8e62bcf6fea762370796b24707ff2a9e1"
FAST_K2_EXTENSION_SOURCE_BUNDLE_SHA256 = "9f27d9911108712b6a7366490f51144d58bd19a8182de2105f000fa81db17266"
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
    config: Mapping[str, Any], global_step: int
) -> tuple[dict[str, float], float, list[int]]:
    """Resolve David's sealed PRE recipe: lower LR, warmup, then true cosine."""
    if config.get("recipe_id") != PUBLISHED_PRE_RECIPE_ID:
        raise ArtifactError("published PRE recipe id drift")
    if config.get("published_pre_checkpoint_sha256") != PUBLISHED_PRE_SHA256:
        raise ArtifactError("published PRE checkpoint declaration drift")
    lr_scalar = float(config.get("lr_scale", 1.0))
    if not math.isfinite(lr_scalar) or lr_scalar <= 0.0:
        raise ArtifactError("published PRE lr_scale must be finite and positive")
    step = int(global_step)
    if not 0 <= step < 64:
        raise ArtifactError("published PRE recipe cursor must be within U0..U63")
    if step < 16:
        multiplier = (step + 1) / 16.0
    else:
        relative = step - 16
        multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * relative / 48.0))
    return dict(PUBLISHED_PRE_BASE_LRS), multiplier * lr_scalar, [28, 56]


def _published_pre_controlled_schedule(
    schedule: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[int], int, str]:
    """Bind the declared fresh-PRE U1..U4 schedule in file order."""
    if config.get("tailfix_wholesale") is True:
        from .tailfix_wholesale import build_four_update_schedule, validate_wholesale_config

        try:
            validate_wholesale_config(config)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        rows_value = schedule.get("updates")
        expected = build_four_update_schedule()
        if not isinstance(rows_value, list) or rows_value != expected or not all(
            isinstance(row, Mapping) for row in rows_value
        ):
            raise ArtifactError("tailfix wholesale four-update schedule drift")
        membership = schedule.get("train_bank_membership_sha256")
        if not isinstance(membership, str) or not membership:
            raise ArtifactError("tailfix wholesale schedule membership is missing")
        return list(rows_value), [1, 2, 3, 4], 4, membership
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
    """Select one accepted teacher source for training loss and held-out scoring."""
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


def _resolve_runtime_provider_files(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the exact provider files that the trainer will import at runtime."""
    wrapper_value = config.get("fast_k2_wrapper_source")
    expert_value = config.get("fast_v7_expert_source")
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
    return {
        "wrapper_path": wrapper,
        "wrapper_sha256": wrapper_sha,
        "expert_path": expert,
        "expert_sha256": expert_sha,
    }


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
    selection = _resolve_runtime_provider_files(config)
    wrapper = selection["wrapper_path"]
    wrapper_sha = selection["wrapper_sha256"]
    expert = selection["expert_path"]
    expert_sha = selection["expert_sha256"]
    extension_sha = str(config.get("fast_k2_extension_sha256", ""))
    extension_source_sha = str(config.get("fast_k2_extension_source_bundle_sha256", ""))
    if sealed_published_pre and extension_sha != ACCEPTED_W28_EXTENSION_SHA256:
        raise ArtifactError(
            "accepted W28 fast K2 extension SHA mismatch: "
            f"{extension_sha} != {ACCEPTED_W28_EXTENSION_SHA256}"
        )
    if not sealed_published_pre and extension_source_sha != FAST_K2_EXTENSION_SOURCE_BUNDLE_SHA256:
        raise ArtifactError(
            "official resident fast K2 extension source bundle SHA mismatch: "
            f"{extension_source_sha} != {FAST_K2_EXTENSION_SOURCE_BUNDLE_SHA256}"
        )
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
        self.tailfix_wholesale = config.get("tailfix_wholesale") is True
        if self.tailfix_wholesale:
            from .tailfix_wholesale import validate_wholesale_config

            try:
                validate_wholesale_config(config)
            except ValueError as exc:
                raise ArtifactError(str(exc)) from exc
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
        self.expert_parallel_all_layers = bool(
            config.get("expert_parallel_all_layers", False)
        )
        if self.expert_parallel_all_layers:
            self.first, self.last = (0, 42)
        else:
            self.first, self.last = layer_ranges[rank]
        self.payload = payload
        self.state = payload.get("state")
        if not isinstance(self.state, Mapping):
            raise ArtifactError("U16 checkpoint state must contain official trainable surfaces")
        if set(self.state) != {"luts", "norms", "outputs"}:
            raise ArtifactError("U16 state must contain exactly luts, norms, and outputs")
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
        self.corpus_path = Path(str(config["corpus"])).expanduser().resolve()
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
        self.teacher_root = _resolve_scorer_aligned_training_teacher_root(config)
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
        if self.expert_parallel_all_layers:
            for expert in self.student.experts.values():
                expert.configure_tensor_parallel(self.rank, 2, None)
            if self.rank == 1:
                from torch import nn
                self.student.model.model.embed_tokens.weight = nn.Parameter(
                    self.student.get_tensor("embed.weight")
                    .to(self.device).to(torch.bfloat16),
                    requires_grad=False,
                )
        self.luts, self.norms, self.outputs = self.trainer.expose_local_dense(torch, self.student, admission)
        self._load_local_trainable_state()
        if self.tailfix_wholesale:
            from .tailfix_wholesale import build_fresh_adam_cosine

            self.optimizer, self.scheduler = build_fresh_adam_cosine(
                torch,
                [p for _name, p in self.luts],
                [p for _name, p in self.norms],
                [p for _name, p in self.outputs],
                steps=4,
            )
        else:
            optimizer_lrs = (
                PUBLISHED_PRE_BASE_LRS if self.published_pre_recipe
                else HISTORICAL_BASE_LRS if self.controlled_arm else BASE_LRS
            )
            self.optimizer = torch.optim.Adam(
                [
                    {"params": [p for _name, p in self.luts], "lr": optimizer_lrs["luts"], "group_name": "luts"},
                    {"params": [p for _name, p in self.norms], "lr": optimizer_lrs["norms"], "group_name": "norms"},
                    {"params": [p for _name, p in self.outputs], "lr": optimizer_lrs["outputs"], "group_name": "outputs"},
                ],
                foreach=True,
            )
            if self.published_pre_recipe:
                lr_lambda = lambda local_step: _published_pre_recipe_policy(
                    self.config, int(local_step)
                )[1]
            elif self.controlled_arm:
                lr_lambda = lambda local_step: _controlled_arm_policy(
                    self.config, _controlled_scheduler_step(str(self.controlled_arm_id), int(local_step))
                )[1]
            else:
                lr_lambda = self.trainer.current_multiplier
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=[lr_lambda] * 3
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
                    "resident_validation_expert_implementation", "accepted_static_w28"
                )
            ).lower()
            if expert_implementation not in {
                "accepted_static_w28", "sealed_bf16_full_weight", "packed_cuda_bf16_boundary",
            }:
                raise ArtifactError(
                    "resident validation expert implementation must be accepted_static_w28, "
                    "sealed_bf16_full_weight, or packed_cuda_bf16_boundary"
                )
            os.environ["BR_ATTN_IMPL"] = attention
            packed_validation = expert_implementation == "packed_cuda_bf16_boundary"
            os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = (
                "1" if expert_implementation == "sealed_bf16_full_weight" else "0"
            )
            # The first physical PRE tap localized packed-provider drift to the
            # extra FP32→BF16→FP32 projection round at L000/w1.  Packed wire/CUDA
            # arithmetic is already the accepted provider; do not add that
            # scientifically divergent boundary.
            os.environ["FAST_K2_SEALED_PROJECTION_BF16"] = "0"
            os.environ["FAST_K2_SEALED_NO_SWIGLU_CLAMP"] = "1"
            # Keep the immutable accepted grouped provider installed above.
            # Replacing it with the trainer's separate-projection PlaneSource
            # implementation changed W28 from 0.13712959240533734/877 to
            # 0.14319767370156203/871 even though the LUT bytes were identical.
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
                self.config, self.global_step
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
            _published_pre_recipe_policy(self.config, self.global_step)
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

    def _batch_p2p_isend(self, tensor: Any, *, dst: int) -> Any:
        """Start one rank-pipeline send while retaining its tensor owner."""
        print(f"BANANA_P2P rank={self.rank} op=isend phase=enter peer={dst}", flush=True)
        requests = self.dist.batch_isend_irecv([
            self.dist.P2POp(self.dist.isend, tensor, dst)
        ])
        if len(requests) != 1:
            raise ArtifactError("official resident batched P2P isend request drift")
        return requests[0]

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
    def _attention_workspace_key(
        query: Any, key: Any, chunk_size: int, logits_dtype: Any
    ) -> tuple[Any, ...]:
        return (
            str(query.device), str(query.dtype), tuple(int(value) for value in query.shape),
            int(key.shape[-2]), int(chunk_size), str(logits_dtype),
        )

    def _attention_workspace_for(
        self, query: Any, key: Any, chunk_size: int, logits_dtype: Any
    ) -> tuple[Any, Any, Any]:
        """Own one reusable eager workspace per exact-pair CUDA stream."""
        batch, heads, query_rows, width = query.shape
        key_rows = int(key.shape[-2])
        rows = min(int(chunk_size), int(query_rows))
        device = getattr(query, "device", None)
        stream_key: Any = "cpu"
        if getattr(device, "type", None) == "cuda":
            stream_key = int(self.torch.cuda.current_stream(device=device).cuda_stream)
        workspace_key = (
            stream_key,
            *self._attention_workspace_key(query, key, rows, logits_dtype),
        )
        workspaces = getattr(self, "_attention_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._attention_workspaces = workspaces
        current = workspaces.get(workspace_key)
        if current is None:
            output = query.new_empty((batch, query_rows, heads, width))
            weights = query.new_empty((batch, heads, rows, key_rows))
            logits = self.torch.empty(
                (batch, heads, rows, key_rows + 1), device=query.device, dtype=logits_dtype
            )
            current = (workspace_key, output, weights, logits)
            workspaces[workspace_key] = current
        return current[1], current[2], current[3]

    @staticmethod
    def _chunked_eager_attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        """Run the exact eager equation over bounded query-row chunks."""
        torch = __import__("torch")
        chunk_size = int(kwargs.pop("query_chunk_size", 512))
        observer = kwargs.pop("_chunk_observer", None)
        workspace_observer = kwargs.pop("_workspace_observer", None)
        workspace_factory = kwargs.pop("_resident_workspace_factory", None)
        if chunk_size <= 0:
            raise ArtifactError("resident eager attention query chunk must be positive")
        if not callable(workspace_factory):
            raise ArtifactError("resident eager attention caller workspace is required")

        def repeat_kv(states: Any, repeats: int) -> Any:
            batch, heads, length, width = states.shape
            if repeats == 1:
                return states
            states = states[:, :, None, :, :].expand(batch, heads, repeats, length, width)
            return states.reshape(batch, heads * repeats, length, width)

        batch, heads, query_rows, _width = query.shape
        repeats = int(module.num_key_value_groups)
        if int(key.shape[1]) == 1 and repeats == heads:
            key_states = key
            value_states = value
        else:
            key_states = repeat_kv(key, repeats)
            value_states = repeat_kv(value, repeats)
        logits_dtype = torch.promote_types(query.dtype, module.sinks.dtype)
        factory = cast(Callable[..., tuple[Any, Any, Any]], workspace_factory)
        output, weight_workspace, logits_workspace = factory(
            query, key, chunk_size, logits_dtype
        )
        for start in range(0, query_rows, chunk_size):
            end = min(start + chunk_size, query_rows)
            rows = end - start
            if observer is not None:
                observer(rows)
            query_chunk = query[:, :, start:end]
            logits = logits_workspace[:, :, :rows]
            if workspace_observer is not None:
                workspace_observer(output, weight_workspace, logits_workspace)
            weights = weight_workspace[:, :, :rows]
            torch.matmul(query_chunk, key_states.transpose(2, 3), out=weights)
            weights.mul_(scaling)
            if attention_mask is not None:
                mask = attention_mask
                if int(mask.shape[-2]) == query_rows:
                    mask = mask[..., start:end, :]
                weights.add_(mask)
            logits[..., :-1].copy_(weights)
            logits[..., -1:].copy_(
                module.sinks.reshape(1, -1, 1, 1).expand(batch, -1, rows, -1)
            )
            logits.sub_(logits.max(dim=-1, keepdim=True).values)
            torch.ops.aten._softmax.out(logits, -1, False, out=logits)
            scores = logits[..., :-1]
            scores = torch.nn.functional.dropout(
                scores, p=dropout, training=bool(getattr(module, "training", False))
            ).to(value_states.dtype)
            torch.matmul(scores, value_states, out=output[:, start:end].transpose(1, 2))
        return output, None

    def _install_chunked_eager(self) -> None:
        if getattr(self, "_chunked_eager_installed", False):
            return
        layer = self.student.model.model.layers[self.first]
        attention = layer.self_attn
        forward = attention.forward
        function = getattr(forward, "__func__", forward)
        namespace = getattr(function, "__globals__", None)
        interface = namespace.get("ALL_ATTENTION_FUNCTIONS") if isinstance(namespace, dict) else None
        register = getattr(interface, "register", None)
        if not callable(register):
            raise ArtifactError("chunked eager attention registration seam drift")

        def chunked_eager(module: Any, query: Any, key: Any, value: Any,
                          attention_mask: Any, scaling: float, dropout: float = 0.0,
                          **kwargs: Any) -> tuple[Any, None]:
            return self._chunked_eager_attention_forward(
                module, query, key, value, attention_mask, scaling, dropout,
                query_chunk_size=int(self.config["attention_query_chunk_size"]),
                _resident_workspace_factory=self._attention_workspace_for, **kwargs,
            )

        implementation = "banana_smasher_chunked_eager"
        register(implementation, chunked_eager)
        self.student.model.config._attn_implementation = implementation
        self.student.model.config._attn_implementation_internal = implementation
        self._chunked_eager_installed = True

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

        The ordinary Transformers SDPA adapter drops ``module.sinks``. Encode the
        sink as one extra zero-value KV token: an added constant query coordinate
        dotted with a per-head sink coordinate reproduces the exact sink logit,
        while the zero value reproduces eager's dropped sink contribution.
        """
        torch = __import__("torch")
        heads = int(query.shape[1])
        if int(key.shape[1]) != heads:
            if heads % int(key.shape[1]):
                raise ArtifactError("sink-corrected SDPA GQA head geometry drift")
            repeats = heads // int(key.shape[1])
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        width = int(query.shape[-1])
        pad_width = 8 - (width % 8)
        query_aug = torch.nn.functional.pad(query, (0, pad_width))
        query_aug[..., width] = 1
        key_aug = torch.nn.functional.pad(key, (0, pad_width))
        value_aug = torch.nn.functional.pad(value, (0, pad_width))
        sink_key = key_aug.new_zeros((*key_aug.shape[:-2], 1, width + pad_width))
        sink_key[..., width] = module.sinks.reshape(1, -1, 1) / float(scaling)
        sink_value = value_aug.new_zeros((*value_aug.shape[:-2], 1, width + pad_width))
        key_aug = torch.cat((key_aug, sink_key), dim=-2)
        value_aug = torch.cat((value_aug, sink_value), dim=-2)
        mask_aug = (
            torch.nn.functional.pad(attention_mask, (0, 1), value=0.0)
            if attention_mask is not None else None
        )
        output = torch.nn.functional.scaled_dot_product_attention(
            query_aug, key_aug, value_aug, attn_mask=mask_aug,
            dropout_p=float(dropout), is_causal=False, scale=float(scaling),
        )
        return output[..., :width].transpose(1, 2).contiguous(), None

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

    def _run_layers(self, hidden: Any, ids: Any, train: bool) -> Any:
        from transformers.cache_utils import DynamicCache
        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        cache = DynamicCache(config=self.student.config)
        attention_implementation = str(
            self.config.get("resident_validation_attention_implementation", "eager")
        ).lower()
        if attention_implementation == "sdpa":
            self._install_sink_corrected_sdpa()
        elif int(self.config.get("attention_query_chunk_size", 0)) > 0:
            self._install_chunked_eager()
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
        token_values = []
        for row, window in enumerate(group):
            idx, lp_n, p_n = self.teacher_cache[window]
            length = self.real_lengths[window]
            logits = self.student.model.lm_head(final[row, :length].to(self.torch.bfloat16))
            support_logits = logits.gather(1, idx[:length]).float()
            if self.tailfix_wholesale:
                from .tailfix_wholesale import support_renormalized_teacher_student_kld

                values = support_renormalized_teacher_student_kld(
                    self.torch, lp_n[:length], support_logits
                )
            else:
                student_normalized = support_logits - support_logits.logsumexp(-1, keepdim=True)
                values = (p_n[:length] * (lp_n[:length] - student_normalized)).sum(-1)
            token_values.append(values)
        if self.tailfix_wholesale:
            from .tailfix_wholesale import detached_tail_weighted_loss

            loss, evidence = detached_tail_weighted_loss(self.torch, token_values)
            self.tailfix_loss_evidence = evidence
            return loss
        return self.torch.stack([values.mean() for values in token_values]).mean()

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
        backward_started = time.perf_counter()
        (loss / float(loss_divisor)).backward()
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
                (loss / float(loss_divisor)).backward()
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
        return [*self.luts, *self.norms, *self.outputs]

    def _local_norm(self, values: list[Any]) -> float:
        return sum(float(value.detach().float().pow(2).sum().cpu()) for value in values) ** 0.5

    def _step(self, global_step: int) -> dict[str, Any]:
        torch = self.torch
        params = self._local_params()
        before = [parameter.detach().clone() for _name, parameter in params]
        self.optimizer.zero_grad(set_to_none=True)
        if self.published_pre_recipe:
            if self.tailfix_wholesale:
                from .tailfix_wholesale import (
                    CODEBOOK_LR,
                    OUTPUT_GAIN_LR,
                    RMSNORM_LR,
                    cosine_multiplier,
                )

                base_lrs = {
                    "luts": CODEBOOK_LR,
                    "norms": RMSNORM_LR,
                    "outputs": OUTPUT_GAIN_LR,
                }
                multiplier = cosine_multiplier(global_step, 4)
                default_windows = []
            else:
                base_lrs, multiplier, default_windows = _published_pre_recipe_policy(
                    self.config, global_step
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
            base_lrs = BASE_LRS
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
            "trainable_surfaces": ["luts", "rmsnorms", "output_gains"],
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

    def _score_pair_stream_pool(self, concurrency: int) -> list[Any]:
        """Reuse one bounded stream set so stream-keyed workspaces cannot grow per group."""
        if concurrency < 1:
            raise ArtifactError("resident validation pair stream concurrency must be positive")
        pool = getattr(self, "_score_pair_streams", None)
        if pool is None:
            pool = [
                self.torch.cuda.Stream(device=self.student.device)
                for _ in range(concurrency)
            ]
            self._score_pair_streams = pool
        elif len(pool) != concurrency:
            raise ArtifactError("resident validation pair stream pool concurrency drift")
        return pool

    def _validate_preloaded(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        """Run the sealed readout on inputs already resident outside timing."""
        from .balanced64 import POSITIONS_PER_WINDOW, SUPPORT
        from .official_k2_resident_score import (
            PayloadModelReadCounter,
            _sealed_pair_groups,
        )
        import numpy as np

        ordered = tuple(int(value) for value in prepared["windows"])
        physical = tuple(int(value) for value in prepared.get("physical_windows", ordered))
        physical_batch_size = int(prepared.get("physical_batch_size", self.score_pipeline_microbatch))
        if physical_batch_size < 1 or len(physical) % physical_batch_size:
            raise ArtifactError("resident validation physical fixture batch drift")
        pair_stream_concurrency = int(
            self.config.get("score_pair_stream_concurrency", 1)
        )
        if pair_stream_concurrency > 1:
            if physical_batch_size != 2:
                raise ArtifactError(
                    "resident validation pair concurrency requires exact sealed mb=2"
                )
            scheduled_batches = tuple(
                tuple(window for pair in group for window in pair)
                for group in _sealed_pair_groups(physical, pair_stream_concurrency)
            )
        else:
            scheduled_batches = tuple(
                physical[offset:offset + physical_batch_size]
                for offset in range(0, len(physical), physical_batch_size)
            )
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
        rank_phase_profiles: list[dict[str, Any]] = []
        previous_send: Any | None = None
        previous_hidden: Any | None = None
        try:
            for batch in scheduled_batches:
                pair_parallel = pair_stream_concurrency > 1 and len(batch) > 2
                pair_windows = (
                    tuple(batch[index:index + 2] for index in range(0, len(batch), 2))
                    if pair_parallel else (batch,)
                )
                mechanism_before = self._local_resident_mechanism_snapshot()
                ids = torch.cat([ids_cache[window] for window in batch], dim=0)
                pair_ids = tuple(
                    torch.cat([ids_cache[window] for window in pair], dim=0)
                    for pair in pair_windows
                )
                shape = (
                    len(batch), ids.shape[1], int(self.student.config.hc_mult),
                    int(self.student.config.hidden_size),
                )
                if self.rank == 0 and not self.expert_parallel_all_layers:
                    forward_started = time.perf_counter()
                    if pair_parallel:
                        launch_stream = torch.cuda.current_stream(device=self.student.device)
                        pair_streams = self._score_pair_stream_pool(
                            len(pair_windows)
                        )
                        hidden_pairs = []
                        for pair_id_tensor, stream in zip(pair_ids, pair_streams):
                            stream.wait_stream(launch_stream)
                            pair_id_tensor.record_stream(stream)
                            with torch.cuda.stream(stream):
                                pair_embeds = self.student.model.model.embed_tokens(
                                    pair_id_tensor
                                )
                                pair_hidden = pair_embeds.unsqueeze(2).expand(
                                    -1, -1, self.student.config.hc_mult, -1
                                ).contiguous()
                                pair_hidden = self._run_layers(
                                    pair_hidden, pair_id_tensor, False
                                )
                                pair_hidden.record_stream(launch_stream)
                                hidden_pairs.append(pair_hidden)
                        for stream in pair_streams:
                            launch_stream.wait_stream(stream)
                        hidden = torch.cat(hidden_pairs, dim=0)
                    else:
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
                    if bool(self.config.get("score_pipeline_overlap", False)):
                        if previous_send is not None:
                            previous_send.wait()
                            print(
                                f"BANANA_P2P rank={self.rank} op=isend phase=complete peer=1",
                                flush=True,
                            )
                            previous_send = None
                            previous_hidden = None
                        hidden = hidden.detach().contiguous()
                        previous_send = self._batch_p2p_isend(hidden, dst=1)
                        previous_hidden = hidden
                    else:
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
                        "sealed_pair_stream_concurrency": len(pair_windows),
                        "rank_pipeline_inflight": previous_send is not None,
                        "mechanism_before": mechanism_before,
                        "mechanism_after": mechanism_after,
                        "mechanism_counter_delta": mechanism_delta,
                    })
                    del ids
                else:
                    p2p_started = time.perf_counter()
                    if self.expert_parallel_all_layers:
                        embeds = self.student.model.model.embed_tokens(ids)
                        hidden = embeds.unsqueeze(2).expand(
                            -1, -1, self.student.config.hc_mult, -1
                        ).contiguous()
                    else:
                        hidden = torch.empty(
                            shape, dtype=torch.bfloat16, device=self.student.device
                        )
                        self._batch_p2p_recv(hidden, src=0)
                        _cuda_sync(torch)
                    p2p_ms = (time.perf_counter() - p2p_started) * 1000.0
                    forward_started = time.perf_counter()
                    if pair_parallel:
                        launch_stream = torch.cuda.current_stream(device=self.student.device)
                        pair_streams = self._score_pair_stream_pool(
                            len(pair_windows)
                        )
                        final_pairs = []
                        for pair_hidden, pair_id_tensor, stream in zip(
                            hidden.split(2, dim=0), pair_ids, pair_streams
                        ):
                            stream.wait_stream(launch_stream)
                            pair_hidden.record_stream(stream)
                            pair_id_tensor.record_stream(stream)
                            with torch.cuda.stream(stream):
                                pair_output = self._run_layers(
                                    pair_hidden, pair_id_tensor, False
                                )
                                pair_final = self.student.model.model.norm(
                                    self.student.model.model.hc_head(pair_output)
                                )
                                pair_final.record_stream(launch_stream)
                                final_pairs.append(pair_final)
                        for stream in pair_streams:
                            launch_stream.wait_stream(stream)
                        final = torch.cat(final_pairs, dim=0)
                    else:
                        hidden = self._run_layers(hidden, ids, False)
                        if self.expert_parallel_all_layers and self.rank == 0:
                            _cuda_sync(torch)
                            forward_ms = (
                                time.perf_counter() - forward_started
                            ) * 1000.0
                            mechanism_after = self._local_resident_mechanism_snapshot()
                            mechanism_delta = self._mechanism_counter_delta(
                                mechanism_before, mechanism_after
                            )
                            if mechanism_delta.get("reconstruction_calls", 0) != 0:
                                raise ArtifactError(
                                    "resident validation reconstructed weights inside a batch"
                                )
                            rank_phase_profiles.append({
                                "rank": self.rank,
                                "batch_windows": list(batch),
                                "weight_reconstruction_ms": 0.0,
                                "forward_ms": forward_ms,
                                "p2p_ms": p2p_ms,
                                "readout_ms": 0.0,
                                "sealed_pair_stream_concurrency": len(pair_windows),
                                "rank_pipeline_inflight": False,
                                "expert_parallel_all_layers": True,
                                "mechanism_before": mechanism_before,
                                "mechanism_after": mechanism_after,
                                "mechanism_counter_delta": mechanism_delta,
                            })
                            del hidden, ids
                            continue
                        final = self.student.model.model.norm(
                            self.student.model.model.hc_head(hidden)
                        )
                    _cuda_sync(torch)
                    forward_ms = (time.perf_counter() - forward_started) * 1000.0
                    readout_started = time.perf_counter()
                    for batch_index, window in enumerate(batch):
                        if window not in ordered:
                            continue
                        teacher_idx, teacher_logprob = teacher_cache[window]
                        logits = self.student.model.lm_head(
                            final[batch_index, :POSITIONS_PER_WINDOW].to(torch.bfloat16)
                        ).float()
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
                        "sealed_pair_stream_concurrency": len(pair_windows),
                        "rank_pipeline_inflight": False,
                        "mechanism_before": mechanism_before,
                        "mechanism_after": mechanism_after,
                        "mechanism_counter_delta": mechanism_delta,
                    })
                    del hidden, ids, final
            if self.rank == 0 and previous_send is not None:
                previous_send.wait()
                print(
                    f"BANANA_P2P rank={self.rank} op=isend phase=complete peer=1",
                    flush=True,
                )
                previous_send = None
                previous_hidden = None
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
                "execution_mode": "resident_in_memory",
                "timed_wall_seconds": time.perf_counter() - started,
                "phase_profiles_by_rank": phase_profiles_by_rank,
                "runtime_counters": {
                    "timed_model_payload_reads": 0, "timed_score_file_reads": 0,
                    "fallback_calls": 0, "reconstruction_calls": 0,
                    "cpu_relay_bytes": 0, "resident_ready": terminals,
                    "sealed_pair_stream_concurrency": pair_stream_concurrency,
                    "rank_pipeline_overlap": bool(
                        self.config.get("score_pipeline_overlap", False)
                    ),
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
