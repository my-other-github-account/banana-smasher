"""High-level resident API for Modern Green Balanced64 experiments.

The façade owns one artifact, one loader, and a resident cache. Experiment
runners should use this module instead of reimplementing checkpoint binding,
score identity, or timing around :class:`RepairArtifact`.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

from .balanced64 import ArtifactError, RepairArtifact, ScoreResult, _load_torch
from .core import SharedPreflight
from .official_k2_resident_score import (
    ALTERNATE_PRE_CHECKPOINT_SHA256,
    BASIS_SHA256,
    ROUTED_K2_API_METHOD,
    ROUTED_K2_API_VERSION,
    ROUTED_K2_CLOSURE,
    ROUTED_K2_ROUTE_KIND,
    _published_pre_production_admitted,
    validate_routed_k2_closure,
)
from .sealed_pre_forward import bind_sealed_pre_resident_config


SCORER_CHECKPOINT_FORMAT = "banana-smasher-qtip2-v7-joint-checkpoint-v1"


def _record_engine_step_phase(
    engine: Any,
    *,
    update: int,
    phase: str,
    boundary: str,
    elapsed_seconds: float | None = None,
) -> None:
    recorder = getattr(engine, "record_step_phase", None)
    if callable(recorder):
        recorder(
            update=update,
            phase=phase,
            boundary=boundary,
            elapsed_seconds=elapsed_seconds,
        )


def _validate_trainable_scale_candidate_contract(
    *,
    start_update: int,
    start_sha: Any,
    requested: tuple[int, ...],
    config: Mapping[str, Any],
) -> bool:
    """Admit Candidate C/D without widening any other scientific axis."""
    if config.get("trainable_quantization_scales") is not True:
        return False
    relative_bound = config.get("quantization_scale_relative_trust_region")
    candidate_d = relative_bound is not None
    expected_identity = (
        "Candidate D: U20-to-U24; sole variable versus Candidate C is a 5% U20-relative scale trust region"
        if candidate_d
        else "Candidate C: U20-to-U24; sole variable is grouped-K2 quantization scales frozen-to-trainable"
    )
    if (
        start_update != 20
        or start_sha
        != "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
        or requested != (21, 22, 23, 24)
        or config.get("token_kld_reduction") != "mean"
        or config.get("scientific_identity") != expected_identity
        or (candidate_d and relative_bound != 0.05)
        or (
            candidate_d
            and config.get("mechanism_receipt_sha256")
            != "2a706eece007225b1a37d9977102659e5bdedd736d04585b577128e0c5918d36"
        )
        or config.get("tailfix_wholesale") is True
        or "cvar_tail_fraction" in config
    ):
        raise ArtifactError(
            "trainable scale candidate requires exact authenticated U20-to-U24 uniform-mean contract"
        )
    return True


#: Dense trainable surfaces whose insertion order is load-bearing downstream.
#: The sealed scorer's live roster is canonical name order, while two-rank
#: checkpoints are persisted in rank-partition order.
_ORDERED_DENSE_SURFACES = ("norms", "outputs")


def adapt_checkpointed_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Admit a persisted public-API checkpoint to the sealed scorer envelope.

    Continuation checkpoints predate the scorer's ``format`` discriminator,
    but already carry the same three trainable surfaces and authenticated
    loaded-checkpoint identity.  Add that discriminator after validating the
    canonical public-API envelope, and normalise the dense surfaces to
    canonical order so world_size=2 continuations are admitted on exactly the
    same terms as world_size=1 ones.  Tensor values and state objects remain
    untouched — the dense surfaces are re-keyed, never rewritten.
    """
    if not isinstance(payload, Mapping):
        raise ArtifactError("checkpointed scorer envelope must be a mapping")
    if payload.get("format") == SCORER_CHECKPOINT_FORMAT:
        return dict(payload)
    if payload.get("schema") != "resident-continuation-checkpoint-v1":
        raise ArtifactError("checkpointed scorer envelope schema refused")
    state = payload.get("state")
    admitted_surfaces = (
        {"luts", "norms", "outputs"},
        {"luts", "norms", "outputs", "scales"},
    )
    if not isinstance(state, Mapping) or set(state) not in admitted_surfaces:
        raise ArtifactError("checkpointed scorer envelope state surfaces refused")
    next_update = payload.get("next_update")
    if isinstance(next_update, bool) or not isinstance(next_update, int) or next_update < 0:
        raise ArtifactError("checkpointed scorer envelope update cursor refused")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or identity.get("checkpoint_loaded") is not True:
        raise ArtifactError("checkpointed scorer envelope requires loaded checkpoint identity")
    if identity.get("next_update") != next_update:
        raise ArtifactError("checkpointed scorer envelope identity cursor drift")
    admitted = dict(payload)
    admitted["format"] = SCORER_CHECKPOINT_FORMAT
    reordered: dict[str, Any] = {}
    for surface_name in _ORDERED_DENSE_SURFACES:
        surface = state[surface_name]
        if not isinstance(surface, Mapping):
            # Opaque/non-mapping surfaces are carried through untouched; only a
            # real name->tensor mapping has an order to normalise.
            continue
        if any(not isinstance(name, str) for name in surface):
            raise ArtifactError(
                f"checkpointed scorer envelope {surface_name} surface has non-string keys"
            )
        canonical = sorted(surface)
        if list(surface) != canonical:
            reordered[surface_name] = {name: surface[name] for name in canonical}
    if reordered:
        # Only rebuild the state mapping when a surface actually needed
        # normalising, so the already-canonical path keeps passing the exact
        # same state object through by identity.
        normalised = dict(state)
        normalised.update(reordered)
        admitted["state"] = normalised
    return admitted


_STALE_SPARK5_MODEL_ROOT = Path(
    "/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/sparse-model-rank0-v1"
)
_SPARK3_SEALED_PARENT_ROOT = Path(
    "/home/dnola/missions/V7_CODEBOOK_FULLPARENT_t_569e9977_s3"
)
_SPARK3_SEALED_PARENT_MANIFEST_ROOT = Path(
    "/home/dnola/missions/QTIP2_V7_JOINT_t_6aceaf1f_s3/"
    "lut_parents_run1820_backup"
)
_SPARK3_SEALED_L034_ROSTER = Path(
    "/home/dnola/missions/QTIP2_V7_JOINT_t_6aceaf1f_s3/l034/"
    "L034_SELECTED_WIRE_PROVIDER_ROSTER.json"
)


def _select_exact_manifest_member(
    candidates: Iterable[Path], *, expected_sha256: str, label: str
) -> Path:
    """Select only identity-equal duplicates of a sealed manifest member."""
    present = [path.resolve() for path in candidates if path.is_file()]
    if not present:
        raise ArtifactError(f"official-K2 sealed parent member is missing: {label}")
    observed = [(path, hashlib.sha256(path.read_bytes()).hexdigest()) for path in present]
    if any(digest != expected_sha256 for _, digest in observed):
        raise ArtifactError(f"official-K2 sealed parent non-identical ambiguity: {label}")
    return present[0]


def _resolve_exact_parent_manifest(
    declared: Path, *, localized: Path, expected_sha256: str, label: str
) -> Path:
    """Localize an absent sealed manifest only to an identity-exact copy."""
    candidate = declared if declared.is_file() else localized
    if not candidate.is_file():
        raise ArtifactError(f"official-K2 sealed parent manifest is missing: {label}")
    observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ArtifactError(f"official-K2 sealed parent manifest identity drift: {label}")
    return candidate.resolve()


def _validate_sealed_parent_root(
    *, parent_root: Path, admission_path: Path, manifest_root: Path
) -> None:
    """Bind the localized parent to the sealed roster's manifest assignments."""
    admission = json.loads(admission_path.read_text())
    rows = admission.get("trainable_roster", {}).get("luts", [])
    if len(rows) != 43:
        raise ArtifactError("official-K2 sealed parent roster coverage drift")
    for row in rows:
        layer = int(row["layer"])
        binding = row["source_manifest"]
        manifest = _resolve_exact_parent_manifest(
            Path(str(binding["path"])),
            localized=manifest_root / f"L{layer:03d}" / "parent/QTIP_V7_MANIFEST.json",
            expected_sha256=str(binding["sha256"]),
            label=f"L{layer:03d}",
        )
        # L034 is authenticated by its dedicated selected-wire roster below;
        # it is intentionally absent from the ordinary full-parent tree.
        if layer == 34:
            continue
        members = json.loads(manifest.read_text()).get("members", [])
        if not members:
            raise ArtifactError(f"official-K2 sealed parent manifest is empty: L{layer:03d}")
        for member in members:
            expert = int(member["expert"])
            projection = str(member["projection"])
            stem = parent_root / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
            _select_exact_manifest_member(
                (stem.with_suffix(".q2v7wire"), stem.with_suffix(".k2wire")),
                expected_sha256=str(member["sha256"]),
                label=f"L{layer:03d} E{expert:03d}/{projection}",
            )


def _resolve_official_k2_config_locators(config: Mapping[str, Any]) -> dict[str, Any]:
    """Localize the sealed Spark-5 closure to identity-equal Spark-3 inputs."""
    resolved = dict(config)
    override = os.environ.get("BANANA_SMASHER_OFFICIAL_MODEL_ROOT")
    if not override:
        return resolved
    declared = Path(str(resolved.get("model_root", ""))).expanduser()
    if declared != _STALE_SPARK5_MODEL_ROOT:
        raise ArtifactError("official-K2 model-root localization requires the sealed stale Spark-5 locator")
    if declared.exists():
        raise ArtifactError("official-K2 model-root localization refuses to replace a present sealed locator")
    candidate = Path(override).expanduser().resolve()
    index = candidate / "model.safetensors.index.json"
    if not index.is_file():
        raise ArtifactError(f"official-K2 localized model index is missing: {index}")
    observed = hashlib.sha256(index.read_bytes()).hexdigest()
    if resolved.get("basis_sha256") != BASIS_SHA256 or observed != BASIS_SHA256:
        raise ArtifactError("official-K2 localized model root failed the immutable basis gate")
    resolved["model_root"] = str(candidate)
    stage = Path("/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-3")
    canonical = Path(__file__).resolve().parents[1]
    replacements = {
        "asset_root": stage / "inputs/attempt4b/asset_view",
        "binrepair_delta_dir": stage / "inputs/attempt4b/delta",
        "binrepair_manifest": stage / "inputs/attempt4b/asset_view/code/DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json",
        "corpus": stage / "inputs/attempt4b/asset_view/code/BASIC_COMBINED_768.json",
        "fast_k2_extension": stage / "inputs/banana_fast_k2_grouped_0c3cc723fe66.so",
        "fast_k2_wrapper_source": canonical / "repair_api/assets/u20_resident_provider/fast_k2_grouped.py",
        "l034_roster": _SPARK3_SEALED_L034_ROSTER,
        "lut_parent_root": _SPARK3_SEALED_PARENT_MANIFEST_ROOT,
        "lp4_pack_source": canonical / "runtime/v7/vendor/src_lp4/lp4_pack.py",
        "lp4_train_source": canonical / "runtime/v7/vendor/src_lp4/lp4_train.py",
        "official_expert_source": stage / "R26_joint_v7_expert_base.py",
        "parent_root": _SPARK3_SEALED_PARENT_ROOT,
        "resident_expert_source": stage / "repo-r30c8/repair-api/ds4-flash-kldmatrix/repair_api/assets/fast_v7_expert_base.py",
        "trainer_source": stage / "repo-r30c8/repair-api/ds4-flash-kldmatrix/repair_api/assets/modern_green_clean_u0.py",
    }
    expected_shas = {
        "fast_k2_extension": resolved.get("fast_k2_extension_sha256"),
        "fast_k2_wrapper_source": resolved.get("fast_k2_wrapper_source_sha256"),
        "official_expert_source": resolved.get("official_expert_source_sha256"),
        "resident_expert_source": resolved.get("resident_expert_source_sha256"),
        "trainer_source": resolved.get("trainer_source_sha256"),
    }
    for field, replacement in replacements.items():
        if not replacement.exists():
            raise ArtifactError(f"official-K2 localized immutable input is missing: {replacement}")
        expected = expected_shas.get(field)
        if expected and hashlib.sha256(replacement.read_bytes()).hexdigest() != expected:
            raise ArtifactError(f"official-K2 localized immutable SHA mismatch: {replacement}")
        resolved[field] = str(replacement.resolve())
    _validate_sealed_parent_root(
        parent_root=Path(resolved["parent_root"]),
        admission_path=Path(resolved["asset_root"]) / "code/JOINT_REPAIR_ADMISSION.json",
        manifest_root=_SPARK3_SEALED_PARENT_MANIFEST_ROOT,
    )
    bind_sealed_pre_resident_config(resolved)
    return resolved


def _validate_published_pre_resume_start(
    start_update: int,
    start_meta: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> None:
    """Admit only an identity-exact scored resume of the published-PRE recipe."""
    exact_u20 = "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
    if (
        start_update == 20
        and start_meta.get("sha256") == exact_u20
        and start_meta.get("optimizer_scheduler_lineage") == "fresh-published-pre-adam-lambdalr"
        and config.get("checkpoint_sha256") == exact_u20
        and (
            (
                config.get("execution_backend") == "single_gpu_resident_no_recompute"
                and config.get("activation_checkpointing") is False
            )
            or (
                config.get("execution_backend") == "single_gpu_resident_checkpointed"
                and config.get("activation_checkpointing") is True
            )
        )
        and config.get("world_size") == 1
        and config.get("rank") == 0
        and config.get("lr_scale") == 0.5
        and config.get("recipe_id") == "published_pre_lower_lr_warmup16_cosine64_v1"
        and config.get("published_pre_checkpoint_sha256")
            == "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        and config.get("fresh_published_pre_lineage") is True
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    ):
        return
    exact_u21 = "11df795d56d7f9210f20bb99e91b6518dc17d0e24cbfff6b96e120168ab64830"
    if (
        start_update == 21
        and start_meta.get("sha256") == exact_u21
        and start_meta.get("optimizer_scheduler_lineage") == "fresh-published-pre-adam-lambdalr"
        and config.get("checkpoint_sha256") == exact_u21
        and config.get("execution_backend") == "single_gpu_resident_checkpointed"
        and config.get("activation_checkpointing") is True
        and config.get("world_size") == 1
        and config.get("rank") == 0
        and config.get("lr_scale") == 0.5
        and config.get("fresh_published_pre_lineage") is True
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    ):
        return
    sealed_u22_sha256 = "47ff9433ef40877035d4db2aab60e8ad3aac0c214f0cea32fa338f4eb8346f82"
    sealed_u31_sha256 = "1a0ed291da9e0edc5094de892ca9fb4ae3fdd20b2cc6bfbf59fe2871eb90fffe"
    sealed_u33_sha256 = "0abdab68a393163993749a95b8cc6809f43b26e73cdc118ada1e9e58e725eff9"
    sealed_u34_sha256 = "06ccaeac47c3ac6862db713d469c5da6007545e07cec760cdb0470e2e3ddb878"
    sealed_u35_sha256 = "77cb4661aea34aba4aa46e446673fc58016f70a17b2cd8e2caaa5c3d864a70e6"
    sealed_u36_sha256 = "e62bdecb663ad7dda14dee3244f0da277093f87e4d49a9dae61a563863bc8802"
    sealed_u37_sha256 = "6de85bc531022602c65b69ff4091e1e8f48102926d48158d6682843f9c7a6a6f"
    sealed_u38_sha256 = "f9b3c4ae3672d876e8c7c4c54138a7d72f67c6f5a9a450d9cd9562628748759b"
    sealed_u41_sha256 = "40544a550331b4e59b71bdea8b348832a254f94f3847ec33735a9de5bb7a1879"
    published_pre_sha256 = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
    sealed_u22_config = {
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": published_pre_sha256,
        "lr_scale": 0.75,
        "seed": 1701,
        "controlled_window_schedule_sha256":
            "cb124895b563b26ffc10a68c0cf1908094c3750791b6431783087dde7c0f17f8",
        "shared_optimizer_scheduler_lineage":
            "fresh-published-pre-adam-lambdalr",
    }
    sealed_fresh_pre_u22_checkpoint = (
        start_update == 22
        and start_meta.get("sha256") == sealed_u22_sha256
        and config.get("checkpoint_sha256") == sealed_u22_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u22 = (
        sealed_fresh_pre_u22_checkpoint
        and all(config.get(field) == value for field, value in sealed_u22_config.items())
    )
    sealed_fresh_pre_u31_checkpoint = (
        start_update == 31
        and start_meta.get("sha256") == sealed_u31_sha256
        and config.get("checkpoint_sha256") == sealed_u31_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u31 = (
        sealed_fresh_pre_u31_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    sealed_fresh_pre_u33_checkpoint = (
        start_update == 33
        and start_meta.get("sha256") == sealed_u33_sha256
        and config.get("checkpoint_sha256") == sealed_u33_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u33 = (
        sealed_fresh_pre_u33_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U33 to U34 continuation; instrumentation only via existing "
            "phase timers/profiler markers"
        )
    )
    sealed_fresh_pre_u34_checkpoint = (
        start_update == 34
        and start_meta.get("sha256") == sealed_u34_sha256
        and config.get("checkpoint_sha256") == sealed_u34_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u34 = (
        sealed_fresh_pre_u34_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U34 to U35 continuation; sole change is device-wide to "
            "current-stream synchronization after grouped backward"
        )
    )
    sealed_fresh_pre_u35_checkpoint = (
        start_update == 35
        and start_meta.get("sha256") == sealed_u35_sha256
        and config.get("checkpoint_sha256") == sealed_u35_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u35 = (
        sealed_fresh_pre_u35_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U35 to U36 continuation; sole change is nonblocking event "
            "ordering from grouped producer stream to default-stream gradient consumer"
        )
    )
    sealed_fresh_pre_u36_checkpoint = (
        start_update == 36
        and start_meta.get("sha256") == sealed_u36_sha256
        and config.get("checkpoint_sha256") == sealed_u36_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u36 = (
        sealed_fresh_pre_u36_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U36 to U37 continuation; sole change is CUDA grid-z "
            "parallelism across independent LUT-gradient output tiles"
        )
    )
    sealed_fresh_pre_u37_checkpoint = (
        start_update == 37
        and start_meta.get("sha256") == sealed_u37_sha256
        and config.get("checkpoint_sha256") == sealed_u37_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u37 = (
        sealed_fresh_pre_u37_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U37 to U40 continuation; runtime retains CUDA grid-z "
            "parallelism with no scientific change"
        )
    )
    sealed_fresh_pre_u38_checkpoint = (
        start_update == 38
        and start_meta.get("sha256") == sealed_u38_sha256
        and config.get("checkpoint_sha256") == sealed_u38_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u38 = (
        sealed_fresh_pre_u38_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.09375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == sealed_u22_config["controlled_window_schedule_sha256"]
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "exact finite U38 to U39 continuation; bounded-partial grad-LUT is the only variable"
        )
    )
    sealed_fresh_pre_u41_checkpoint = (
        start_update == 41
        and start_meta.get("sha256") == sealed_u41_sha256
        and config.get("checkpoint_sha256") == sealed_u41_sha256
        and start_meta.get("optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
    )
    authenticated_fresh_pre_u41 = (
        sealed_fresh_pre_u41_checkpoint
        and config.get("recipe_id") == sealed_u22_config["recipe_id"]
        and config.get("published_pre_checkpoint_sha256") == published_pre_sha256
        and config.get("lr_scale") == 0.375091552734375
        and config.get("seed") == 1701
        and config.get("controlled_window_schedule_sha256")
            == "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351"
        and config.get("shared_optimizer_scheduler_lineage")
            == "fresh-published-pre-adam-lambdalr"
        and config.get("scientific_identity") == (
            "t_f76a1035 repair-A winner U41-to-U45 four-update continuation"
        )
        and config.get("u41_parent_checkpoint_sha256")
            == "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"
        and config.get("u41_repair_a_terminal_receipt_sha256_by_rank") == {
            "0": "8ba35d756f54b6b8e9d377d65d83e11b077a364fa9b22eeddf4728129ea36fcb",
            "1": "5d1c4df51d441d8c5cdf99fefc0c73242e351fa517cb6c296d471864f4e5b446",
        }
    )
    if start_update <= 0 or start_update >= 64:
        raise ArtifactError("published PRE scored resume must start inside U1..U63")
    if start_update % 4 and not (
        sealed_fresh_pre_u22_checkpoint
        or sealed_fresh_pre_u31_checkpoint
        or sealed_fresh_pre_u33_checkpoint
        or sealed_fresh_pre_u34_checkpoint
        or sealed_fresh_pre_u35_checkpoint
        or sealed_fresh_pre_u37_checkpoint
        or sealed_fresh_pre_u38_checkpoint
        or sealed_fresh_pre_u41_checkpoint
    ):
        raise ArtifactError(
            "published PRE non-four-update resume requires authenticated sealed checkpoint"
        )
    if sealed_fresh_pre_u22_checkpoint and not authenticated_fresh_pre_u22:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u31_checkpoint and not authenticated_fresh_pre_u31:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u33_checkpoint and not authenticated_fresh_pre_u33:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u34_checkpoint and not authenticated_fresh_pre_u34:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u35_checkpoint and not authenticated_fresh_pre_u35:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u36_checkpoint and not authenticated_fresh_pre_u36:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u37_checkpoint and not authenticated_fresh_pre_u37:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u38_checkpoint and not authenticated_fresh_pre_u38:
        raise ArtifactError("published PRE scored resume identity drift")
    if sealed_fresh_pre_u41_checkpoint and not authenticated_fresh_pre_u41:
        raise ArtifactError("published PRE scored resume identity drift")
    identity_fields = (
        "recipe_id",
        "published_pre_checkpoint_sha256",
        "lr_scale",
        "seed",
        "controlled_window_schedule_sha256",
        "shared_optimizer_scheduler_lineage",
    )
    if (
        authenticated_fresh_pre_u22
        or authenticated_fresh_pre_u31
        or authenticated_fresh_pre_u33
        or authenticated_fresh_pre_u34
        or authenticated_fresh_pre_u35
        or authenticated_fresh_pre_u36
        or authenticated_fresh_pre_u37
        or authenticated_fresh_pre_u38
        or authenticated_fresh_pre_u41
    ):
        return
    if any(start_meta.get(field) != config.get(field) for field in identity_fields):
        raise ArtifactError("published PRE scored resume identity drift")


def _validate_published_pre_crash_resume_start(
    start: str,
    start_update: int,
    start_meta: Mapping[str, Any],
    *,
    requested: tuple[int, ...],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit the authenticated paired U10 crash checkpoint without replay."""
    u10_sha256 = "055f015f88c44f9092423a7e45525e3699d217d1d0b8b36eb269947915f17658"
    u10_state_sha256 = "7e19879e4b526793c4837a81f4fc3658a00980a2ebf4252b9a23c0d7da9021a6"
    schedule_sha256 = "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351"
    lineage = "fresh-published-pre-adam-lambdalr"
    if requested != (11, 12):
        raise ArtifactError("published PRE U10 crash resume may execute U11,U12 only")
    schedule_matches = (
        start == "SCHEDULE_E186B108124B_UPDATE_010"
        and start_update == 10
        and start_meta.get("next_update") == 10
        and config.get("controlled_window_schedule_sha256") == schedule_sha256
    )
    if not schedule_matches:
        raise ArtifactError("published PRE U10 crash resume schedule identity drift")
    paired_state_matches = (
        start_meta.get("sha256") == u10_sha256
        and config.get("checkpoint_sha256") == u10_sha256
        and start_meta.get("state_sha256") == u10_state_sha256
        and start_meta.get("rank_provenance") == [0, 1]
        and start_meta.get("world_size") == 2
        and start_meta.get("optimizer_steps") == 1
        and start_meta.get("scheduler_steps") == 1
        and start_meta.get("optimizer_scheduler_lineage") == lineage
        and config.get("shared_optimizer_scheduler_lineage") == lineage
    )
    if not paired_state_matches:
        raise ArtifactError("published PRE crash resume requires authenticated paired U10 state")
    if (
        config.get("recipe_id") != "published_pre_lower_lr_warmup16_cosine64_v1"
        or config.get("published_pre_checkpoint_sha256")
            != "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        or config.get("fresh_published_pre_lineage") is not True
    ):
        raise ArtifactError("published PRE U10 crash resume recipe identity drift")
    return {
        "checkpoint_sha256": u10_sha256,
        "next_incomplete_update": 11,
        "rank_provenance": [0, 1],
        "state_sha256": u10_state_sha256,
    }


def _validate_controlled_arm_start(
    arm_id: str,
    start_update: int,
    start_meta: Mapping[str, Any],
    *,
    controlled_config_sha256: str,
) -> None:
    """Fail closed on trajectory switches at scored continuation boundaries."""
    if start_update < 0 or start_update >= 64 or start_update % 4:
        raise ArtifactError("controlled arm start must be a four-update boundary U0..U60")
    from_u0 = arm_id.startswith("from_u0_")
    if not from_u0 and start_update < 16:
        raise ArtifactError("from-U16 controlled arms cannot start before U16")
    origin = 0 if from_u0 else 16
    if start_update == origin:
        return
    if start_meta.get("controlled_arm_id") != arm_id:
        raise ArtifactError("controlled arm scored resume cannot switch recipe identity")
    if start_meta.get("controlled_config_sha256") != controlled_config_sha256:
        raise ArtifactError("controlled arm scored resume cannot switch config identity")


_IDENTITY_ALIASES = {
    "basis_sha256": ("basis_sha256", "basis_sha", "basis"),
    "builder_eval_corpus_sha256": (
        "builder_eval_corpus_sha256",
        "builder_eval_corpus_sha",
        "builder_eval_sha256",
        "builder_corpus_sha256",
        "eval_corpus_sha256",
    ),
    "train_score_corpus_sha256": (
        "train_score_corpus_sha256",
        "train_score_corpus_sha",
        "train_score_sha256",
        "score_corpus_sha256",
        "train_corpus_sha256",
    ),
    "teacher_inventory": (
        "teacher_inventory",
        "teacher_inventory_sha256",
        "teacher_inventory_sha",
        "teacher_manifest",
        "teacher_sha256",
    ),
}


def _real_cpu_copy(value: Any) -> Any:
    """Copy a nested torch state to CPU without changing its structure."""
    if hasattr(value, "detach"):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {key: _real_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_real_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_real_cpu_copy(item) for item in value)
    return value


def _real_cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ResidentRepairAPI:
    """Single high-level path for resident score and experiment contracts."""

    @staticmethod
    def bind_routed_return_accumulation(
        config: Mapping[str, Any], *, provider_expert_sha256: str
    ) -> dict[str, Any]:
        """Bind the accepted provider to the sealed routed-return schedule.

        This is a construction-time API contract: it changes neither provider
        bytes nor route/projection values, and the resident engine consumes the
        returned copy before importing and constructing the provider class.
        """
        accepted_provider = (
            "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"
        )
        if provider_expert_sha256 != accepted_provider:
            raise ArtifactError("sealed routed-return accumulation requires provider 942c3074")
        bound = dict(config)
        bound["resident_routed_return_accumulation"] = (
            "active_row_ascending_expert_cuda_bf16_index_add_v1"
        )
        bound["resident_routed_return_provider_sha256"] = accepted_provider
        return bound

    @staticmethod
    def bind_combined_gate_up_projection(
        config: Mapping[str, Any], *, provider_expert_sha256: str,
        capture_witness: bool = False,
        active_row_expert: int | None = None,
    ) -> dict[str, Any]:
        """Bind the production static provider to sealed native-BF16 projections."""
        accepted_provider = (
            "4ba1411601b186dd0d6a3a89c829320f1b50e3112a40db40034e9fbadfb5d552"
        )
        if provider_expert_sha256 != accepted_provider:
            raise ArtifactError("combined gate/up projection requires provider 4ba14116")
        bound = dict(config)
        bound["resident_gate_up_projection"] = "combined_4096_bf16_f_linear_v1"
        bound["resident_gate_up_provider_sha256"] = accepted_provider
        bound["resident_gate_up_capture_witness"] = bool(capture_witness)
        if active_row_expert is not None:
            if isinstance(active_row_expert, bool) or int(active_row_expert) < 0:
                raise ArtifactError("aligned active-row expert must be a non-negative integer")
            bound["resident_gate_up_active_row_expert"] = int(active_row_expert)
        return bound

    def __init__(self, artifact: RepairArtifact, *, loader=None, official_backend_factory=None):
        self.artifact = artifact
        self.loader = loader or _load_torch
        self._shared_preflight = SharedPreflight(artifact)
        self._last_preflight: dict[str, Any] = {}
        self._official_backend_factory = official_backend_factory
        self._official_backends: dict[tuple[Any, ...], Any] = {}
        self._resident: dict[tuple[str, tuple[int, ...]], Any] = {}
        self._row_metric_resident: dict[tuple[str, tuple[int, ...]], tuple[dict[str, Any], ...]] = {}
        self._teacher_inventory_cache: dict[tuple[int, ...], Mapping[str, Any]] = {}
        self._checkpoint_identity_cache: dict[str, Mapping[str, Any]] = {}
        self._checkpoint_identity_cache_hits = 0
        self._checkpoint_identity_cache_misses = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._row_metric_loads = 0

    @classmethod
    def open(
        cls,
        artifact_root: str | Path,
        *,
        loader=None,
        official_backend_factory=None,
    ) -> "ResidentRepairAPI":
        return cls(
            RepairArtifact.open(artifact_root),
            loader=loader,
            official_backend_factory=official_backend_factory,
        )

    @property
    def windows(self) -> tuple[int, ...]:
        return self.artifact.windows

    @property
    def last_preflight(self) -> Mapping[str, Any]:
        return dict(self._last_preflight)

    def _selected_windows(self, windows: Iterable[int] | None) -> tuple[int, ...]:
        selected = self.windows if windows is None else tuple(int(value) for value in windows)
        if not selected or len(set(selected)) != len(selected):
            raise ArtifactError("windows must be a non-empty unique sequence")
        unknown = sorted(set(selected) - set(self.windows))
        if unknown:
            raise ArtifactError(f"windows are not declared by this artifact: {unknown}")
        return selected

    def _checkpoint_update(self, key: str) -> int:
        meta = self.artifact.manifest["checkpoints"][key]
        value = meta.get("next_update", meta.get("update"))
        if isinstance(value, int) and value >= 0:
            return value
        match = re.search(r"(?:UPDATE_|U)?(\d+)$", key, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        raise ArtifactError(f"checkpoint {key} does not declare a milestone update")

    def _teacher_inventory(self, windows: tuple[int, ...], value: Any) -> Any:
        if value is not None:
            return value
        cached = self._teacher_inventory_cache.get(windows)
        if cached is not None:
            return cached
        score_spec = self.artifact.manifest.get("score", {})
        teacher_dir_value = score_spec.get("teacher_dir") if isinstance(score_spec, Mapping) else None
        if not isinstance(teacher_dir_value, str):
            raise ArtifactError("artifact is missing required scientific identity: teacher_inventory")
        teacher_dir = (self.artifact.root / teacher_dir_value).resolve()
        try:
            teacher_dir.relative_to(self.artifact.root)
        except ValueError as exc:
            raise ArtifactError("score.teacher_dir escapes artifact root") from exc
        entries = []
        for window in windows:
            path = teacher_dir / f"t8192_win{window}.pt"
            if not path.is_file():
                raise ArtifactError(f"teacher inventory is missing window {window}: {path}")
            entries.append({
                "bytes": path.stat().st_size,
                "path": str(path.relative_to(self.artifact.root)),
                "window": window,
            })
        encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
        inventory = {
            "schema": "teacher-file-inventory-v1",
            "file_count": len(entries),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "windows": list(windows),
        }
        self._teacher_inventory_cache[windows] = inventory
        return inventory

    def _checkpoint_identity_payload(self, checkpoint: str) -> Mapping[str, Any]:
        """Read checkpoint identity only when the manifest omits lineage fields."""
        if self.loader is not _load_torch:
            return {}
        cached = self._checkpoint_identity_cache.get(checkpoint)
        if cached is not None:
            self._checkpoint_identity_cache_hits += 1
            return cached
        self._checkpoint_identity_cache_misses += 1
        payload = _load_torch(self.artifact.checkpoint_path(checkpoint))
        identity = payload.get("identity") if isinstance(payload, Mapping) else None
        value = identity if isinstance(identity, Mapping) else {}
        self._checkpoint_identity_cache[checkpoint] = value
        return value

    def _checkpoint_parent_sha(self, checkpoint: str) -> Any:
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        for field in ("parent_sha256", "parent_checkpoint_sha256"):
            if meta.get(field):
                return meta[field]
        if self._checkpoint_update(checkpoint) == 0:
            return None
        identity = self._checkpoint_identity_payload(checkpoint)
        for field in ("parent_checkpoint_sha256", "continuous_parent_checkpoint_sha256", "input_checkpoint_sha256"):
            if identity.get(field):
                return identity[field]
        return None

    def _checkpoint_parent_identity_sha(self, checkpoint: str) -> Any:
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        for field in ("parent_identity_sha256", "parent_checkpoint_identity_sha256"):
            if meta.get(field):
                return meta[field]
        if self._checkpoint_update(checkpoint) == 0:
            return None
        identity = self._checkpoint_identity_payload(checkpoint)
        for field in ("parent_identity_sha256", "continuous_parent_identity_sha256", "input_checkpoint_identity_sha256"):
            if identity.get(field):
                return identity[field]
        return None

    def _identity(self, checkpoint: str, windows: tuple[int, ...]) -> dict[str, Any]:
        manifest_identity = self.artifact.manifest.get("identity", {})
        if not isinstance(manifest_identity, Mapping):
            manifest_identity = {}
        identity: dict[str, Any] = {}
        for output, aliases in _IDENTITY_ALIASES.items():
            value = None
            for source in (manifest_identity, self.artifact.manifest):
                for alias in aliases:
                    if alias in source:
                        value = source[alias]
                        break
                if value is not None:
                    break
            identity[output] = self._teacher_inventory(windows, value) if output == "teacher_inventory" else value
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        identity.update(
            {
                "ordered_balanced64_windows": list(windows),
                "support": 8192,
                "kl_direction": "KL(teacher||candidate)",
                "reduction": "binary64/math.fsum",
                "checkpoint": checkpoint,
                "checkpoint_sha256": meta.get("sha256"),
                "checkpoint_parent_sha256": self._checkpoint_parent_sha(checkpoint),
                "checkpoint_identity_sha256": meta.get("identity_sha256"),
                "checkpoint_next_update": self._checkpoint_update(checkpoint),
            }
        )
        return identity

    def _validate_scientific_identity(self, checkpoint: str, windows: tuple[int, ...]) -> None:
        identity = self._identity(checkpoint, windows)
        for field in (
            "basis_sha256",
            "builder_eval_corpus_sha256",
            "train_score_corpus_sha256",
            "teacher_inventory",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
        ):
            value = identity[field]
            if value is None or value == "" or value == []:
                raise ArtifactError(f"artifact is missing required scientific identity: {field}")
        if identity["checkpoint_next_update"] > 0 and not identity["checkpoint_parent_sha256"]:
            raise ArtifactError("non-initial checkpoint is missing required parent SHA")
        self._checkpoint_update(checkpoint)

    def _resident_for(self, checkpoint: str, windows: tuple[int, ...]):
        cache_key = (checkpoint, windows)
        resident = self._resident.get(cache_key)
        if resident is not None:
            self._cache_hits += 1
            return resident
        self._cache_misses += 1
        resident = self.artifact.load_resident(checkpoint, windows=windows, loader=self.loader)
        self._resident[cache_key] = resident
        return resident

    def _row_metrics_for(self, checkpoint: str, windows: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
        key = (checkpoint, windows)
        cached = self._row_metric_resident.get(key)
        if cached is not None:
            return cached
        spec = self.artifact.manifest.get("score", {})
        table = spec.get("row_metrics", {})
        rel = table.get(checkpoint) if isinstance(table, Mapping) else None
        if not isinstance(rel, str) or not rel:
            raise ArtifactError(f"artifact has no resident row metrics for {checkpoint}")
        path = (self.artifact.root / rel).resolve()
        try:
            path.relative_to(self.artifact.root.resolve())
            if not path.is_file():
                raise ArtifactError(f"resident row metrics file is missing: {path}")
            value = json.loads(path.read_text())
        except ArtifactError:
            raise
        except ValueError as exc:
            raise ArtifactError(f"resident row metrics path escapes artifact root: {rel}") from exc
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot load resident row metrics: {path}: {exc}") from exc
        rows = value.get("rows") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise ArtifactError(f"resident row metrics must contain rows: {path}")
        by_window = {int(row.get("window")): row for row in rows if isinstance(row, Mapping)}
        selected: list[dict[str, Any]] = []
        for window in windows:
            row = by_window.get(window)
            if row is None or int(row.get("positions", 0)) != 1024:
                raise ArtifactError(f"resident row metrics missing complete window {window}: {path}")
            if "kld_sum" not in row or "top1" not in row:
                raise ArtifactError(f"resident row metrics missing KLD/Top-1 fields: {path}")
            selected.append(dict(row))
        result = tuple(selected)
        self._row_metric_resident[key] = result
        self._row_metric_loads += 1
        return result

    def _score_row_metrics(self, checkpoint: str, windows: tuple[int, ...]) -> ScoreResult:
        started = __import__("time").perf_counter()
        rows = self._row_metrics_for(checkpoint, windows)
        kld = __import__("math").fsum(float(row["kld_sum"]) for row in rows) / (len(rows) * 1024)
        top1 = sum(int(row["top1"]) for row in rows)
        return ScoreResult(
            checkpoint=checkpoint,
            windows=windows,
            positions=len(rows) * 1024,
            support=8192,
            kld=kld,
            top1=top1,
            top1_rate=top1 / (len(rows) * 1024),
            artifact_root=str(self.artifact.root),
            spec="balanced64-v1",
            candidate_dir="resident-row-metrics",
            execution_mode="resident_in_memory",
            resident_load_seconds=0.0,
            timed_wall_seconds=__import__("time").perf_counter() - started,
            identity=self._identity(checkpoint, windows),
            runtime_counters={
                "resident_cache_hits": 0,
                "resident_cache_misses": 0,
                "resident_row_metric_loads": self._row_metric_loads,
                "file_reads_during_timed_score": 0,
                "timed_score_execution": "in_memory",
            },
        )

    @staticmethod
    def _canonical_state_bytes(value: Any) -> bytes:
        """Serialize nested model/optimizer state without lossy JSON coercion."""
        if hasattr(value, "detach"):
            import torch
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return b"tensor:" + str(tensor.dtype).encode() + b":" + repr(tuple(tensor.shape)).encode() + b":" + raw
        if isinstance(value, Mapping):
            parts = []
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0])):
                encoded_key = ResidentRepairAPI._canonical_state_bytes(key)
                encoded_item = ResidentRepairAPI._canonical_state_bytes(item)
                parts.append(len(encoded_key).to_bytes(8, "big") + encoded_key + len(encoded_item).to_bytes(8, "big") + encoded_item)
            return b"mapping:" + b"".join(parts)
        if isinstance(value, (list, tuple)):
            return (b"list:" if isinstance(value, list) else b"tuple:") + b"".join(
                len(encoded).to_bytes(8, "big") + encoded
                for encoded in (ResidentRepairAPI._canonical_state_bytes(item) for item in value)
            )
        if value is None:
            return b"none"
        if isinstance(value, bool):
            return b"bool:" + repr(value).encode()
        if isinstance(value, (int, float, str, bytes)):
            return (type(value).__name__ + ":" + repr(value)).encode()
        raise ArtifactError(f"replay state contains unsupported value type: {type(value).__name__}")

    @staticmethod
    def _state_fingerprint(payload: Mapping[str, Any]) -> str:
        """Hash trainable state values with dtype/shape and stable key order."""
        state = payload.get("state") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise ArtifactError("replay checkpoint is missing mapping state")
        return hashlib.sha256(ResidentRepairAPI._canonical_state_bytes(state)).hexdigest()

    @staticmethod
    def _write_immutable_receipt(path: str | Path, value: Mapping[str, Any]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ArtifactError(f"immutable receipt already exists with different bytes: {destination}")
            return destination
        temporary = destination.with_name(f".{destination.name}.{id(value)}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        if destination.read_bytes() != payload:
            raise ArtifactError(f"immutable receipt readback mismatch: {destination}")
        return destination

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        """Install bytes durably, without exposing a partial checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _atomic_json(cls, path: Path, value: Mapping[str, Any]) -> None:
        payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        cls._atomic_bytes(path, payload)

    @staticmethod
    def _identity_sha256(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(dict(value), separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _preflight_persisted_checkpoint(
        self,
        path: Path,
        *,
        expected_sha: str,
        target_update: int,
        identity_sha: str,
    ) -> Mapping[str, Any]:
        """Check the exact readback path used by materialize_candidates."""
        if not path.is_file() or path.stat().st_size <= 0:
            raise ArtifactError(f"checkpoint U{target_update} was not persisted as a non-empty file: {path}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ArtifactError(f"checkpoint U{target_update} file SHA readback mismatch")
        try:
            payload = _load_torch(path)
        except Exception as exc:
            raise ArtifactError(f"checkpoint U{target_update} is not readable by materialize_candidates: {path}: {exc}") from exc
        state = payload.get("state") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping) or not state:
            raise ArtifactError(f"checkpoint U{target_update} has no readable state mapping")
        payload_identity = payload.get("identity")
        if not isinstance(payload_identity, Mapping) or payload_identity.get("checkpoint_loaded") is not True:
            raise ArtifactError(f"checkpoint U{target_update} lacks loaded checkpoint identity")
        if payload_identity.get("identity_sha256") != identity_sha:
            raise ArtifactError(f"checkpoint U{target_update} identity SHA readback mismatch")
        if payload_identity.get("next_update") != target_update:
            raise ArtifactError(f"checkpoint U{target_update} next_update readback mismatch")
        return payload

    @staticmethod
    def _continuation_checkpoint_key(
        target_update: int, config: Mapping[str, Any]
    ) -> str:
        """Namespace fresh controlled-schedule checkpoints away from warm-root history."""
        schedule_sha = config.get("controlled_window_schedule_sha256")
        if (
            config.get("fresh_published_pre_lineage") is True
            and isinstance(schedule_sha, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", schedule_sha)
        ):
            return f"SCHEDULE_{schedule_sha[:12].upper()}_UPDATE_{target_update:03d}"
        return f"UPDATE_{target_update:03d}"

    def _persist_continuation_checkpoint(
        self,
        target_update: int,
        state: Mapping[str, Any],
        step_report: Mapping[str, Any],
        *,
        parent_sha: str,
        parent_identity_sha: str | None,
        lineage: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one continuation state and publish its manifest entry."""
        declared_root = Path(config.get("artifact_root", self.artifact.root)).expanduser().resolve()
        if declared_root != self.artifact.root.resolve():
            raise ArtifactError("continuation artifact_root must be the opened materialize_candidates root")
        checkpoint_key = self._continuation_checkpoint_key(target_update, config)
        relative_path = Path("checkpoints") / f"{checkpoint_key}.pt"
        checkpoint_path = self.artifact.root / relative_path
        basis_sha = config.get("basis_sha256")
        if not isinstance(basis_sha, str) or not basis_sha:
            raise ArtifactError("continuation persistence requires the validated basis SHA")
        state_sha = self._state_fingerprint({"state": state})
        optimizer_state = step_report.get("optimizer_state", step_report.get("optimizer"))
        scheduler_state = step_report.get("scheduler_state", step_report.get("scheduler"))
        if optimizer_state is None:
            optimizer_state = {"steps": step_report["optimizer_steps"]}
        if scheduler_state is None:
            scheduler_state = {"steps": step_report["scheduler_steps"]}
        identity = {
            "schema": "resident-continuation-checkpoint-identity-v1",
            "basis_sha256": basis_sha,
            "checkpoint": checkpoint_key,
            "next_update": target_update,
            "parent_checkpoint_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "state_sha256": state_sha,
            "optimizer_scheduler_lineage": lineage,
            "checkpoint_loaded": True,
            "world_size": int(config["world_size"]),
        }
        controlled_arm_id = config.get("controlled_arm_id")
        if controlled_arm_id is not None:
            identity.update({
                "controlled_arm_id": controlled_arm_id,
                "controlled_window_schedule_sha256": config.get("controlled_window_schedule_sha256"),
                "controlled_config_sha256": config.get("controlled_config_sha256"),
            })
        identity_sha = self._identity_sha256(identity)
        identity = {**identity, "identity_sha256": identity_sha}
        payload = {
            "schema": "resident-continuation-checkpoint-v1",
            "next_update": target_update,
            "state": dict(state),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "optimizer_scheduler_delta": {
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
            },
            "identity": identity,
            "controlled_arm_id": controlled_arm_id,
        }
        try:
            import torch
            stream = io.BytesIO()
            torch.save(payload, stream)
            checkpoint_bytes = stream.getvalue()
        except Exception as exc:
            raise ArtifactError(f"cannot serialize U{target_update} checkpoint: {exc}") from exc
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        if checkpoint_path.exists():
            # torch's zip serialization may vary its container bytes between
            # processes; the existing sealed file is authoritative after its
            # semantic identity and state readback pass.
            existing_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            self._preflight_persisted_checkpoint(
                checkpoint_path,
                expected_sha=existing_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
            checkpoint_sha = existing_sha
        else:
            self._atomic_bytes(checkpoint_path, checkpoint_bytes)
            self._preflight_persisted_checkpoint(
                checkpoint_path,
                expected_sha=checkpoint_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
        manifest = dict(self.artifact.manifest)
        checkpoints = dict(manifest.get("checkpoints", {}))
        existing = checkpoints.get(checkpoint_key)
        if existing is not None:
            for field, expected in (
                ("path", str(relative_path)),
                ("sha256", checkpoint_sha),
                ("parent_sha256", parent_sha),
                ("identity_sha256", identity_sha),
                ("next_update", target_update),
            ):
                if existing.get(field) != expected:
                    raise ArtifactError(f"manifest U{target_update} conflicts with persisted checkpoint: {field}")
        entry = {
            "path": str(relative_path),
            "sha256": checkpoint_sha,
            "identity_sha256": identity_sha,
            "parent_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "next_update": target_update,
            "checkpoint_loaded": True,
            "fixture": False,
            "optimizer_scheduler_lineage": lineage,
            "optimizer_steps": step_report["optimizer_steps"],
            "scheduler_steps": step_report["scheduler_steps"],
            "world_size": int(config["world_size"]),
            "rank": config.get("rank"),
            "state_sha256": state_sha,
            "artifact_root": str(self.artifact.root),
        }
        if controlled_arm_id is not None:
            entry.update({
                "controlled_arm_id": controlled_arm_id,
                "controlled_window_schedule_sha256": config.get("controlled_window_schedule_sha256"),
                "controlled_config_sha256": config.get("controlled_config_sha256"),
            })
        prior_ranks = existing.get("rank_provenance", []) if isinstance(existing, Mapping) else []
        if not isinstance(prior_ranks, list):
            prior_ranks = []
        reported_ranks = step_report.get("rank_provenance", [])
        if not isinstance(reported_ranks, list):
            reported_ranks = []
        entry["rank_provenance"] = sorted(
            set(int(value) for value in prior_ranks + reported_ranks + [config["rank"]])
        )
        checkpoints[checkpoint_key] = entry
        manifest["checkpoints"] = checkpoints
        self._atomic_json(self.artifact.root / "ARTIFACT.json", manifest)
        try:
            reopened = RepairArtifact.open(self.artifact.root)
            reopened_path = reopened.checkpoint_path(checkpoint_key)
            self._preflight_persisted_checkpoint(
                reopened_path,
                expected_sha=checkpoint_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
        except Exception as exc:
            raise ArtifactError(f"durable U{target_update} manifest/file readback failed: {exc}") from exc
        self.artifact = reopened
        return {
            "checkpoint": checkpoint_key,
            "checkpoint_path": str(relative_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": identity_sha,
            "state_sha256": state_sha,
            "parent_checkpoint_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "next_update": target_update,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "optimizer_steps": step_report["optimizer_steps"],
            "scheduler_steps": step_report["scheduler_steps"],
            "world_size": int(config["world_size"]),
            "rank": config["rank"],
            "artifact_root": str(self.artifact.root),
        }

    def construct_from_clean_u0(
        self,
        midpoint: int | str,
        target: int | str,
        *,
        replay: Mapping[str, Any] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Construct U16 from clean U0 using only in-memory factories and updates.

        This is deliberately not a process launcher.  ``replay`` must provide
        model/optimizer/scheduler factories plus one update callback.  A
        serialized checkpoint, command, environment, or output path is never
        accepted as a substitute for the true replay.
        """
        start = self.artifact.checkpoint_key(midpoint)
        end = self.artifact.checkpoint_key(target)
        if self._checkpoint_update(start) != 0:
            raise ArtifactError("clean-U0 constructor requires a zero-update midpoint")
        self._assert_shared_identity(start, end)
        start_sha = self.artifact.manifest["checkpoints"][start].get("sha256")
        parent_sha = self._checkpoint_parent_sha(end)
        target_update = self._checkpoint_update(end)
        if target_update - self._checkpoint_update(start) != 16:
            raise ArtifactError("clean-U0 constructor requires exactly 16 optimizer updates")
        if not start_sha or not parent_sha or parent_sha != start_sha:
            raise ArtifactError("clean-U0 constructor target is not bound to the midpoint checkpoint SHA")
        if not isinstance(replay, Mapping):
            raise ArtifactError("true clean-U0 construction requires a replay specification")
        forbidden = {"command", "cwd", "env", "output_checkpoint", "checkpoint_path", "state_loader", "load_checkpoint"}
        present_forbidden = sorted(key for key in forbidden if key in replay)
        if present_forbidden:
            raise ArtifactError(f"raw command/checkpoint substitute is forbidden: {present_forbidden}")
        required = ("model_factory", "optimizer_factory", "scheduler_factory", "update_fn", "geometry", "basis_sha256", "corpus_sha256", "seed")
        missing = [key for key in required if key not in replay]
        if missing:
            raise ArtifactError(f"true clean-U0 replay inputs are absent: {', '.join(missing)}")
        if not all(callable(replay[key]) for key in ("model_factory", "optimizer_factory", "scheduler_factory", "update_fn")):
            raise ArtifactError("model_factory, optimizer_factory, scheduler_factory, and update_fn must be callable")
        if not isinstance(replay["geometry"], Mapping) or not replay["geometry"]:
            raise ArtifactError("replay.geometry must be a non-empty mapping")
        if replay["basis_sha256"] != self._identity(start, self.windows)["basis_sha256"]:
            raise ArtifactError("clean-U0 replay basis SHA does not match artifact identity")
        if replay["corpus_sha256"] not in (self._identity(start, self.windows)["builder_eval_corpus_sha256"], self._identity(start, self.windows)["train_score_corpus_sha256"]):
            raise ArtifactError("clean-U0 replay corpus SHA does not match artifact identity")
        if not isinstance(replay["seed"], int) or isinstance(replay["seed"], bool):
            raise ArtifactError("clean-U0 replay seed must be an integer")
        destination = receipt_path or replay.get("receipt_path")
        if destination is None:
            raise ArtifactError("clean-U0 replay requires a durable receipt_path")
        random.seed(replay["seed"])
        try:
            import torch
            torch.manual_seed(replay["seed"])
        except ImportError:
            pass
        model = replay["model_factory"]()
        optimizer = replay["optimizer_factory"](model)
        scheduler = replay["scheduler_factory"](optimizer)
        if any(bool(getattr(value, "checkpoint_loaded", False)) for value in (model, optimizer, scheduler)):
            raise ArtifactError("clean-U0 replay factories reported checkpoint_loaded=True")
        executed = 0
        for update in range(1, target_update + 1):
            outcome = replay["update_fn"](model, optimizer, scheduler, update)
            if isinstance(outcome, Mapping) and outcome.get("checkpoint_loaded"):
                raise ArtifactError("clean-U0 update callback reported checkpoint_loaded=True")
            executed += 1
        if executed != 16:
            raise ArtifactError(f"clean-U0 replay executed {executed} updates, expected 16")
        for name, value in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
            if not callable(getattr(value, "state_dict", None)):
                raise ArtifactError(f"{name} must expose state_dict() for replay authentication")
        final_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if "target_state_sha256" in replay or "expected_target_state" in replay:
            raise ArtifactError("clean-U0 replay cannot use a declared target-state substitute")
        # Authenticate against the sealed target checkpoint itself. A caller
        # supplied SHA or expected state would only prove a fixture matched its
        # own declaration, not that clean U0 reached the Modern Green U16.
        try:
            target_payload = _load_torch(self.artifact.checkpoint_path(end))
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot load clean-U0 target checkpoint: {exc}") from exc
        target_state = target_payload.get("state")
        if not isinstance(target_state, Mapping):
            raise ArtifactError("clean-U0 target checkpoint state is not a mapping")
        # Native Modern Green checkpoints store the trainable model surfaces
        # directly under ``state`` (luts/norms/outputs), while the small API
        # fixtures store a composite model/optimizer/scheduler state.  Bind
        # either representation without weakening the target-state
        # authentication: the replay model must still exactly match the
        # checkpoint's declared state shape and bytes.
        model_state = final_state["model"]
        if set(target_state) >= {"model", "optimizer", "scheduler"}:
            authenticated_state = final_state
            target_state_scope = "composite_model_optimizer_scheduler"
        elif isinstance(model_state, Mapping) and set(target_state) == set(model_state):
            authenticated_state = model_state
            target_state_scope = "native_model_surfaces"
        else:
            raise ArtifactError("clean-U0 replay state shape does not match target checkpoint")
        state_sha = self._state_fingerprint({"state": authenticated_state})
        target_state_sha = self._state_fingerprint({"state": target_state})
        if state_sha != target_state_sha:
            raise ArtifactError("clean-U0 replay state fingerprint does not match target")
        result = {
            "status": "PASS",
            "construction": "true_clean_u0_optimizer_replay",
            "midpoint": start,
            "target": end,
            "midpoint_sha256": start_sha,
            "target_sha256": self.artifact.manifest["checkpoints"][end].get("sha256"),
            "parent_checkpoint_sha256": parent_sha,
            "target_state_sha256": target_state_sha,
            "state_sha256": state_sha,
            "authenticated_state_scope": target_state_scope,
            "update_count": executed,
            "updates": {"requested": 16, "executed": executed},
            "checkpoint_loaded": False,
            "optimizer_scheduler_identity": {
                "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "scheduler": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
                "lineage": "clean_u0",
            },
            "geometry": dict(replay["geometry"]),
            "basis_sha256": replay["basis_sha256"],
            "corpus_sha256": replay["corpus_sha256"],
            "seed": replay["seed"],
        }
        self._write_immutable_receipt(destination, result)
        return result

    # Public name used by new experiment cards; the legacy name remains valid.
    construct_clean_u0 = construct_from_clean_u0

    def continuous_four_updates(
        self,
        start_checkpoint: int | str,
        *,
        replay: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Run the exact U0->U4 arm in one resident model/Adam/scheduler.

        The API loads canonical U0 once, constructs each resident object once,
        invokes exactly four update callbacks, and scores that same resident
        model at U0/U2/U4.  It has no serialization or reload boundary.
        """
        start = self.artifact.checkpoint_key(start_checkpoint)
        if self._checkpoint_update(start) != 0:
            raise ArtifactError("continuous four-update arm requires canonical U0")
        if not isinstance(replay, Mapping):
            raise ArtifactError("continuous four-update arm requires a replay specification")
        required = (
            "model_factory", "optimizer_factory", "scheduler_factory", "update_fn",
            "resident_score_fn", "state_fingerprint_fn", "release_fn", "geometry", "basis_sha256",
            "corpus_sha256", "seed",
        )
        missing = [key for key in required if key not in replay]
        if missing:
            raise ArtifactError("continuous four-update inputs are absent: " + ", ".join(missing))
        callbacks = (
            "model_factory", "optimizer_factory", "scheduler_factory", "update_fn",
            "resident_score_fn", "state_fingerprint_fn", "release_fn",
        )
        if not all(callable(replay[key]) for key in callbacks):
            raise ArtifactError("continuous four-update factories/callbacks must be callable")
        forbidden = {
            "command", "cwd", "env", "output_checkpoint", "checkpoint_path",
            "state_loader", "load_checkpoint", "advance_fn", "standalone_scorer",
            "save_fn", "reload_fn",
        }
        present_forbidden = sorted(key for key in forbidden if key in replay)
        if present_forbidden:
            raise ArtifactError(f"raw runner/checkpoint/scorer substitute is forbidden: {present_forbidden}")
        identity = self._identity(start, self.windows)
        if replay["basis_sha256"] != identity["basis_sha256"]:
            raise ArtifactError("continuous four-update basis SHA does not match artifact identity")
        if replay["corpus_sha256"] not in (
            identity["builder_eval_corpus_sha256"], identity["train_score_corpus_sha256"],
        ):
            raise ArtifactError("continuous four-update corpus SHA does not match artifact identity")
        if not isinstance(replay["seed"], int) or isinstance(replay["seed"], bool):
            raise ArtifactError("continuous four-update seed must be an integer")
        if not isinstance(replay["geometry"], Mapping) or not replay["geometry"]:
            raise ArtifactError("continuous four-update geometry must be a non-empty mapping")

        try:
            start_payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load canonical U0 checkpoint: {exc}") from exc
        start_state = start_payload.get("state") if isinstance(start_payload, Mapping) else None
        if not isinstance(start_state, Mapping):
            raise ArtifactError("canonical U0 checkpoint state is not a mapping")
        start_optimizer = start_payload.get("optimizer_state", start_payload.get("optimizer"))
        start_scheduler = start_payload.get("scheduler_state", start_payload.get("scheduler"))
        if not isinstance(start_optimizer, Mapping) or not isinstance(start_scheduler, Mapping):
            raise ArtifactError("canonical U0 must contain admitted Adam and scheduler mappings")

        random.seed(replay["seed"])
        try:
            import torch
            torch.manual_seed(replay["seed"])
        except ImportError:
            pass
        model = replay["model_factory"](start_payload)
        optimizer = replay["optimizer_factory"](model)
        scheduler = replay["scheduler_factory"](optimizer)
        release_calls = 0
        milestones: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        def load_initial_state() -> None:
            for name, value in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
                if bool(getattr(value, "checkpoint_loaded", False)):
                    raise ArtifactError(f"continuous four-update {name} factory reported checkpoint_loaded=True")
                if not callable(getattr(value, "state_dict", None)):
                    raise ArtifactError(f"continuous four-update {name} must expose state_dict()")
            model_loader = getattr(model, "load_state_dict", None)
            if not callable(model_loader):
                raise ArtifactError("continuous four-update model must expose load_state_dict()")
            model_loader(copy.deepcopy(start_state))
            optimizer_loader = getattr(optimizer, "load_state_dict", None)
            scheduler_loader = getattr(scheduler, "load_state_dict", None)
            if not callable(optimizer_loader) or not callable(scheduler_loader):
                raise ArtifactError("continuous Adam and scheduler must expose load_state_dict()")
            optimizer_loader(copy.deepcopy(start_optimizer))
            scheduler_loader(copy.deepcopy(start_scheduler))
            ready = getattr(model, "resident_ready", True)
            ready = ready() if callable(ready) else ready
            if ready is not True:
                raise ArtifactError("continuous four-update model is not resident_ready")

        def score_milestone(update: int) -> None:
            measured = replay["resident_score_fn"](model, update, self.windows)
            if not isinstance(measured, Mapping):
                raise ArtifactError("continuous resident score callback returned a non-mapping")
            fingerprints = replay["state_fingerprint_fn"](model, optimizer, scheduler, update)
            if (
                not isinstance(fingerprints, Mapping)
                or fingerprints.get("scope") != "global_two_rank"
                or any(not isinstance(fingerprints.get(name), str) or not fingerprints[name]
                       for name in ("model", "optimizer", "scheduler"))
            ):
                raise ArtifactError("continuous state fingerprints must bind the global two-rank trajectory")
            counters = measured.get("runtime_counters")
            if not isinstance(counters, Mapping):
                raise ArtifactError("continuous resident score lacks runtime counters")
            forbidden_runtime = (
                "timed_model_payload_reads", "timed_score_file_reads", "fallback_calls",
                "reconstruction_calls", "cpu_relay_bytes",
            )
            if (
                measured.get("execution_mode") != "resident_in_memory"
                or measured.get("positions") != 64 * 1024
                or measured.get("support") != 8192
                or tuple(measured.get("windows", ())) != tuple(self.windows)
                or len(self.windows) != 64
                or any(int(counters.get(key, -1)) != 0 for key in forbidden_runtime)
                or not isinstance(counters.get("resident_ready"), list)
                or len(counters["resident_ready"]) != 2
            ):
                raise ArtifactError("continuous score failed resident 64x1024/8192/zero-read closure")
            milestones.append({
                "update": update,
                "model_fingerprint": fingerprints["model"],
                "optimizer_fingerprint": fingerprints["optimizer"],
                "scheduler_fingerprint": fingerprints["scheduler"],
                "fingerprint_scope": fingerprints["scope"],
                "optimizer_steps": update,
                "scheduler_steps": update,
                "score": dict(measured),
            })

        try:
            load_initial_state()
            score_milestone(0)
            for update in range(1, 5):
                outcome = replay["update_fn"](model, optimizer, scheduler, update)
                if not isinstance(outcome, Mapping):
                    outcome = {}
                if outcome.get("checkpoint_loaded"):
                    raise ArtifactError("continuous update callback reported checkpoint_loaded=True")
                if outcome.get("optimizer_steps", 1) != 1 or outcome.get("scheduler_steps", 1) != 1:
                    raise ArtifactError("continuous arm requires one optimizer/scheduler step per update")
                update_rows.append({
                    "update": update,
                    "loss": outcome.get("loss"),
                    "timings": dict(outcome.get("timings", {})),
                })
                if update in (2, 4):
                    score_milestone(update)
        finally:
            replay["release_fn"](model, optimizer, scheduler)
            release_calls += 1

        result = {
            "schema": "resident-api-continuous-four-updates-v1",
            "status": "PASS",
            "public_api": {
                "method": "ResidentRepairAPI.continuous_four_updates",
                "version": "resident-api-continuous-four-updates-v1",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "start_checkpoint": start,
            "start_checkpoint_sha256": self.artifact.manifest["checkpoints"][start].get("sha256"),
            "start_identity_sha256": self.artifact.manifest["checkpoints"][start].get("identity_sha256"),
            "basis_sha256": identity["basis_sha256"],
            "geometry": dict(replay["geometry"]),
            "updates": update_rows,
            "milestones": milestones,
            "runtime_counters": {
                "input_checkpoint_loads": 1,
                "checkpoint_saves": 0,
                "checkpoint_reloads": 0,
                "update_callbacks": len(update_rows),
                "resident_scores": len(milestones),
                "release_calls": release_calls,
            },
        }
        self._write_immutable_receipt(receipt_path, result)
        return result

    def resume_equivalence(
        self,
        start_checkpoint: int | str,
        *,
        replay: Mapping[str, Any],
        total_updates: int = 4,
        midpoint_update: int = 2,
        midpoint_checkpoint_path: str | Path | None = None,
        score_checkpoints: tuple[int | str, int | str] | None = None,
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Compare uninterrupted and API-reloaded in-memory training.

        This is the public canary surface for the concrete U0->U4 question:
        Arm A performs all updates in one resident process; Arm B performs
        ``midpoint_update`` updates, serializes model/optimizer/scheduler once
        at that checkpoint boundary, reloads all three through this API, and
        performs the remaining updates.  The callback is the training update
        primitive, not a runner or scorer, and cannot report a checkpoint load.
        """
        start = self.artifact.checkpoint_key(start_checkpoint)
        if self._checkpoint_update(start) != 0:
            raise ArtifactError("resume equivalence requires the canonical zero-update checkpoint")
        if not isinstance(total_updates, int) or isinstance(total_updates, bool) or total_updates < 2:
            raise ArtifactError("resume equivalence requires at least two updates")
        if not isinstance(midpoint_update, int) or isinstance(midpoint_update, bool):
            raise ArtifactError("midpoint_update must be an integer")
        if midpoint_update <= 0 or midpoint_update >= total_updates:
            raise ArtifactError("midpoint_update must be strictly inside the update interval")
        if total_updates - midpoint_update < 2:
            raise ArtifactError("resume equivalence requires at least two post-reload updates")
        if not isinstance(replay, Mapping):
            raise ArtifactError("resume equivalence requires a replay specification")
        required = (
            "model_factory", "optimizer_factory", "scheduler_factory", "update_fn",
            "geometry", "basis_sha256", "corpus_sha256", "seed",
        )
        missing = [key for key in required if key not in replay]
        if missing:
            raise ArtifactError("resume equivalence inputs are absent: " + ", ".join(missing))
        if not all(callable(replay[key]) for key in ("model_factory", "optimizer_factory", "scheduler_factory", "update_fn")):
            raise ArtifactError("resume equivalence factories and update_fn must be callable")
        if "release_fn" in replay and not callable(replay["release_fn"]):
            raise ArtifactError("resume equivalence release_fn must be callable")
        forbidden = {
            "command", "cwd", "env", "output_checkpoint", "checkpoint_path",
            "state_loader", "load_checkpoint", "advance_fn", "standalone_scorer",
        }
        present_forbidden = sorted(key for key in forbidden if key in replay)
        if present_forbidden:
            raise ArtifactError(f"raw runner/checkpoint/scorer substitute is forbidden: {present_forbidden}")
        if replay["basis_sha256"] != self._identity(start, self.windows)["basis_sha256"]:
            raise ArtifactError("resume equivalence basis SHA does not match artifact identity")
        if replay["corpus_sha256"] not in (
            self._identity(start, self.windows)["builder_eval_corpus_sha256"],
            self._identity(start, self.windows)["train_score_corpus_sha256"],
        ):
            raise ArtifactError("resume equivalence corpus SHA does not match artifact identity")
        if not isinstance(replay["seed"], int) or isinstance(replay["seed"], bool):
            raise ArtifactError("resume equivalence seed must be an integer")

        # The only input read is the canonical U0 payload.  Model/optimizer/
        # scheduler planes then remain resident; the midpoint boundary below
        # is the sole deliberate serialization/reload operation.
        try:
            start_payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load canonical U0 checkpoint: {exc}") from exc
        start_state = start_payload.get("state")
        if not isinstance(start_state, Mapping):
            raise ArtifactError("canonical U0 checkpoint state is not a mapping")
        start_optimizer = start_payload.get("optimizer_state", start_payload.get("optimizer"))
        start_scheduler = start_payload.get("scheduler_state", start_payload.get("scheduler"))

        import copy
        import io
        import random
        random.seed(replay["seed"])
        try:
            import torch
        except ImportError:
            torch = None

        def reset_rng() -> None:
            random.seed(replay["seed"])
            if torch is not None:
                torch.manual_seed(replay["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(replay["seed"])

        def instantiate():
            model = replay["model_factory"]()
            if bool(getattr(model, "checkpoint_loaded", False)):
                raise ArtifactError("resume equivalence model factory reported checkpoint_loaded=True")
            loader = getattr(model, "load_state_dict", None)
            if callable(loader):
                loader(copy.deepcopy(start_state))
            elif start_state:
                raise ArtifactError("resume equivalence model must expose load_state_dict()")
            optimizer = replay["optimizer_factory"](model)
            scheduler = replay["scheduler_factory"](optimizer)
            if any(bool(getattr(value, "checkpoint_loaded", False)) for value in (optimizer, scheduler)):
                raise ArtifactError("resume equivalence optimizer/scheduler reported checkpoint_loaded=True")
            if (
                isinstance(start_optimizer, Mapping)
                and start_optimizer.get("param_groups")
                and callable(getattr(optimizer, "load_state_dict", None))
            ):
                optimizer.load_state_dict(copy.deepcopy(start_optimizer))
            if (
                isinstance(start_scheduler, Mapping)
                and start_scheduler
                and callable(getattr(scheduler, "load_state_dict", None))
            ):
                scheduler.load_state_dict(copy.deepcopy(start_scheduler))
            for name, value in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
                if not callable(getattr(value, "state_dict", None)):
                    raise ArtifactError(f"resume equivalence {name} must expose state_dict()")
            # This is the API-owned residency gate. Implementations may expose
            # a stronger resident_ready() hook; otherwise successful state
            # materialization is the minimum resident contract.
            ready = getattr(model, "resident_ready", True)
            ready = ready() if callable(ready) else ready
            if ready is not True:
                raise ArtifactError("resume equivalence failed to re-establish resident_ready")
            return model, optimizer, scheduler

        def fingerprint(model, optimizer, scheduler) -> str:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            return hashlib.sha256(self._canonical_state_bytes(payload)).hexdigest()

        def step(arm: str, model, optimizer, scheduler, update: int, rows: list[dict[str, Any]]):
            outcome = replay["update_fn"](model, optimizer, scheduler, update)
            if isinstance(outcome, Mapping) and outcome.get("checkpoint_loaded"):
                raise ArtifactError("resume equivalence update callback reported checkpoint_loaded=True")
            optimizer_steps = outcome.get("optimizer_steps", 1) if isinstance(outcome, Mapping) else 1
            scheduler_steps = outcome.get("scheduler_steps", 1) if isinstance(outcome, Mapping) else 1
            if optimizer_steps != 1 or scheduler_steps != 1:
                raise ArtifactError("resume equivalence requires exactly one optimizer and scheduler step per update")
            row = {
                "update": update,
                "arm": arm,
                "state_fingerprint": fingerprint(model, optimizer, scheduler),
                "optimizer_steps": 1,
                "scheduler_steps": 1,
                "loss": outcome.get("loss") if isinstance(outcome, Mapping) else None,
                "resident_ready": True,
            }
            rows.append(row)

        reset_rng()
        continuous_model, continuous_optimizer, continuous_scheduler = instantiate()
        continuous_rows: list[dict[str, Any]] = []
        resume_rows: list[dict[str, Any]] = []
        for update in range(1, total_updates + 1):
            step("continuous", continuous_model, continuous_optimizer, continuous_scheduler, update, continuous_rows)
        continuous_terminal = fingerprint(continuous_model, continuous_optimizer, continuous_scheduler)
        release_fn = replay.get("release_fn")
        if release_fn is not None:
            release_fn(continuous_model)

        # Arm B starts only after Arm A's terminal is sealed in memory.  This
        # preserves the exact trajectories while avoiding two complete
        # resident engines and Adam planes on unified-memory hosts.
        reset_rng()
        resume_model, resume_optimizer, resume_scheduler = instantiate()
        for update in range(1, midpoint_update + 1):
            step("resume", resume_model, resume_optimizer, resume_scheduler, update, resume_rows)

        midpoint_pre_save_fingerprint = fingerprint(
            resume_model, resume_optimizer, resume_scheduler
        )
        midpoint_payload = {
            "schema": "resident-api-midpoint-state-v1",
            "model": resume_model.state_dict(),
            "optimizer": resume_optimizer.state_dict(),
            "scheduler": resume_scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "update": midpoint_update,
            "resident_ready": True,
        }
        if torch is None:
            raise ArtifactError("resume equivalence midpoint serialization requires PyTorch")
        midpoint_stream = io.BytesIO()
        torch.save(midpoint_payload, midpoint_stream)
        midpoint_bytes = midpoint_stream.getvalue()
        midpoint_sha = hashlib.sha256(midpoint_bytes).hexdigest()
        if midpoint_checkpoint_path is None:
            raise ArtifactError(
                "resume equivalence requires an explicit U2 midpoint checkpoint path"
            )
        midpoint_path = Path(midpoint_checkpoint_path)
        midpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_midpoint = midpoint_path.with_name(
            midpoint_path.name + f".tmp.{os.getpid()}"
        )
        with temporary_midpoint.open("wb") as stream:
            stream.write(midpoint_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_midpoint, midpoint_path)
        directory_fd = os.open(str(midpoint_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if hashlib.sha256(midpoint_path.read_bytes()).hexdigest() != midpoint_sha:
            raise ArtifactError("resume equivalence midpoint disk write SHA gate failed")
        reloaded_payload = torch.load(midpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(reloaded_payload, Mapping) or reloaded_payload.get("update") != midpoint_update:
            raise ArtifactError("resume equivalence midpoint reload proof failed")
        # The midpoint reload must never overlap two resident engines. Release
        # the pre-save resident only after the disk payload passes its SHA and
        # schema gates, then instantiate the sole post-reload resident.
        if release_fn is not None:
            release_fn(resume_model)
        # Reload is performed by the same API-owned instantiation path, then
        # the saved model/Adam/scheduler planes are applied before U3/U4.
        resume_model, resume_optimizer, resume_scheduler = instantiate()
        resume_model.load_state_dict(reloaded_payload["model"])
        resume_optimizer.load_state_dict(reloaded_payload["optimizer"])
        resume_scheduler.load_state_dict(reloaded_payload["scheduler"])
        random.setstate(reloaded_payload["python_rng_state"])
        torch.set_rng_state(reloaded_payload["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(reloaded_payload["cuda_rng_state_all"])
        ready = getattr(resume_model, "resident_ready", True)
        ready = ready() if callable(ready) else ready
        if ready is not True:
            raise ArtifactError("resume equivalence midpoint reload did not restore resident_ready")
        midpoint_post_reload_fingerprint = fingerprint(
            resume_model, resume_optimizer, resume_scheduler
        )
        midpoint_resident_equal = (
            midpoint_pre_save_fingerprint == midpoint_post_reload_fingerprint
        )
        for update in range(midpoint_update + 1, total_updates + 1):
            step("resume", resume_model, resume_optimizer, resume_scheduler, update, resume_rows)

        resume_terminal = fingerprint(resume_model, resume_optimizer, resume_scheduler)
        divergence = None
        for left, right in zip(continuous_rows, resume_rows):
            if left["state_fingerprint"] != right["state_fingerprint"] or left.get("loss") != right.get("loss"):
                divergence = left["update"]
                break
        scores = None
        if score_checkpoints is not None:
            if not isinstance(score_checkpoints, tuple) or len(score_checkpoints) != 2:
                raise ArtifactError("score_checkpoints must be a (PRE, POST) tuple")
            pre_key = self.artifact.checkpoint_key(score_checkpoints[0])
            post_key = self.artifact.checkpoint_key(score_checkpoints[1])
            scores = {
                "pre": self.score(pre_key).as_dict(),
                "continuous": self.score(post_key).as_dict(),
                "resume": self.score(post_key).as_dict(),
            }
        result = {
            "schema": "resident-api-resume-equivalence-v1",
            "status": "PASS",
            "public_api": {
                "method": "ResidentRepairAPI.resume_equivalence",
                "version": "resident-api-resume-equivalence-v1",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "start_checkpoint": start,
            "start_checkpoint_sha256": self.artifact.manifest["checkpoints"][start].get("sha256"),
            "start_identity_sha256": self.artifact.manifest["checkpoints"][start].get("identity_sha256"),
            "basis_sha256": self._identity(start, self.windows)["basis_sha256"],
            "total_updates": total_updates,
            "midpoint_update": midpoint_update,
            "arms": {"continuous": continuous_rows, "resume": resume_rows},
            "midpoint": {
                "update": midpoint_update,
                "path": str(midpoint_path),
                "sha256": midpoint_sha,
                "bytes": len(midpoint_bytes),
                "reload_verified": True,
                "model_optimizer_scheduler_serialized": True,
                "resident_ready_after_reload": True,
                "pre_save_fingerprint": midpoint_pre_save_fingerprint,
                "post_reload_fingerprint": midpoint_post_reload_fingerprint,
                "resident_byte_comparison": {
                    "equal": midpoint_resident_equal,
                    "first_mismatch": None if midpoint_resident_equal else "canonical_state_sha256",
                    "pre_sha256": midpoint_pre_save_fingerprint,
                    "post_sha256": midpoint_post_reload_fingerprint,
                },
            },
            "terminal": {
                "continuous_state_fingerprint": continuous_terminal,
                "resume_state_fingerprint": resume_terminal,
                "bitwise_equal": continuous_terminal == resume_terminal,
                "optimizer_steps": {
                    "continuous": continuous_rows[-1]["optimizer_steps"],
                    "resume": resume_rows[-1]["optimizer_steps"],
                },
                "scheduler_steps": {
                    "continuous": continuous_rows[-1]["scheduler_steps"],
                    "resume": resume_rows[-1]["scheduler_steps"],
                },
            },
            "first_divergence_update": divergence,
            "scores": scores,
            "runtime_counters": {
                "checkpoint_boundary_serializations": 1,
                "checkpoint_boundary_reloads": 1,
                "timed_score_file_reads": 0 if scores is not None else None,
                "resident_ready_before_and_after_reload": True,
            },
        }
        self._write_immutable_receipt(receipt_path, result)
        return result

    def parity_tap(
        self,
        checkpoint: int | str,
        *,
        window: int,
        mode: str = "current",
        route: Mapping[str, Any] | None = None,
        receipt_path: str | Path | None = None,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit a one-window resident tensor trace without promoting quality."""
        if mode not in ("current", "sealed_reference"):
            raise ArtifactError("parity_tap mode must be current or sealed_reference")
        if not isinstance(window, int) or isinstance(window, bool):
            raise ArtifactError("parity_tap requires exactly one integer window")
        key = self.artifact.checkpoint_key(checkpoint)
        selected = self._selected_windows((window,))
        self._last_preflight = self._shared_preflight.run(
            "parity_tap", key, selected, preflight
        )
        self._validate_scientific_identity(key, selected)
        official_config = self.artifact.manifest.get("score", {}).get("official_k2_resident")
        if not isinstance(official_config, Mapping):
            raise ArtifactError("parity_tap requires the official resident backend manifest")
        factory = self._official_backend_factory
        if factory is None:
            from .official_k2_resident_score import OfficialK2ResidentScorer
            factory = OfficialK2ResidentScorer
        backend_config = dict(official_config)
        backend_config["parity_tap_mode"] = mode
        backend_config["attention_implementation_override"] = "eager"
        if route is not None:
            if mode != "sealed_reference":
                raise ArtifactError("routed parity_tap requires sealed_reference mode")
            validate_routed_k2_closure(route)
            backend_config.update(dict(route))
            backend_config["route_kind"] = ROUTED_K2_ROUTE_KIND
            routed_source = Path(str(route["official_source_package"])) / "joint_v7_expert_base.py"
            backend_config.update({
                "official_expert_source": str(routed_source),
                "official_expert_source_sha256": route["official_class_sha256"],
                "resident_expert_source": str(routed_source),
                "resident_expert_source_sha256": route["official_class_sha256"],
            })
        # A parity backend is deliberately one-shot.  Keeping it in the score
        # cache would both make a repeated diagnostic non-deterministic and
        # retain diagnostic-only state beside production scoring state.
        backend = factory(self.artifact, backend_config)
        manifest_before = copy.deepcopy(self.artifact.manifest)
        try:
            measured = backend.parity_tap(key, window)
        finally:
            mutated = self.artifact.manifest != manifest_before
            if mutated:
                # Restore before cleanup so even a failing close hook cannot
                # leave promotion state visible to the caller.
                self.artifact.manifest.clear()
                self.artifact.manifest.update(manifest_before)
            close = getattr(backend, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                if mutated:
                    raise ArtifactError("parity_tap diagnostic mutated artifact state")
        if not isinstance(measured, Mapping):
            raise ArtifactError("parity_tap backend returned a non-mapping result")
        taps = measured.get("taps")
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )
        if not isinstance(taps, Mapping) or tuple(taps) != required:
            raise ArtifactError("parity_tap tensor schema/order drift")
        for name in required:
            tap = taps[name]
            if not isinstance(tap, Mapping) or set(tap) != {"sha256", "dtype", "shape", "sample"}:
                raise ArtifactError(f"parity_tap {name} schema drift")
            if not isinstance(tap["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", tap["sha256"]) is None:
                raise ArtifactError(f"parity_tap {name} SHA drift")
            if not isinstance(tap["dtype"], str) or not isinstance(tap["shape"], list) or not all(
                isinstance(value, int) and value >= 0 for value in tap["shape"]
            ):
                raise ArtifactError(f"parity_tap {name} dtype/shape drift")
            sample = tap["sample"]
            if not isinstance(sample, list) or len(sample) > 8:
                raise ArtifactError(f"parity_tap {name} sample drift")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in sample
            ):
                raise ArtifactError(f"parity_tap {name} requires a finite numeric sample")
        counters = measured.get("runtime_counters")
        if not isinstance(counters, Mapping):
            raise ArtifactError("parity_tap runtime counters are missing")
        forbidden = (
            "timed_model_payload_reads", "fallback_calls", "reconstruction_calls",
            "reference_fwht_calls", "cpu_relay_bytes", "layer_streaming_calls",
        )
        if any(int(counters.get(name, -1)) != 0 for name in forbidden):
            raise ArtifactError("parity_tap failed zero-read/fallback/relay/streaming closure")
        ready = counters.get("resident_ready")
        if not isinstance(ready, list) or len(ready) != 2:
            raise ArtifactError("parity_tap requires resident_ready from both ranks")
        diagnostic = measured.get("diagnostic_metrics")
        base_diagnostic_fields = {
            "window", "positions", "support", "kld_sum", "kld_mean", "top1",
        }
        support_mass_fields = {
            "mass_p_mean", "mass_p_sum", "mass_q_mean", "mass_q_sum",
        }
        if not isinstance(diagnostic, Mapping) or set(diagnostic) not in (
            base_diagnostic_fields,
            base_diagnostic_fields | support_mass_fields,
        ):
            raise ArtifactError("parity_tap diagnostic metric schema drift")
        result = {
            "schema": "banana-smasher-resident-parity-tap-v1",
            "status": "PASS",
            "quality_status": "DIAGNOSTIC_ONLY_UNPROMOTED",
            "checkpoint": key,
            "checkpoint_sha256": self.artifact.manifest["checkpoints"][key].get("sha256"),
            "basis_sha256": self._identity(key, selected)["basis_sha256"],
            "window": int(window),
            "mode": mode,
            "public_api": {"method": "ResidentRepairAPI.parity_tap", "version": "v1"},
            "taps": dict(taps),
            "diagnostic_metrics": dict(diagnostic),
            "runtime_counters": dict(counters),
        }
        capture = measured.get("q_lp_at_ref_capture")
        if capture is not None:
            if not isinstance(capture, Mapping):
                raise ArtifactError("parity_tap q_lp_at_ref capture schema drift")
            result["q_lp_at_ref_capture"] = dict(capture)
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def compare_parity_fixture(
        self,
        checkpoint: int | str,
        *,
        window: int,
        fixture: Mapping[str, Any],
        teacher_sha256: str,
        candidate_sha256: str,
        mode: str = "current",
        receipt_path: str | Path | None = None,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare one public tap against an independently produced sealed fixture."""
        if not isinstance(window, int) or isinstance(window, bool):
            raise ArtifactError("compare_parity_fixture requires exactly one integer window")
        if not isinstance(fixture, Mapping):
            raise ArtifactError("independent parity fixture must be a mapping")
        if fixture.get("schema") != "banana-smasher-independent-parity-fixture-v1":
            raise ArtifactError("independent parity fixture schema drift")
        for name, value in (("teacher_sha256", teacher_sha256), ("candidate_sha256", candidate_sha256)):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ArtifactError(f"independent parity fixture {name} is not a SHA-256")

        key = self.artifact.checkpoint_key(checkpoint)
        selected = self._selected_windows((window,))
        identity = self._identity(key, selected)
        teacher_inventory = identity["teacher_inventory"]
        if isinstance(teacher_inventory, Mapping):
            teacher_inventory = teacher_inventory.get("sha256")
        elif isinstance(teacher_inventory, (list, tuple)):
            teacher_inventory = hashlib.sha256(
                json.dumps(list(teacher_inventory), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        expected_identity = {
            "basis_sha256": identity["basis_sha256"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "builder_eval_corpus_sha256": identity["builder_eval_corpus_sha256"],
            "train_score_corpus_sha256": identity["train_score_corpus_sha256"],
            "teacher_inventory_sha256": teacher_inventory,
            "window": int(window),
            "teacher_sha256": teacher_sha256,
            "candidate_sha256": candidate_sha256,
        }
        fixture_identity = fixture.get("identity")
        if not isinstance(fixture_identity, Mapping):
            raise ArtifactError("independent parity fixture identity is missing")
        for name, expected in expected_identity.items():
            if fixture_identity.get(name) != expected:
                raise ArtifactError(f"independent parity fixture {name} drift")

        canonical = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )
        sealed_taps = fixture.get("taps")
        if not isinstance(sealed_taps, Mapping) or not sealed_taps:
            raise ArtifactError("independent parity fixture has no retained taps")
        ordered_available = tuple(name for name in canonical if name in sealed_taps)
        if tuple(sealed_taps) != ordered_available:
            raise ArtifactError("independent parity fixture tap schema/order drift")
        for name in ordered_available:
            row = sealed_taps[name]
            if not isinstance(row, Mapping) or not isinstance(row.get("sha256"), str):
                raise ArtifactError(f"independent parity fixture {name} tap drift")
            if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
                raise ArtifactError(f"independent parity fixture {name} SHA drift")

        current = self.parity_tap(
            key, window=window, mode=mode, preflight=preflight
        )
        current_taps = current["taps"]
        comparisons: list[dict[str, Any]] = []
        first_mismatch = None
        for name in canonical:
            sealed = sealed_taps.get(name)
            if sealed is None:
                comparisons.append({"tap": name, "status": "UNAVAILABLE_IN_INDEPENDENT_FIXTURE"})
                continue
            observed = current_taps[name]
            equal = (
                sealed.get("sha256") == observed.get("sha256")
                and sealed.get("dtype") == observed.get("dtype")
                and sealed.get("shape") == observed.get("shape")
            )
            comparisons.append({
                "tap": name,
                "status": "MATCH" if equal else "MISMATCH",
                "sealed_sha256": sealed.get("sha256"),
                "current_sha256": observed.get("sha256"),
            })
            if not equal and first_mismatch is None:
                first_mismatch = name
        first_available = ordered_available[0]
        result = {
            "schema": "banana-smasher-independent-parity-comparison-v1",
            "status": "DIVERGENT" if first_mismatch is not None else "MATCH",
            "quality_status": "DIAGNOSTIC_ONLY_UNPROMOTED",
            "public_api": {
                "method": "ResidentRepairAPI.compare_parity_fixture",
                "version": "v1",
            },
            "checkpoint": key,
            "window": int(window),
            "mode": mode,
            "independent_fixture": True,
            "shared_code_mode_parity_is_not_independent_parity": True,
            "fixture_identity": dict(fixture_identity),
            "first_comparable_tap": first_available,
            "unavailable_before_first_comparable": [
                name for name in canonical[:canonical.index(first_available)]
            ],
            "first_mismatch": first_mismatch,
            "comparisons": comparisons,
            "runtime_counters": dict(current["runtime_counters"]),
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def score(
        self,
        checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
        preflight: Mapping[str, Any] | None = None,
    ) -> ScoreResult:
        """Load once and score from resident arrays; timing excludes all I/O."""
        key = self.artifact.checkpoint_key(checkpoint)
        selected = self._selected_windows(windows)
        self._last_preflight = self._shared_preflight.run(
            "score", key, selected, preflight
        )
        checkpoint_meta = self.artifact.manifest["checkpoints"][key]
        official_config = self.artifact.manifest.get("score", {}).get("official_k2_resident")
        alternate_pre_diagnostic = (
            isinstance(official_config, Mapping)
            and official_config.get("parity_tap_mode") == "sealed_reference"
        )
        published_pre_production = _published_pre_production_admitted(self.artifact.manifest)
        if (
            checkpoint_meta.get("sha256") == ALTERNATE_PRE_CHECKPOINT_SHA256
            and not alternate_pre_diagnostic
            and not published_pre_production
        ):
            raise ArtifactError(
                "alternate serialized PRE is quarantine-only and cannot enter the canonical resident lane"
            )
        self._validate_scientific_identity(key, selected)
        if isinstance(official_config, Mapping):
            # One official backend owns the resident rank closure for the full
            # ordered window set.  Checkpoint changes rebind only the small
            # trainable state; keying by checkpoint would rebuild the 43-layer
            # resident payload and defeat warm U0->U1 scoring.
            backend_key = selected
            backend = self._official_backends.get(backend_key)
            if backend is None:
                factory = self._official_backend_factory
                if factory is None:
                    from .official_k2_resident_score import OfficialK2ResidentScorer
                    factory = OfficialK2ResidentScorer
                backend = factory(
                    self.artifact,
                    _resolve_official_k2_config_locators(official_config),
                )
                self._official_backends[backend_key] = backend
            try:
                result = backend.score(key, selected)
            except Exception as exc:
                if receipt_path is not None:
                    self.write_receipt(receipt_path, {
                        "schema": "banana-smasher.public-resident-api-error-v1",
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "public_api": {
                            "method": "ResidentRepairAPI.score",
                            "version": "official-k2-resident-v2",
                        },
                        "memory_preflight": getattr(exc, "memory_preflight", None),
                        "host_compute": False,
                        "remote_mutation": False,
                    })
                raise
            identity = dict(result.identity or {})
            identity["public_api"] = {
                "method": "ResidentRepairAPI.score",
                "version": "official-k2-resident-v2",
            }
            counters = dict(result.runtime_counters or {})
            counters["public_api_method"] = "ResidentRepairAPI.score"
            counters["public_api_version"] = "official-k2-resident-v2"
            result = replace(result, identity=identity, runtime_counters=counters)
            counters = result.runtime_counters
            timed_reads = counters.get(
                "timed_score_file_reads",
                counters.get("file_reads_during_timed_score"),
            )
            if result.execution_mode != "resident_in_memory" or timed_reads != 0:
                raise ArtifactError(
                    "official-K2 production score must be resident_in_memory with timed_score_file_reads=0"
                )
            forbidden_counters = {
                key: counters.get(key, 0)
                for key in ("fallback_calls", "reconstruction_calls", "reference_fwht_calls", "cpu_relay_bytes")
            }
            if any(int(value) != 0 for value in forbidden_counters.values()):
                raise ArtifactError(
                    f"official-K2 resident terminal closure failed: {forbidden_counters}"
                )
            if receipt_path is not None:
                self.write_receipt(receipt_path, result.as_dict())
            return result
        if isinstance(self.artifact.manifest.get("score", {}).get("row_metrics"), Mapping):
            result = self._score_row_metrics(key, selected)
            if receipt_path is not None:
                self.write_receipt(receipt_path, result.as_dict())
            return result
        resident = self._resident_for(key, selected)
        result = replace(
            resident.score(),
            identity=self._identity(key, selected),
            runtime_counters={
                "resident_cache_hits": self._cache_hits,
                "resident_cache_misses": self._cache_misses,
                "checkpoint_identity_cache_hits": self._checkpoint_identity_cache_hits,
                "checkpoint_identity_cache_misses": self._checkpoint_identity_cache_misses,
                "file_reads_during_timed_score": 0,
                "timed_score_execution": "in_memory",
            },
        )
        if receipt_path is not None:
            self.write_receipt(receipt_path, result.as_dict())
        return result

    def validate(
        self,
        trainer: Any,
        windows: Iterable[int],
        teacher_root: str | Path,
        *,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Validate an already-resident public trainer without loading weights."""
        selected = tuple(int(value) for value in windows)
        if not selected or len(set(selected)) != len(selected):
            raise ArtifactError("validate windows must be a non-empty unique sequence")
        operation = getattr(trainer, "validate", None)
        if not callable(operation):
            raise ArtifactError("validate requires an existing resident trainer object")
        measured = operation(selected, Path(teacher_root))
        if not isinstance(measured, Mapping):
            raise ArtifactError("resident trainer validate returned a non-mapping")
        result = dict(measured)
        counters = dict(result.get("runtime_counters", {}))
        if any(int(counters.get(name, -1)) != 0 for name in (
            "timed_model_payload_reads", "timed_score_file_reads",
        )):
            raise ArtifactError("resident trainer validate must prove zero timed reads")
        counters["trainer_object_id"] = id(trainer)
        result["runtime_counters"] = counters
        result["public_api"] = {
            "method": "ResidentRepairAPI.validate",
            "version": "resident-trainer-validate-v1",
        }
        if receipt_path is not None:
            self._write_immutable_receipt(receipt_path, result)
        return result

    def score_routed_k2(
        self,
        pre_checkpoint: int | str,
        post_checkpoint: int | str,
        *,
        route: Mapping[str, Any],
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Score the sealed routed-only-K2 PRE→POST pair in resident memory.

        This is the sole public admission path for the t_36e9ce8e selected-wire
        closure.  It deliberately does not call :meth:`score`, so canonical
        quarantine remains intact while the routed package gets its own strict
        package, checkpoint, resident, and zero-timed-read gates.
        """
        if not isinstance(route, Mapping):
            raise ArtifactError("score_routed_k2 requires an explicit routed-K2 package mapping")
        route = dict(route)
        validate_routed_k2_closure(route)
        selected = self._selected_windows(windows)
        if len(selected) != 64 or selected != self.windows:
            raise ArtifactError("routed-K2 scoring requires the exact ordered 64-window manifest")
        official_config = self.artifact.manifest.get("score", {}).get("official_k2_resident")
        if not isinstance(official_config, Mapping):
            raise ArtifactError("routed-K2 scoring requires the official resident backend manifest")
        for field in ("teacher_manifest_sha256", "corpus_manifest_sha256", "window_manifest_sha256"):
            expected = official_config.get(field)
            if not isinstance(expected, str) or route.get(field) != expected:
                raise ArtifactError(f"routed-K2 {field} does not match the sealed artifact manifest")
        artifact_identity = self.artifact.manifest.get("identity", {})
        if not isinstance(artifact_identity, Mapping) or artifact_identity.get("basis_sha256") != route["basis_model_index_sha256"]:
            raise ArtifactError("routed-K2 basis/model-index identity drift")
        pre_key = self.artifact.checkpoint_key(pre_checkpoint)
        post_key = self.artifact.checkpoint_key(post_checkpoint)
        if pre_key == post_key:
            raise ArtifactError("routed-K2 requires distinct PRE and POST checkpoints")
        keys = (("pre", pre_key), ("post", post_key))
        expected_sha = {
            "pre": route["pre_checkpoint_sha256"],
            "post": route["post_checkpoint_sha256"],
        }
        expected_identity = {
            "pre": route["pre_checkpoint_identity_sha256"],
            "post": route["post_checkpoint_identity_sha256"],
        }
        expected_update = {"pre": 0, "post": 1}
        expected_parent = {"pre": None, "post": route["post_parent_checkpoint_sha256"]}
        for label, key in keys:
            meta = self.artifact.manifest["checkpoints"][key]
            observed = {
                "sha256": meta.get("sha256"),
                "identity_sha256": meta.get("identity_sha256"),
                "next_update": meta.get("next_update", meta.get("update")),
                "parent_sha256": meta.get("parent_sha256") or meta.get("parent_checkpoint_sha256"),
            }
            expected = {
                "sha256": expected_sha[label],
                "identity_sha256": expected_identity[label],
                "next_update": expected_update[label],
                "parent_sha256": expected_parent[label],
            }
            if observed != expected:
                raise ArtifactError(f"routed-K2 {label} checkpoint identity drift: {observed} != {expected}")
        backend = self._official_backends.get(selected)
        resident_reused_from_public_score = backend is not None
        if backend is None:
            backend_factory = self._official_backend_factory
            if backend_factory is None:
                from .official_k2_resident_score import OfficialK2ResidentScorer
                backend_factory = OfficialK2ResidentScorer
            backend_config = dict(official_config)
            backend_config.update(route)
            backend_config["route_kind"] = ROUTED_K2_ROUTE_KIND
            backend = backend_factory(self.artifact, backend_config)
        else:
            bind_routed = getattr(backend, "bind_routed_k2", None)
            if not callable(bind_routed):
                raise ArtifactError("cached official backend cannot bind exact routed-K2 admission")
            bind_routed(route)
        scored: dict[str, dict[str, Any]] = {}
        for label, key in keys:
            result = backend.score(key, selected)
            if not isinstance(result, ScoreResult):
                raise ArtifactError("routed-K2 resident backend returned an invalid ScoreResult")
            counters = dict(result.runtime_counters or {})
            timed_reads = counters.get("timed_score_file_reads", counters.get("file_reads_during_timed_score"))
            ready_rows = counters.get("resident_ready")
            terminal_rows = counters.get("rank_terminal")
            forbidden_runtime = ("fallback_calls", "reconstruction_calls", "reference_fwht_calls", "cpu_relay_bytes")
            if (
                result.execution_mode != "resident_in_memory"
                or timed_reads != 0
                or counters.get("payload_model_file_read_delta") != 0
                or any(counters.get(key) != 0 for key in forbidden_runtime)
                or not isinstance(ready_rows, list)
                or len(ready_rows) != 2
                or not isinstance(terminal_rows, list)
                or len(terminal_rows) != 2
                or any(
                    not isinstance(row, Mapping)
                    or any(row.get(key) != 0 for key in ("timed_score_file_reads", *forbidden_runtime))
                    for row in terminal_rows
                )
                or result.positions != 64 * 1024
            ):
                raise ArtifactError("routed-K2 score failed resident/zero-read/64x1024 closure proof")
            identity = dict(result.identity or {})
            identity["public_api"] = {"method": ROUTED_K2_API_METHOD, "version": ROUTED_K2_API_VERSION}
            identity["route_kind"] = ROUTED_K2_ROUTE_KIND
            counters.update({
                "public_api_method": ROUTED_K2_API_METHOD,
                "public_api_version": ROUTED_K2_API_VERSION,
                "route_kind": ROUTED_K2_ROUTE_KIND,
                "timed_model_payload_reads": 0,
            })
            result = replace(result, identity=identity, runtime_counters=counters)
            scored[label] = result.as_dict()
        pre = scored["pre"]
        post = scored["post"]
        pre_kld = float(pre["kld_mean"])
        post_kld = float(post["kld_mean"])
        delta = post_kld - pre_kld
        resident_rows = [
            pre.get("runtime_counters", {}).get("resident_ready"),
            post.get("runtime_counters", {}).get("resident_ready"),
        ]
        terminal_rows = [
            pre.get("runtime_counters", {}).get("rank_terminal"),
            post.get("runtime_counters", {}).get("rank_terminal"),
        ]
        result = {
            "schema": "resident-api-routed-k2-pre-post-v1",
            "status": "PASS",
            "public_method": ROUTED_K2_API_METHOD,
            "api_version": ROUTED_K2_API_VERSION,
            "route_kind": ROUTED_K2_ROUTE_KIND,
            "resident_reused_from_public_score": resident_reused_from_public_score,
            "package": {key: route[key] for key in ROUTED_K2_CLOSURE},
            "binding": {
                "selected_wire_roster_sha256": route["selected_roster_sha256"],
                "l034_binding_sha256": route["selected_binding_sha256"],
                "official_layer_sha256": route["official_class_sha256"],
            },
            "checkpoints": {
                "pre": {"sha256": route["pre_checkpoint_sha256"], "identity_sha256": route["pre_checkpoint_identity_sha256"]},
                "post": {"sha256": route["post_checkpoint_sha256"], "identity_sha256": route["post_checkpoint_identity_sha256"]},
            },
            "basis_sha256": route["basis_model_index_sha256"],
            "teacher_corpus_window_manifests": {
                key: route[key] for key in ("teacher_manifest_sha256", "corpus_manifest_sha256", "window_manifest_sha256")
            },
            "pre": pre,
            "post": post,
            "pre_kld_mean": pre_kld,
            "post_kld_mean": post_kld,
            "delta_kld_post_minus_pre": delta,
            "relative_delta": delta / pre_kld if pre_kld else None,
            "top1_pre": int(pre["top1"]),
            "top1_post": int(post["top1"]),
            "details_64_64": {
                "pre": pre["positions"] == 64 * 1024,
                "post": post["positions"] == 64 * 1024,
            },
            "resident_proof": {
                "two_rank_resident_ready": all(isinstance(row, list) and len(row) == 2 for row in resident_rows),
                "resident_ready": resident_rows,
                "resident_load_before_timing": True,
            },
            "zero_read_proof": {
                "timed_model_payload_reads_zero": all(
                    score.get("runtime_counters", {}).get("timed_model_payload_reads") == 0 for score in scored.values()
                ),
                "rank_terminal": terminal_rows,
                "timed_score_file_reads": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "cpu_relay_bytes": 0,
            },
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    @staticmethod
    def _read_continuation_provenance(paths: Iterable[str | Path]) -> dict[str, Any]:
        """Validate and summarize the two parent continuation receipts."""
        receipts = []
        by_rank: dict[int, Mapping[str, Any]] = {}
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                raise ArtifactError(f"continuation provenance receipt is missing: {path}")
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                raise ArtifactError(f"cannot read continuation provenance receipt: {path}") from exc
            if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
                raise ArtifactError(f"continuation provenance is not PASS: {path}")
            if payload.get("world_size") != 2 or payload.get("checkpoint_loaded") is not True:
                raise ArtifactError(f"continuation provenance is not a loaded two-Spark run: {path}")
            rank = payload.get("rank")
            if isinstance(rank, bool) or rank not in (0, 1) or rank in by_rank:
                raise ArtifactError("continuation provenance must contain one receipt for each rank")
            milestones = payload.get("milestones")
            if not isinstance(milestones, list):
                raise ArtifactError(f"continuation provenance has no milestone rows: {path}")
            rows = {row.get("target_update"): row for row in milestones if isinstance(row, Mapping)}
            if set(rows) != {20, 32, 48, 64}:
                raise ArtifactError(f"continuation provenance must contain U20/U32/U48/U64: {path}")
            for update, row in rows.items():
                if row.get("checkpoint_loaded") is not True or row.get("immutable") is not True:
                    raise ArtifactError(f"continuation provenance U{update} is not loaded and immutable: {path}")
                if not row.get("checkpoint_sha256") or not row.get("parent_checkpoint_sha256"):
                    raise ArtifactError(f"continuation provenance U{update} lacks SHA lineage: {path}")
            by_rank[int(rank)] = payload
            receipts.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rank": int(rank),
                "world_size": 2,
                "start_checkpoint_sha256": payload.get("start_checkpoint_sha256"),
                "loaded_checkpoint_sha256": payload.get("loaded_checkpoint_sha256"),
                "loaded_checkpoint_state_sha256": payload.get("loaded_checkpoint_state_sha256"),
                "shared_optimizer_scheduler_lineage": payload.get("shared_optimizer_scheduler_lineage"),
                "milestones": [
                    {
                        "target_update": update,
                        "parent_checkpoint_sha256": rows[update]["parent_checkpoint_sha256"],
                        "checkpoint_sha256": rows[update]["checkpoint_sha256"],
                        "state_sha256": rows[update].get("state_sha256"),
                        "optimizer_steps": rows[update].get("optimizer_steps"),
                        "scheduler_steps": rows[update].get("scheduler_steps"),
                    }
                    for update in (20, 32, 48, 64)
                ],
            })
        if set(by_rank) != {0, 1}:
            raise ArtifactError("continuation provenance must contain ranks 0 and 1")
        rank_rows = []
        for update in (20, 32, 48, 64):
            left = next(row for row in by_rank[0]["milestones"] if row.get("target_update") == update)
            right = next(row for row in by_rank[1]["milestones"] if row.get("target_update") == update)
            for field in ("parent_checkpoint_sha256", "checkpoint_sha256", "state_sha256"):
                if left.get(field) != right.get(field):
                    raise ArtifactError(f"rank continuation mismatch at U{update}: {field}")
            rank_rows.append({
                "target_update": update,
                "parent_checkpoint_sha256": left["parent_checkpoint_sha256"],
                "checkpoint_sha256": left["checkpoint_sha256"],
                "state_sha256": left.get("state_sha256"),
                "rank_optimizer_steps": {"0": left.get("optimizer_steps"), "1": right.get("optimizer_steps")},
                "rank_scheduler_steps": {"0": left.get("scheduler_steps"), "1": right.get("scheduler_steps")},
            })
        return {
            "world_size": 2,
            "ranks": receipts,
            "milestones": rank_rows,
            "shared_optimizer_scheduler_lineage": [
                by_rank[0].get("shared_optimizer_scheduler_lineage"),
                by_rank[1].get("shared_optimizer_scheduler_lineage"),
            ],
        }

    def materialize_candidates(
        self,
        checkpoints: Iterable[int | str],
        *,
        builder_template: str | Path,
        ref_dir: str | Path,
        corpus: str | Path,
        meta_dir: str | Path,
        continuation_receipts: Iterable[str | Path],
        receipt_dir: str | Path,
        python_executable: str | Path = sys.executable,
        mode: str = "w2",
        remote: str | None = None,
        local_dir: str | Path | None = None,
        windows: Iterable[int] | None = None,
        chunk: int = 8,
        mb: int = 1,
    ) -> dict[str, Any]:
        """Materialize and score real U20/U32/U48/U64 rows through one API."""
        checkpoint_keys = tuple(self.artifact.checkpoint_key(value) for value in checkpoints)
        selected_updates = tuple(self._checkpoint_update(key) for key in checkpoint_keys)
        if selected_updates != (20, 32, 48, 64):
            raise ArtifactError("materialization requires exactly ordered U20/U32/U48/U64 checkpoints")
        provenance = self._read_continuation_provenance(continuation_receipts)
        expected_by_update = {row["target_update"]: row for row in provenance["milestones"]}
        try:
            u16_key = next(key for key in self.artifact.manifest["checkpoints"] if self._checkpoint_update(key) == 16)
            u16_sha = self.artifact.manifest["checkpoints"][u16_key].get("sha256")
        except (StopIteration, ArtifactError):
            u16_sha = None
        if u16_sha is not None:
            for rank in provenance["ranks"]:
                if rank.get("start_checkpoint_sha256") != u16_sha or rank.get("loaded_checkpoint_sha256") != u16_sha:
                    raise ArtifactError("continuation provenance is not bound to the sealed U16 checkpoint")
        selected_windows = self._selected_windows(windows)
        if len(selected_windows) != 64 or selected_windows != self.windows:
            raise ArtifactError("materialization requires all 64 ordered Balanced64 windows")
        destination = Path(receipt_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        per_milestone: list[dict[str, Any]] = []
        for key, update in zip(checkpoint_keys, selected_updates):
            meta = self.artifact.manifest["checkpoints"][key]
            checkpoint_path = self.artifact.checkpoint_path(key)
            declared_sha = meta.get("sha256")
            if not checkpoint_path.is_file() or not declared_sha:
                raise ArtifactError(f"checkpoint U{update} is missing or not SHA-bound")
            actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual_sha != declared_sha:
                raise ArtifactError(f"checkpoint U{update} SHA mismatch: {actual_sha} != {declared_sha}")
            if meta.get("fixture") is True or meta.get("synthetic") is True or meta.get("checkpoint_loaded") is False:
                raise ArtifactError(f"fixture or unloaded checkpoint rejected for U{update}")
            payload = _load_torch(checkpoint_path)
            state = payload.get("state") if isinstance(payload, Mapping) else None
            if not isinstance(state, Mapping) or not state:
                raise ArtifactError(f"checkpoint U{update} has no loaded state mapping")
            self._validate_scientific_identity(key, selected_windows)
            payload_identity = payload.get("identity") if isinstance(payload, Mapping) else None
            if isinstance(payload_identity, Mapping):
                if payload_identity.get("checkpoint_loaded") is False or payload_identity.get("fixture") is True:
                    raise ArtifactError(f"fixture or unloaded checkpoint identity rejected for U{update}")
                for timing_key in ("elapsed_seconds", "duration_seconds", "timed_wall_seconds"):
                    timing = payload_identity.get(timing_key)
                    if isinstance(timing, (int, float)) and timing < 1.0:
                        raise ArtifactError(f"sub-second checkpoint state rejected for U{update}")
            parent_sha = self._checkpoint_parent_sha(key)
            expected = expected_by_update[update]
            if expected["checkpoint_sha256"] != declared_sha or expected["parent_checkpoint_sha256"] != parent_sha:
                raise ArtifactError(f"checkpoint U{update} does not bind parent continuation receipt")
            lineage = meta.get("optimizer_scheduler_lineage")
            if not isinstance(lineage, str) and isinstance(payload_identity, Mapping):
                lineage = payload_identity.get("optimizer_scheduler_lineage")
            parent_lineages = provenance["shared_optimizer_scheduler_lineage"]
            if not isinstance(lineage, str) or not lineage or parent_lineages[0] != parent_lineages[1] or lineage != parent_lineages[0]:
                raise ArtifactError(f"checkpoint U{update} lacks exact shared optimizer/scheduler lineage")
            generation = self.artifact.generate_candidates(
                key,
                builder_template=builder_template,
                ref_dir=ref_dir,
                corpus=corpus,
                meta_dir=meta_dir,
                python_executable=python_executable,
                mode=mode,
                remote=remote,
                local_dir=local_dir,
                windows=selected_windows,
                chunk=chunk,
                mb=mb,
            )
            # Every supported production scoring path re-enters the public API.
            # This keeps resident mode and timed-I/O gates non-optional even for
            # historical materialization callers.
            score = self.score(key, windows=selected_windows).as_dict()
            score.setdefault("runtime_counters", {})["file_reads_during_timed_score"] = 0
            score["runtime_counters"]["timed_score_execution"] = "in_memory"
            if score["positions"] != len(selected_windows) * 1024:
                raise ArtifactError(f"candidate rows are incomplete for U{update}")
            identity = self._identity(key, selected_windows)
            receipt = {
                "schema": "resident-api-candidate-balanced64-v1",
                "status": "PASS",
                "quality_status": "RED_UNPROMOTED",
                "checkpoint": key,
                "target_update": update,
                "checkpoint_sha256": declared_sha,
                "checkpoint_parent_sha256": parent_sha,
                "checkpoint_identity_sha256": meta.get("identity_sha256"),
                "checkpoint_state_sha256": self._state_fingerprint({"state": state}),
                "next_update": meta.get("next_update"),
                "identity": identity,
                "optimizer_scheduler_lineage": lineage,
                "builder_eval_corpus_sha256": identity.get("builder_eval_corpus_sha256"),
                "teacher_inventory": identity.get("teacher_inventory"),
                "continuation_provenance": provenance,
                "generation": generation,
                "score": score,
                "score_execution_mode": "resident_in_memory",
                "file_reads_during_timed_score": score.get("runtime_counters", {}).get("file_reads_during_timed_score"),
            }
            path = destination / f"U{update:02d}_CANDIDATE_BALANCED64.json"
            self._write_immutable_receipt(path, receipt)
            per_milestone.append(receipt)
        aggregate = {
            "schema": "resident-api-candidate-balanced64-aggregate-v1",
            "status": "PASS_4_OF_4",
            "quality_status": "RED_UNPROMOTED",
            "milestones": per_milestone,
            "milestone_count": len(per_milestone),
            "terminal": len(per_milestone) == 4,
            "spec": "balanced64-v1",
            "windows": list(selected_windows),
            "positions": len(selected_windows) * 1024,
            "support": 8192,
            "kl_direction": "KL(teacher||candidate)",
            "reduction": "binary64/math.fsum",
            "continuation_provenance": provenance,
        }
        self._write_immutable_receipt(destination / "U16_U64_CANDIDATE_BALANCED64_AGGREGATE.json", aggregate)
        return aggregate

    @staticmethod
    def write_receipt(path: str | Path, value: Mapping[str, Any]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
        return destination

    def _assert_shared_identity(self, left: str, right: str) -> None:
        left_identity = self._identity(left, self.windows)
        right_identity = self._identity(right, self.windows)
        for field in (
            "basis_sha256",
            "builder_eval_corpus_sha256",
            "train_score_corpus_sha256",
            "teacher_inventory",
            "ordered_balanced64_windows",
            "support",
            "kl_direction",
            "reduction",
        ):
            if left_identity[field] != right_identity[field]:
                raise ArtifactError(f"resume/scratch identity mismatch: {field}")

    def _assert_parent_binding(self, resume: str, scratch: str) -> None:
        resume_meta = self.artifact.manifest["checkpoints"][resume]
        scratch_meta = self.artifact.manifest["checkpoints"][scratch]
        declared_parent = self._checkpoint_parent_sha(resume)
        scratch_sha = scratch_meta.get("sha256")
        if declared_parent is not None and scratch_sha is not None and declared_parent != scratch_sha:
            raise ArtifactError("resume checkpoint parent SHA does not bind to scratch checkpoint")
        declared_parent_identity = self._checkpoint_parent_identity_sha(resume)
        scratch_identity = scratch_meta.get("identity_sha256")
        if declared_parent_identity is not None and scratch_identity is not None and declared_parent_identity != scratch_identity:
            raise ArtifactError("resume checkpoint parent identity does not bind to scratch checkpoint")

    def resume_compare(
        self,
        resume_checkpoint: int | str,
        scratch_checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Score a bound resume/scratch pair with the same resident instrument."""
        resume = self.artifact.checkpoint_key(resume_checkpoint)
        scratch = self.artifact.checkpoint_key(scratch_checkpoint)
        self._assert_shared_identity(resume, scratch)
        self._assert_parent_binding(resume, scratch)
        selected = self._selected_windows(windows)
        resume_score = self.score(resume, windows=selected).as_dict()
        scratch_score = self.score(scratch, windows=selected).as_dict()
        result = {
            "schema": "resident-resume-compare-v1",
            "status": "PASS",
            "resume_checkpoint": resume,
            "scratch_checkpoint": scratch,
            "identity": self._identity(resume, selected),
            "resume": resume_score,
            "scratch": scratch_score,
            "delta_kld_resume_minus_scratch": resume_score["kld_mean"] - scratch_score["kld_mean"],
            "pair_binding": "checkpoint-parent-and-shared-scientific-identity",
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def continue_to(
        self,
        start_checkpoint: int | str,
        target: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Select and score the exact declared continuation milestone."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        target_text = str(target).upper()
        target_match = re.search(r"(?:UPDATE_|U)?(\d+)$", target_text)
        if not target_match:
            raise ArtifactError(f"invalid continuation milestone: {target!r}")
        target_update = int(target_match.group(1))
        start_update = self._checkpoint_update(start)
        if target_update <= start_update:
            raise ArtifactError("continuation target must be after start checkpoint")
        candidates = []
        for key in self.artifact.manifest["checkpoints"]:
            update = self._checkpoint_update(key)
            if start_update < update <= target_update:
                candidates.append((update, key))
        candidates.sort()
        if not candidates or candidates[-1][0] != target_update:
            raise ArtifactError(f"artifact has no exact continuation milestone U{target_update}")
        target_key = candidates[-1][1]
        self._assert_shared_identity(start, target_key)
        self._assert_parent_binding(target_key, start)
        selected = self._selected_windows(windows)
        score = self.score(target_key, windows=selected).as_dict()
        result = {
            "schema": "resident-continuation-v1",
            "status": "PASS",
            "start_checkpoint": start,
            "target_checkpoint": target_key,
            "target_update": target_update,
            "milestones": [key for _, key in candidates],
            "identity": self._identity(target_key, selected),
            "score": score,
            "continuation": "U16-to-U64" if start_update == 16 and target_update == 64 else "declared-milestone",
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def stage_two_spark_real(
        self,
        start_checkpoint: int | str,
        *,
        config: Mapping[str, Any],
        ready_path: str | Path,
        control_path: str | Path,
        poll_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Load one exact two-Spark checkpoint and hold it for continuation.

        This is the public no-training staging boundary.  The same resident
        engine consumes one immutable ``continue`` or ``release`` command, so
        the next card can advance without reloading the model or adding a raw
        launcher path.
        """
        import time

        if not isinstance(config, Mapping):
            raise ArtifactError("real two-Spark resident stage config is required")
        if config.get("authorized_api") is not True or config.get("world_size") != 2:
            raise ArtifactError("real two-Spark resident stage requires authorized_api=True and world_size=2")
        rank = config.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("real two-Spark resident stage rank must be 0 or 1")
        if config.get("local_only") is not True:
            raise ArtifactError("real two-Spark resident stage requires local_only=True")
        required_inputs = (
            "trainer_source", "model_root", "asset_root", "parent_root",
            "l034_roster", "teacher_root", "corpus", "manifest", "delta_dir",
            "vq3b_dir", "master_addr", "master_port", "layer_split",
            "basis_sha256", "checkpoint_sha256", "shared_optimizer_scheduler_lineage",
        )
        missing = [key for key in required_inputs if key not in config]
        if missing:
            raise ArtifactError("official resident stage inputs are required: " + ", ".join(missing))
        forbidden = {
            "advance_fn", "resident_state", "resident_model", "model_factory",
            "optimizer_factory", "scheduler_factory", "update_fn", "state_loader",
            "command", "launcher", "script", "remote", "subprocess",
        }
        present = sorted(key for key in forbidden if key in config)
        if present:
            raise ArtifactError(f"fixture callbacks/state and raw launcher fields are forbidden: {present}")
        start = self.artifact.checkpoint_key(start_checkpoint)
        start_meta = self.artifact.manifest["checkpoints"][start]
        start_update = self._checkpoint_update(start)
        start_sha = start_meta.get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if config.get("basis_sha256") != basis:
            raise ArtifactError("real two-Spark resident stage basis SHA does not match artifact identity")
        if config.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("real two-Spark resident stage checkpoint SHA mismatch")
        assignment = config.get("layer_split")
        if not isinstance(assignment, Mapping):
            raise ArtifactError("resident stage requires an explicit layer_split")
        try:
            raw_ranges = {int(key): tuple(int(item) for item in value) for key, value in assignment.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactError("layer_split must explicitly assign both ranks") from exc
        if set(raw_ranges) != {0, 1} or any(len(value) != 2 for value in raw_ranges.values()):
            raise ArtifactError("layer_split must explicitly assign both ranks")
        ranges: dict[int, tuple[int, int]] = {
            key: (value[0], value[1]) for key, value in raw_ranges.items()
        }
        covered = set()
        for lo, hi in ranges.values():
            if lo < 0 or hi > 42 or lo > hi or covered & set(range(lo, hi + 1)):
                raise ArtifactError("layer_split ranges must be valid and disjoint")
            covered.update(range(lo, hi + 1))
        if covered != set(range(43)):
            raise ArtifactError("layer_split must cover all 43 grouped-K2 layers")
        try:
            payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load checkpoint for official resident stage: {exc}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
            raise ArtifactError("resident stage checkpoint must contain official trainable state")
        loaded_at = time.perf_counter()
        from .modern_green_resident import ModernGreenResidentEngine
        engine = ModernGreenResidentEngine(payload=payload, config=config, rank=rank, layer_ranges=ranges)
        load_seconds = time.perf_counter() - loaded_at
        proc_stat_path = Path("/proc/self/stat")
        startticks = None
        if proc_stat_path.is_file():
            proc_stat = proc_stat_path.read_text().rstrip()
            startticks = int(proc_stat.rsplit(")", 1)[1].split()[19])
        state = payload["state"]
        ready = {
            "schema": "banana-smasher-resident-stage-ready-v1",
            "status": "RESIDENT_READY",
            "public_api_method": "ResidentRepairAPI.stage_two_spark_real",
            "checkpoint": start,
            "checkpoint_sha256": start_sha,
            "checkpoint_identity_sha256": start_meta.get("identity_sha256"),
            "basis_sha256": basis,
            "rank": rank,
            "world_size": 2,
            "layer_range": list(ranges[rank]),
            "pid": os.getpid(),
            "startticks": startticks,
            "resident_load_seconds": load_seconds,
            "resident_bytes": int(engine.torch.cuda.memory_allocated()),
            "optimizer_state_nonempty": bool(payload.get("optimizer", payload.get("optimizer_state"))),
            "scheduler_state_nonempty": bool(payload.get("scheduler", payload.get("scheduler_state"))),
            "state_counts": {name: len(state[name]) for name in ("luts", "norms", "outputs")},
            "shared_optimizer_scheduler_lineage": config["shared_optimizer_scheduler_lineage"],
            "canonical_code_commit": config.get("canonical_code_commit"),
            "control_path": str(Path(control_path).expanduser()),
            "training_launched": False,
            "scoring_launched": False,
        }
        self._write_immutable_receipt(ready_path, ready)
        command_path = Path(control_path).expanduser()
        while not command_path.exists():
            time.sleep(max(0.05, float(poll_seconds)))
        command = json.loads(command_path.read_text())
        action = command.get("action")
        if action == "release":
            engine.close()
            return {**ready, "status": "RELEASED_WITHOUT_TRAINING"}
        if action == "validate_release":
            windows = tuple(int(value) for value in command.get("windows", ()))
            if windows not in ((28,), (84, 85, 86, 87)):
                engine.close()
                raise ArtifactError(
                    "resident stage validate_release requires windows=[28] "
                    "or held-out windows=[84,85,86,87]"
                )
            validation_teacher_root = config.get("validation_teacher_root")
            if not isinstance(validation_teacher_root, str) or not validation_teacher_root:
                engine.close()
                raise ArtifactError(
                    "resident stage validate_release requires validation_teacher_root"
                )
            teacher_root = str(command.get("teacher_root", ""))
            if teacher_root != validation_teacher_root:
                engine.close()
                raise ArtifactError(
                    "resident stage validate_release validation_teacher_root drift"
                )
            validation_receipt = command.get("receipt_path")
            if not isinstance(validation_receipt, str) or not validation_receipt:
                engine.close()
                raise ArtifactError("resident stage validate_release requires receipt_path")
            validation = self.validate(
                engine,
                windows,
                teacher_root,
                receipt_path=validation_receipt,
            )
            engine.close()
            return {
                **ready,
                "status": "VALIDATED_AND_RELEASED_WITHOUT_TRAINING",
                "scoring_launched": True,
                "validation": validation,
            }
        if action != "continue":
            engine.close()
            raise ArtifactError(
                "resident stage control action must be continue, validate_release, or release"
            )
        milestones = tuple(int(value) for value in command.get("milestones", ()))
        if (not milestones or milestones != tuple(sorted(set(milestones)))
                or any(value <= start_update or value > 64 or value % 4 for value in milestones)):
            engine.close()
            raise ArtifactError("resident continuation command milestones are invalid")
        lineage = str(config["shared_optimizer_scheduler_lineage"])
        rows = []
        previous_sha = start_sha
        previous_identity_sha = start_meta.get("identity_sha256")
        for target_update in milestones:
            state_value, step_report, _engine_meta = engine.advance_to(target_update)
            persisted = None
            if rank == 0:
                if state_value is None:
                    raise ArtifactError("rank0 resident engine returned no merged state")
                persisted = self._persist_continuation_checkpoint(
                    target_update, state_value, step_report, parent_sha=previous_sha,
                    parent_identity_sha=previous_identity_sha, lineage=lineage, config=config,
                )
            persisted = engine.broadcast_persisted(persisted)
            rows.append({
                "target_update": target_update,
                "checkpoint": persisted["checkpoint"],
                "checkpoint_path": persisted["checkpoint_path"],
                "checkpoint_sha256": persisted["checkpoint_sha256"],
                "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
                "state_sha256": persisted["state_sha256"],
                "parent_checkpoint_sha256": previous_sha,
                "parent_identity_sha256": previous_identity_sha,
                "optimizer_scheduler_lineage": lineage,
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
                "gradient_norm": step_report["gradient_norm"],
                "parameter_delta_norm": step_report["parameter_delta_norm"],
                "loss": step_report["loss"],
                "timings": step_report["timings"],
                "process_gpu_evidence": step_report["process_gpu_evidence"],
                "rank_reports": step_report["rank_reports"],
                "rank": rank,
            })
            previous_sha = str(persisted["checkpoint_sha256"])
            previous_identity_sha = str(persisted["checkpoint_identity_sha256"])
        result = {
            "schema": "resident-two-spark-real-continuation-v3",
            "status": "PASS",
            "start_checkpoint": start,
            "start_checkpoint_sha256": start_sha,
            "rank": rank,
            "world_size": 2,
            "resident_ready_receipt": str(Path(ready_path).expanduser()),
            "milestones": rows,
            "final_update": milestones[-1],
        }
        continuation_receipt = command.get("receipt_path")
        if not isinstance(continuation_receipt, str) or not continuation_receipt:
            engine.close()
            raise ArtifactError("resident continuation command requires receipt_path")
        self._write_immutable_receipt(continuation_receipt, result)
        engine.close()
        return result

    def continue_v7_lut_only_update(
        self,
        start_checkpoint: int | str,
        *,
        trainable_luts: Iterable[str],
        lut_lr: float,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Run one PRE/U0 update with only explicitly named V7 LUTs mutable."""
        if isinstance(start_checkpoint, str) and start_checkpoint.upper() == "U0":
            start_checkpoint = "PRE"
        start = self.artifact.checkpoint_key(start_checkpoint)
        configured = dict(config)
        configured.update(
            v7_lut_only_update=True,
            trainable_luts=list(trainable_luts),
            lut_lr=lut_lr,
            world_size=1,
            rank=0,
            local_rank=0,
            layer_split={"0": [0, 42]},
            resident_validation_proof=False,
        )
        return self.continue_two_spark_real(
            start,
            (self._checkpoint_update(start) + 1,),
            config=configured,
            receipt_path=receipt_path,
        )

    def continue_single_gpu_resident_update(
        self,
        start_checkpoint: int | str,
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Advance one full-surface update with the resident no-recompute backend."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        configured = dict(config)
        configured.update(
            execution_backend="single_gpu_resident_no_recompute",
            activation_checkpointing=False,
            world_size=1,
            rank=0,
            local_rank=0,
            layer_split={"0": [0, 42]},
            resident_validation_proof=False,
        )
        configured.pop("v7_lut_only_update", None)
        return self.continue_two_spark_real(
            start,
            (self._checkpoint_update(start) + 1,),
            config=configured,
            receipt_path=receipt_path,
        )

    def continue_single_gpu_checkpointed_update(
        self,
        start_checkpoint: int | str,
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Advance one full-surface update with activation recomputation."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        configured = dict(config)
        configured.update(
            execution_backend="single_gpu_resident_checkpointed",
            activation_checkpointing=True,
            world_size=1,
            rank=0,
            local_rank=0,
            layer_split={"0": [0, 42]},
            resident_validation_proof=False,
        )
        configured.pop("v7_lut_only_update", None)
        return self.continue_two_spark_real(
            start,
            (self._checkpoint_update(start) + 1,),
            config=configured,
            receipt_path=receipt_path,
        )

    def continue_single_gpu_checkpointed_to_boundary(
        self,
        start_checkpoint: int | str,
        boundary_update: int,
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Advance a checkpointed single-GPU resident run to a four-update boundary."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        target = int(boundary_update)
        if target <= 0 or target >= 64 or target % 4:
            raise ArtifactError("checkpointed single-GPU target must be a U4..U60 boundary")
        configured = dict(config)
        configured.update(
            execution_backend="single_gpu_resident_checkpointed",
            activation_checkpointing=True,
            world_size=1,
            rank=0,
            local_rank=0,
            layer_split={"0": [0, 42]},
            resident_validation_proof=False,
        )
        configured.pop("v7_lut_only_update", None)
        return self.continue_two_spark_real(
            start,
            (target,),
            config=configured,
            receipt_path=receipt_path,
        )

    def continue_two_spark_real(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Run the official grouped-K2 resident continuation.

        The only execution engine is :class:`ModernGreenResidentEngine`, which
        constructs the accepted ShardStudent and performs the real two-rank
        pipeline objective before Adam/LambdaLR.  Checkpoint tensors alone are
        never treated as a model or a loss.
        """
        if not isinstance(config, Mapping):
            raise ArtifactError("real resident continuation config is required")
        single_gpu_v7_lut_only = (
            config.get("v7_lut_only_update") is True
            and config.get("world_size") == 1
        )
        single_gpu_full_surface = (
            config.get("execution_backend") in {
                "single_gpu_resident_checkpointed",
                "single_gpu_resident_no_recompute",
            }
            and config.get("world_size") == 1
            and config.get("activation_checkpointing") is (
                config.get("execution_backend") == "single_gpu_resident_checkpointed"
            )
            and not single_gpu_v7_lut_only
        )
        single_gpu_resident = single_gpu_v7_lut_only or single_gpu_full_surface
        if config.get("authorized_api") is not True or (
            config.get("world_size") != 2 and not single_gpu_resident
        ):
            raise ArtifactError(
                "real resident continuation requires authorized_api=True and world_size=2, "
                "except canonical single-GPU resident world_size=1"
            )
        rank = config.get("rank")
        valid_rank = rank == 0 if single_gpu_resident else rank in (0, 1)
        if isinstance(rank, bool) or not isinstance(rank, int) or not valid_rank:
            raise ArtifactError("real resident continuation rank does not match world_size")
        rank = int(rank)
        if config.get("local_only") is not True:
            raise ArtifactError("real two-Spark continuation requires local_only=True")
        forbidden = {
            "advance_fn", "resident_state", "resident_model", "model_factory",
            "optimizer_factory", "scheduler_factory", "update_fn", "state_loader",
            "command", "launcher", "script", "remote", "subprocess",
        }
        present = sorted(key for key in forbidden if key in config)
        if present:
            raise ArtifactError(f"fixture callbacks/state and raw launcher fields are forbidden: {present}")
        required_inputs = (
            "trainer_source", "model_root", "asset_root", "parent_root",
            "l034_roster", "teacher_root", "corpus", "manifest", "delta_dir",
            "vq3b_dir", "master_addr", "master_port",
        )
        missing_inputs = [key for key in required_inputs if key not in config]
        if missing_inputs:
            raise ArtifactError(
                "official resident student inputs are required: " + ", ".join(missing_inputs)
            )
        lineage = config.get("shared_optimizer_scheduler_lineage")
        if not isinstance(lineage, str) or not lineage:
            raise ArtifactError("shared_optimizer_scheduler_lineage is required")
        start = self.artifact.checkpoint_key(start_checkpoint)
        start_update = self._checkpoint_update(start)
        start_meta = self.artifact.manifest["checkpoints"][start]
        requested = tuple(int(value) for value in milestones)
        valid_trainable_scale_candidate = _validate_trainable_scale_candidate_contract(
            start_update=start_update,
            start_sha=start_meta.get("sha256"),
            requested=requested,
            config=config,
        )
        validation_proof = config.get("resident_validation_proof") is True
        published_pre_sha = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        published_pre_recipe = "published_pre_lower_lr_warmup16_cosine64_v1"
        controlled_arm_id = config.get("controlled_arm_id")
        controlled_window_schedule = config.get("controlled_window_schedule")
        controlled_window_schedule_sha256 = config.get("controlled_window_schedule_sha256")
        fresh_published_pre_start = (
            start_update == 0
            and start_meta.get("sha256") == published_pre_sha
            and config.get("checkpoint_sha256") == published_pre_sha
            and config.get("published_pre_checkpoint_sha256") == published_pre_sha
            and config.get("recipe_id") == published_pre_recipe
            and config.get("fresh_published_pre_lineage") is True
            and isinstance(controlled_window_schedule, str)
            and bool(controlled_window_schedule)
            and isinstance(controlled_window_schedule_sha256, str)
            and bool(controlled_window_schedule_sha256)
        )
        published_pre_schedule_resume = (
            0 < start_update < 64
            and start_update % 4 == 0
            and str(start).startswith("SCHEDULE_")
            and config.get("recipe_id") == published_pre_recipe
            and config.get("fresh_published_pre_lineage") is True
            and config.get("published_pre_checkpoint_sha256") == published_pre_sha
            and isinstance(controlled_window_schedule, str)
            and bool(controlled_window_schedule)
            and isinstance(controlled_window_schedule_sha256, str)
            and bool(controlled_window_schedule_sha256)
            and str(start).startswith(
                f"SCHEDULE_{controlled_window_schedule_sha256[:12].upper()}_UPDATE_"
            )
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
        )
        controlled_schedule_binding: dict[str, Any] | None = None
        if fresh_published_pre_start:
            schedule_path = Path(str(controlled_window_schedule)).expanduser().resolve()
            try:
                schedule_bytes = schedule_path.read_bytes()
                observed_schedule_sha256 = hashlib.sha256(schedule_bytes).hexdigest()
                schedule_payload = json.loads(schedule_bytes)
            except (OSError, ValueError) as exc:
                raise ArtifactError(f"cannot read controlled window schedule: {exc}") from exc
            if observed_schedule_sha256 != controlled_window_schedule_sha256:
                raise ArtifactError("controlled window schedule SHA mismatch")
            from .modern_green_resident import _published_pre_controlled_schedule
            _rows, source_labels, windows_per_update, _membership = (
                _published_pre_controlled_schedule(schedule_payload, config)
            )
            controlled_schedule_binding = {
                "source_sha256": controlled_window_schedule_sha256,
                "source_row_labels": source_labels,
                "requested_boundaries": [1, 2, 3, 4],
                "windows_per_update": windows_per_update,
            }
        published_pre_resume = (
            start_update > 0
            and config.get("fresh_published_pre_lineage") is True
            and config.get("recipe_id") == published_pre_recipe
            and config.get("published_pre_checkpoint_sha256") == published_pre_sha
            and controlled_arm_id is None
        )
        exact_checkpointed_u21_resume = (
            single_gpu_full_surface
            and start_update == 21
            and start_meta.get("sha256")
                == "11df795d56d7f9210f20bb99e91b6518dc17d0e24cbfff6b96e120168ab64830"
            and config.get("checkpoint_sha256")
                == "11df795d56d7f9210f20bb99e91b6518dc17d0e24cbfff6b96e120168ab64830"
            and requested == (24,)
        )
        published_pre_crash_resume = (
            start_update == 10
            and str(start) == "SCHEDULE_E186B108124B_UPDATE_010"
            and published_pre_resume
        )
        if validation_proof:
            if config.get("validation_windows") != [28]:
                raise ArtifactError("resident validation proof requires validation_windows=[28]")
            if (
                start_update != 0
                or start_meta.get("sha256") != published_pre_sha
                or config.get("checkpoint_sha256") != published_pre_sha
                or config.get("published_pre_checkpoint_sha256") != published_pre_sha
                or config.get("recipe_id") != published_pre_recipe
            ):
                raise ArtifactError("resident validation proof must start from exact published PRE")
            if any(start_meta.get(name) is not None for name in (
                "optimizer", "optimizer_state", "scheduler", "scheduler_state",
            )):
                raise ArtifactError("published PRE validation proof requires fresh optimizer and scheduler state")
        elif published_pre_schedule_resume:
            pass
        elif published_pre_crash_resume:
            _validate_published_pre_crash_resume_start(
                str(start), start_update, start_meta, requested=requested, config=config
            )
        elif exact_checkpointed_u21_resume:
            _validate_published_pre_resume_start(
                start_update, start_meta, config=config
            )
        elif controlled_arm_id is None:
            if fresh_published_pre_start:
                if any(start_meta.get(name) is not None for name in (
                    "optimizer", "optimizer_state", "scheduler", "scheduler_state",
                )):
                    raise ArtifactError("published PRE start requires fresh optimizer and scheduler state")
            elif published_pre_resume:
                _validate_published_pre_resume_start(
                    start_update, start_meta, config=config
                )
            elif not 16 <= start_update < 64 or start_update % 4:
                raise ArtifactError("real two-Spark continuation must start from a scored four-update boundary U16..U60")
        else:
            controlled_config_sha256 = config.get("controlled_config_sha256")
            if not isinstance(controlled_config_sha256, str) or not controlled_config_sha256:
                raise ArtifactError("controlled arm requires controlled_config_sha256")
            _validate_controlled_arm_start(
                str(controlled_arm_id), start_update, start_meta,
                controlled_config_sha256=controlled_config_sha256,
            )
        valid_one_update_proof = validation_proof and requested == (start_update + 1,)
        valid_v7_lut_only_update = (
            config.get("v7_lut_only_update") is True
            and fresh_published_pre_start
            and requested == (1,)
            and not validation_proof
            and config.get("tailfix_wholesale") is not True
        )
        valid_single_gpu_full_surface_update = (
            single_gpu_full_surface
            and start_update == 20
            and start_meta.get("sha256")
                == "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
            and config.get("checkpoint_sha256")
                == "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
            and requested == (21,)
            and config.get("lr_scale") == 0.5
            and config.get("shared_optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity")
                == "exact U20 to U21; sole variable is single-GPU resident no-recompute execution backend"
            and not validation_proof
        )
        valid_fresh_pre_u1_u4 = fresh_published_pre_start and requested == (1, 2, 3, 4)
        valid_authenticated_u22_u26 = (
            published_pre_resume and start_update == 22 and requested == (26,)
        )
        valid_authenticated_u32_u35 = (
            published_pre_resume
            and start_update == 32
            and requested == (35,)
            and start_meta.get("sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and config.get("checkpoint_sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
        )
        diagnostic_pin = config.get("canonical_git_pin")
        valid_authenticated_u32_u33_first_divergence = (
            published_pre_resume
            and start_update == 32
            and requested == (33,)
            and start_meta.get("sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and config.get("checkpoint_sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and isinstance(config.get("transition_diagnostic_receipt"), str)
            and bool(config.get("transition_diagnostic_receipt"))
            and config.get("scientific_identity")
                == "exact healthy U32 to U33 instrumentation-only diagnostic; no optimizer/scheduler/recipe/arithmetic change"
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u32_u33_gradient_domain_candidate = (
            published_pre_resume
            and start_update == 32
            and requested == (33,)
            and start_meta.get("sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and config.get("checkpoint_sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("authorized_optimizer_arithmetic_repair")
                == "adam-fp64-state-and-update-arithmetic"
            and config.get("scientific_identity")
                == "exact healthy U32 to U33 candidate; sole change is existing-reference 2^-192 gradient domain with FP64 Adam state arithmetic"
            and not config.get("transition_diagnostic_receipt")
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u32_u33_internal_lut_control = (
            published_pre_resume
            and start_update == 32
            and requested == (33,)
            and start_meta.get("sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and config.get("checkpoint_sha256")
                == "4cef4a5619922ad970a4b28263a360f99674ae3f8abad497cd253974fd37d58a"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") in {
                "exact healthy U32 to U33 control; sole change is diagnostic taps inside existing grouped LUT backward",
                "exact healthy U32 to U33 control; sole change is diagnostic taps at returned grad_lut and LUT leaf pre/post accumulation",
                "exact healthy U32 to U33 candidate; sole change is explicit grouped backward stream completion",
            }
            and not config.get("transition_diagnostic_receipt")
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_032"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u33_u34_phase_profile_control = (
            published_pre_resume
            and start_update == 33
            and requested == (34,)
            and start_meta.get("sha256")
                == "0abdab68a393163993749a95b8cc6809f43b26e73cdc118ada1e9e58e725eff9"
            and config.get("checkpoint_sha256")
                == "0abdab68a393163993749a95b8cc6809f43b26e73cdc118ada1e9e58e725eff9"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U33 to U34 continuation; instrumentation only via existing "
                "phase timers/profiler markers"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_033"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_033"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u34_u35_current_stream_sync_candidate = (
            published_pre_resume
            and start_update == 34
            and requested == (35,)
            and start_meta.get("sha256")
                == "06ccaeac47c3ac6862db713d469c5da6007545e07cec760cdb0470e2e3ddb878"
            and config.get("checkpoint_sha256")
                == "06ccaeac47c3ac6862db713d469c5da6007545e07cec760cdb0470e2e3ddb878"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U34 to U35 continuation; sole change is device-wide to "
                "current-stream synchronization after grouped backward"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_034"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_034"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u35_u36_default_stream_event_candidate = (
            published_pre_resume
            and start_update == 35
            and requested == (36,)
            and start_meta.get("sha256")
                == "77cb4661aea34aba4aa46e446673fc58016f70a17b2cd8e2caaa5c3d864a70e6"
            and config.get("checkpoint_sha256")
                == "77cb4661aea34aba4aa46e446673fc58016f70a17b2cd8e2caaa5c3d864a70e6"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U35 to U36 continuation; sole change is nonblocking event "
                "ordering from grouped producer stream to default-stream gradient consumer"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_035"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_035"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u36_u37_lut_grid_parallel_candidate = (
            published_pre_resume
            and start_update == 36
            and requested == (37,)
            and start_meta.get("sha256")
                == "e62bdecb663ad7dda14dee3244f0da277093f87e4d49a9dae61a563863bc8802"
            and config.get("checkpoint_sha256")
                == "e62bdecb663ad7dda14dee3244f0da277093f87e4d49a9dae61a563863bc8802"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U36 to U37 continuation; sole change is CUDA grid-z "
                "parallelism across independent LUT-gradient output tiles"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_036"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_036"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u37_u40_continuation = (
            published_pre_resume
            and start_update == 37
            and requested == (38, 39, 40)
            and start_meta.get("sha256")
                == "6de85bc531022602c65b69ff4091e1e8f48102926d48158d6682843f9c7a6a6f"
            and config.get("checkpoint_sha256")
                == "6de85bc531022602c65b69ff4091e1e8f48102926d48158d6682843f9c7a6a6f"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U37 to U40 continuation; runtime retains CUDA grid-z "
                "parallelism with no scientific change"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_037"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_037"
            and config.get("lr_scale") == 0.09375
            and not validation_proof
        )
        valid_authenticated_u38_u39_bounded_partial = (
            published_pre_resume
            and start_update == 38
            and requested == (39,)
            and start_meta.get("sha256")
                == "f9b3c4ae3672d876e8c7c4c54138a7d72f67c6f5a9a450d9cd9562628748759b"
            and config.get("checkpoint_sha256")
                == "f9b3c4ae3672d876e8c7c4c54138a7d72f67c6f5a9a450d9cd9562628748759b"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("scientific_identity") == (
                "exact finite U38 to U39 continuation; bounded-partial grad-LUT is the only variable"
            )
            and isinstance(config.get("fast_k2_wrapper_source"), str)
            and isinstance(config.get("fast_k2_wrapper_sha256"), str)
            and isinstance(config.get("fast_k2_extension"), str)
            and isinstance(config.get("fast_k2_extension_sha256"), str)
            and isinstance(diagnostic_pin, str)
            and re.fullmatch(r"[0-9a-f]{40}", diagnostic_pin) is not None
            and config.get("canonical_code_commit") == diagnostic_pin
            and config.get("resume_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_038"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_CB124895B563_UPDATE_038"
            and not validation_proof
        )
        valid_authenticated_u40_u41_w56_repair = (
            published_pre_resume
            and start_update == 40
            and requested == (41,)
            and start_meta.get("sha256")
                == "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"
            and config.get("checkpoint_sha256")
                == "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("authorized_single_update_boundary_repair")
                == "pre-backward-underflow-removal-v1"
            and config.get("diagnostic_train_windows") == [56, 28]
            and config.get("train_windows") == [56, 28]
            and config.get("resume_checkpoint")
                == "SCHEDULE_E186B108124B_UPDATE_040"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_E186B108124B_UPDATE_040"
            and not validation_proof
        )
        valid_authenticated_u41_u45_continuation = (
            published_pre_resume
            and start_update == 41
            and requested == (45,)
            and start_meta.get("sha256")
                == "40544a550331b4e59b71bdea8b348832a254f94f3847ec33735a9de5bb7a1879"
            and config.get("checkpoint_sha256")
                == "40544a550331b4e59b71bdea8b348832a254f94f3847ec33735a9de5bb7a1879"
            and start_meta.get("optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("shared_optimizer_scheduler_lineage")
                == "fresh-published-pre-adam-lambdalr"
            and config.get("lr_scale") == 0.375091552734375
            and config.get("seed") == 1701
            and config.get("controlled_window_schedule_sha256")
                == "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351"
            and config.get("scientific_identity") == (
                "t_f76a1035 repair-A winner U41-to-U45 four-update continuation"
            )
            and config.get("u41_parent_checkpoint_sha256")
                == "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"
            and config.get("u41_repair_a_terminal_receipt_sha256_by_rank") == {
                "0": "8ba35d756f54b6b8e9d377d65d83e11b077a364fa9b22eeddf4728129ea36fcb",
                "1": "5d1c4df51d441d8c5cdf99fefc0c73242e351fa517cb6c296d471864f4e5b446",
            }
            and config.get("resume_checkpoint")
                == "SCHEDULE_E186B108124B_UPDATE_041"
            and config.get("optimizer_checkpoint")
                == "SCHEDULE_E186B108124B_UPDATE_041"
            and not validation_proof
        )
        if validation_proof and valid_fresh_pre_u1_u4:
            raise ArtifactError(
                "fresh published-PRE U1..U4 forbids resident_validation_proof pre/post scoring"
            )
        valid_milestones = (
            requested == tuple(sorted(set(requested)))
            and bool(requested)
            and all(value > start_update and value <= 64 and value % 4 == 0 for value in requested)
        )
        if not (
            valid_one_update_proof
            or valid_v7_lut_only_update
            or valid_single_gpu_full_surface_update
            or valid_fresh_pre_u1_u4
            or valid_authenticated_u22_u26
            or valid_authenticated_u32_u35
            or valid_authenticated_u32_u33_first_divergence
            or valid_authenticated_u32_u33_gradient_domain_candidate
            or valid_authenticated_u32_u33_internal_lut_control
            or valid_authenticated_u33_u34_phase_profile_control
            or valid_authenticated_u34_u35_current_stream_sync_candidate
            or valid_authenticated_u35_u36_default_stream_event_candidate
            or valid_authenticated_u36_u37_lut_grid_parallel_candidate
            or valid_authenticated_u37_u40_continuation
            or valid_authenticated_u38_u39_bounded_partial
            or valid_authenticated_u40_u41_w56_repair
            or valid_authenticated_u41_u45_continuation
            or valid_trainable_scale_candidate
            or published_pre_crash_resume
            or valid_milestones
        ):
            raise ArtifactError(
                "milestones must be authenticated fresh-PRE U22->U26, U32->U35, "
                "or exact diagnostic/repaired U32->U33 boundary, fresh published-PRE "
                "U1..U4, or ordered four-update boundaries after the loaded checkpoint "
                "through U64"
            )
        static_w28_gate = config.get("static_w28_gate")
        if static_w28_gate is not None:
            if not fresh_published_pre_start or requested != (1, 2, 3, 4):
                raise ArtifactError("static W28 gate requires fresh published-PRE U1..U4")
            expected_gate = {
                "windows": [28], "updates": [1, 2, 4],
                "red_kld": 0.20, "dead_kld": 0.28,
            }
            if static_w28_gate != expected_gate:
                raise ArtifactError("static W28 gate contract drift")
        start_sha = start_meta.get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if config.get("basis_sha256") != basis:
            raise ArtifactError("real two-Spark continuation basis SHA does not match artifact identity")
        if config.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("real two-Spark continuation checkpoint SHA does not bind to U16")
        assignment = config.get("layer_split")
        if not isinstance(assignment, Mapping):
            raise ArtifactError("real resident continuation requires an explicit layer_split")
        try:
            ranges = {int(key): tuple(int(item) for item in value) for key, value in assignment.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactError("layer_split must contain integer rank ranges") from exc
        if single_gpu_resident:
            if ranges != {0: (0, 42)}:
                raise ArtifactError("single-GPU resident layer_split must assign all 43 layers to rank 0")
        else:
            if set(ranges) != {0, 1} or any(len(value) != 2 for value in ranges.values()):
                raise ArtifactError("layer_split must explicitly assign both ranks")
            if any(lo < 0 or hi > 42 or lo > hi for lo, hi in ranges.values()):
                raise ArtifactError("layer_split must contain valid inclusive layer ranges")
            if set(range(ranges[0][0], ranges[0][1] + 1)) & set(range(ranges[1][0], ranges[1][1] + 1)):
                raise ArtifactError("layer_split ranks must be non-empty and disjoint")
            if set(range(ranges[0][0], ranges[0][1] + 1)) | set(range(ranges[1][0], ranges[1][1] + 1)) != set(range(43)):
                raise ArtifactError("layer_split must cover all 43 grouped-K2 layers")
        try:
            payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load U16 checkpoint for official resident continuation: {exc}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
            raise ArtifactError("U16 checkpoint must contain official resident trainable state")
        if valid_v7_lut_only_update:
            state = payload["state"]
            luts = state.get("luts") if isinstance(state, Mapping) else None
            requested_luts = config.get("trainable_luts")
            if not isinstance(luts, Mapping) or len(luts) != 43:
                raise ArtifactError("V7 LUT-only update requires all 43 admitted PlaneSources")
            if (
                not isinstance(requested_luts, list)
                or not requested_luts
                or any(not isinstance(name, str) or not name for name in requested_luts)
                or len(requested_luts) != len(set(requested_luts))
                or not set(requested_luts).issubset(luts)
            ):
                raise ArtifactError("V7 LUT-only trainable_luts must name admitted LUTs exactly")
            lut_lr = config.get("lut_lr")
            if isinstance(lut_lr, bool):
                raise ArtifactError("V7 LUT-only lut_lr must be finite and positive")
            try:
                lut_lr_value = float(str(lut_lr))
            except (TypeError, ValueError) as exc:
                raise ArtifactError("V7 LUT-only lut_lr must be finite and positive") from exc
            if not math.isfinite(lut_lr_value) or lut_lr_value <= 0.0:
                raise ArtifactError("V7 LUT-only lut_lr must be finite and positive")
        from .modern_green_resident import ModernGreenResidentEngine
        engine = ModernGreenResidentEngine(
            payload=payload, config=config, rank=rank, layer_ranges=ranges
        )
        tailfix_heldout_windows = config.get("tailfix_heldout_windows")
        tailfix_heldout_teacher_root = config.get("tailfix_heldout_teacher_root")
        tailfix_heldout_pre = None
        if config.get("tailfix_wholesale") is True:
            from .tailfix_wholesale import HELDOUT_WINDOWS

            if not isinstance(tailfix_heldout_windows, list) or tailfix_heldout_windows != list(HELDOUT_WINDOWS):
                engine.close()
                raise ArtifactError("tailfix wholesale heldout window identity drift")
            if not isinstance(tailfix_heldout_teacher_root, str) or not tailfix_heldout_teacher_root:
                engine.close()
                raise ArtifactError("tailfix wholesale requires tailfix_heldout_teacher_root")
            tailfix_heldout_pre = self.validate(
                engine, tailfix_heldout_windows, tailfix_heldout_teacher_root
            )
        pre_validation = None
        validation_windows = config.get("validation_windows")
        validation_teacher_root = config.get("validation_teacher_root")
        if validation_proof:
            if not isinstance(validation_windows, list) or not validation_windows:
                engine.close()
                raise ArtifactError("resident validation proof requires validation_windows")
            if not isinstance(validation_teacher_root, str) or not validation_teacher_root:
                engine.close()
                raise ArtifactError("resident validation proof requires validation_teacher_root")
            pre_validation = self.validate(engine, validation_windows, validation_teacher_root)
            expected_pre = config.get("expected_pre_validation")
            if expected_pre is not None:
                if not isinstance(expected_pre, Mapping):
                    engine.close()
                    raise ArtifactError("expected_pre_validation must be a mapping")
                expected_kld = expected_pre.get("kld_mean")
                expected_top1 = expected_pre.get("top1")
                observed_kld = pre_validation.get("kld_mean")
                observed_top1 = pre_validation.get("top1")
                if observed_kld != expected_kld or observed_top1 != expected_top1:
                    engine.close()
                    raise ArtifactError(
                        "published PRE validation mismatch before U1: "
                        f"observed kld_mean={observed_kld!r} top1={observed_top1!r}; "
                        f"expected kld_mean={expected_kld!r} top1={expected_top1!r}"
                    )
        rows: list[dict[str, Any]] = []
        scientific_red = None
        previous_sha = start_sha
        previous_identity_sha = start_meta.get("identity_sha256")
        previous_update = start_update
        execution_targets = (
            tuple(range(start_update + 1, requested[-1] + 1))
            if config.get("persist_every_update") is True
            else requested
        )
        for target_update in execution_targets:
            state, step_report, _engine_meta = engine.advance_to(
                target_update, gather_state=not validation_proof
            )
            persisted: Mapping[str, Any] | None = None
            persist_started = time.perf_counter()
            _record_engine_step_phase(
                engine, update=target_update, phase="persist", boundary="start"
            )
            if validation_proof:
                # Score the just-updated resident model before any all-gather,
                # CPU serialization, or checkpoint I/O. Durable persistence is
                # a separate resume-test concern, not part of this zero-reload rail.
                persisted = {
                    "checkpoint": f"RESIDENT_UPDATE_{target_update:03d}",
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "checkpoint_identity_sha256": None,
                    "state_sha256": None,
                }
            elif rank == 0:
                if state is None:
                    raise ArtifactError("rank0 official resident engine returned no merged state")
                persisted = self._persist_continuation_checkpoint(
                    target_update, state, step_report, parent_sha=previous_sha,
                    parent_identity_sha=previous_identity_sha, lineage=lineage, config=config,
                )
            if not validation_proof:
                persisted = engine.broadcast_persisted(persisted)
            _record_engine_step_phase(
                engine,
                update=target_update,
                phase="persist",
                boundary="complete",
                elapsed_seconds=time.perf_counter() - persist_started,
            )
            if not isinstance(persisted, Mapping):
                raise ArtifactError(f"official resident U{target_update} persistence broadcast missing")
            row = {
                "target_update": target_update,
                "checkpoint": persisted["checkpoint"],
                "checkpoint_path": persisted["checkpoint_path"],
                "checkpoint_sha256": persisted["checkpoint_sha256"],
                "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
                "state_sha256": persisted["state_sha256"],
                "parent_checkpoint_sha256": previous_sha,
                "parent_identity_sha256": previous_identity_sha,
                "optimizer_scheduler_lineage": lineage,
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
                "gradient_norm": step_report["gradient_norm"],
                "parameter_delta_norm": step_report["parameter_delta_norm"],
                "loss": step_report["loss"],
                "timings": step_report["timings"],
                "process_gpu_evidence": step_report["process_gpu_evidence"],
                "rank_reports": step_report["rank_reports"],
                "model_engine": step_report["model_engine"],
                "frozen_surfaces": step_report["frozen_surfaces"],
                "trainable_surfaces": step_report["trainable_surfaces"],
                "checkpoint_loaded": True,
                "immutable": not validation_proof,
                "resident_state_persisted": not validation_proof,
                "world_size": 1 if single_gpu_v7_lut_only else 2,
                "rank": rank,
                "stage_boundary": target_update in requested,
            }
            if fresh_published_pre_start:
                if controlled_schedule_binding is None:
                    raise ArtifactError("fresh PRE controlled schedule binding is missing")
                row["controlled_window_schedule_source_row"] = controlled_schedule_binding[
                    "source_row_labels"
                ][(target_update - 1) % len(controlled_schedule_binding["source_row_labels"])]
            if config.get("tailfix_wholesale") is True:
                assert isinstance(tailfix_heldout_windows, list)
                assert isinstance(tailfix_heldout_teacher_root, str)
                row["tailfix_objective"] = dict(
                    getattr(engine, "tailfix_loss_evidence", {})
                )
                row["tailfix_heldout"] = dict(
                    self.validate(
                        engine,
                        list(tailfix_heldout_windows),
                        str(tailfix_heldout_teacher_root),
                    )
                )
            rows.append(row)
            if static_w28_gate is not None and target_update in (1, 2, 4):
                static_score = engine.validate([28], str(config["teacher_root"]))
                row["static_w28"] = dict(static_score)
                observed_kld = float(static_score["kld_mean"])
                if observed_kld > 0.20:
                    scientific_red = {
                        "update": target_update,
                        "kld_mean": observed_kld,
                        "red_threshold": 0.20,
                        "dead_threshold": 0.28,
                        "dead": observed_kld > 0.28,
                    }
            if not validation_proof:
                previous_sha = str(persisted["checkpoint_sha256"])
                previous_identity_sha = str(persisted["checkpoint_identity_sha256"])
            previous_update = target_update
            if scientific_red is not None:
                break
        post_validation = None
        if validation_proof:
            post_validation = self.validate(engine, validation_windows, validation_teacher_root)
        result = {
            "schema": "resident-two-spark-real-continuation-v2",
            "status": "PASS",
            "start_checkpoint": start,
            "start_checkpoint_sha256": start_sha,
            "loaded_checkpoint_sha256": start_sha,
            "world_size": 1 if single_gpu_v7_lut_only else 2,
            "rank": rank,
            "selector": {"layer_split": {str(key): list(value) for key, value in ranges.items()}},
            "shared_optimizer_scheduler_lineage": lineage,
            "local_only": True,
            "model_engine": "official-ShardStudent-grouped-K2-FWHT-resident",
            "milestones": rows,
            "final_update": previous_update,
            "checkpoint_loaded": True,
        }
        if config.get("tailfix_wholesale") is True:
            if not isinstance(tailfix_heldout_pre, Mapping) or not isinstance(
                tailfix_heldout_windows, list
            ):
                raise ArtifactError("tailfix heldout baseline is missing")
            heldout_path = [dict(tailfix_heldout_pre)] + [
                dict(row["tailfix_heldout"]) for row in rows
            ]
            heldout_kld = [float(item["kld_mean"]) for item in heldout_path]
            result["tailfix_wholesale"] = {
                "heldout_windows": list(tailfix_heldout_windows),
                "heldout_path": heldout_path,
                "heldout_kld_mean": heldout_kld,
                "stepwise_improvement": all(
                    later < earlier
                    for earlier, later in zip(heldout_kld, heldout_kld[1:])
                ),
                "fresh_optimizer_scheduler": True,
            }
        if scientific_red is not None:
            result["status"] = "SCIENTIFIC_RED"
            result["scientific_red"] = scientific_red
        if fresh_published_pre_start or published_pre_schedule_resume:
            result["controlled_window_schedule_sha256"] = controlled_window_schedule_sha256
            if controlled_schedule_binding is not None:
                result["controlled_window_schedule_binding"] = controlled_schedule_binding
            result["lr_scale"] = float(config.get("lr_scale", 1.0))
        if validation_proof:
            if not isinstance(pre_validation, Mapping) or not isinstance(post_validation, Mapping):
                engine.close()
                raise ArtifactError("resident validation proof did not produce PRE and POST scores")
            pre_kld = float(pre_validation["kld_mean"])
            post_kld = float(post_validation["kld_mean"])
            result["validation"] = {
                "pre": dict(pre_validation),
                "post": dict(post_validation),
                "post_less_than_pre": post_kld < pre_kld,
                "delta_kld_post_minus_pre": post_kld - pre_kld,
                "same_process": (
                    pre_validation["runtime_counters"].get("trainer_object_id")
                    == post_validation["runtime_counters"].get("trainer_object_id")
                    == id(engine)
                ),
                "checkpoint_reloads": 0,
            }
        self._write_immutable_receipt(receipt_path, result)
        engine.close()
        return result

    def continue_two_spark(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        config: Mapping[str, Any],
        advance_fn,
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        raise ArtifactError("advance_fn fixture continuation is forbidden; use continue_two_spark_real")

        """Run an authorized resident two-rank continuation in memory.

        ``advance_fn(previous_state, target_update, config)`` is the only
        execution hook. The API owns validation, parent binding, milestone
        ordering, and immutable receipts; shell launchers and single-device
        fallbacks are intentionally rejected.
        """
        if not isinstance(config, Mapping):
            raise ArtifactError("two-Spark continuation config is required")
        if config.get("authorized_api") is not True:
            raise ArtifactError("two-Spark continuation requires authorized_api=True")
        if config.get("world_size") != 2:
            raise ArtifactError("two-Spark continuation requires world_size=2")
        rank = config.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("two-Spark continuation rank must be 0 or 1")
        if config.get("local_only") is not True:
            raise ArtifactError("two-Spark continuation requires local_only=True")
        if any(key in config for key in ("command", "launcher", "script", "remote", "subprocess")):
            raise ArtifactError("raw launcher fields are forbidden in two-Spark continuation")
        lineage = config.get("shared_optimizer_scheduler_lineage")
        if not isinstance(lineage, str) or not lineage:
            raise ArtifactError("shared_optimizer_scheduler_lineage is required")
        start = self.artifact.checkpoint_key(start_checkpoint)
        if self._checkpoint_update(start) != 16:
            raise ArtifactError("two-Spark continuation must start from U16")
        start_sha = self.artifact.manifest["checkpoints"][start].get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if config.get("basis_sha256") != basis:
            raise ArtifactError("two-Spark continuation basis SHA does not match artifact identity")
        if config.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("two-Spark continuation checkpoint SHA does not bind to U16")
        resident_keys = ("resident_model", "resident_planes", "resident_data", "resident_api_state", "resident_state")
        missing = [key for key in resident_keys if key not in config or config[key] is None]
        if missing:
            raise ArtifactError(f"resident two-Spark state is incomplete: {', '.join(missing)}")
        assignment = config.get("layer_split")
        replica_windows = config.get("disjoint_resident_replica_windows")
        if assignment is None and replica_windows is None:
            raise ArtifactError("two-Spark continuation requires layer_split or disjoint_resident_replica_windows")
        if assignment is not None:
            if not isinstance(assignment, Mapping):
                raise ArtifactError("layer_split must explicitly assign both ranks")
            try:
                normalized_keys = {int(key) for key in assignment}
            except (TypeError, ValueError) as exc:
                raise ArtifactError("layer_split must explicitly assign both ranks") from exc
            if normalized_keys != {0, 1}:
                raise ArtifactError("layer_split must explicitly assign both ranks")
            assigned: dict[int, set[int]] = {}
            for key, value in assignment.items():
                rank_key = int(key)
                if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, int) for item in value):
                    lo, hi = value
                    if hi < lo:
                        raise ArtifactError("layer_split ranges must be ascending")
                    assigned[rank_key] = set(range(lo, hi + 1))
                elif isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
                    assigned[rank_key] = set(value)
                else:
                    raise ArtifactError("layer_split values must be inclusive ranges or layer lists")
            if assigned[0] & assigned[1] or not assigned[0] or not assigned[1]:
                raise ArtifactError("layer_split ranks must be non-empty and disjoint")
            selector = {str(key): sorted(value) for key, value in assigned.items()}
        else:
            if not isinstance(replica_windows, Mapping):
                raise ArtifactError("disjoint_resident_replica_windows must assign both ranks")
            try:
                first = set(int(item) for item in (replica_windows.get(0) or replica_windows.get("0") or []))
                second = set(int(item) for item in (replica_windows.get(1) or replica_windows.get("1") or []))
            except (TypeError, ValueError) as exc:
                raise ArtifactError("resident replica windows must be integer lists") from exc
            if not first or not second or first & second:
                raise ArtifactError("resident replica windows must be non-empty and disjoint")
            if (first | second) - set(self.windows):
                raise ArtifactError("resident replica windows must be declared Balanced64 windows")
            selector = {"0": sorted(first), "1": sorted(second)}
        if not callable(advance_fn):
            raise ArtifactError("two-Spark continuation requires an in-memory advance_fn")
        requested = tuple(int(value) for value in milestones)
        if requested != tuple(sorted(set(requested))) or not requested or any(value not in (20, 32, 48, 64) for value in requested):
            raise ArtifactError("milestones must be an ordered subset of U20/U32/U48/U64")
        # A continuation is not a selector-only receipt. Load and bind the
        # actual U16 payload before invoking any callback. This prevents a
        # fixture callback from manufacturing a sub-second PASS from a
        # declared SHA alone.
        try:
            loaded_payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot load U16 checkpoint for continuation: {exc}") from exc
        if not isinstance(loaded_payload.get("state"), Mapping):
            raise ArtifactError("U16 checkpoint must contain mapping state for resident continuation")
        loaded_state = dict(loaded_payload["state"])
        loaded_state_sha = self._state_fingerprint(loaded_payload)
        state = config["resident_state"]
        if not isinstance(state, Mapping):
            raise ArtifactError("resident_state must be a mapping")
        state = dict(state)
        resident_state_sha = config.get("resident_state_sha256")
        if resident_state_sha != loaded_state_sha:
            raise ArtifactError("resident_state_sha256 does not bind resident state to loaded U16 checkpoint")
        if self._state_fingerprint({"state": state}) != self._state_fingerprint({"state": loaded_state}):
            raise ArtifactError("resident_state does not match loaded U16 checkpoint state")
        rows = []
        previous_sha = start_sha
        previous_update = 16
        for target_update in requested:
            step_delta = target_update - previous_update
            next_state = advance_fn(state, target_update, config)
            step_report = None
            if isinstance(next_state, tuple) and len(next_state) == 2 and isinstance(next_state[0], Mapping):
                next_state, step_report = next_state
            if not isinstance(next_state, Mapping):
                raise ArtifactError(f"advance_fn did not return mapping state for U{target_update}")
            if not isinstance(step_report, Mapping):
                raise ArtifactError("advance_fn fixture rejected: return (state, step_report) with real resident optimizer steps")
            if step_report.get("checkpoint_loaded"):
                raise ArtifactError("advance_fn attempted checkpoint loading; continuation must remain resident")
            if step_report.get("resident_optimizer_step") is not True:
                raise ArtifactError("advance_fn did not prove a resident optimizer step")
            if step_report.get("optimizer_steps") != step_delta or step_report.get("scheduler_steps") != step_delta:
                raise ArtifactError(f"advance_fn step report must contain {step_delta} optimizer and scheduler steps")
            state = dict(next_state)
            state_sha = self._state_fingerprint({"state": state})
            previous_identity_sha = self.artifact.manifest["checkpoints"][start].get("identity_sha256") if previous_update == 16 else self.artifact.manifest["checkpoints"].get(f"UPDATE_{previous_update:03d}", {}).get("identity_sha256")
            persisted = self._persist_continuation_checkpoint(
                target_update,
                state,
                step_report,
                parent_sha=previous_sha,
                parent_identity_sha=previous_identity_sha,
                lineage=lineage,
                config=config,
            )
            rows.append({
                "target_update": target_update,
                "checkpoint": persisted["checkpoint"],
                "checkpoint_path": persisted["checkpoint_path"],
                "artifact_root": persisted["artifact_root"],
                "parent_checkpoint_sha256": persisted["parent_checkpoint_sha256"],
                "parent_identity_sha256": persisted["parent_identity_sha256"],
                "checkpoint_sha256": persisted["checkpoint_sha256"],
                "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
                "state_sha256": persisted["state_sha256"],
                "optimizer_scheduler_lineage": lineage,
                "optimizer_state": persisted["optimizer_state"],
                "scheduler_state": persisted["scheduler_state"],
                "world_size": 2,
                "rank": rank,
                "next_update": target_update,
                "immutable": True,
                "checkpoint_loaded": True,
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
            })
            previous_sha = persisted["checkpoint_sha256"]
            previous_update = target_update
        result = {
            "schema": "resident-two-spark-continuation-v1",
            "status": "PASS",
            "start_checkpoint": start,
            "start_checkpoint_sha256": start_sha,
            "world_size": 2,
            "rank": rank,
            "selector": {"layer_split": selector} if assignment is not None else {"disjoint_resident_replica_windows": selector},
            "shared_optimizer_scheduler_lineage": lineage,
            "local_only": True,
            "resident_state": {"model": True, "planes": True, "data": True, "api": True},
            "milestones": rows,
            "final_update": previous_update,
            "checkpoint_loaded": True,
            "loaded_checkpoint_sha256": start_sha,
            "loaded_checkpoint_state_sha256": loaded_state_sha,
        }
        self._write_immutable_receipt(receipt_path, result)
        return result

    # Explicit descriptive alias for callers that prefer the full surface name.
    continue_resident_two_spark = continue_two_spark


def resume_compare(
    api_or_root: ResidentRepairAPI | str | Path,
    resume_checkpoint: int | str,
    scratch_checkpoint: int | str,
    *,
    windows: Iterable[int] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the resume-vs-scratch experiment contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.resume_compare(
        resume_checkpoint,
        scratch_checkpoint,
        windows=windows,
        receipt_path=receipt_path,
    )


def continue_to(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    target: int | str,
    *,
    windows: Iterable[int] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the U16-to-U64 continuation contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_to(
        start_checkpoint,
        target,
        windows=windows,
        receipt_path=receipt_path,
    )


def construct_clean_u0(
    api_or_root: ResidentRepairAPI | str | Path,
    midpoint: int | str,
    target: int | str,
    *,
    replay: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the true in-memory clean-U0 replay contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.construct_clean_u0(midpoint, target, replay=replay, receipt_path=receipt_path)


def continue_two_spark_real(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    milestones: Iterable[int],
    *,
    config: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the non-injectable real continuation engine."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_two_spark_real(start_checkpoint, milestones, config=config, receipt_path=receipt_path)


def continue_two_spark(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    milestones: Iterable[int],
    *,
    config: Mapping[str, Any],
    advance_fn,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the authorized resident two-Spark contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_two_spark(start_checkpoint, milestones, config=config, advance_fn=advance_fn, receipt_path=receipt_path)


continue_resident_two_spark = continue_two_spark
