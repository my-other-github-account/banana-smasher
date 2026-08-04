"""Public bs-pack, teacher-bank, and paired-evaluation APIs."""

from pathlib import Path
from typing import Any

from .contract import (
    MANIFEST_NAME,
    PackValidationError,
    export_pack,
    load_manifest,
    verify_pack,
)
from .anchor import (
    AnchorEvaluationError,
    aggregate_scores,
    build_bank_manifest,
    compare_training_rails,
    create_balanced_subset,
    emit_solver_row,
    import_producer,
    materialize_candidate_producer,
    materialize_bank,
    register_bank,
    resolve_bank_identities,
    score_bank,
    status_report,
    validate_bank_manifest,
)
from .bank import build_bank, verify_bank
from .backpack import (
    BackpackPlan,
    anchor_backpack,
    anchor_backpack_candidates,
    build_backpack,
    candidate_artifact_root,
    export_backpack_lifecycle,
    generate_backpack_candidates,
    generate_qtip_backpack_candidate,
    generate_vector_vq_backpack_candidate,
    inspect_backpack,
    materialize_backpack_source,
    predict_backpack,
    repair_backpack,
    reuse_backpack_receipts,
    score_backpack,
    solve_backpack,
    status_backpack,
)
from .evaluate import evaluate_paired, verify_evaluation
from .fixed_d4 import (
    materialize_fixed_d4,
    persist_fixed_d4_solve,
    prepare_fixed_d4_solve_config,
    produce_fixed_d4_layerwise_logits,
    produce_fixed_d4_logits,
    solve_fixed_d4_exact,
    verify_fixed_d4_model,
)
from .persistent import UpdateQueue
from .update_service import serve_persistent_updates


def solve_qtip_profiles(
    config_root: str | Path,
    root: str | Path,
    layer: int,
    *,
    batch_size: int = 1,
    limit: int | None = None,
    tier: str | None = None,
    all_cells: bool = False,
    profile_mode: bool = False,
    resume: bool = True,
    resume_flag_explicit: bool = False,
    kernel_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Solve a discoverable QTIP config set through one resident process.

    ``batch_size > 1`` selects the exact K2/full16 cross-unit build path and
    fails closed rather than falling back to serial unit solves.
    """
    from .solver_qtip_profile import main_many

    return main_many(
        Path(config_root),
        Path(root),
        int(layer),
        batch_size=batch_size,
        limit=limit,
        tier=tier,
        all_cells=all_cells,
        profile_mode=profile_mode,
        resume=resume,
        resume_flag_explicit=resume_flag_explicit,
        kernel_cache_root=(
            Path(kernel_cache_root) if kernel_cache_root is not None else None
        ),
    )

__all__ = [
    "MANIFEST_NAME",
    "PackValidationError",
    "AnchorEvaluationError",
    "UpdateQueue",
    "aggregate_scores",
    "anchor_backpack",
    "anchor_backpack_candidates",
    "build_bank",
    "build_bank_manifest",
    "build_backpack",
    "BackpackPlan",
    "candidate_artifact_root",
    "compare_training_rails",
    "create_balanced_subset",
    "emit_solver_row",
    "evaluate_paired",
    "export_pack",
    "export_backpack_lifecycle",
    "generate_backpack_candidates",
    "generate_qtip_backpack_candidate",
    "generate_vector_vq_backpack_candidate",
    "import_producer",
    "inspect_backpack",
    "load_manifest",
    "materialize_backpack_source",
    "materialize_candidate_producer",
    "materialize_bank",
    "materialize_fixed_d4",
    "persist_fixed_d4_solve",
    "prepare_fixed_d4_solve_config",
    "predict_backpack",
    "produce_fixed_d4_layerwise_logits",
    "produce_fixed_d4_logits",
    "register_bank",
    "repair_backpack",
    "reuse_backpack_receipts",
    "resolve_bank_identities",
    "score_bank",
    "score_backpack",
    "serve_persistent_updates",
    "solve_fixed_d4_exact",
    "solve_backpack",
    "solve_qtip_profiles",
    "status_backpack",
    "status_report",
    "validate_bank_manifest",
    "verify_bank",
    "verify_evaluation",
    "verify_fixed_d4_model",
    "verify_pack",
]
