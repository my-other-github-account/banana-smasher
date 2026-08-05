"""Public bs-pack, teacher-bank, and paired-evaluation APIs."""

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

__all__ = [
    "MANIFEST_NAME",
    "PackValidationError",
    "AnchorEvaluationError",
    "UpdateQueue",
    "aggregate_scores",
    "build_bank",
    "build_bank_manifest",
    "compare_training_rails",
    "create_balanced_subset",
    "emit_solver_row",
    "evaluate_paired",
    "export_pack",
    "import_producer",
    "load_manifest",
    "materialize_candidate_producer",
    "materialize_bank",
    "materialize_fixed_d4",
    "persist_fixed_d4_solve",
    "prepare_fixed_d4_solve_config",
    "produce_fixed_d4_layerwise_logits",
    "produce_fixed_d4_logits",
    "register_bank",
    "resolve_bank_identities",
    "score_bank",
    "serve_persistent_updates",
    "solve_fixed_d4_exact",
    "status_report",
    "validate_bank_manifest",
    "verify_bank",
    "verify_evaluation",
    "verify_fixed_d4_model",
    "verify_pack",
]
