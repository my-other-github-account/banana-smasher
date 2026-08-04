"""Shared bs-pack v1 contract, exporter, validator, repacker, and loader."""

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
from .fixed_d4 import materialize_fixed_d4

__all__ = [
    "MANIFEST_NAME",
    "PackValidationError",
    "AnchorEvaluationError",
    "aggregate_scores",
    "build_bank_manifest",
    "compare_training_rails",
    "create_balanced_subset",
    "emit_solver_row",
    "export_pack",
    "import_producer",
    "load_manifest",
    "materialize_candidate_producer",
    "materialize_bank",
    "materialize_fixed_d4",
    "register_bank",
    "resolve_bank_identities",
    "score_bank",
    "status_report",
    "validate_bank_manifest",
    "verify_pack",
]
