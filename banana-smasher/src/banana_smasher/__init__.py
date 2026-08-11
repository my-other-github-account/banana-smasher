"""Public bs-pack, teacher-bank, and paired-evaluation APIs."""

from pathlib import Path
from collections.abc import Sequence
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
from .bpw import (
    BPW_ACCOUNTING_SCHEMA,
    BpwAccountingError,
    build_bpw_accounting,
    require_comparable_bpw,
    verify_bpw_accounting,
)
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
from .backpack_providers import (
    BQ23_PROVIDER_IDS,
    BackpackFamilyActivation,
    BackpackFamilyProvider,
    BackpackWireArtifact,
    BackpackWirePrice,
    backpack_provider_from_declaration,
    bq23_backpack_family_providers,
    bind_native_mxfp4_backpack_candidate,
    builtin_backpack_family_providers,
    fixed_d4_backpack_provider,
    generate_backpack_candidate,
    materialize_backpack_assignment,
    native_mxfp4_backpack_provider,
    price_backpack_candidate,
    predict_backpack_candidate,
    qtip_ring_backpack_provider,
    resolve_backpack_family_provider,
    vector_vq_backpack_provider,
    verify_backpack_candidate,
)
from .backpack_contextual import (
    ContextualValuationError,
    build_contextual_delta_ledger,
    run_contextual_trust_solve,
    run_contextual_value_update,
    solve_contextual_trust_region,
)
from .backpack_contextual_candidate import materialize_contextual_change
from .backpack_contextual_measure import record_contextual_swap_measurement
from .backpack_contextual_prepare import prepare_contextual_iteration
from .backpack_exact64 import EXACT64_TERMINAL_SCHEMA, bind_backpack_exact64
from .backpack_runtime_exact64 import run_backpack_exact64
from .backpack_selection import select_measured_nonworse
from .backpack_virtual import materialize_virtual_backpack, verify_virtual_backpack
from .locality import require_local_backpack_inputs, require_local_path
from .staging import stage_qsfp_manifest
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
from .qtip1 import (
    EncodedQtip,
    QTIP1_GEOMETRY,
    QTIP2_GEOMETRY,
    QtipGeometry,
    QtipProviderComponent,
    QtipProviderDeclaration,
    QtipWireConsumer,
    assign_qtip_provider_components,
    decode_qtip,
    encode_qtip,
    gaussian_tlut,
    pack_qtip_states,
    qtip1_5_provider_declaration,
    qtip1_provider_declaration,
    qtip_provider_counts,
    unpack_qtip_states,
    verify_qtip_wire,
    write_encoded_qtip_wire,
    write_qtip_wire,
)
from .qtip25_native_v4_api import (
    anchor_qtip25_native_v4_cell,
    build_qtip25_native_v4_cell,
)
from .qtip_v7_batch import produce_qtip2_v7_batch10
from .update_service import serve_persistent_updates


def solve_qtip_profiles(
    config_root: str | Path,
    root: str | Path,
    layer: int,
    *,
    batch_size: int = 1,
    config_paths: Sequence[str | Path] | None = None,
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
        config_paths=(
            [Path(path) for path in config_paths] if config_paths is not None else None
        ),
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
    "BQ23_PROVIDER_IDS",
    "BPW_ACCOUNTING_SCHEMA",
    "BpwAccountingError",
    "PackValidationError",
    "AnchorEvaluationError",
    "BackpackFamilyActivation",
    "BackpackFamilyProvider",
    "BackpackPlan",
    "BackpackWireArtifact",
    "BackpackWirePrice",
    "ContextualValuationError",
    "EXACT64_TERMINAL_SCHEMA",
    "EncodedQtip",
    "QTIP1_GEOMETRY",
    "QTIP2_GEOMETRY",
    "QtipGeometry",
    "QtipProviderComponent",
    "QtipProviderDeclaration",
    "QtipWireConsumer",
    "UpdateQueue",
    "aggregate_scores",
    "anchor_backpack",
    "anchor_backpack_candidates",
    "anchor_qtip25_native_v4_cell",
    "assign_qtip_provider_components",
    "backpack_provider_from_declaration",
    "bq23_backpack_family_providers",
    "bind_native_mxfp4_backpack_candidate",
    "bind_backpack_exact64",
    "build_bank",
    "build_bank_manifest",
    "build_bpw_accounting",
    "build_backpack",
    "build_qtip25_native_v4_cell",
    "build_contextual_delta_ledger",
    "builtin_backpack_family_providers",
    "candidate_artifact_root",
    "compare_training_rails",
    "create_balanced_subset",
    "decode_qtip",
    "emit_solver_row",
    "encode_qtip",
    "evaluate_paired",
    "export_backpack_lifecycle",
    "export_pack",
    "fixed_d4_backpack_provider",
    "generate_backpack_candidate",
    "generate_backpack_candidates",
    "generate_qtip_backpack_candidate",
    "generate_vector_vq_backpack_candidate",
    "gaussian_tlut",
    "import_producer",
    "inspect_backpack",
    "load_manifest",
    "materialize_backpack_assignment",
    "materialize_backpack_source",
    "materialize_contextual_change",
    "materialize_virtual_backpack",
    "materialize_bank",
    "materialize_candidate_producer",
    "materialize_fixed_d4",
    "native_mxfp4_backpack_provider",
    "pack_qtip_states",
    "persist_fixed_d4_solve",
    "prepare_fixed_d4_solve_config",
    "price_backpack_candidate",
    "predict_backpack",
    "predict_backpack_candidate",
    "prepare_contextual_iteration",
    "produce_fixed_d4_layerwise_logits",
    "produce_fixed_d4_logits",
    "produce_qtip2_v7_batch10",
    "qtip1_5_provider_declaration",
    "qtip1_provider_declaration",
    "qtip_provider_counts",
    "qtip_ring_backpack_provider",
    "register_bank",
    "require_comparable_bpw",
    "repair_backpack",
    "record_contextual_swap_measurement",
    "resolve_bank_identities",
    "resolve_backpack_family_provider",
    "reuse_backpack_receipts",
    "require_local_backpack_inputs",
    "require_local_path",
    "run_contextual_trust_solve",
    "run_contextual_value_update",
    "run_backpack_exact64",
    "score_bank",
    "score_backpack",
    "serve_persistent_updates",
    "solve_backpack",
    "solve_contextual_trust_region",
    "solve_fixed_d4_exact",
    "solve_qtip_profiles",
    "status_backpack",
    "status_report",
    "stage_qsfp_manifest",
    "select_measured_nonworse",
    "unpack_qtip_states",
    "validate_bank_manifest",
    "vector_vq_backpack_provider",
    "verify_bank",
    "verify_bpw_accounting",
    "verify_evaluation",
    "verify_backpack_candidate",
    "verify_virtual_backpack",
    "verify_fixed_d4_model",
    "verify_pack",
    "verify_qtip_wire",
    "write_encoded_qtip_wire",
    "write_qtip_wire",
]
