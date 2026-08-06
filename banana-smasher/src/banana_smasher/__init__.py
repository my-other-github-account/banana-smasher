"""Public bs-pack, teacher-bank, and paired-evaluation APIs."""

from .contract import (
    MANIFEST_NAME,
    PackValidationError,
    export_pack,
    load_manifest,
    verify_pack,
)
from .bank import build_bank, verify_bank
from .evaluate import evaluate_paired, verify_evaluation
from .persistent import UpdateQueue
from .backpack_contextual import (
    run_contextual_trust_solve,
    run_contextual_value_update,
)
from .backpack_contextual_prepare import prepare_contextual_iteration
from .backpack_contextual_measure import record_contextual_swap_measurement
from .backpack_contextual_candidate import materialize_contextual_change
from .backpack_selection import select_measured_nonworse
from .locality import require_local_backpack_inputs
from .staging import stage_qsfp_manifest
from .update_service import serve_persistent_updates

__all__ = [
    "MANIFEST_NAME",
    "PackValidationError",
    "UpdateQueue",
    "build_bank",
    "evaluate_paired",
    "export_pack",
    "load_manifest",
    "materialize_contextual_change",
    "prepare_contextual_iteration",
    "record_contextual_swap_measurement",
    "require_local_backpack_inputs",
    "run_contextual_trust_solve",
    "run_contextual_value_update",
    "select_measured_nonworse",
    "serve_persistent_updates",
    "stage_qsfp_manifest",
    "verify_bank",
    "verify_evaluation",
    "verify_pack",
]
