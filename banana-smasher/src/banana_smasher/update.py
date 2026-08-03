"""Public physical-update APIs.

The package owns durable segmented updates, persistent exactly-once queueing,
explicit token semantics, and checkpoint relocation authorization. Accelerator
execution remains an injected callable and is never replaced by a fallback.
"""

from .checkpoint_rebind import (
    authorize_checkpoint_identity_rebind,
    build_checkpoint_identity_rebind_receipt,
    validate_checkpoint_identity_rebind,
)
from .persistent import (
    DuplicateSegment,
    IdentityMismatch,
    SegmentStateConflict,
    UpdateQueue,
    recover_committed_cycle,
    serve_queue,
)
from .update_engine import run_segmented_update
from .update_inputs import PhysicalBatch, prepare_physical_batch

__all__ = [
    "DuplicateSegment",
    "IdentityMismatch",
    "PhysicalBatch",
    "SegmentStateConflict",
    "UpdateQueue",
    "authorize_checkpoint_identity_rebind",
    "build_checkpoint_identity_rebind_receipt",
    "prepare_physical_batch",
    "recover_committed_cycle",
    "run_segmented_update",
    "serve_queue",
    "validate_checkpoint_identity_rebind",
]
