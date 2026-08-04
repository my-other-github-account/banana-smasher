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
from .bank import build_bank, verify_bank
from .evaluate import evaluate_paired, verify_evaluation
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
    "UpdateQueue",
    "build_bank",
    "evaluate_paired",
    "export_pack",
    "load_manifest",
    "serve_persistent_updates",
    "solve_qtip_profiles",
    "verify_bank",
    "verify_evaluation",
    "verify_pack",
]
