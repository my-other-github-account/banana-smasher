"""Hard production fence against non-resident scoring entry points."""
from __future__ import annotations

from .balanced64 import ArtifactError


def reject_standalone_score_runner(entrypoint: str) -> None:
    """Refuse historical builder/rail scorers in every production context.

    Candidate builders may still exist as sealed scientific references, but a
    production score must enter through ``ResidentRepairAPI.score``.  Keeping
    this as an unconditional exception prevents a flag or environment variable
    from silently reviving layer streaming.
    """
    raise ArtifactError(
        f"standalone production scorer {entrypoint!r} is forbidden; "
        "route through ResidentRepairAPI.score(), which is resident_in_memory "
        "by default and requires timed_score_file_reads=0"
    )
