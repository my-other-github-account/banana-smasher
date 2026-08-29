"""Focused tests for L028 SU/SV ordered continuation admission and phase teardown.

The first expert-plane canary is rooted at the published PRE (U0) and runs exactly
one U0->U1 update.  Continuing that same authenticated expansion from a checkpoint
it sealed itself is the same experiment, not a new root, so the API admits it only
when the parent chain walks contiguously back to U0 under one unchanged
optimizer/scheduler lineage.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from banana_smasher.resident_balanced64 import ArtifactError
from banana_smasher.resident_continuation import ModernGreenResidentEngine
from banana_smasher.resident_proven_api import ResidentRepairAPI

LINEAGE = "pre-f9bffe04-l028-su-sv"


def _api(checkpoints: dict[str, dict[str, object]]) -> ResidentRepairAPI:
    api = ResidentRepairAPI.__new__(ResidentRepairAPI)
    api.artifact = SimpleNamespace(manifest={"checkpoints": checkpoints})
    return api


def _sealed(update: int, *, parent_sha: str | None, lineage: str = LINEAGE) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": f"checkpoints/UPDATE_{update:03d}.pt",
        "sha256": f"sha-{update:03d}",
        "identity_sha256": f"id-{update:03d}",
        "next_update": update,
        "checkpoint_loaded": True,
        "fixture": False,
        "world_size": 2,
        "optimizer_scheduler_lineage": lineage,
    }
    if parent_sha is not None:
        entry["parent_sha256"] = parent_sha
    return entry


def _pre_root() -> dict[str, object]:
    return {
        "path": "checkpoints/UPDATE_000.pt",
        "sha256": "sha-000",
        "identity_sha256": "id-000",
        "next_update": 0,
        "checkpoint_loaded": True,
        "fixture": False,
        "world_size": 2,
        "optimizer_scheduler_lineage": LINEAGE,
    }


def test_sealed_u1_resumes_and_reports_pre_rooted_ancestry() -> None:
    api = _api(
        {
            "UPDATE_000": _pre_root(),
            "UPDATE_001": _sealed(1, parent_sha="sha-000"),
        }
    )

    admission = api._expert_plane_continuation_start("UPDATE_001", 1, lineage=LINEAGE)

    assert admission["schema"] == "resident-expert-plane-continuation-admission-v1"
    assert admission["pre_root_checkpoint"] == "UPDATE_000"
    assert admission["pre_root_sha256"] == "sha-000"
    assert admission["optimizer_scheduler_lineage"] == LINEAGE
    assert admission["ancestry"] == ["UPDATE_000", "UPDATE_001"]


def test_deeper_ordered_chain_through_u4_is_admissible() -> None:
    api = _api(
        {
            "UPDATE_000": _pre_root(),
            "UPDATE_001": _sealed(1, parent_sha="sha-000"),
            "UPDATE_002": _sealed(2, parent_sha="sha-001"),
            "UPDATE_003": _sealed(3, parent_sha="sha-002"),
        }
    )

    admission = api._expert_plane_continuation_start("UPDATE_003", 3, lineage=LINEAGE)

    assert admission["ancestry"] == [
        "UPDATE_000",
        "UPDATE_001",
        "UPDATE_002",
        "UPDATE_003",
    ]


def test_broken_parent_binding_is_rejected() -> None:
    api = _api(
        {
            "UPDATE_000": _pre_root(),
            "UPDATE_001": _sealed(1, parent_sha="sha-000"),
            "UPDATE_002": _sealed(2, parent_sha="foreign-sha"),
        }
    )

    with pytest.raises(ArtifactError, match="parent SHA does not bind"):
        api._expert_plane_continuation_start("UPDATE_002", 2, lineage=LINEAGE)


def test_lineage_swap_is_rejected() -> None:
    api = _api(
        {
            "UPDATE_000": _pre_root(),
            "UPDATE_001": _sealed(1, parent_sha="sha-000", lineage="some-other-lineage"),
        }
    )

    with pytest.raises(ArtifactError, match="one unchanged optimizer/scheduler lineage"):
        api._expert_plane_continuation_start("UPDATE_001", 1, lineage=LINEAGE)


def test_missing_ancestor_is_rejected() -> None:
    api = _api({"UPDATE_002": _sealed(2, parent_sha="sha-001")})

    with pytest.raises(ArtifactError, match="parent UPDATE_001 is absent"):
        api._expert_plane_continuation_start("UPDATE_002", 2, lineage=LINEAGE)


def test_fixture_ancestor_is_rejected() -> None:
    entry = _sealed(1, parent_sha="sha-000")
    entry["fixture"] = True
    api = _api({"UPDATE_000": _pre_root(), "UPDATE_001": entry})

    with pytest.raises(ArtifactError, match="not a real loaded continuation checkpoint"):
        api._expert_plane_continuation_start("UPDATE_001", 1, lineage=LINEAGE)


def test_single_rank_ancestor_is_rejected() -> None:
    entry = _sealed(1, parent_sha="sha-000")
    entry["world_size"] = 1
    api = _api({"UPDATE_000": _pre_root(), "UPDATE_001": entry})

    with pytest.raises(ArtifactError, match="not produced by the two-rank pipeline"):
        api._expert_plane_continuation_start("UPDATE_001", 1, lineage=LINEAGE)


def test_u16_lineage_start_without_pre_root_is_rejected() -> None:
    """A U16-rooted chain has no U0 ancestor and must never be admitted here."""
    api = _api(
        {
            "UPDATE_016": _sealed(16, parent_sha="u15-sha"),
            "UPDATE_017": _sealed(17, parent_sha="sha-016"),
        }
    )

    with pytest.raises(ArtifactError, match="UPDATE_015 is absent"):
        api._expert_plane_continuation_start("UPDATE_017", 17, lineage=LINEAGE)


def test_engine_close_requires_keyword_phase() -> None:
    """The API's teardown call sites must pass the keyword-only phase argument."""
    signature = inspect.signature(ModernGreenResidentEngine.close)
    phase = signature.parameters["phase"]

    assert phase.kind is inspect.Parameter.KEYWORD_ONLY
    assert phase.default is inspect.Parameter.empty


def test_no_bare_engine_close_call_sites_remain() -> None:
    from pathlib import Path

    import banana_smasher.resident_proven_api as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "engine.close()" not in text
    assert 'engine.close(phase="repair_train")' in text
    assert 'engine.close(phase="diagnostic_zero_update")' in text
