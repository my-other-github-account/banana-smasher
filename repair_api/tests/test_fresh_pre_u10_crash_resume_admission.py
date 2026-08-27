from copy import deepcopy
import pathlib

import pytest

from repair_api.api import _validate_published_pre_crash_resume_start
from repair_api.balanced64 import ArtifactError


U10_SHA256 = "055f015f88c44f9092423a7e45525e3699d217d1d0b8b36eb269947915f17658"
U10_STATE_SHA256 = "7e19879e4b526793c4837a81f4fc3658a00980a2ebf4252b9a23c0d7da9021a6"
SCHEDULE_SHA256 = "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351"


def _meta() -> dict:
    return {
        "sha256": U10_SHA256,
        "state_sha256": U10_STATE_SHA256,
        "next_update": 10,
        "rank_provenance": [0, 1],
        "world_size": 2,
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
    }


def _config() -> dict:
    return {
        "checkpoint_sha256": U10_SHA256,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256":
            "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
        "fresh_published_pre_lineage": True,
        "controlled_window_schedule_sha256": SCHEDULE_SHA256,
        "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
    }


def test_exact_paired_u10_state_admits_only_u11_u12_crash_resume() -> None:
    proof = _validate_published_pre_crash_resume_start(
        "SCHEDULE_E186B108124B_UPDATE_010",
        10,
        _meta(),
        requested=(11, 12),
        config=_config(),
    )

    assert proof == {
        "checkpoint_sha256": U10_SHA256,
        "next_incomplete_update": 11,
        "rank_provenance": [0, 1],
        "state_sha256": U10_STATE_SHA256,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank_provenance", [0]),
        ("optimizer_steps", 0),
        ("scheduler_steps", 0),
        ("state_sha256", "0" * 64),
        ("optimizer_scheduler_lineage", "drift"),
    ],
)
def test_u10_crash_resume_rejects_missing_or_mismatched_state(field: str, value: object) -> None:
    meta = deepcopy(_meta())
    meta[field] = value

    with pytest.raises(ArtifactError, match="authenticated paired U10"):
        _validate_published_pre_crash_resume_start(
            "SCHEDULE_E186B108124B_UPDATE_010",
            10,
            meta,
            requested=(11, 12),
            config=_config(),
        )


def test_u10_crash_resume_rejects_replay_or_schedule_drift() -> None:
    with pytest.raises(ArtifactError, match="U11,U12 only"):
        _validate_published_pre_crash_resume_start(
            "SCHEDULE_E186B108124B_UPDATE_010",
            10,
            _meta(),
            requested=(10, 11, 12),
            config=_config(),
        )

    drifted = _config()
    drifted["controlled_window_schedule_sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="schedule identity"):
        _validate_published_pre_crash_resume_start(
            "SCHEDULE_E186B108124B_UPDATE_010",
            10,
            _meta(),
            requested=(11, 12),
            config=drifted,
        )


def test_public_api_routes_u10_through_crash_resume_admission() -> None:
    source = (pathlib.Path(__file__).resolve().parents[1] / "api.py").read_text()
    assert "published_pre_crash_resume" in source
    assert "_validate_published_pre_crash_resume_start(" in source
