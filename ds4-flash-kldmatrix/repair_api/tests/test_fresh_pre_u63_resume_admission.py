import pytest

from repair_api.api import _validate_published_pre_resume_start
from repair_api.balanced64 import ArtifactError


U63_SHA256 = "ed05a36fbb71fc9818142e060bd013378f639386592f1b638b6c5d3da04227a3"


def _config() -> dict:
    return {
        "checkpoint_sha256": U63_SHA256,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": (
            "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        ),
        "lr_scale": 0.375091552734375,
        "seed": 1701,
        "controlled_window_schedule_sha256": (
            "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351"
        ),
        "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        "scientific_identity": (
            "t_f76a1035 exact U45-era unchanged U63-to-U64 continuation"
        ),
    }


def _meta() -> dict:
    return {
        "sha256": U63_SHA256,
        "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
    }


def test_exact_authenticated_u63_terminal_continuation_is_admitted() -> None:
    _validate_published_pre_resume_start(63, _meta(), config=_config())


def test_u63_terminal_continuation_rejects_identity_drift() -> None:
    for field in ("checkpoint_sha256", "scientific_identity", "lr_scale"):
        config = _config()
        config[field] = "drift"
        with pytest.raises(ArtifactError, match="identity drift|authenticated sealed"):
            _validate_published_pre_resume_start(63, _meta(), config=config)
