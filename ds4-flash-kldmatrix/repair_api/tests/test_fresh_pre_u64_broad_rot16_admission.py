import pytest

from repair_api.api import _validate_published_pre_resume_start
from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import _published_pre_recipe_policy

U64_SHA256 = "ee4d245a624d52669a145d11ce0a21870e5a1b0db66d44bc857faf977b9fda0b"
U65_SHA256 = "f3fbc6e251b412f9dcbbce794825b1a0dc7e6979a0cc75d3650aa8a11926b7b1"
IDENTITY = "t_f76a1035 sealed U64 broad ROT16 sole-variable diet U65-to-U68"


def _config() -> dict:
    return {
        "checkpoint_sha256": U64_SHA256,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
        "lr_scale": 0.375091552734375,
        "seed": 1701,
        "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        "scientific_identity": IDENTITY,
        "controlled_windows_per_update": 16,
        "train_windows": list(range(20, 84)),
    }


def _meta() -> dict:
    return {"sha256": U64_SHA256, "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr"}


def test_authenticated_u64_broad_rot16_resume_is_admitted() -> None:
    _validate_published_pre_resume_start(64, _meta(), config=_config())


def test_u64_broad_rot16_resume_rejects_identity_drift() -> None:
    for field in ("checkpoint_sha256", "scientific_identity", "lr_scale", "controlled_windows_per_update"):
        config = _config()
        config[field] = "drift"
        with pytest.raises(ArtifactError, match="identity drift|authenticated"):
            _validate_published_pre_resume_start(64, _meta(), config=config)


def test_broad_rot16_scheduler_admits_u68_but_not_u81() -> None:
    _published_pre_recipe_policy(_config(), 68)
    with pytest.raises(ArtifactError, match="U0..U80"):
        _published_pre_recipe_policy(_config(), 81)


def test_imported_static_w28_admits_exact_sealed_u68_cursor_only() -> None:
    config = _config()
    config["scientific_identity"] = "t_f76a1035 repair-A U68 decision-boundary imported static W28 retry4"
    config["checkpoint_sha256"] = "41d7ecd80020a7192117adae3da1b2c2cd73abab2207582c20265e42ac8714cf"
    config["score_checkpoint"] = "SCHEDULE_7E8DEC81D865_UPDATE_068"
    config["controlled_windows_per_update"] = 2
    config["train_windows"] = [56, 28]
    _published_pre_recipe_policy(config, 68)
    with pytest.raises(ArtifactError, match="U0..U68"):
        _published_pre_recipe_policy(config, 69)


def test_authenticated_u65_broad_rot16_resume_is_admitted() -> None:
    config = _config()
    config["checkpoint_sha256"] = U65_SHA256
    meta = _meta()
    meta["sha256"] = U65_SHA256
    _validate_published_pre_resume_start(65, meta, config=config)
