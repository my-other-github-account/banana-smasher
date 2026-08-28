import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import _published_pre_recipe_policy


CONFIG = {
    "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
    "published_pre_checkpoint_sha256": (
        "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
    ),
    "lr_scale": 0.375091552734375,
}


def test_published_pre_schedule_admits_terminal_u64_scheduler_cursor() -> None:
    _lrs, multiplier, windows = _published_pre_recipe_policy(CONFIG, 64)

    assert multiplier == pytest.approx(0.1 * CONFIG["lr_scale"])
    assert windows == [28, 56]


def test_published_pre_schedule_rejects_cursor_after_terminal_u64() -> None:
    with pytest.raises(ArtifactError, match="U0..U64"):
        _published_pre_recipe_policy(CONFIG, 65)
