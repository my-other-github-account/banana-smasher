from copy import deepcopy
import pathlib

import pytest

from repair_api.api import _validate_published_pre_resume_start
from repair_api.balanced64 import ArtifactError


U41_SHA256 = "40544a550331b4e59b71bdea8b348832a254f94f3847ec33735a9de5bb7a1879"
U40_SHA256 = "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"
REPAIR_A_RANK0_RECEIPT_SHA256 = "8ba35d756f54b6b8e9d377d65d83e11b077a364fa9b22eeddf4728129ea36fcb"
REPAIR_A_RANK1_RECEIPT_SHA256 = "5d1c4df51d441d8c5cdf99fefc0c73242e351fa517cb6c296d471864f4e5b446"


def _config() -> dict:
    return {
        "checkpoint_sha256": U41_SHA256,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256":
            "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
        "lr_scale": 0.375091552734375,
        "seed": 1701,
        "controlled_window_schedule_sha256":
            "e186b108124b7c0c2e070016612ebb1de7dc208ef5806acf0f8f5bc4b7377351",
        "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        "scientific_identity":
            "t_f76a1035 repair-A winner U41-to-U45 four-update continuation",
        "u41_parent_checkpoint_sha256": U40_SHA256,
        "u41_repair_a_terminal_receipt_sha256_by_rank": {
            "0": REPAIR_A_RANK0_RECEIPT_SHA256,
            "1": REPAIR_A_RANK1_RECEIPT_SHA256,
        },
    }


def test_exact_u41_checkpoint_and_sealed_method_receipts_are_admitted() -> None:
    start_meta = {
        "sha256": U41_SHA256,
        "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
    }

    _validate_published_pre_resume_start(41, start_meta, config=_config())

    for field in (
        "u41_parent_checkpoint_sha256",
        "u41_repair_a_terminal_receipt_sha256_by_rank",
    ):
        drifted = _config()
        drifted[field] = "drift"
        with pytest.raises(ArtifactError, match="identity drift"):
            _validate_published_pre_resume_start(41, start_meta, config=drifted)


def test_u41_admission_rejects_any_other_checkpoint() -> None:
    start_meta = {
        "sha256": "0" * 64,
        "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
    }
    config = deepcopy(_config())
    config["checkpoint_sha256"] = "0" * 64

    with pytest.raises(
        ArtifactError,
        match="non-four-update resume requires authenticated sealed checkpoint",
    ):
        _validate_published_pre_resume_start(41, start_meta, config=config)


def test_public_method_admits_only_exact_u41_to_u45_continuation() -> None:
    source = (pathlib.Path(__file__).resolve().parents[1] / "api.py").read_text()

    assert "valid_authenticated_u41_u45_continuation" in source
    assert "and start_update == 41" in source
    assert "and requested == (45,)" in source
    assert U41_SHA256 in source
    assert 'config.get("u41_parent_checkpoint_sha256")' in source
    assert 'config.get("u41_repair_a_terminal_receipt_sha256_by_rank")' in source
