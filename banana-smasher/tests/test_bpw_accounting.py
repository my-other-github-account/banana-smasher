from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

import banana_smasher
from banana_smasher.bpw import (
    BPW_ACCOUNTING_SCHEMA,
    BpwAccountingError,
    build_bpw_accounting,
    require_comparable_bpw,
)
from banana_smasher.cli import main


BASE_PARAMETERS = 284_334_567_511
BASE_INVENTORY_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


def test_bpw_accounting_is_on_the_public_package_api() -> None:
    assert banana_smasher.build_bpw_accounting is build_bpw_accounting
    assert banana_smasher.require_comparable_bpw is require_comparable_bpw


def test_ff0731_publication_bpw_uses_base_model_denominator() -> None:
    accounting = build_bpw_accounting(
        weight_bytes=106_623_252_108,
        base_model_parameters=BASE_PARAMETERS,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
    )

    assert accounting["schema"] == BPW_ACCOUNTING_SCHEMA
    assert accounting["wire"] == {
        "scope": "whole_shipped_model_weights",
        "bytes": 106_623_252_108,
        "decimal_gb": "106.623252108",
    }
    assert accounting["parameters"]["base_model"] == {
        "scope": "canonical_base_model_logical_parameters",
        "logical_parameters": BASE_PARAMETERS,
        "inventory_sha256": BASE_INVENTORY_SHA256,
    }
    assert accounting["bpw"]["comparison"] == (
        "2.999937799792846799052242162608052699084192147372562734142065965737483798472331993952175189063236474"
    )
    assert accounting["bpw"]["including_auxiliary"] == accounting["bpw"]["comparison"]
    assert accounting["publication"] == {
        "bpw": "3.0",
        "decimal_places": 1,
        "label": "3.0bpw",
        "source": "comparison",
    }


def test_auxiliary_parameters_never_change_publication_bpw() -> None:
    accounting = build_bpw_accounting(
        weight_bytes=93_691_352_992,
        base_model_parameters=BASE_PARAMETERS,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
        auxiliary_model_parameters={"dwarfstar_drafter": 19_845_850_983},
    )

    assert accounting["bpw"]["comparison"].startswith("2.63608758687774759058")
    assert accounting["bpw"]["including_auxiliary"].startswith(
        "2.46409952240493941306"
    )
    assert accounting["publication"]["label"] == "2.6bpw"


def test_smaller_artifact_has_lower_comparable_publication_bpw() -> None:
    banana = build_bpw_accounting(
        weight_bytes=106_623_252_108,
        base_model_parameters=BASE_PARAMETERS,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
    )
    unsloth_iq3 = build_bpw_accounting(
        weight_bytes=104_207_848_032,
        base_model_parameters=BASE_PARAMETERS,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
    )

    basis = require_comparable_bpw([banana, unsloth_iq3])

    assert basis["base_model_parameters"] == BASE_PARAMETERS
    assert basis["base_parameter_inventory_sha256"] == BASE_INVENTORY_SHA256
    assert unsloth_iq3["publication"]["label"] == "2.9bpw"
    assert banana["publication"]["label"] == "3.0bpw"


def test_comparison_rejects_same_inventory_with_different_parameter_counts() -> None:
    correct = build_bpw_accounting(
        weight_bytes=106_623_252_108,
        base_model_parameters=BASE_PARAMETERS,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
    )
    wrong_hf_total = build_bpw_accounting(
        weight_bytes=106_623_252_108,
        base_model_parameters=304_180_418_494,
        base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
    )

    with pytest.raises(BpwAccountingError, match="base-model parameter count mismatch"):
        require_comparable_bpw([correct, wrong_hf_total])


def test_smash_bpw_emits_canonical_publication_label(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "bpw",
                "--weight-bytes",
                "106623252108",
                "--base-model-parameters",
                str(BASE_PARAMETERS),
                "--base-parameter-inventory-sha256",
                BASE_INVENTORY_SHA256,
            ]
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["command"] == "bpw"
    assert receipt["publication"]["label"] == "3.0bpw"
    assert receipt["bpw"]["comparison"].startswith("2.999937799792846")


def test_published_balanced64_rows_use_the_package_accounting_contract() -> None:
    result_path = (
        Path(__file__).parents[2]
        / "Evals/results/deepseek-v4-flash-0731-balanced64-v1.json"
    )
    rows = json.loads(result_path.read_text(encoding="utf-8"))["results"]

    for row in rows:
        wire = row["wire"]
        components = wire["total_model_parameter_components"]
        auxiliary = (
            {"auxiliary_models": components["auxiliary_models"]}
            if components["auxiliary_models"]
            else None
        )
        accounting = build_bpw_accounting(
            weight_bytes=wire["bytes"],
            base_model_parameters=components["base_model"],
            base_parameter_inventory_sha256=BASE_INVENTORY_SHA256,
            auxiliary_model_parameters=auxiliary,
        )

        for api_field, published_field in (
            ("comparison", "normalized_bpw"),
            ("including_auxiliary", "total_model_bpw"),
        ):
            published = Decimal(wire[published_field])
            with localcontext() as context:
                context.prec = len(published.as_tuple().digits)
                assert +Decimal(accounting["bpw"][api_field]) == published


def test_public_table_uses_comparison_bpw_not_auxiliary_inclusive_bpw() -> None:
    readme = (Path(__file__).parents[2] / "Evals/README.md").read_text(
        encoding="utf-8"
    )
    assert "| Comparison BPW |" in readme
    dwarfstar_row = next(
        line for line in readme.splitlines() if "**DwarfStar Q2**" in line
    )
    assert "| 2.636 |" in dwarfstar_row
    assert "| 2.464 |" not in dwarfstar_row
