from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from banana_smasher import (
    Qtip25CodecProvider,
    builtin_qtip25_codec_providers,
    resolve_qtip25_codec_provider,
    verify_qtip25_avg_member_baseline,
)
from banana_smasher.cli import main


ROOT = Path(__file__).parents[2]
BASELINE = ROOT / "notes/receipts/2026-08-06-qtip25-avg-member-ff0731-baseline.json"
TAXONOMY = ROOT / "notes/receipts/2026-08-06-qtip25-codec-taxonomy.json"


def test_qtip25_provider_names_are_collision_free_and_runtime_explicit() -> None:
    providers = builtin_qtip25_codec_providers()

    assert set(providers) == {
        "qtip25_avg_member",
        "qtip25_periodic_23",
        "qtip25_twostep_5b",
    }
    assert [provider.public_name for provider in providers.values()] == [
        "QTIP2.5-AVG-MEMBER",
        "QTIP2.5-PERIODIC",
        "QTIP2.5-TWOSTEP",
    ]
    assert all(isinstance(provider, Qtip25CodecProvider) for provider in providers.values())
    assert all(provider.runtime_family == "qtip2" for provider in providers.values())
    assert all(
        provider.runtime_payload_families == ("qtip2", "qtip3")
        for provider in providers.values()
    )
    assert all((provider.rate_num, provider.rate_den) == (5, 2) for provider in providers.values())
    assert providers["qtip25_avg_member"].codec_form == "avg_member_50_50"
    assert providers["qtip25_periodic_23"].codec_form == "periodic_2_3"
    assert providers["qtip25_twostep_5b"].codec_form == "twostep_5b"


def test_legacy_qtip_at_250_is_only_an_avg_member_compatibility_alias() -> None:
    provider = resolve_qtip25_codec_provider("qtip@2.50")

    assert provider.provider_id == "qtip25_avg_member"
    assert provider.public_name == "QTIP2.5-AVG-MEMBER"
    assert provider.codec_form == "avg_member_50_50"
    assert provider.compatibility_aliases == ("qtip@2.50",)
    assert provider.as_dict(requested_id="qtip@2.50")["compatibility_alias"] is True
    assert provider.as_dict()["compatibility_alias"] is False


def test_cli_lists_and_resolves_the_public_codec_taxonomy(capsys) -> None:
    assert main(["qtip25-codecs"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["status"] == "PASS"
    assert [row["machine_id"] for row in listed["codecs"]] == [
        "qtip25_avg_member",
        "qtip25_periodic_23",
        "qtip25_twostep_5b",
    ]

    assert main(["qtip25-codecs", "qtip@2.50"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["codec"]["machine_id"] == "qtip25_avg_member"
    assert resolved["codec"]["codec_form"] == "avg_member_50_50"
    assert resolved["codec"]["compatibility_alias"] is True
    assert (resolved["codec"]["rate_num"], resolved["codec"]["rate_den"]) == (5, 2)


def test_taxonomy_and_avg_member_baseline_receipts_are_decision_grade() -> None:
    taxonomy = json.loads(TAXONOMY.read_text())
    baseline = json.loads(BASELINE.read_text())

    assert taxonomy["schema"] == "banana-smasher-qtip25-codec-taxonomy-v1"
    assert taxonomy["status"] == "PASS"
    assert [row["machine_id"] for row in taxonomy["codecs"]] == [
        "qtip25_avg_member",
        "qtip25_periodic_23",
        "qtip25_twostep_5b",
    ]
    assert taxonomy["compatibility_aliases"] == {
        "qtip@2.50": {
            "machine_id": "qtip25_avg_member",
            "codec_form": "avg_member_50_50",
        }
    }

    assert baseline["schema"] == "banana-smasher-qtip25-avg-member-baseline-v1"
    assert baseline["status"] == "PASS"
    assert baseline["codec"] == {
        "public_name": "QTIP2.5-AVG-MEMBER",
        "machine_id": "qtip25_avg_member",
        "codec_form": "avg_member_50_50",
        "compatibility_alias": "qtip@2.50",
        "rate_num": 5,
        "rate_den": 2,
    }
    assert baseline["runtime"] == {
        "family": "qtip2",
        "payload_families": ["qtip2", "qtip3"],
        "payload_tiers": ["qtip25k2", "qtip25k3"],
    }
    size = baseline["size"]
    assert size["nominal_code_bpw"] == "2.5"
    assert (size["rate_num"], size["rate_den"]) == (5, 2)
    assert size["code_bytes"] + size["auxiliary_bytes"] == size["qtip_expert_payload_bytes"]
    assert (
        size["qtip_expert_payload_bytes"]
        + size["retained_non_routed_bytes"]
        + size["required_weight_pack_index_bytes"]
        == size["whole_model_shipping_bytes"]
    )
    assert size["routing_bytes"] == 108834816
    assert baseline["train64"]["mean_kld"] == "0.14997021151401604"
    assert baseline["train64"]["top1_matches"] == 58390
    assert baseline["train64"]["top1_positions"] == 65536
    assert baseline["ff0731_ancestry"]["model_index_sha256"] == (
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
    )
    assert baseline["ff0731_ancestry"]["member_manifest_sha256"] == (
        "4bb99e3192b02aeac9fd54e11a2f7685a9df02e0ab20586b49fcfeffb612f41f"
    )
    assert baseline["ff0731_ancestry"]["pack_admission_sha256"] == (
        "a3241fb207a2f2f9c7bc4a496e27aee1e2752ef1930846a58ec425b3fa1d7f70"
    )
    assert [row["machine_id"] for row in baseline["sibling_comparisons"]] == [
        "qtip25_periodic_23",
        "qtip25_twostep_5b",
    ]
    assert all(row["status"] == "AWAITING_SIBLING_RECEIPT" for row in baseline["sibling_comparisons"])
    comparison_fields = set(baseline["sibling_comparison_fields"])
    for row in baseline["sibling_comparisons"]:
        assert set(row) == comparison_fields | {"public_name", "status"}
        assert row["runtime_family"] == "qtip2"
        assert row["nominal_code_bpw"] == "2.5"
        assert (row["rate_num"], row["rate_den"]) == (5, 2)
        assert all(
            row[field] is None
            for field in (
                "code_bytes",
                "auxiliary_bytes",
                "routing_bytes",
                "whole_model_shipping_bytes",
                "train64_mean_kld",
                "train64_top1_matches",
                "model_index_sha256",
                "receipt_sha256",
            )
        )

    schema = ROOT / "banana-smasher/schema/qtip25-avg-member-baseline-v1.schema.json"
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == taxonomy["baseline_receipt"]["sha256"]
    assert taxonomy["baseline_receipt"]["bytes"] == BASELINE.stat().st_size
    assert json.loads(schema.read_text())["$id"].endswith(
        "/qtip25-avg-member-baseline-v1.schema.json"
    )


def test_baseline_verifier_rejects_historical_or_self_asserted_drift() -> None:
    baseline = json.loads(BASELINE.read_text())
    assert verify_qtip25_avg_member_baseline(baseline)["status"] == "PASS"

    mutations = []
    wrong_basis = deepcopy(baseline)
    wrong_basis["ff0731_ancestry"]["model_index_sha256"] = "0" * 64
    mutations.append(wrong_basis)
    wrong_metric = deepcopy(baseline)
    wrong_metric["train64"]["mean_kld"] = "0.0"
    mutations.append(wrong_metric)
    wrong_top1 = deepcopy(baseline)
    wrong_top1["train64"]["top1_matches"] = 0
    mutations.append(wrong_top1)
    wrong_bytes = deepcopy(baseline)
    wrong_bytes["size"]["whole_model_shipping_bytes"] += 1
    mutations.append(wrong_bytes)
    historical = deepcopy(baseline)
    historical["immutability"]["historical_tensor_adoption"] = True
    mutations.append(historical)

    for mutation in mutations:
        with pytest.raises(ValueError, match="AVG-MEMBER baseline"):
            verify_qtip25_avg_member_baseline(mutation)


def test_schemas_reject_cross_wired_taxonomy_and_hollow_baseline() -> None:
    taxonomy = json.loads(TAXONOMY.read_text())
    taxonomy_schema = json.loads(
        (ROOT / "banana-smasher/schema/qtip25-codec-taxonomy-v1.schema.json").read_text()
    )
    baseline = json.loads(BASELINE.read_text())
    baseline_schema = json.loads(
        (ROOT / "banana-smasher/schema/qtip25-avg-member-baseline-v1.schema.json").read_text()
    )
    taxonomy_validator = Draft202012Validator(taxonomy_schema)
    baseline_validator = Draft202012Validator(baseline_schema)
    assert not list(taxonomy_validator.iter_errors(taxonomy))
    assert not list(baseline_validator.iter_errors(baseline))

    taxonomy_mutations = []
    duplicate = deepcopy(taxonomy)
    duplicate["codecs"] = [duplicate["codecs"][0]] * 3
    taxonomy_mutations.append(duplicate)
    cross_wired = deepcopy(taxonomy)
    cross_wired["codecs"][0]["machine_id"] = "qtip25_periodic_23"
    taxonomy_mutations.append(cross_wired)
    omitted = deepcopy(taxonomy)
    omitted["codecs"].pop()
    taxonomy_mutations.append(omitted)
    extra = deepcopy(taxonomy)
    extra["codecs"].append(deepcopy(extra["codecs"][0]))
    taxonomy_mutations.append(extra)
    no_immutability = deepcopy(taxonomy)
    no_immutability.pop("immutability")
    taxonomy_mutations.append(no_immutability)
    decimal_string_rate = deepcopy(taxonomy)
    decimal_string_rate["codecs"][0].pop("rate_num")
    decimal_string_rate["codecs"][0].pop("rate_den")
    decimal_string_rate["codecs"][0]["nominal_code_bpw"] = "2.50"
    taxonomy_mutations.append(decimal_string_rate)
    for mutation in taxonomy_mutations:
        assert list(taxonomy_validator.iter_errors(mutation))

    baseline_mutations = []
    wrong_metric = deepcopy(baseline)
    wrong_metric["train64"]["mean_kld"] = "not-a-number"
    baseline_mutations.append(wrong_metric)
    fake_ancestry = deepcopy(baseline)
    fake_ancestry["ff0731_ancestry"] = {f"fake_{index}": index for index in range(8)}
    baseline_mutations.append(fake_ancestry)
    hollow_siblings = deepcopy(baseline)
    hollow_siblings["sibling_comparisons"] = [{}, {}]
    baseline_mutations.append(hollow_siblings)
    wrong_fields = deepcopy(baseline)
    wrong_fields["sibling_comparison_fields"][0] = "arbitrary"
    baseline_mutations.append(wrong_fields)
    no_historical_exclusion = deepcopy(baseline)
    no_historical_exclusion["immutability"].pop("historical_tensor_adoption")
    baseline_mutations.append(no_historical_exclusion)
    decimal_string_rate = deepcopy(baseline)
    decimal_string_rate["size"].pop("rate_num")
    decimal_string_rate["size"].pop("rate_den")
    decimal_string_rate["size"]["nominal_code_bpw"] = "2.50"
    baseline_mutations.append(decimal_string_rate)
    for mutation in baseline_mutations:
        assert list(baseline_validator.iter_errors(mutation))