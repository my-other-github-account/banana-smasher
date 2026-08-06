"""Collision-free public identities for the three QTIP2.5 codecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Qtip25CodecProvider:
    """Immutable public identity for one QTIP2.5 construction codec."""

    provider_id: str
    public_name: str
    codec_form: str
    runtime_family: str = "qtip2"
    runtime_payload_families: tuple[str, ...] = ("qtip2", "qtip3")
    nominal_code_bpw: str = "2.5"
    rate_num: int = 5
    rate_den: int = 2
    compatibility_aliases: tuple[str, ...] = ()

    def as_dict(self, *, requested_id: str | None = None) -> dict[str, object]:
        """Return the stable JSON provider declaration used by CLI and manifests."""

        result: dict[str, object] = {
            "public_name": self.public_name,
            "machine_id": self.provider_id,
            "codec_form": self.codec_form,
        }
        if requested_id is not None:
            result["requested_id"] = requested_id
        result.update(
            {
                "compatibility_alias": requested_id in self.compatibility_aliases,
                "runtime_family": self.runtime_family,
                "runtime_payload_families": list(self.runtime_payload_families),
                "nominal_code_bpw": self.nominal_code_bpw,
                "rate_num": self.rate_num,
                "rate_den": self.rate_den,
            }
        )
        return result


def builtin_qtip25_codec_providers() -> dict[str, Qtip25CodecProvider]:
    """Return the ordered stock QTIP2.5 codec taxonomy."""

    providers = (
        Qtip25CodecProvider(
            provider_id="qtip25_avg_member",
            public_name="QTIP2.5-AVG-MEMBER",
            codec_form="avg_member_50_50",
            compatibility_aliases=("qtip@2.50",),
        ),
        Qtip25CodecProvider(
            provider_id="qtip25_periodic_23",
            public_name="QTIP2.5-PERIODIC",
            codec_form="periodic_2_3",
        ),
        Qtip25CodecProvider(
            provider_id="qtip25_twostep_5b",
            public_name="QTIP2.5-TWOSTEP",
            codec_form="twostep_5b",
        ),
    )
    return {provider.provider_id: provider for provider in providers}


def resolve_qtip25_codec_provider(value: str) -> Qtip25CodecProvider:
    """Resolve a machine identity or the immutable legacy AVG-MEMBER alias."""

    if not isinstance(value, str) or not value:
        raise ValueError("QTIP2.5 codec identity must be a non-empty string")
    providers = builtin_qtip25_codec_providers()
    if value in providers:
        return providers[value]
    matches = [
        provider for provider in providers.values() if value in provider.compatibility_aliases
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown QTIP2.5 codec identity {value!r}")


_SEALED_AVG_MEMBER_EVIDENCE: dict[tuple[str, ...], object] = {
    ("schema",): "banana-smasher-qtip25-avg-member-baseline-v1",
    ("status",): "PASS",
    ("size", "rate_num"): 5,
    ("size", "rate_den"): 2,
    ("size", "code_bytes"): 86_570_434_560,
    ("size", "auxiliary_bytes"): 315_713_536,
    ("size", "routing_bytes"): 108_834_816,
    ("size", "qtip_expert_payload_bytes"): 86_886_148_096,
    ("size", "retained_non_routed_bytes"): 19_708_797_688,
    ("size", "required_weight_pack_index_bytes"): 28_306_324,
    ("size", "whole_model_shipping_bytes"): 106_623_252_108,
    ("train64", "bank"): "train_balanced64",
    ("train64", "candidate_id"): "qtip25_genuine_all43_ff0731",
    ("train64", "mean_kld"): "0.14997021151401604",
    ("train64", "top1_matches"): 58_390,
    ("train64", "top1_positions"): 65_536,
    ("train64", "top1_rate"): "0.890960693359375",
    (
        "train64",
        "scorer_sha256",
    ): "53db95bcc809909d649a9f6765e5cbcac3d1da1b2ad4fe7d0efa9421a57c12c5",
    (
        "train64",
        "teacher_sha256",
    ): "a3144885f78ead45dd121ae733652a3e2df61b62d159621502ccd43d497335d1",
    (
        "train64",
        "bank_manifest_sha256",
    ): "78a07d83774f31ffc519724f1cd626b9ece1eda3ff3eab4971d71a1a1c0a3a07",
    (
        "train64",
        "aggregate_sha256",
    ): "ff38f0658acd21cf648ad32fb12123e743101697ee4380034d04bd1baf6be8fd",
    ("ff0731_ancestry", "base_model"): "DeepSeek-V4-Flash-0731",
    (
        "ff0731_ancestry",
        "legacy_identity",
    ): "qtip@2.50 deterministic per-layer/per-projection 50/50 K2+K3 L16/V2 ring",
    (
        "ff0731_ancestry",
        "model_index_sha256",
    ): "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
    (
        "ff0731_ancestry",
        "immutable_handoff_sha256",
    ): "bf0086c03c9040daffd0cc647fa1e9aebeeb18cde77bb912868eb8ed0b3d3132",
    (
        "ff0731_ancestry",
        "member_manifest_sha256",
    ): "4bb99e3192b02aeac9fd54e11a2f7685a9df02e0ab20586b49fcfeffb612f41f",
    (
        "ff0731_ancestry",
        "pack_admission_sha256",
    ): "a3241fb207a2f2f9c7bc4a496e27aee1e2752ef1930846a58ec425b3fa1d7f70",
    (
        "ff0731_ancestry",
        "train64_candidate_sha256",
    ): "1e2a72af04a4eac589a3987c25a4cf8d8faca68f3d928c13ffa24e7d1b2151e4",
    (
        "ff0731_ancestry",
        "competitive_candidate_artifact_sha256",
    ): "9eb76ed85235fd15f8fb6e28eaa62395bafd927034b6258343dff9d0f74bdca0",
    (
        "ff0731_ancestry",
        "shipping_artifact_tree_sha256",
    ): "ce4c7a8d77fc3bab29756b731e37f2b41742724b9daa113c5a65d28d41673230",
    (
        "ff0731_ancestry",
        "whole_model_weight_audit_sha256",
    ): "43ffaf0b43d995544e5ded2165bc97d98948844c2b5f491e44ef69cf1b2ae18b",
    (
        "ff0731_ancestry",
        "shared_tlut_sha256",
    ): "000c7985f6ac0cbece4a9850d3913102f9a6cf6f4c051d05444fad1cfb29d",
    ("immutability", "legacy_alias_only"): True,
    ("immutability", "legacy_alias"): "qtip@2.50",
    ("immutability", "historical_tensor_adoption"): False,
    ("immutability", "bytes_relabelled"): False,
    ("immutability", "sealed_receipts_mutated"): False,
}


def verify_qtip25_avg_member_baseline(receipt: Mapping[str, object]) -> dict[str, object]:
    """Verify the baseline against independently pinned sealed FF0731 evidence."""

    for path, expected in _SEALED_AVG_MEMBER_EVIDENCE.items():
        value: object = receipt
        try:
            for key in path:
                if not isinstance(value, Mapping):
                    raise KeyError(key)
                value = value[key]
        except (KeyError, TypeError):
            value = None
        if value != expected or (type(expected) is int and type(value) is not int):
            dotted = ".".join(path)
            raise ValueError(f"AVG-MEMBER baseline mismatch at {dotted}")

    size = receipt["size"]
    assert isinstance(size, Mapping)
    if size["code_bytes"] + size["auxiliary_bytes"] != size["qtip_expert_payload_bytes"]:
        raise ValueError("AVG-MEMBER baseline expert byte accounting mismatch")
    if (
        size["qtip_expert_payload_bytes"]
        + size["retained_non_routed_bytes"]
        + size["required_weight_pack_index_bytes"]
        != size["whole_model_shipping_bytes"]
    ):
        raise ValueError("AVG-MEMBER baseline whole-model byte accounting mismatch")

    return {
        "status": "PASS",
        "machine_id": "qtip25_avg_member",
        "model_index_sha256": _SEALED_AVG_MEMBER_EVIDENCE[
            ("ff0731_ancestry", "model_index_sha256")
        ],
    }
