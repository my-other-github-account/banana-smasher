from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
from itertools import chain
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "banana-smasher.evaluation-comparison.v1"
WINDOW_SCHEMA = "banana-smasher.balanced64-window.v1"
SUITE_LOCK_SCHEMA = "banana-smasher.balanced64-suite-lock.v1"
BALANCED64_V1_LOCK_SHA256 = "d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9"
MTP_PARAMETER_COUNT = 10_215_806_828
MTP_INCLUSIVE_PARAMETER_DENOMINATOR = 294_550_374_339
ARTIFACT_PAYLOAD_SCOPES = {
    "base-model-only",
    "base-plus-native-mtp",
    "base-plus-separate-drafter",
}
SOURCE_CLASSES = (
    "agentic",
    "chat",
    "code",
    "multilingual",
    "prose",
    "reasoning",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?$")
_INTERNAL_LABEL = re.compile(r"(?i)(?:\bRUN\d+\b|\bSPARK\d+\b|\bt_[0-9a-f]{8}\b)")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_VERIFICATION_SCOPE = {
    "publicly_verifiable": (
        "suite-lock consistency, SHA-256 field syntax, Top-1/GB/BPW arithmetic, "
        "rankings, and standardized per-position reaggregation when window receipts "
        "are supplied"
    ),
    "not_publicly_authenticated": (
        "protected source-receipt contents and historical KLD values; SHA-256 values "
        "identify those sources but do not prove availability or authenticity"
    ),
    "full_gpu_replay": "blocked; see each result replay.blockers",
}
EXPECTED_RESULT_MODEL_COUNT = 14
_PROTECTED_EXL_PUBLICATION_ROWS = {
    "EXL3-K2P5-greedy-full": {
        "display_name": "EXL3 K2.5 greedy optimizer full",
        "kld_mean": "0.30277489559979315",
        "top1_matches": 54_732,
        "wire_bytes": 94_832_865_520,
        "normalized_bpw": "2.6682050332506607541984592911089396395596594633708887538020512532588125649342761",
        "total_model_bpw": "2.5756644372376514306392840126330402137510080619975999995940035714829300785993458",
        "candidate_artifact_sha256": "7c8d1aa6d5fea5c22374346b0e18450881cc97cee118f7bc75f064f56f828044",
        "candidate_manifest_sha256": "a226f60c6193f6fb2a8b1240cbf83b8ecea3bea3de9d905460244501545cc503",
        "classes": {
            "agentic": ("0.3964723762975632", 16_355, 19_456, 19),
            "chat": ("0.1511617236051681", 6_159, 7_168, 7),
            "code": ("0.16400093568881688", 8_113, 9_216, 9),
            "multilingual": ("0.4480915980718887", 8_020, 10_240, 10),
            "prose": ("0.39144750210645024", 7_845, 10_240, 10),
            "reasoning": ("0.10167629783490464", 8_240, 9_216, 9),
        },
        "sources": {
            "EXL3 K2.5 greedy exact-rate optimizer solution": "07a8eb1fbd59bc34aff90aac0d8635647fd9c09074adc93e79aec7ea40497f6f",
            "EXL3 corrected level-3 optimizer measurement": "989eca997272e168ede2a92b2bbbbc018c197fae6e3963396a3c9d61cab3c089",
            "EXL3 K2.5 greedy physical provenance": "8271f49c24e9f2ab302f642c9cfbbf97ab75b5c894b25ed2d35f2c1d7122cd56",
            "EXL3 K2.5 greedy optimized payload manifest": "a226f60c6193f6fb2a8b1240cbf83b8ecea3bea3de9d905460244501545cc503",
            "EXL3 K2.5 greedy BALANCED64 measurement bindings": "ae0c5e616c23867df6a59d55540a915e3b0d7d16461515c306bf767e6abfbc5d",
            "EXL3 K2.5 greedy independent Exact64 recompute": "7ea0919f389f34a6de691d6ea9d74c7b33ac71d9c2b487bfb035ff072acb611a",
            "EXL3 K2.5 greedy Exact64 capture manifest": "5f438d19df70fe5b2dd3da171e82b08e766b40a1425db3169351611d172cd9bb",
        },
    },
    "EXL3-K3-routed-native-rest": {
        "display_name": "EXL3 K3 routed-only + native rest",
        "kld_mean": "0.07686796725357639",
        "top1_matches": 60_447,
        "wire_bytes": 123_999_250_168,
        "normalized_bpw": "3.4888265961739699744459889854268040102450967587235200857315432779975056108576297",
        "total_model_bpw": "3.3678246159765781019987434251990662176430186852148307430503292387805231170862566",
        "candidate_artifact_sha256": "4ae72ab897698d94f12e93918294d4e7e1c79f5b5f0df2112d18113670ea0ee3",
        "candidate_manifest_sha256": "42f3d57f5f112a9dbb7badd4dc76536f0ef6da7a3fe0422bc271891b487f83c8",
        "upstream_repository": "https://github.com/turboderp-org/exllamav3",
        "upstream_revision": "791c83073f7f90c44f765a0ceeab7a05fa15b96b",
        "classes": {
            "agentic": ("0.12174968537781196", 17_871, 19_456, 19),
            "chat": ("0.028756867612099144", 6_753, 7_168, 7),
            "code": ("0.031304263141010216", 8_722, 9_216, 9),
            "multilingual": ("0.10764323388773885", 9_161, 10_240, 10),
            "prose": ("0.09123982241898246", 9_089, 10_240, 10),
            "reasoning": ("0.014937653047718106", 8_851, 9_216, 9),
        },
        "sources": {
            "EXL3 routed-only K3 native-rest arm terminal": "69234474ce937b5685007dacd1657bf60d4f19ed56a6e4d2c28eb58fb231e2ff",
            "EXL3 routed-only K3 overlay identity": "71f9a2d74d0b3f39d7daba0624dacdd86d24b03e246c27323496d96130acbda8",
            "EXL3 routed-only K3 selected-payload proof": "a9284944717bcbddfa9b60ff54b0468317d5a234daff29136d3be48764d0823e",
            "EXL3 routed-only K3 BALANCED64 measurement bindings": "6fab4d832299f782d27af03c8333b8b921faa7ffb14db1fd764c6224c9544b51",
            "EXL3 routed-only K3 independent Exact64 recompute": "014869075e2ef73bb328fdc83aef0a6f4943e93cb513fae47479df3dffc49aeb",
            "EXL3 routed-only K3 Exact64 capture terminal": "556fb2293635ce017407f791a20eb04baa64e7be37c6293cf82898dc2f1afb4e",
            "EXL3 routed-only K3 functional readback": "04317b764ad511e5735cb05e5928f579bee0dce43f35a4a3726a6ee3f7a88c55",
            "EXL routed-native reporting handoff": "47d837d3eefb2a12843fb185419f2de72e1467fc3aabd86942fedea3fe426dca",
            "EXL routed-native pair decision terminal": "599760e7624018cbc9204726b69f329d09acdb01523909cb9c9b8d3d7be90533",
            "EXL routed-native durable evidence mirror seal": "836864b810e729e50aca2922f0e3e7b0f0cd7b1cb374ba53019f6174c194f9bf",
        },
    },
    "EXL3-K2P5-greedy-routed-native-rest": {
        "display_name": "EXL3 K2.5 greedy-upcast routed-only + native rest",
        "kld_mean": "0.1746041415211709",
        "top1_matches": 57_885,
        "wire_bytes": 106_282_510_072,
        "normalized_bpw": "2.990350726677318761133570203797083967844437614338647678644542139727383081773899285391977912993952953",
        "total_model_bpw": "2.886637243235786808211855375258090922632836912396751146876837308679676476518714841150246405221221918",
        "candidate_artifact_sha256": "5bedb489dfe62bad9107948d011a42cf888f7e2789a6386b680b2da7681be051",
        "candidate_manifest_sha256": "6e77d799bbc6516375fddeda848df972143639880140099b840ef364b035aad7",
        "classes": {
            "agentic": ("0.2567325914331791", 17_107, 19_456, 19),
            "chat": ("0.0654209493349179", 6_534, 7_168, 7),
            "code": ("0.08451931125373331", 8_460, 9_216, 9),
            "multilingual": ("0.2674554124711736", 8_616, 10_240, 10),
            "prose": ("0.21185796815801314", 8_509, 10_240, 10),
            "reasoning": ("0.03166572968940458", 8_659, 9_216, 9),
        },
        "sources": {
            "EXL3 K2.5 greedy exact-rate optimizer solution": "07a8eb1fbd59bc34aff90aac0d8635647fd9c09074adc93e79aec7ea40497f6f",
            "EXL3 corrected level-3 optimizer measurement": "989eca997272e168ede2a92b2bbbbc018c197fae6e3963396a3c9d61cab3c089",
            "EXL3 K2.5 greedy physical provenance": "8271f49c24e9f2ab302f642c9cfbbf97ab75b5c894b25ed2d35f2c1d7122cd56",
            "EXL3 K2.5 greedy optimized payload manifest": "a226f60c6193f6fb2a8b1240cbf83b8ecea3bea3de9d905460244501545cc503",
            "EXL3 K2.5 greedy routed-native Exact64 terminal": "6d344da2022930818319bdc0c5f82709324a0145e74b68a099aaee2456cae549",
            "EXL3 K2.5 greedy routed-native overlay identity": "5bedb489dfe62bad9107948d011a42cf888f7e2789a6386b680b2da7681be051",
            "EXL3 K2.5 greedy routed-native selected-payload proof": "6e77d799bbc6516375fddeda848df972143639880140099b840ef364b035aad7",
            "EXL3 K2.5 greedy routed tensor-source manifest": "24191b3074b9a778f5eff8af48f832bfc26afd180fa9747ae6643687dd098146",
            "EXL3 K2.5 greedy routed-native BALANCED64 measurement bindings": "5f277890ad9239427963b9875ba3d0d8c94cd495e186b74ddd59f2467c184a33",
            "EXL3 K2.5 greedy routed-native independent Exact64 recompute": "6449da9c85d7297d0d05804747bad9782b54ac5bb63f38e16d586fc6f7a09f20",
            "EXL3 K2.5 greedy routed-native Exact64 capture terminal": "d1f8cb18d42c819cf20a234ab22a861078a6980680a343dcf880fcdae02329f9",
            "EXL3 K2.5 greedy routed-native functional readback": "fa80ee530a2c83f2bf9c30ef0c7f9b715368905631f093d2e300937b33dab772",
            "EXL3 K2.5 greedy routed-native durable mirror manifest": "6126d283d20be77e4ba686a64cd72a318d1392f60e5db60b634113b0e7095fa1",
            "EXL3 K2.5 greedy routed-native release terminal": "d06e15d7875262611a697c03902de311207fb14e843024cab7553ab3aed49236",
        },
    },
    "Physical-K2K3-2P5-alternating-comparator": {
        "display_name": "Physical alternating K2/K3 2.5-BPW comparator",
        "kld_mean": "0.29960352599248635",
        "top1_matches": 54_585,
        "wire_bytes": 94_832_907_712,
        "normalized_bpw": "2.6682062203592242845254773071874904890730094033332647693753735642321114923652284",
        "total_model_bpw": "2.5756655831740664070723314987285659393901029185138807891100977264136044544482164",
        "candidate_artifact_sha256": "0e1a0d7da3c3917a72e827052d57b5278005b2a228896e89a3b96d7cc01fb24d",
        "candidate_manifest_sha256": "047c0ff5ba2de074c1e4f11712b73755849267d62d785d6908993e550bf5e628",
        "classes": {
            "agentic": ("0.38563935297463126", 16_310, 19_456, 19),
            "chat": ("0.15400998095795138", 6_134, 7_168, 7),
            "code": ("0.1626202665162457", 8_100, 9_216, 9),
            "multilingual": ("0.438331370333361", 8_022, 10_240, 10),
            "prose": ("0.39809882999045076", 7_801, 10_240, 10),
            "reasoning": ("0.10461374315679389", 8_218, 9_216, 9),
        },
        "sources": {
            "EXL3 K2.5 physical provenance": "1cd820cfa3a12251d69e5121793cbc618ce8810f524944808a728499c10fc62c",
            "EXL3 K2.5 terminal acceptance": "9a633e8adb351ae9e638e192d0f9a818497b62dc22831fa7367b91a4c74e622e",
            "EXL3 K2.5 Exact64 capture manifest": "7fa640d61e0bf179ebf65d03c9dcd5d51d991d86e97e19698d35dfb4a967d7f8",
            "EXL3 K2.5 measurement bindings": "2937b661a968c965fef6f9451c94e6fd425192f5007d497ca44793f5c0eca316",
            "EXL3 K2.5 independent exact recompute": "b90c04e2643b8bb9e371343bec67dbca907572ed9581b1dbd5f46b2d18b4d4f4",
            "EXL3 K2.5 native MTP closure": "6d2b42005c11cb4f2ec8289926d2261dcd489a1cdf7aef19395f0d475efbbc84",
            "EXL3 K2.5 physical alternating assignment": "ffbf48cfe16c5ae29c1254995176d32177ebef9e2030bce135586bfa03ffa5ef",
            "EXL3 corrected level-3 measurement": "989eca997272e168ede2a92b2bbbbc018c197fae6e3963396a3c9d61cab3c089",
        },
    },
}


class ReceiptError(ValueError):
    """Raised when an evaluation receipt violates a fail-closed contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReceiptError(f"{label} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReceiptError(f"{label} key drift: missing={missing} extra={extra}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiptError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReceiptError(f"{label} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, label: str, *, minimum: Decimal = Decimal(0)) -> Decimal:
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise ReceiptError(f"{label} must be a canonical nonnegative decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ReceiptError(f"{label} must be a decimal") from exc
    if not parsed.is_finite() or parsed < minimum:
        raise ReceiptError(f"{label} must be finite and >= {minimum}")
    return parsed


def _binary64(value: Any, label: str) -> float:
    if not isinstance(value, str):
        raise ReceiptError(f"{label} must be a round-trip binary64 decimal string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ReceiptError(f"{label} must be a binary64 value") from exc
    if (
        not math.isfinite(parsed)
        or parsed < 0
        or math.copysign(1.0, parsed) < 0
    ):
        raise ReceiptError(f"{label} must be finite and nonnegative; no clamp is applied")
    if value != repr(parsed):
        raise ReceiptError(f"{label} must use Python's shortest round-trip binary64 repr")
    return parsed


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_sha256_fields(value: Any, label: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_label = f"{label}.{key}"
            if key.endswith("sha256"):
                _sha256(item, child_label)
            _verify_sha256_fields(item, child_label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _verify_sha256_fields(item, f"{label}[{index}]")


def _verify_protected_exl_publication(rows: Sequence[Any]) -> None:
    if len(rows) != EXPECTED_RESULT_MODEL_COUNT:
        raise ReceiptError(
            f"published result population drift: expected {EXPECTED_RESULT_MODEL_COUNT}, "
            f"found {len(rows)}"
        )
    by_id = {
        str(_mapping(row, f"results[{index}]").get("model_id")): _mapping(
            row, f"results[{index}]"
        )
        for index, row in enumerate(rows)
    }
    if "EXL3-K2P5-physical-alternating" in by_id:
        raise ReceiptError("the in-house physical comparator must not use the EXL K2.5 identity")

    for model_id, expected in _PROTECTED_EXL_PUBLICATION_ROWS.items():
        row = by_id.get(model_id)
        if row is None:
            raise ReceiptError(f"{model_id}: required protected EXL publication row is missing")
        artifact = _mapping(row.get("artifact"), f"{model_id}.artifact")
        wire = _mapping(row.get("wire"), f"{model_id}.wire")
        kld = _mapping(row.get("kld"), f"{model_id}.kld")
        top1 = _mapping(row.get("top1"), f"{model_id}.top1")
        observed = {
            "display_name": row.get("display_name"),
            "kld_mean": kld.get("mean"),
            "top1_matches": top1.get("matches"),
            "wire_bytes": wire.get("bytes"),
            "normalized_bpw": wire.get("normalized_bpw"),
            "total_model_bpw": wire.get("total_model_bpw"),
            "candidate_artifact_sha256": artifact.get("candidate_artifact_sha256"),
            "candidate_manifest_sha256": artifact.get("candidate_manifest_sha256"),
        }
        for optional_identity in ("upstream_repository", "upstream_revision"):
            if optional_identity in expected:
                observed[optional_identity] = artifact.get(optional_identity)
        classes = _mapping(row.get("classes"), f"{model_id}.classes")
        observed["classes"] = {
            name: (
                _mapping(classes.get(name), f"{model_id}.classes.{name}").get("kld_mean"),
                _mapping(classes.get(name), f"{model_id}.classes.{name}").get(
                    "top1_matches"
                ),
                _mapping(classes.get(name), f"{model_id}.classes.{name}").get("positions"),
                _mapping(classes.get(name), f"{model_id}.classes.{name}").get("windows"),
            )
            for name in SOURCE_CLASSES
        }
        observed["sources"] = {
            str(
                _mapping(source, f"{model_id}.source_receipts[{index}]").get("label")
            ): _mapping(source, f"{model_id}.source_receipts[{index}]").get("sha256")
            for index, source in enumerate(
                _sequence(row.get("source_receipts"), f"{model_id}.source_receipts")
            )
        }
        if observed != expected:
            raise ReceiptError(f"{model_id}: protected EXL publication value drift")


def _ratio(numerator: int | Decimal, denominator: int | Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        return Decimal(numerator) / Decimal(denominator)


def _ratio_at_stored_precision(
    numerator: int | Decimal, denominator: int | Decimal, stored: Any, label: str
) -> tuple[Decimal, Decimal]:
    parsed = _decimal(stored, label)
    significant_digits = len(parsed.as_tuple().digits)
    if significant_digits < 30:
        raise ReceiptError(f"{label} must preserve at least 30 significant digits")
    with localcontext() as context:
        context.prec = significant_digits
        expected = Decimal(numerator) / Decimal(denominator)
    return parsed, expected


def _canonical_digest(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_suite_lock(suite_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_keys = {
        "canonicalization",
        "class_map_sha256",
        "class_windows",
        "fp",
        "metrics",
        "name",
        "positions",
        "positions_per_window",
        "retired_class_map",
        "schema",
        "source_provenance_sha256",
        "source_suite_sha256",
        "source_windows_sha256",
        "suite_lock_sha256",
        "support",
        "teacher_bank",
        "teacher_source_model_index_sha256",
        "window_count",
        "window_population_sha256",
        "windows",
        "wire_parameter_denominator",
    }
    _require_exact_keys(suite_lock, expected_keys, "suite lock")
    if suite_lock.get("schema") != SUITE_LOCK_SCHEMA:
        raise ReceiptError(f"suite lock schema must be {SUITE_LOCK_SCHEMA}")
    if suite_lock.get("name") != "BALANCED64_V1":
        raise ReceiptError("suite lock name must be BALANCED64_V1")
    _verify_sha256_fields(suite_lock, "suite lock")

    stored_lock_digest = _sha256(
        suite_lock.get("suite_lock_sha256"), "suite lock suite_lock_sha256"
    )
    recomputed_lock_digest = _canonical_digest(suite_lock, omit="suite_lock_sha256")
    if stored_lock_digest != recomputed_lock_digest:
        raise ReceiptError("suite lock canonical digest does not match its content")
    if stored_lock_digest != BALANCED64_V1_LOCK_SHA256:
        raise ReceiptError("suite lock is not the published BALANCED64_V1 authority")

    window_count = _integer(suite_lock.get("window_count"), "suite lock window_count", minimum=1)
    positions_per_window = _integer(
        suite_lock.get("positions_per_window"),
        "suite lock positions_per_window",
        minimum=1,
    )
    positions = _integer(suite_lock.get("positions"), "suite lock positions", minimum=1)
    if positions != window_count * positions_per_window:
        raise ReceiptError("suite lock position denominator drift")
    _integer(suite_lock.get("support"), "suite lock support", minimum=1)
    _integer(
        suite_lock.get("wire_parameter_denominator"),
        "suite lock wire_parameter_denominator",
        minimum=1,
    )
    _nonempty_string(suite_lock.get("fp"), "suite lock fp")
    _nonempty_string(suite_lock.get("teacher_bank"), "suite lock teacher_bank")

    class_windows = _mapping(suite_lock.get("class_windows"), "suite lock class_windows")
    if set(class_windows) != set(SOURCE_CLASSES):
        raise ReceiptError("suite lock class_windows must contain exactly six source classes")
    validated_class_windows = {
        name: _integer(class_windows[name], f"suite lock class_windows.{name}")
        for name in SOURCE_CLASSES
    }
    if sum(validated_class_windows.values()) != window_count:
        raise ReceiptError("suite lock class_windows does not sum to window_count")

    windows = _sequence(suite_lock.get("windows"), "suite lock windows")
    if len(windows) != window_count:
        raise ReceiptError("suite lock window population has the wrong size")
    normalized_windows: list[dict[str, Any]] = []
    seen_window_ids: set[int] = set()
    observed_classes: dict[str, int] = defaultdict(int)
    for ordinal, raw_window in enumerate(windows):
        window = _mapping(raw_window, f"suite lock windows[{ordinal}]")
        _require_exact_keys(
            window,
            {"ordinal", "source_class", "window_id"},
            f"suite lock windows[{ordinal}]",
        )
        if _integer(window.get("ordinal"), f"suite lock windows[{ordinal}].ordinal") != ordinal:
            raise ReceiptError("suite lock ordinals must be contiguous from zero")
        window_id = _integer(window.get("window_id"), f"suite lock windows[{ordinal}].window_id")
        if window_id in seen_window_ids:
            raise ReceiptError(f"duplicate suite lock window_id: {window_id}")
        seen_window_ids.add(window_id)
        source_class = window.get("source_class")
        if source_class not in SOURCE_CLASSES:
            raise ReceiptError(f"suite lock windows[{ordinal}] has unknown source_class")
        observed_classes[str(source_class)] += 1
        normalized_windows.append(
            {"ordinal": ordinal, "window_id": window_id, "source_class": source_class}
        )
    if dict(observed_classes) != validated_class_windows:
        raise ReceiptError("suite lock class counts do not match its window population")

    population_payload = [
        {"ordinal": item["ordinal"], "window_id": item["window_id"]}
        for item in normalized_windows
    ]
    encoded_population = json.dumps(
        population_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded_population).hexdigest() != suite_lock.get(
        "window_population_sha256"
    ):
        raise ReceiptError("suite lock window_population_sha256 drift")
    encoded_class_map = json.dumps(
        normalized_windows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded_class_map).hexdigest() != suite_lock.get("class_map_sha256"):
        raise ReceiptError("suite lock class_map_sha256 drift")

    retired = _mapping(suite_lock.get("retired_class_map"), "suite lock retired_class_map")
    _require_exact_keys(retired, {"sha256", "status"}, "suite lock retired_class_map")
    if retired.get("status") != "invalid-for-subgroup-reporting":
        raise ReceiptError("retired class map status must remain invalid-for-subgroup-reporting")

    metrics = _mapping(suite_lock.get("metrics"), "suite lock metrics")
    _require_exact_keys(metrics, {"kld", "top1"}, "suite lock metrics")
    kld = _mapping(metrics.get("kld"), "suite lock metrics.kld")
    _require_exact_keys(
        kld,
        {
            "direction",
            "negative_policy",
            "per_position_dtype",
            "reduction",
            "serialization",
            "support",
        },
        "suite lock metrics.kld",
    )
    top1 = _mapping(metrics.get("top1"), "suite lock metrics.top1")
    _require_exact_keys(top1, {"definition", "tie_break"}, "suite lock metrics.top1")
    return suite_lock


def _suite_projection(suite_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": suite_lock["name"],
        "suite_lock_sha256": suite_lock["suite_lock_sha256"],
        "source_suite_sha256": suite_lock["source_suite_sha256"],
        "source_windows_sha256": suite_lock["source_windows_sha256"],
        "source_provenance_sha256": suite_lock["source_provenance_sha256"],
        "window_population_sha256": suite_lock["window_population_sha256"],
        "class_map_sha256": suite_lock["class_map_sha256"],
        "teacher_bank": suite_lock["teacher_bank"],
        "teacher_source_model_index_sha256": suite_lock[
            "teacher_source_model_index_sha256"
        ],
        "windows": suite_lock["window_count"],
        "positions_per_window": suite_lock["positions_per_window"],
        "positions": suite_lock["positions"],
        "support": suite_lock["support"],
        "class_windows": suite_lock["class_windows"],
    }


def _verify_artifact(artifact: Mapping[str, Any], label: str) -> None:
    identity_status = artifact.get("identity_status")
    missing_fields = _sequence(
        artifact.get("missing_identity_fields"),
        f"{label}.missing_identity_fields",
    )
    if "repository" in artifact:
        _require_exact_keys(
            artifact,
            {
                "artifact_manifest_sha256",
                "identity_status",
                "missing_identity_fields",
                "repository",
                "revision",
                "variant",
            },
            label,
        )
        if identity_status != "complete-as-recorded" or missing_fields:
            raise ReceiptError(f"{label}: recorded-complete identity status drift")
        for field in ("repository", "revision", "variant"):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
    elif "format" in artifact:
        expected_keys = {
            "base_model",
            "candidate_artifact_sha256",
            "candidate_manifest_sha256",
            "format",
            "identity_status",
            "mechanism",
            "missing_identity_fields",
            "variant",
        }
        if "upstream_repository" in artifact:
            expected_keys.add("upstream_repository")
        if "upstream_revision" in artifact:
            expected_keys.add("upstream_revision")
        _require_exact_keys(artifact, expected_keys, label)
        if identity_status == "complete-as-recorded":
            if (
                missing_fields
                or "upstream_repository" not in artifact
                or "upstream_revision" not in artifact
            ):
                raise ReceiptError(f"{label}: complete EXL3 identity requires recorded upstream lineage")
            upstream_repository = _nonempty_string(
                artifact.get("upstream_repository"), f"{label}.upstream_repository"
            )
            if upstream_repository != "https://github.com/turboderp-org/exllamav3":
                raise ReceiptError(f"{label}.upstream_repository must identify upstream exllamav3")
            upstream_revision = _nonempty_string(
                artifact.get("upstream_revision"), f"{label}.upstream_revision"
            )
            if _GIT_COMMIT.fullmatch(upstream_revision) is None:
                raise ReceiptError(f"{label}.upstream_revision must be a 40-character Git commit")
        elif identity_status == "partial":
            if not missing_fields:
                raise ReceiptError(f"{label}: partial EXL3 identity must name missing lineage")
        else:
            raise ReceiptError(f"{label}: unsupported EXL3 identity status")
        if artifact.get("base_model") != "DeepSeek-V4-Flash-0731":
            raise ReceiptError(f"{label}.base_model must remain DeepSeek-V4-Flash-0731")
        if artifact.get("format") != "EXL3":
            raise ReceiptError(f"{label}.format must be EXL3")
        for field in ("variant", "mechanism"):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
        for digest_field in ("candidate_artifact_sha256", "candidate_manifest_sha256"):
            _sha256(artifact.get(digest_field), f"{label}.{digest_field}")
    elif "family" in artifact:
        _require_exact_keys(
            artifact,
            {
                "base_model",
                "candidate_artifact_sha256",
                "description",
                "family",
                "geometry",
                "identity_status",
                "layers",
                "missing_identity_fields",
                "units",
                "variant",
            },
            label,
        )
        if identity_status != "complete-as-recorded" or missing_fields:
            raise ReceiptError(f"{label}: QTIP identity must be complete as recorded")
        if artifact.get("base_model") != "DeepSeek-V4-Flash-0731":
            raise ReceiptError(f"{label}.base_model must remain DeepSeek-V4-Flash-0731")
        if artifact.get("family") not in {"QTIP2", "QTIP2.5", "QTIP3"}:
            raise ReceiptError(f"{label}.family must be QTIP2, QTIP2.5, or QTIP3")
        for field in ("variant", "geometry", "description"):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
        if _integer(artifact.get("layers"), f"{label}.layers", minimum=1) != 43:
            raise ReceiptError(f"{label}.layers must cover all 43 layers")
        if _integer(artifact.get("units"), f"{label}.units", minimum=1) != 22016:
            raise ReceiptError(f"{label}.units must cover all 22,016 QTIP units")
        _sha256(artifact.get("candidate_artifact_sha256"), f"{label}.candidate_artifact_sha256")
    elif "base_model" in artifact:
        _require_exact_keys(
            artifact,
            {
                "base_model",
                "identity_status",
                "missing_identity_fields",
                "source_final_sha256",
                "variant",
            },
            label,
        )
        if identity_status != "partial" or not missing_fields:
            raise ReceiptError(f"{label}: partial identity status drift")
        for field in ("base_model", "variant"):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
    elif "base_repository" in artifact:
        _require_exact_keys(
            artifact,
            {
                "base_repository",
                "base_revision",
                "base_sha256",
                "drafter_repository",
                "drafter_revision",
                "drafter_sha256",
                "engine",
                "engine_commit",
                "identity_status",
                "missing_identity_fields",
            },
            label,
        )
        if identity_status != "complete-as-recorded" or missing_fields:
            raise ReceiptError(f"{label}: recorded-complete identity status drift")
        for field in (
            "base_repository",
            "base_revision",
            "drafter_repository",
            "drafter_revision",
            "engine",
        ):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
        engine_commit = _nonempty_string(
            artifact.get("engine_commit"), f"{label}.engine_commit"
        )
        if _GIT_COMMIT.fullmatch(engine_commit) is None:
            raise ReceiptError(f"{label}.engine_commit must be a 40-character Git commit")
    else:
        raise ReceiptError(f"{label}: unsupported artifact identity shape")

    for missing_index, value in enumerate(missing_fields):
        _nonempty_string(value, f"{label}.missing_identity_fields[{missing_index}]")


def _verify_classes(
    row: Mapping[str, Any],
    suite_lock: Mapping[str, Any],
    *,
    model_id: str,
    kld_mean: Decimal,
    matches: int,
    positions: int,
) -> None:
    classes = _mapping(row.get("classes"), f"{model_id}.classes")
    _require_exact_keys(classes, set(SOURCE_CLASSES), f"{model_id}.classes")
    expected_class_windows = _mapping(
        suite_lock.get("class_windows"), "suite lock class_windows"
    )
    positions_per_window = int(suite_lock["positions_per_window"])
    class_matches = 0
    class_positions = 0
    class_windows = 0
    weighted_kld = Decimal(0)
    for name in SOURCE_CLASSES:
        class_row = _mapping(classes.get(name), f"{model_id}.classes.{name}")
        _require_exact_keys(
            class_row,
            {"kld_mean", "positions", "top1_matches", "top1_rate", "windows"},
            f"{model_id}.classes.{name}",
        )
        windows = _integer(
            class_row.get("windows"), f"{model_id}.classes.{name}.windows", minimum=1
        )
        if windows != expected_class_windows[name]:
            raise ReceiptError(f"{model_id}: {name} window count differs from suite lock")
        expected_positions = windows * positions_per_window
        observed_positions = _integer(
            class_row.get("positions"),
            f"{model_id}.classes.{name}.positions",
            minimum=1,
        )
        if observed_positions != expected_positions:
            raise ReceiptError(f"{model_id}: {name} positions differ from suite lock")
        observed_matches = _integer(
            class_row.get("top1_matches"), f"{model_id}.classes.{name}.top1_matches"
        )
        if observed_matches > observed_positions:
            raise ReceiptError(f"{model_id}: {name} Top-1 matches exceed positions")
        observed_rate = _decimal(
            class_row.get("top1_rate"), f"{model_id}.classes.{name}.top1_rate"
        )
        if observed_rate != _ratio(observed_matches, observed_positions):
            raise ReceiptError(f"{model_id}: {name} Top-1 rate is not integer-derived")
        observed_kld = _decimal(
            class_row.get("kld_mean"), f"{model_id}.classes.{name}.kld_mean"
        )
        class_windows += windows
        class_positions += observed_positions
        class_matches += observed_matches
        weighted_kld += observed_kld * observed_positions

    if class_windows != int(suite_lock["window_count"]):
        raise ReceiptError(f"{model_id}: class windows do not sum to global windows")
    if class_positions != positions:
        raise ReceiptError(f"{model_id}: class positions do not sum to global positions")
    if class_matches != matches:
        raise ReceiptError(f"{model_id}: class Top-1 matches do not sum to global matches")
    weighted_kld /= Decimal(positions)
    rounded_weighted_kld = float(weighted_kld)
    global_kld = float(kld_mean)
    if (
        repr(rounded_weighted_kld) != str(kld_mean)
        and abs(rounded_weighted_kld - global_kld) > math.ulp(global_kld)
    ):
        raise ReceiptError(
            f"{model_id}: weighted class KLD differs from global KLD by more than one "
            "binary64 ULP"
        )


def _verify_external_measurement(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    suite_lock: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    measurement = _mapping(row.get("measurement"), f"{model_id}.measurement")
    expected_keys = {
        "artifacts",
        "candidate_artifact_sha256",
        "class_map_sha256",
        "fallback_used",
        "holdout_used",
        "limitations",
        "numeric_semantics",
        "positions",
        "scorer_source_sha256",
        "status",
        "suite_file_sha256",
        "suite_lock_sha256",
        "support",
        "teacher_bank",
        "teacher_source_model_index_sha256",
        "unavailable_fields",
        "window_population_sha256",
        "windows",
    }
    complete_identity = artifact.get("identity_status") == "complete-as-recorded"
    if complete_identity or "censoring_used" in measurement:
        expected_keys.add("censoring_used")
    _require_exact_keys(measurement, expected_keys, f"{model_id}.measurement")
    if measurement.get("status") != "PASS":
        raise ReceiptError(f"{model_id}: measurement status must be PASS")
    if measurement.get("candidate_artifact_sha256") != artifact.get(
        "candidate_artifact_sha256"
    ):
        raise ReceiptError(f"{model_id}: candidate identity drift between artifact and measurement")
    expected_measurement = {
        "teacher_bank": suite_lock["teacher_bank"],
        "teacher_source_model_index_sha256": suite_lock[
            "teacher_source_model_index_sha256"
        ],
        "suite_lock_sha256": suite_lock["suite_lock_sha256"],
        "window_population_sha256": suite_lock["window_population_sha256"],
        "class_map_sha256": suite_lock["class_map_sha256"],
        "windows": suite_lock["window_count"],
        "positions": suite_lock["positions"],
        "support": suite_lock["support"],
    }
    for field, expected in expected_measurement.items():
        if measurement.get(field) != expected:
            raise ReceiptError(f"{model_id}: measurement {field} differs from suite lock")
    for digest_field in ("scorer_source_sha256", "suite_file_sha256"):
        _sha256(measurement.get(digest_field), f"{model_id}.measurement.{digest_field}")
    for flag in ("holdout_used", "fallback_used"):
        if measurement.get(flag) is not False:
            raise ReceiptError(f"{model_id}: measurement {flag} must be false")
    if complete_identity or "censoring_used" in measurement:
        if measurement.get("censoring_used") is not False:
            raise ReceiptError(f"{model_id}: measurement censoring_used must be false")
    _nonempty_string(
        measurement.get("numeric_semantics"), f"{model_id}.measurement.numeric_semantics"
    )

    artifacts = _mapping(measurement.get("artifacts"), f"{model_id}.measurement.artifacts")
    required_artifacts = {
        "candidate_artifact_tree_sha256",
        "candidate_manifest_sha256",
        "independent_exact_recompute_sha256",
        "main_measurement_receipt_sha256",
        "teacher_tree_sha256",
    }
    if complete_identity:
        if len(artifacts) < 6 or not required_artifacts.issubset(artifacts):
            raise ReceiptError(f"{model_id}: measurement artifact hash set is incomplete")
    else:
        required_artifacts.add("independent_validation_sha256")
        if set(artifacts) != required_artifacts:
            raise ReceiptError(f"{model_id}: partial external artifact hash set has drifted")
    for name, digest in artifacts.items():
        _sha256(digest, f"{model_id}.measurement.artifacts.{name}")
    if artifacts["candidate_artifact_tree_sha256"] != artifact.get(
        "candidate_artifact_sha256"
    ):
        raise ReceiptError(f"{model_id}: candidate artifact tree hash drift")
    if artifacts["candidate_manifest_sha256"] != artifact.get("candidate_manifest_sha256"):
        raise ReceiptError(f"{model_id}: candidate manifest hash drift")

    limitations = _sequence(
        measurement.get("limitations"), f"{model_id}.measurement.limitations"
    )
    if not limitations:
        raise ReceiptError(f"{model_id}: measurement limitations must not be empty")
    for value_index, value in enumerate(limitations):
        _nonempty_string(value, f"{model_id}.measurement.limitations[{value_index}]")
    unavailable = _sequence(
        measurement.get("unavailable_fields"), f"{model_id}.measurement.unavailable_fields"
    )
    for value_index, value in enumerate(unavailable):
        _nonempty_string(value, f"{model_id}.measurement.unavailable_fields[{value_index}]")


def _verify_external_weight_components(
    row: Mapping[str, Any], *, model_id: str, wire_bytes: int
) -> None:
    components = _mapping(row.get("weight_components"), f"{model_id}.weight_components")
    _require_exact_keys(
        components,
        {
            "artifact_metadata_bytes",
            "format_retained_payload",
            "quantized_eligible_payload",
            "repair_payload",
            "runtime_and_headroom_bytes",
            "safetensors_container_bytes",
        },
        f"{model_id}.weight_components",
    )

    component_total = 0
    for group_name in ("quantized_eligible_payload", "format_retained_payload"):
        group = _mapping(components.get(group_name), f"{model_id}.{group_name}")
        _require_exact_keys(group, {"bytes", "components"}, f"{model_id}.{group_name}")
        group_bytes = _integer(group.get("bytes"), f"{model_id}.{group_name}.bytes")
        ledger = _mapping(group.get("components"), f"{model_id}.{group_name}.components")
        if not ledger:
            raise ReceiptError(f"{model_id}: {group_name} component ledger is empty")
        ledger_total = sum(
            _integer(value, f"{model_id}.{group_name}.components.{name}")
            for name, value in ledger.items()
        )
        if ledger_total != group_bytes:
            raise ReceiptError(f"{model_id}: {group_name} components do not sum to bytes")
        component_total += group_bytes

    for field in ("safetensors_container_bytes", "artifact_metadata_bytes"):
        component_total += _integer(components.get(field), f"{model_id}.{field}")
    repair = _mapping(components.get("repair_payload"), f"{model_id}.repair_payload")
    _require_exact_keys(repair, {"bytes"}, f"{model_id}.repair_payload")
    component_total += _integer(repair.get("bytes"), f"{model_id}.repair_payload.bytes")
    if _integer(
        components.get("runtime_and_headroom_bytes"),
        f"{model_id}.runtime_and_headroom_bytes",
    ) != 0:
        raise ReceiptError(f"{model_id}: runtime_and_headroom_bytes must remain excluded")
    if component_total != wire_bytes:
        raise ReceiptError(f"{model_id}: EXL3 weight components do not sum to whole-model bytes")


def _verify_qtip_details(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    suite_lock: Mapping[str, Any],
    *,
    model_id: str,
    kld_mean: Decimal,
    matches: int,
    positions: int,
    wire_bytes: int,
    accounting_receipt_sha256: str,
) -> None:
    _verify_classes(
        row,
        suite_lock,
        model_id=model_id,
        kld_mean=kld_mean,
        matches=matches,
        positions=positions,
    )

    components = _mapping(row.get("weight_components"), f"{model_id}.weight_components")
    _require_exact_keys(
        components,
        {
            "qtip_expert_payload",
            "mtp_accounting_correction",
            "repair_payload",
            "retained_non_routed_payload",
            "runtime_and_headroom_bytes",
            "torch_container_inflation_bytes",
            "weight_pack_index_metadata",
        },
        f"{model_id}.weight_components",
    )

    payload_total = 0
    for group_name in ("qtip_expert_payload", "retained_non_routed_payload"):
        group = _mapping(components.get(group_name), f"{model_id}.{group_name}")
        _require_exact_keys(group, {"bytes", "components"}, f"{model_id}.{group_name}")
        group_bytes = _integer(group.get("bytes"), f"{model_id}.{group_name}.bytes")
        ledger = _mapping(group.get("components"), f"{model_id}.{group_name}.components")
        if not ledger:
            raise ReceiptError(f"{model_id}: {group_name} component ledger is empty")
        ledger_total = sum(
            _integer(value, f"{model_id}.{group_name}.components.{name}")
            for name, value in ledger.items()
        )
        if ledger_total != group_bytes:
            raise ReceiptError(f"{model_id}: {group_name} components do not sum to bytes")
        payload_total += group_bytes

    index = _mapping(
        components.get("weight_pack_index_metadata"),
        f"{model_id}.weight_pack_index_metadata",
    )
    _require_exact_keys(
        index,
        {"corrected_bytes", "corrected_index_materialized", "source_bytes", "source_sha256"},
        f"{model_id}.weight_pack_index_metadata",
    )
    _integer(
        index.get("source_bytes"),
        f"{model_id}.weight_pack_index_metadata.source_bytes",
        minimum=1,
    )
    index_bytes = _integer(
        index.get("corrected_bytes"),
        f"{model_id}.weight_pack_index_metadata.corrected_bytes",
        minimum=1,
    )
    index_sha256 = _sha256(
        index.get("source_sha256"),
        f"{model_id}.weight_pack_index_metadata.source_sha256",
    )
    if index.get("corrected_index_materialized") is not False:
        raise ReceiptError(f"{model_id}: corrected index must remain explicitly unmaterialized")

    correction = _mapping(
        components.get("mtp_accounting_correction"),
        f"{model_id}.mtp_accounting_correction",
    )
    _require_exact_keys(
        correction,
        {
            "accounting_receipt_sha256",
            "corrected_retained_payload_bytes",
            "corrected_retained_tensor_count",
            "missing_tensor_count",
            "missing_tensor_payload_bytes",
        },
        f"{model_id}.mtp_accounting_correction",
    )
    if _integer(
        correction.get("missing_tensor_count"),
        f"{model_id}.mtp_accounting_correction.missing_tensor_count",
    ) != 10:
        raise ReceiptError(f"{model_id}: MTP correction must contain ten tensors")
    if _integer(
        correction.get("missing_tensor_payload_bytes"),
        f"{model_id}.mtp_accounting_correction.missing_tensor_payload_bytes",
    ) != 33_843_220:
        raise ReceiptError(f"{model_id}: MTP correction payload-byte drift")
    _integer(
        correction.get("corrected_retained_tensor_count"),
        f"{model_id}.mtp_accounting_correction.corrected_retained_tensor_count",
        minimum=1,
    )
    corrected_retained_bytes = _integer(
        correction.get("corrected_retained_payload_bytes"),
        f"{model_id}.mtp_accounting_correction.corrected_retained_payload_bytes",
        minimum=1,
    )
    if corrected_retained_bytes != components["retained_non_routed_payload"]["bytes"]:
        raise ReceiptError(f"{model_id}: corrected retained payload does not match ledger")
    if _sha256(
        correction.get("accounting_receipt_sha256"),
        f"{model_id}.mtp_accounting_correction.accounting_receipt_sha256",
    ) != accounting_receipt_sha256:
        raise ReceiptError(f"{model_id}: MTP correction receipt hash drift")
    repair = _mapping(components.get("repair_payload"), f"{model_id}.repair_payload")
    _require_exact_keys(repair, {"bytes"}, f"{model_id}.repair_payload")
    repair_bytes = _integer(repair.get("bytes"), f"{model_id}.repair_payload.bytes")
    for excluded in ("torch_container_inflation_bytes", "runtime_and_headroom_bytes"):
        if _integer(components.get(excluded), f"{model_id}.{excluded}") != 0:
            raise ReceiptError(f"{model_id}: {excluded} must remain explicitly excluded")
    if payload_total + index_bytes + repair_bytes != wire_bytes:
        raise ReceiptError(f"{model_id}: weight components do not sum to whole-model bytes")

    measurement = _mapping(row.get("measurement"), f"{model_id}.measurement")
    _require_exact_keys(
        measurement,
        {
            "artifacts",
            "candidate_artifact_sha256",
            "censoring_used",
            "class_map_sha256",
            "fallback_used",
            "holdout_used",
            "limitations",
            "numeric_semantics",
            "positions",
            "repository_commit",
            "scorer_source_sha256",
            "status",
            "suite_file_sha256",
            "support",
            "teacher_bank",
            "teacher_source_model_index_sha256",
            "window_population_sha256",
            "windows",
        },
        f"{model_id}.measurement",
    )
    if measurement.get("status") != "PASS":
        raise ReceiptError(f"{model_id}: measurement status must be PASS")
    if measurement.get("candidate_artifact_sha256") != artifact.get(
        "candidate_artifact_sha256"
    ):
        raise ReceiptError(f"{model_id}: candidate identity drift between artifact and measurement")
    expected_measurement = {
        "teacher_bank": suite_lock["teacher_bank"],
        "teacher_source_model_index_sha256": suite_lock[
            "teacher_source_model_index_sha256"
        ],
        "window_population_sha256": suite_lock["window_population_sha256"],
        "class_map_sha256": suite_lock["class_map_sha256"],
        "windows": suite_lock["window_count"],
        "positions": suite_lock["positions"],
        "support": suite_lock["support"],
    }
    for field, expected in expected_measurement.items():
        if measurement.get(field) != expected:
            raise ReceiptError(f"{model_id}: measurement {field} differs from suite lock")
    for digest_field in ("scorer_source_sha256", "suite_file_sha256"):
        _sha256(measurement.get(digest_field), f"{model_id}.measurement.{digest_field}")
    repository_commit = _nonempty_string(
        measurement.get("repository_commit"), f"{model_id}.measurement.repository_commit"
    )
    if _GIT_COMMIT.fullmatch(repository_commit) is None:
        raise ReceiptError(f"{model_id}: measurement repository_commit is invalid")
    for flag in ("holdout_used", "fallback_used", "censoring_used"):
        if measurement.get(flag) is not False:
            raise ReceiptError(f"{model_id}: measurement {flag} must be false")
    _nonempty_string(
        measurement.get("numeric_semantics"), f"{model_id}.measurement.numeric_semantics"
    )
    artifacts = _mapping(measurement.get("artifacts"), f"{model_id}.measurement.artifacts")
    required_artifacts = {
        "aggregate_sha256",
        "independent_verification_sha256",
        "row_collection_sha256",
        "terminal_handoff_sha256",
        "weight_pack_index_sha256",
        "whole_model_receipt_sha256",
    }
    if not required_artifacts <= set(artifacts):
        raise ReceiptError(f"{model_id}: measurement artifact hash set is incomplete")
    for name, digest in artifacts.items():
        _sha256(digest, f"{model_id}.measurement.artifacts.{name}")
    if artifacts["weight_pack_index_sha256"] != index_sha256:
        raise ReceiptError(f"{model_id}: weight-pack index hash drift")
    limitations = _sequence(
        measurement.get("limitations"), f"{model_id}.measurement.limitations"
    )
    if not limitations:
        raise ReceiptError(f"{model_id}: measurement limitations must not be empty")
    for limitation_index, limitation in enumerate(limitations):
        _nonempty_string(
            limitation, f"{model_id}.measurement.limitations[{limitation_index}]"
        )


def verify_result_receipt(
    receipt: Mapping[str, Any], suite_lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify compact-result structure and arithmetic against the published suite lock."""
    verify_suite_lock(suite_lock)
    _require_exact_keys(
        receipt,
        {
            "comparison_id",
            "fp",
            "metrics",
            "mtp_inclusive_parameter_denominator",
            "results",
            "schema",
            "size_accounting",
            "suite",
            "title",
            "verification_scope",
            "wire_parameter_denominator",
        },
        "receipt",
    )
    if receipt.get("schema") != RESULT_SCHEMA:
        raise ReceiptError(f"schema must be {RESULT_SCHEMA}")
    _verify_sha256_fields(receipt)
    _nonempty_string(receipt.get("comparison_id"), "comparison_id")
    _nonempty_string(receipt.get("title"), "title")
    if receipt.get("fp") != suite_lock.get("fp"):
        raise ReceiptError("receipt FP basis differs from suite lock")
    if receipt.get("wire_parameter_denominator") != suite_lock.get(
        "wire_parameter_denominator"
    ):
        raise ReceiptError("receipt BPW denominator differs from suite lock")
    mtp_inclusive_denominator = _integer(
        receipt.get("mtp_inclusive_parameter_denominator"),
        "mtp_inclusive_parameter_denominator",
        minimum=1,
    )
    if mtp_inclusive_denominator != MTP_INCLUSIVE_PARAMETER_DENOMINATOR:
        raise ReceiptError("receipt MTP-inclusive parameter denominator drift")
    if mtp_inclusive_denominator != int(receipt["wire_parameter_denominator"]) + MTP_PARAMETER_COUNT:
        raise ReceiptError("receipt base and MTP parameter counts do not close")
    size_accounting = _mapping(receipt.get("size_accounting"), "size_accounting")
    _require_exact_keys(
        size_accounting,
        {"comparison_bpw_kind", "matched_physical_bpw_kind", "receipt_path", "receipt_sha256"},
        "size_accounting",
    )
    if size_accounting.get("receipt_path") != (
        "deepseek-v4-flash-0731-mtp-size-accounting-v1.json"
    ):
        raise ReceiptError("size-accounting receipt path drift")
    accounting_receipt_sha256 = _sha256(
        size_accounting.get("receipt_sha256"), "size_accounting.receipt_sha256"
    )
    for field in ("comparison_bpw_kind", "matched_physical_bpw_kind"):
        _nonempty_string(size_accounting.get(field), f"size_accounting.{field}")
    if _mapping(receipt.get("suite"), "suite") != _suite_projection(suite_lock):
        raise ReceiptError("receipt suite fields differ from the published suite lock")
    if _mapping(receipt.get("metrics"), "metrics") != suite_lock.get("metrics"):
        raise ReceiptError("receipt metric semantics differ from the published suite lock")

    verification_scope = _mapping(receipt.get("verification_scope"), "verification_scope")
    if verification_scope != EXPECTED_VERIFICATION_SCOPE:
        raise ReceiptError("verification_scope differs from the published limitations")

    positions = int(suite_lock["positions"])
    denominator = int(suite_lock["wire_parameter_denominator"])
    fp = str(suite_lock["fp"])
    rows = _sequence(receipt.get("results"), "results")
    if not rows:
        raise ReceiptError("results must not be empty")
    _verify_protected_exl_publication(rows)

    seen_models: set[str] = set()
    kld_rows: list[tuple[Decimal, str]] = []
    top1_rows: list[tuple[Decimal, str]] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"results[{index}]")
        artifact = _mapping(row.get("artifact"), f"results[{index}].artifact")
        is_qtip = "family" in artifact
        is_external = "format" in artifact
        has_external_measurement = "measurement" in row and not is_qtip
        has_complete_external_identity = (
            is_external and artifact.get("identity_status") == "complete-as-recorded"
        )
        expected_row_keys = {
            "artifact",
            "display_name",
            "fp",
            "kld",
            "model_id",
            "replay",
            "source_receipts",
            "top1",
            "vendor",
            "wire",
        }
        expected_row_keys.add("classes")
        if is_qtip:
            expected_row_keys.update({"measurement", "weight_components"})
        elif has_external_measurement or has_complete_external_identity:
            expected_row_keys.add("measurement")
            if has_complete_external_identity:
                expected_row_keys.add("weight_components")
        _require_exact_keys(
            row,
            expected_row_keys,
            f"results[{index}]",
        )
        model_id = _nonempty_string(row.get("model_id"), f"results[{index}].model_id")
        _nonempty_string(row.get("display_name"), f"{model_id}.display_name")
        _nonempty_string(row.get("vendor"), f"{model_id}.vendor")
        if model_id in seen_models:
            raise ReceiptError(f"duplicate model_id: {model_id}")
        seen_models.add(model_id)
        if row.get("fp") != fp:
            raise ReceiptError(f"{model_id}: FP basis differs from suite lock")

        _verify_artifact(artifact, f"{model_id}.artifact")

        replay = _mapping(row.get("replay"), f"{model_id}.replay")
        _require_exact_keys(replay, {"blockers", "status"}, f"{model_id}.replay")
        if replay.get("status") != "blocked":
            raise ReceiptError(f"{model_id}: historical full replay must remain explicitly blocked")
        blockers = _sequence(replay.get("blockers"), f"{model_id}.replay.blockers")
        if not blockers:
            raise ReceiptError(f"{model_id}: blocked replay must name blockers")
        for blocker_index, blocker in enumerate(blockers):
            _nonempty_string(blocker, f"{model_id}.replay.blockers[{blocker_index}]")

        kld = _mapping(row.get("kld"), f"{model_id}.kld")
        _require_exact_keys(kld, {"direction", "mean"}, f"{model_id}.kld")
        if kld.get("direction") != suite_lock["metrics"]["kld"]["direction"]:
            raise ReceiptError(f"{model_id}: unsupported KLD direction")
        kld_mean = _decimal(kld.get("mean"), f"{model_id}.kld.mean")

        top1 = _mapping(row.get("top1"), f"{model_id}.top1")
        _require_exact_keys(top1, {"matches", "positions", "rate"}, f"{model_id}.top1")
        matches = _integer(top1.get("matches"), f"{model_id}.top1.matches")
        top1_positions = _integer(
            top1.get("positions"), f"{model_id}.top1.positions", minimum=1
        )
        if matches > top1_positions:
            raise ReceiptError(f"{model_id}: Top-1 matches exceed positions")
        if top1_positions != positions:
            raise ReceiptError(f"{model_id}: Top-1 denominator differs from suite lock")
        stored_rate = _decimal(top1.get("rate"), f"{model_id}.top1.rate")
        expected_rate = _ratio(matches, top1_positions)
        if stored_rate != expected_rate:
            raise ReceiptError(f"{model_id}: Top-1 rate does not match numerator/denominator")

        wire = _mapping(row.get("wire"), f"{model_id}.wire")
        _require_exact_keys(
            wire,
            {
                "bytes",
                "decimal_gb",
                "artifact_payload_scope",
                "normalized_bpw",
                "parameter_denominator",
                "total_model_bpw",
                "total_model_parameter_components",
                "total_model_parameters",
            },
            f"{model_id}.wire",
        )
        if wire.get("parameter_denominator") != denominator:
            raise ReceiptError(f"{model_id}: parameter denominator differs from suite lock")
        payload_scope = wire.get("artifact_payload_scope")
        if payload_scope not in ARTIFACT_PAYLOAD_SCOPES:
            raise ReceiptError(f"{model_id}: unsupported artifact payload scope")
        wire_bytes = _integer(wire.get("bytes"), f"{model_id}.wire.bytes", minimum=1)
        expected_gb = _ratio(wire_bytes, 1_000_000_000)
        stored_gb = _decimal(wire.get("decimal_gb"), f"{model_id}.wire.decimal_gb")
        if stored_gb != expected_gb:
            raise ReceiptError(f"{model_id}: decimal GB does not match bytes")
        stored_bpw, expected_bpw = _ratio_at_stored_precision(
            wire_bytes * 8,
            denominator,
            wire.get("normalized_bpw"),
            f"{model_id}.wire.normalized_bpw",
        )
        if stored_bpw != expected_bpw:
            raise ReceiptError(f"{model_id}: normalized BPW does not match bytes/denominator")

        total_parameters = _integer(
            wire.get("total_model_parameters"),
            f"{model_id}.wire.total_model_parameters",
            minimum=1,
        )
        parameter_components = _mapping(
            wire.get("total_model_parameter_components"),
            f"{model_id}.wire.total_model_parameter_components",
        )
        _require_exact_keys(
            parameter_components,
            {"auxiliary_models", "base_model"},
            f"{model_id}.wire.total_model_parameter_components",
        )
        base_parameters = _integer(
            parameter_components.get("base_model"),
            f"{model_id}.wire.total_model_parameter_components.base_model",
            minimum=1,
        )
        auxiliary_parameters = _integer(
            parameter_components.get("auxiliary_models"),
            f"{model_id}.wire.total_model_parameter_components.auxiliary_models",
        )
        if base_parameters != denominator:
            raise ReceiptError(f"{model_id}: total-model base parameter count differs from suite lock")
        if total_parameters != base_parameters + auxiliary_parameters:
            raise ReceiptError(f"{model_id}: total-model parameter components do not sum")
        has_drafter = "drafter_repository" in artifact
        if payload_scope == "base-model-only":
            if auxiliary_parameters != 0 or total_parameters != denominator or has_drafter:
                raise ReceiptError(f"{model_id}: base-only parameter scope drift")
        elif payload_scope == "base-plus-native-mtp":
            if (
                auxiliary_parameters != MTP_PARAMETER_COUNT
                or total_parameters != mtp_inclusive_denominator
                or has_drafter
            ):
                raise ReceiptError(f"{model_id}: native-MTP parameter scope drift")
        elif not has_drafter or auxiliary_parameters <= 0:
            raise ReceiptError(f"{model_id}: separate-drafter parameter scope drift")
        if is_qtip and payload_scope != "base-plus-native-mtp":
            raise ReceiptError(f"{model_id}: QTIP shipping scope must include native MTP")
        stored_total_bpw, expected_total_bpw = _ratio_at_stored_precision(
            wire_bytes * 8,
            total_parameters,
            wire.get("total_model_bpw"),
            f"{model_id}.wire.total_model_bpw",
        )
        if stored_total_bpw != expected_total_bpw:
            raise ReceiptError(
                f"{model_id}: total-model BPW does not match bytes/total parameters"
            )

        if is_qtip:
            _verify_qtip_details(
                row,
                artifact,
                suite_lock,
                model_id=model_id,
                kld_mean=kld_mean,
                matches=matches,
                positions=top1_positions,
                wire_bytes=wire_bytes,
                accounting_receipt_sha256=accounting_receipt_sha256,
            )
        else:
            _verify_classes(
                row,
                suite_lock,
                model_id=model_id,
                kld_mean=kld_mean,
                matches=matches,
                positions=top1_positions,
            )
            if has_external_measurement:
                _verify_external_measurement(
                    row,
                    artifact,
                    suite_lock,
                    model_id=model_id,
                )
                if has_complete_external_identity:
                    _verify_external_weight_components(
                        row, model_id=model_id, wire_bytes=wire_bytes
                    )

        sources = _sequence(row.get("source_receipts"), f"{model_id}.source_receipts")
        if not sources:
            raise ReceiptError(f"{model_id}: source_receipts must not be empty")
        seen_source_labels: set[str] = set()
        seen_source_digests: set[str] = set()
        for source_index, raw_source in enumerate(sources):
            source = _mapping(raw_source, f"{model_id}.source_receipts[{source_index}]")
            _require_exact_keys(
                source,
                {"availability", "label", "role", "sha256"},
                f"{model_id}.source_receipts[{source_index}]",
            )
            label = _nonempty_string(
                source.get("label"), f"{model_id}.source_receipts[{source_index}].label"
            )
            if _INTERNAL_LABEL.search(label):
                raise ReceiptError(f"{model_id}: source label contains an internal run identifier")
            if label in seen_source_labels:
                raise ReceiptError(f"{model_id}: duplicate source label: {label}")
            seen_source_labels.add(label)
            _nonempty_string(
                source.get("role"), f"{model_id}.source_receipts[{source_index}].role"
            )
            if source.get("availability") != "protected-not-distributed":
                raise ReceiptError(f"{model_id}: unsupported source availability claim")
            digest = _sha256(
                source.get("sha256"),
                f"{model_id}.source_receipts[{source_index}].sha256",
            )
            if digest in seen_source_digests:
                raise ReceiptError(f"{model_id}: duplicate source digest: {digest}")
            seen_source_digests.add(digest)

        kld_rows.append((kld_mean, model_id))
        top1_rows.append((expected_rate, model_id))

    return {
        "schema": RESULT_SCHEMA,
        "suite_lock_sha256": suite_lock["suite_lock_sha256"],
        "models": len(rows),
        "positions": positions,
        "kld_ranking": [model_id for _, model_id in sorted(kld_rows)],
        "top1_ranking": [
            model_id for _, model_id in sorted(top1_rows, key=lambda item: (-item[0], item[1]))
        ],
        "full_gpu_replay": "blocked",
    }


def verify_size_accounting_receipt(
    comparison: Mapping[str, Any], accounting: Mapping[str, Any]
) -> None:
    """Verify the linked MTP accounting receipt against the comparison rows."""
    _require_exact_keys(
        accounting,
        {
            "accounting_policy",
            "basis_model_index_sha256",
            "comparison_id",
            "limitations",
            "qtip_mtp_correction",
            "quality_metrics_changed",
            "rows",
            "schema",
            "source_result_repository_commit",
            "status",
        },
        "size-accounting receipt",
    )
    if accounting.get("schema") != "banana-smasher.mtp-size-accounting-correction.v1":
        raise ReceiptError("unsupported size-accounting receipt schema")
    if accounting.get("comparison_id") != comparison.get("comparison_id"):
        raise ReceiptError("size-accounting comparison_id drift")
    if accounting.get("status") != "PASS" or accounting.get("quality_metrics_changed") is not False:
        raise ReceiptError("size-accounting status or quality-metric scope drift")
    _verify_sha256_fields(accounting, "size-accounting receipt")
    source_commit = _nonempty_string(
        accounting.get("source_result_repository_commit"),
        "size-accounting receipt.source_result_repository_commit",
    )
    if _GIT_COMMIT.fullmatch(source_commit) is None:
        raise ReceiptError("size-accounting source repository commit is invalid")
    if accounting.get("basis_model_index_sha256") != comparison["suite"][
        "teacher_source_model_index_sha256"
    ]:
        raise ReceiptError("size-accounting basis model index drift")

    policy = _mapping(accounting.get("accounting_policy"), "size-accounting policy")
    _require_exact_keys(
        policy,
        {
            "base_equivalent_comparison_bpw",
            "base_model_parameter_denominator",
            "matched_physical_bpw",
            "mtp_inclusive_parameter_denominator",
            "mtp_parameter_denominator",
        },
        "size-accounting policy",
    )
    if policy.get("base_model_parameter_denominator") != comparison.get(
        "wire_parameter_denominator"
    ):
        raise ReceiptError("size-accounting base denominator drift")
    if policy.get("mtp_parameter_denominator") != MTP_PARAMETER_COUNT:
        raise ReceiptError("size-accounting MTP denominator drift")
    if policy.get("mtp_inclusive_parameter_denominator") != comparison.get(
        "mtp_inclusive_parameter_denominator"
    ):
        raise ReceiptError("size-accounting MTP-inclusive denominator drift")
    for field in ("base_equivalent_comparison_bpw", "matched_physical_bpw"):
        _nonempty_string(policy.get(field), f"size-accounting policy.{field}")

    correction = _mapping(
        accounting.get("qtip_mtp_correction"), "size-accounting QTIP correction"
    )
    _require_exact_keys(
        correction,
        {
            "corrected_index_status",
            "corrected_retained_payload_bytes",
            "corrected_retained_tensor_count",
            "missing_tensor_count",
            "missing_tensor_names",
            "missing_tensor_payload_bytes",
        },
        "size-accounting QTIP correction",
    )
    missing_names = _sequence(
        correction.get("missing_tensor_names"),
        "size-accounting QTIP correction.missing_tensor_names",
    )
    missing_count = _integer(
        correction.get("missing_tensor_count"),
        "size-accounting QTIP correction.missing_tensor_count",
    )
    if missing_count != 10 or len(missing_names) != missing_count or len(set(missing_names)) != missing_count:
        raise ReceiptError("size-accounting missing MTP tensor-name closure drift")
    for index, name in enumerate(missing_names):
        _nonempty_string(name, f"size-accounting missing_tensor_names[{index}]")
    if correction.get("missing_tensor_payload_bytes") != 33_843_220:
        raise ReceiptError("size-accounting missing MTP payload-byte drift")
    if correction.get("corrected_retained_tensor_count") != 6_279:
        raise ReceiptError("size-accounting corrected retained tensor-count drift")
    if correction.get("corrected_retained_payload_bytes") != 19_742_640_908:
        raise ReceiptError("size-accounting corrected retained payload-byte drift")
    _nonempty_string(
        correction.get("corrected_index_status"),
        "size-accounting QTIP correction.corrected_index_status",
    )

    comparison_rows = {
        row["model_id"]: row
        for row in _sequence(comparison.get("results"), "comparison results")
    }
    accounting_rows = _mapping(accounting.get("rows"), "size-accounting rows")
    if set(accounting_rows) != set(comparison_rows):
        raise ReceiptError("size-accounting row population drift")
    for model_id, comparison_row in comparison_rows.items():
        accounting_row = _mapping(accounting_rows.get(model_id), f"size-accounting {model_id}")
        wire = _mapping(comparison_row.get("wire"), f"comparison {model_id}.wire")
        if accounting_row.get("artifact_payload_scope") != wire.get("artifact_payload_scope"):
            raise ReceiptError(f"{model_id}: size-accounting payload scope drift")
        if accounting_row.get("comparison_bpw") != wire.get("normalized_bpw"):
            raise ReceiptError(f"{model_id}: size-accounting comparison BPW drift")
        if accounting_row.get("matched_physical_parameter_denominator") != wire.get(
            "total_model_parameters"
        ):
            raise ReceiptError(f"{model_id}: size-accounting physical denominator drift")
        if accounting_row.get("matched_physical_bpw") != wire.get("total_model_bpw"):
            raise ReceiptError(f"{model_id}: size-accounting physical BPW drift")
        accounted_bytes = accounting_row.get(
            "corrected_shipping_bytes", accounting_row.get("shipping_bytes")
        )
        if accounted_bytes != wire.get("bytes"):
            raise ReceiptError(f"{model_id}: size-accounting shipping-byte drift")

        if model_id.startswith("QTIP"):
            components = _mapping(
                comparison_row.get("weight_components"), f"comparison {model_id}.weight_components"
            )
            index = _mapping(
                components.get("weight_pack_index_metadata"),
                f"comparison {model_id}.weight_pack_index_metadata",
            )
            if accounting_row.get("source_index_bytes") != index.get("source_bytes"):
                raise ReceiptError(f"{model_id}: size-accounting source index-byte drift")
            if accounting_row.get("source_index_sha256") != index.get("source_sha256"):
                raise ReceiptError(f"{model_id}: size-accounting source index hash drift")
            if accounting_row.get("corrected_index_bytes") != index.get("corrected_bytes"):
                raise ReceiptError(f"{model_id}: size-accounting corrected index-byte drift")
            if accounting_row.get("corrected_index_materialized") is not False:
                raise ReceiptError(f"{model_id}: corrected index materialization claim drift")
            source_bytes = _integer(
                accounting_row.get("source_shipping_bytes"),
                f"size-accounting {model_id}.source_shipping_bytes",
                minimum=1,
            )
            correction_bytes = _integer(
                accounting_row.get("correction_bytes"),
                f"size-accounting {model_id}.correction_bytes",
                minimum=1,
            )
            expected_correction = 33_843_220 + (
                int(index["corrected_bytes"]) - int(index["source_bytes"])
            )
            if correction_bytes != expected_correction or source_bytes + correction_bytes != wire["bytes"]:
                raise ReceiptError(f"{model_id}: size-accounting correction-byte closure drift")

    limitations = _sequence(accounting.get("limitations"), "size-accounting limitations")
    if not limitations:
        raise ReceiptError("size-accounting limitations must not be empty")
    for index, limitation in enumerate(limitations):
        _nonempty_string(limitation, f"size-accounting limitations[{index}]")


def aggregate_windows(
    rows: Iterable[Mapping[str, Any]], suite_lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate ordered per-position binary64 KLD values under the published suite lock."""
    verify_suite_lock(suite_lock)
    expected_population = _sequence(suite_lock.get("windows"), "suite lock windows")
    expected_windows = _integer(
        suite_lock.get("window_count"), "suite lock window_count", minimum=1
    )
    positions_per_window = _integer(
        suite_lock.get("positions_per_window"),
        "suite lock positions_per_window",
        minimum=1,
    )
    materialized = list(rows)
    if len(materialized) != expected_windows:
        raise ReceiptError(
            f"expected {expected_windows} window receipts, found {len(materialized)}"
        )

    basis_fields = (
        "suite_lock_sha256",
        "teacher_source_model_index_sha256",
        "candidate_artifact_sha256",
    )
    expected_basis: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    seen_window_ids: set[int] = set()
    for index, raw_row in enumerate(materialized):
        row = _mapping(raw_row, f"window row {index}")
        _require_exact_keys(
            row,
            {
                "candidate_artifact_sha256",
                "kld_values",
                "ordinal",
                "positions",
                "schema",
                "source_class",
                "suite_lock_sha256",
                "teacher_source_model_index_sha256",
                "top1_matches",
                "window_id",
            },
            f"window row {index}",
        )
        if row.get("schema") != WINDOW_SCHEMA:
            raise ReceiptError(f"window row {index}: schema must be {WINDOW_SCHEMA}")
        for field in basis_fields:
            value = _sha256(row.get(field), f"window row {index}.{field}")
            if field not in expected_basis:
                expected_basis[field] = value
            elif expected_basis[field] != value:
                raise ReceiptError(f"window row {index}: {field} basis drift")
        if row.get("suite_lock_sha256") != suite_lock.get("suite_lock_sha256"):
            raise ReceiptError(f"window row {index}: suite lock basis drift")
        if row.get("teacher_source_model_index_sha256") != suite_lock.get(
            "teacher_source_model_index_sha256"
        ):
            raise ReceiptError(f"window row {index}: teacher basis differs from suite lock")

        ordinal = _integer(row.get("ordinal"), f"window row {index}.ordinal")
        window_id = _integer(row.get("window_id"), f"window row {index}.window_id")
        if ordinal in seen_ordinals:
            raise ReceiptError(f"duplicate ordinal: {ordinal}")
        if window_id in seen_window_ids:
            raise ReceiptError(f"duplicate window_id: {window_id}")
        seen_ordinals.add(ordinal)
        seen_window_ids.add(window_id)
        if ordinal >= expected_windows:
            raise ReceiptError(f"window row {index}: ordinal outside suite lock")
        expected = _mapping(expected_population[ordinal], f"suite lock window {ordinal}")
        actual_identity = (ordinal, window_id, row.get("source_class"))
        expected_identity = (
            expected.get("ordinal"),
            expected.get("window_id"),
            expected.get("source_class"),
        )
        if actual_identity != expected_identity:
            raise ReceiptError(
                f"window ordinal {ordinal} does not match frozen suite lock: "
                f"actual={actual_identity} expected={expected_identity}"
            )
        if _integer(row.get("positions"), f"window row {index}.positions", minimum=1) != positions_per_window:
            raise ReceiptError(f"window row {index}: positions differ from suite lock")
        matches = _integer(row.get("top1_matches"), f"window row {index}.top1_matches")
        if matches > positions_per_window:
            raise ReceiptError(f"window row {index}: Top-1 matches exceed positions")
        raw_values = _sequence(row.get("kld_values"), f"window row {index}.kld_values")
        if len(raw_values) != positions_per_window:
            raise ReceiptError(f"window row {index}: KLD value count differs from positions")
        kld_values = [
            _binary64(value, f"window row {index}.kld_values[{position}]")
            for position, value in enumerate(raw_values)
        ]
        normalized.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "source_class": str(row["source_class"]),
                "top1_matches": matches,
                "kld_values": kld_values,
            }
        )

    if seen_ordinals != set(range(expected_windows)):
        raise ReceiptError("window ordinals must cover the complete suite lock")
    ordered_rows = sorted(normalized, key=lambda item: int(item["ordinal"]))
    total_positions = expected_windows * positions_per_window
    total_matches = sum(int(row["top1_matches"]) for row in ordered_rows)
    try:
        total_kld = math.fsum(
            chain.from_iterable(row["kld_values"] for row in ordered_rows)
        )
    except (OverflowError, ValueError) as exc:
        raise ReceiptError("global KLD reduction failed") from exc

    classes: dict[str, dict[str, Any]] = {}
    for name in SOURCE_CLASSES:
        class_rows = [row for row in ordered_rows if row["source_class"] == name]
        class_positions = len(class_rows) * positions_per_window
        class_matches = sum(int(row["top1_matches"]) for row in class_rows)
        try:
            class_kld = math.fsum(
                chain.from_iterable(row["kld_values"] for row in class_rows)
            )
        except (OverflowError, ValueError) as exc:
            raise ReceiptError(f"{name} KLD reduction failed") from exc
        classes[name] = {
            "windows": len(class_rows),
            "positions": class_positions,
            "top1_matches": class_matches,
            "top1_rate": _ratio(class_matches, class_positions),
            "kld_mean": repr(class_kld / class_positions),
        }

    return {
        "schema": "banana-smasher.balanced64-aggregate.v1",
        **expected_basis,
        "windows": expected_windows,
        "positions": total_positions,
        "top1_matches": total_matches,
        "top1_rate": _ratio(total_matches, total_positions),
        "kld_mean": repr(total_kld / total_positions),
        "classes": classes,
    }


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Banana Smasher evaluation receipts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify a comparison receipt")
    verify_parser.add_argument("receipt", type=Path)
    verify_parser.add_argument("--suite-lock", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="aggregate window receipts")
    aggregate_parser.add_argument("directory", type=Path)
    aggregate_parser.add_argument("--suite-lock", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    try:
        suite_lock = _mapping(_load_json(args.suite_lock), "suite lock")
        if args.command == "verify":
            comparison = _mapping(_load_json(args.receipt), "receipt")
            summary = verify_result_receipt(
                comparison,
                suite_lock,
            )
            size_accounting = _mapping(
                comparison.get("size_accounting"), "size_accounting"
            )
            linked_name = _nonempty_string(
                size_accounting.get("receipt_path"), "size_accounting.receipt_path"
            )
            if Path(linked_name).name != linked_name:
                raise ReceiptError("size-accounting receipt path must be a basename")
            linked_path = args.receipt.parent / linked_name
            expected_digest = _sha256(
                size_accounting.get("receipt_sha256"), "size_accounting.receipt_sha256"
            )
            observed_digest = hashlib.sha256(linked_path.read_bytes()).hexdigest()
            if observed_digest != expected_digest:
                raise ReceiptError("size-accounting receipt content hash mismatch")
            verify_size_accounting_receipt(
                comparison,
                _mapping(_load_json(linked_path), "size-accounting receipt"),
            )
        else:
            paths = sorted(args.directory.glob("*.json"))
            rows = [_mapping(_load_json(path), str(path)) for path in paths]
            summary = aggregate_windows(rows, suite_lock)

        rendered = json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n"
        if args.command == "aggregate" and args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (
        ArithmeticError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ReceiptError,
    ) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
