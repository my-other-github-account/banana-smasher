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
    if "candidate_manifest_sha256" in artifact:
        _require_exact_keys(
            artifact,
            {
                "artifact_tree_sha256",
                "base_model",
                "candidate_manifest_sha256",
                "identity_status",
                "missing_identity_fields",
                "pack_admission_sha256",
                "variant",
            },
            label,
        )
        if identity_status != "complete-as-recorded" or missing_fields:
            raise ReceiptError(f"{label}: recorded-complete identity status drift")
        for field in ("base_model", "variant"):
            _nonempty_string(artifact.get(field), f"{label}.{field}")
    elif "repository" in artifact:
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
            "results",
            "schema",
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

    seen_models: set[str] = set()
    kld_rows: list[tuple[Decimal, str]] = []
    top1_rows: list[tuple[Decimal, str]] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"results[{index}]")
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
        if "category_metrics" in row:
            expected_row_keys.add("category_metrics")
        _require_exact_keys(row, expected_row_keys, f"results[{index}]")
        model_id = _nonempty_string(row.get("model_id"), f"results[{index}].model_id")
        _nonempty_string(row.get("display_name"), f"{model_id}.display_name")
        _nonempty_string(row.get("vendor"), f"{model_id}.vendor")
        if model_id in seen_models:
            raise ReceiptError(f"duplicate model_id: {model_id}")
        seen_models.add(model_id)
        if row.get("fp") != fp:
            raise ReceiptError(f"{model_id}: FP basis differs from suite lock")

        artifact = _mapping(row.get("artifact"), f"{model_id}.artifact")
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

        if "category_metrics" in row:
            category_metrics = _mapping(
                row.get("category_metrics"), f"{model_id}.category_metrics"
            )
            _require_exact_keys(
                category_metrics,
                set(SOURCE_CLASSES),
                f"{model_id}.category_metrics",
            )
            category_matches = 0
            category_positions = 0
            weighted_category_kld = Decimal(0)
            for name in SOURCE_CLASSES:
                category = _mapping(
                    category_metrics.get(name),
                    f"{model_id}.category_metrics.{name}",
                )
                _require_exact_keys(
                    category,
                    {"kld_mean", "positions", "top1_matches", "top1_rate", "windows"},
                    f"{model_id}.category_metrics.{name}",
                )
                category_windows = _integer(
                    category.get("windows"),
                    f"{model_id}.category_metrics.{name}.windows",
                    minimum=1,
                )
                if category_windows != suite_lock["class_windows"][name]:
                    raise ReceiptError(f"{model_id}: {name} window count differs from suite lock")
                expected_category_positions = category_windows * int(
                    suite_lock["positions_per_window"]
                )
                stored_category_positions = _integer(
                    category.get("positions"),
                    f"{model_id}.category_metrics.{name}.positions",
                    minimum=1,
                )
                if stored_category_positions != expected_category_positions:
                    raise ReceiptError(f"{model_id}: {name} position denominator drift")
                stored_category_matches = _integer(
                    category.get("top1_matches"),
                    f"{model_id}.category_metrics.{name}.top1_matches",
                )
                if stored_category_matches > stored_category_positions:
                    raise ReceiptError(f"{model_id}: {name} Top-1 matches exceed positions")
                stored_category_rate = _decimal(
                    category.get("top1_rate"),
                    f"{model_id}.category_metrics.{name}.top1_rate",
                )
                expected_category_rate = _ratio(
                    stored_category_matches, stored_category_positions
                )
                if stored_category_rate != expected_category_rate:
                    raise ReceiptError(f"{model_id}: {name} Top-1 rate drift")
                category_kld = _decimal(
                    category.get("kld_mean"),
                    f"{model_id}.category_metrics.{name}.kld_mean",
                )
                category_matches += stored_category_matches
                category_positions += stored_category_positions
                weighted_category_kld += category_kld * stored_category_positions
            if category_positions != positions or category_matches != matches:
                raise ReceiptError(f"{model_id}: category Top-1 fan-in differs from global row")
            reaggregated_category_kld = weighted_category_kld / category_positions
            if abs(reaggregated_category_kld - kld_mean) > Decimal("1e-15"):
                raise ReceiptError(f"{model_id}: category KLD fan-in differs from global row")

        wire = _mapping(row.get("wire"), f"{model_id}.wire")
        _require_exact_keys(
            wire,
            {"bytes", "decimal_gb", "normalized_bpw", "parameter_denominator"},
            f"{model_id}.wire",
        )
        if wire.get("parameter_denominator") != denominator:
            raise ReceiptError(f"{model_id}: parameter denominator differs from suite lock")
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
            if source.get("availability") not in {
                "protected-not-distributed",
                "public-distributed",
            }:
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
            summary = verify_result_receipt(
                _mapping(_load_json(args.receipt), "receipt"),
                suite_lock,
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
