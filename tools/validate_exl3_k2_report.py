#!/usr/bin/env python3
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, cast

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA256_SEARCH_RE = re.compile(r"\b[0-9a-f]{64}\b")
FAMILY_RE = re.compile(r"\bEXL3?[ _-]+K[23]\b", re.IGNORECASE)
TIER_RE = re.compile(r"\bK([23])\b", re.IGNORECASE)
METRIC_LABEL_RE = re.compile(r"(?:Top-?1|KLD|bpw|score|matches?|rate|result)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
SHA_LABEL_RE = re.compile(r"\bSHA-?\d+\b", re.IGNORECASE)
QUARANTINE_ID_RE = re.compile(r"\bQ-\d+\b", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
CODED_NUMBER_RE = re.compile(r"^(?:\d[\d,]*/\d[\d,]+|\d[\d,]*(?:\.\d+)?%|\d[\d,]*\.\d+|\d[\d,]*)$")
MISSING_LINE_RE = re.compile(
    r"^[^.!?]*\b(?:is|are|remain|remains)\s+`?MISSING_NOT_A_MEASUREMENT`?\.?$",
    re.IGNORECASE,
)
MEASUREMENT_CLAIM_RE = re.compile(
    r"(?:Top-?1|KLD|bpw|score|matches?|rate|result|\d[\d,]*/\d[\d,]+|%)",
    re.IGNORECASE,
)
FIELD_PREFIX = r"(?<!\S)"
MEASUREMENT_KEY_RE = re.compile(FIELD_PREFIX + r"measurement_key=`([^`]+)`")
BANK_POSITIONS_RE = re.compile(FIELD_PREFIX + r"bank_positions=`([^`]+)`")
INTERVENTION_SCOPE_RE = re.compile(FIELD_PREFIX + r"intervention_scope=`([^`]+)`")
SUPPORT_RE = re.compile(FIELD_PREFIX + r"support=`(\d+)`")
SCORER_RE = re.compile(FIELD_PREFIX + r"scorer_sha256=`([0-9a-f]{64})`")
TERMINAL_RE = re.compile(FIELD_PREFIX + r"terminal_sha256=`([0-9a-f]{64})`")
ARTIFACT_RE = re.compile(FIELD_PREFIX + r"artifact_sha256=`([0-9a-f]{64})`")
TOP1_BINDING_RE = re.compile(FIELD_PREFIX + r"top1=`(\d[\d,]*)/(\d[\d,]*)`")
FLOAT_TEXT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KLD_BINDING_RE = re.compile(FIELD_PREFIX + rf"kld=`({FLOAT_TEXT})`")
WIRE_BPW_BINDING_RE = re.compile(FIELD_PREFIX + rf"wire_bpw=`({FLOAT_TEXT})`")
VISIBLE_HASH_LABEL_PATTERNS = {
    role: re.compile(
        rf"(?<![\w.-]){role}(?:\s+SHA-?256)?\s*(?:=|:|at)\s*[*_`]*([0-9a-f]{{64}})",
        re.IGNORECASE,
    )
    for role in ("terminal", "artifact", "scorer")
}
FRACTION_RE = re.compile(r"(?<![\w.])(\d[\d,]*)/(\d[\d,]*)(?![\w.])")
PERCENT_RE = re.compile(rf"(?<![\w.])({FLOAT_TEXT})%(?![\w.])")
DECIMAL_RE = re.compile(r"(?<![\w.])([+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+))(?![\w.%])")
INTEGER_RE = re.compile(r"(?<![\w.])([+-]?\d[\d,]*)(?![\w./%])")
FIELD_VALUE_RE = re.compile(
    r"(?<!\S)(?:measurement_key|bank_positions|intervention_scope|support|scorer_sha256|terminal_sha256|artifact_sha256|top1|kld|wire_bpw)=`[^`]*`"
)
TOP_LEVEL_KEYS = {"basis_sha256", "decision", "measurement_scopes", "missing_scopes", "quarantine", "schema"}
DECISION_KEYS = {"current_scope_key", "historical_scope_key", "scope_substitution_forbidden", "status", "top1_first"}
RECORD_KEYS = {
    "artifact",
    "authorized_use",
    "evidence",
    "lifecycle",
    "measurement",
    "per_row_evidence",
    "referenced_by",
    "reporting_binding",
    "scope",
    "scope_key",
    "status",
}
MEASUREMENT_KEYS = {
    "kld_semantics",
    "mean_support_renormalized_kld",
    "top1_matches",
    "top1_positions",
    "top1_rate",
    "top1_semantics",
    "wire",
}
REPORTING_BINDING_KEYS = {
    "artifact_sha256",
    "bank_positions",
    "intervention_scope",
    "measurement_key",
    "scorer_sha256",
    "support",
    "terminal_sha256",
}
EXPECTED_MISSING_SCOPE_KEYS = {
    ("EXL3 K2", "ff0731/exl3-k2/routed-only-native-rest/exact64/positions65536/support8192"),
    ("EXL3 K3", "ff0731/exl3-k3/routed-only-native-rest/exact64/positions65536/support8192"),
}
EXPECTED_COMPARISON_GATES = {
    (
        "476ee64e7e919bfaa851ccc8e0e1e3e760831dd547e7c9d1dfdc837b2126a0da",
        "raw-shared-method-history",
        "receipts/Q2_V6_EXACT512_VS_AUTHENTIC_EXL3_K2_ROWS2_GATE.json",
    ),
    (
        "cd30688a38e863921525e233548ce0ecd3326a33a11ca7d741f11672ae3ee395",
        "current-ordinary-q2-rms-one-encode",
        "comparison-gates/current-ordinary-q2-rms-one-encode.json",
    ),
}


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _strict_int(value: Any, field: str) -> int:
    _require(type(value) is int, f"{field} must be an integer")
    assert type(value) is int
    return value


def _finite_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    _require(type(value) is float and math.isfinite(value), f"{field} must be a finite float")
    assert type(value) is float
    if minimum is not None:
        _require(value >= minimum, f"{field} must be >= {minimum}")
    return value


def _exact_keys(value: Any, allowed: set[str], field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(set(value) == allowed, f"unexpected keys at {field}")
    return value


def _sha256(value: Any, field: str) -> None:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{field} must be a lowercase SHA-256")


def _identity(value: Any, field: str) -> None:
    _require(isinstance(value, dict), f"missing {field}")
    status = value.get("status")
    _require(status in {"BOUND", "NOT_APPLICABLE"}, f"invalid {field} status")
    if status == "BOUND":
        _exact_keys(value, {"sha256", "status"}, field)
        _sha256(value.get("sha256"), f"{field}.sha256")
    else:
        _exact_keys(value, {"reason", "status"}, field)
        _require(isinstance(value.get("reason"), str) and value["reason"], f"missing {field} reason")


def _reject_nonfinite(value: Any, field: str = "ledger") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"nonfinite KLD or number at {field}")
    elif isinstance(value, dict):
        for key, member in value.items():
            _reject_nonfinite(member, f"{field}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _reject_nonfinite(member, f"{field}[{index}]")


def _reject_negative_kld(value: Any, field: str = "ledger") -> None:
    if isinstance(value, dict):
        for key, member in value.items():
            if "kld" in key.lower() and isinstance(member, (int, float)) and not isinstance(member, bool):
                _require(math.isfinite(member) and member >= 0, f"nonnegative KLD required at {field}.{key}")
            _reject_negative_kld(member, f"{field}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _reject_negative_kld(member, f"{field}[{index}]")


def _relative_path(value: Any, field: str) -> None:
    _require(isinstance(value, str) and bool(value), f"{field} must be a nonempty relative path")
    assert isinstance(value, str)
    path = PurePosixPath(value)
    _require(
        value != "."
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and not re.match(r"^[A-Za-z]:", value)
        and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) is None,
        f"{field} must be a normalized relative path",
    )


def _safe_description(value: Any, field: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a nonempty description")
    assert isinstance(value, str)
    _require(
        value != "."
        and ".." not in value
        and "\\" not in value
        and not value.startswith(("/", "~"))
        and re.search(r"(?:^|\s)[A-Za-z]:[\\/]", value) is None
        and re.search(r"\b[A-Za-z][A-Za-z0-9+.-]*://", value) is None,
        f"{field} must be a public-safe description, not a locator",
    )


def _sha256_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str) and SHA256_RE.fullmatch(value) is not None:
        values.add(value)
    elif isinstance(value, dict):
        for member in value.values():
            values.update(_sha256_values(member))
    elif isinstance(value, list):
        for member in value:
            values.update(_sha256_values(member))
    return values


def _markdown_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.split("|")]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


def _table_header_names(cells: list[str]) -> list[str] | None:
    names = [re.sub(r"[*_`]", "", cell).strip().lower() for cell in cells]
    recognized = {
        "candidate top-1",
        "k2 top-1",
        "candidate kld",
        "k2 kld",
        "decision",
        "top-1 first",
        "kld",
        "selected wire",
        "terminal sha-256",
        "artifact sha-256",
    }
    return names if any(name in recognized for name in names) else None


def _validate_table_semantics(
    headers: list[str],
    cells: list[str],
    record: dict[str, Any],
    line_number: int,
    failures: list[str],
) -> None:
    if len(headers) != len(cells):
        failures.append(f"line {line_number}: table row does not match governed header")
        return
    measurement = record["measurement"]
    terminal_sha256 = record["evidence"]["terminal"]["sha256"]
    artifact_sha256 = record["artifact"]["artifact_sha256"]
    gates = [gate for gate in record.get("comparison_gates", []) if gate["sha256"] in SHA256_SEARCH_RE.findall(" ".join(cells))]
    gate = gates[0] if len(gates) == 1 else None
    for header, cell in zip(headers, cells):
        display_header = header.replace("top-1", "Top-1").replace("kld", "KLD")
        fractions = FRACTION_RE.findall(cell)
        decimals = DECIMAL_RE.findall(cell)
        hashes = SHA256_SEARCH_RE.findall(cell)
        if header == "top-1 first" and fractions:
            actual = tuple(int(part.replace(",", "")) for part in fractions[0])
            expected = (measurement["top1_matches"], measurement["top1_positions"])
            if actual != expected:
                failures.append(f"line {line_number}: table Top-1 does not bind measurement")
        elif header == "kld" and decimals:
            if _decimal(decimals[0]) != Decimal(str(measurement["mean_support_renormalized_kld"])):
                failures.append(f"line {line_number}: table KLD does not bind measurement")
        elif header == "selected wire" and decimals:
            if _decimal(decimals[0]) != Decimal(str(measurement["wire"]["selected_payload_bpw"])):
                failures.append(f"line {line_number}: table wire does not bind measurement")
        elif header == "terminal sha-256" and hashes and hashes[0] != terminal_sha256:
            failures.append(f"line {line_number}: visible terminal SHA-256 does not bind measurement")
        elif header == "artifact sha-256" and hashes and hashes[0] != artifact_sha256:
            failures.append(f"line {line_number}: visible artifact SHA-256 does not bind measurement")
        elif header in {"candidate top-1", "k2 top-1", "candidate kld", "k2 kld", "decision"}:
            if gate is None:
                failures.append(f"line {line_number}: governed gate row lacks exactly one known gate")
                continue
            arm_name = "q2" if header.startswith("candidate ") else "k2"
            if header.endswith("top-1") and fractions:
                actual = tuple(int(part.replace(",", "")) for part in fractions[0])
                expected = (gate[arm_name]["top1_matches"], gate[arm_name]["top1_positions"])
                if actual != expected:
                    failures.append(f"line {line_number}: {display_header} does not bind gate")
            elif header.endswith("kld") and decimals:
                if _decimal(decimals[0]) != Decimal(str(gate[arm_name]["mean_support_renormalized_kld"])):
                    failures.append(f"line {line_number}: {display_header} does not bind gate")
            elif header == "decision":
                decisions = re.findall(r"\b(?:GREEN|RED)\b", cell)
                if not decisions or decisions[0] != gate["decision"]:
                    failures.append(f"line {line_number}: gate decision does not bind gate")


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"nonfinite JSON constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValidationError(f"invalid decimal: {value}") from exc


def _intervention_slug(intervention: dict[str, Any]) -> str:
    layer = intervention.get("layer")
    projections = intervention.get("projections")
    experts = intervention.get("experts")
    if not isinstance(layer, int) or not isinstance(projections, list) or not projections:
        raise ValidationError("invalid intervention slug inputs")
    if isinstance(experts, list):
        if not experts or not all(isinstance(expert, int) for expert in experts):
            raise ValidationError("invalid intervention experts")
        expert_slug = "-".join(f"e{expert:03d}" for expert in experts)
    elif isinstance(experts, dict):
        first = experts.get("first")
        last = experts.get("last")
        if not isinstance(first, int) or not isinstance(last, int):
            raise ValidationError("invalid intervention expert range")
        expert_slug = f"experts{first:03d}-{last:03d}"
    else:
        raise ValidationError("invalid intervention experts")
    return "-".join((f"l{layer:03d}", expert_slug, *(str(projection) for projection in projections)))


def validate_ledger(data: dict[str, Any]) -> None:
    _require(set(data) == TOP_LEVEL_KEYS, "unexpected top-level keys")
    _reject_nonfinite(data)
    _reject_negative_kld(data)
    _require(data.get("schema") == "banana-smasher-exl3-k2-scope-ledger-v1", "unexpected ledger schema")
    _sha256(data.get("basis_sha256"), "basis_sha256")
    decision = _exact_keys(data.get("decision"), DECISION_KEYS, "decision")
    _require(decision.get("scope_substitution_forbidden") is True, "scope substitution must be forbidden")
    _require(decision.get("status") == "USE_SCOPE_KEYED_CURRENT_WHOLE_L034_ONLY", "invalid decision status")
    _require(isinstance(decision.get("top1_first"), str) and decision["top1_first"], "missing Top-1-first decision")
    scopes_value = data.get("measurement_scopes")
    _require(isinstance(scopes_value, dict) and bool(scopes_value), "measurement_scopes must be a nonempty object")
    assert isinstance(scopes_value, dict)
    scopes = scopes_value
    _require(decision.get("current_scope_key") in scopes, "current scope key is not present")
    _require(decision.get("historical_scope_key") in scopes, "historical scope key is not present")
    _require(
        scopes[decision["current_scope_key"]].get("status") == "CURRENT_AUTHORITATIVE"
        and scopes[decision["historical_scope_key"]].get("status") == "HISTORICAL_AUTHENTICATED",
        "decision scope role mismatch",
    )

    for scope_key, record in scopes.items():
        status = record.get("status") if isinstance(record, dict) else None
        _require(status in {"CURRENT_AUTHORITATIVE", "HISTORICAL_AUTHENTICATED"}, f"invalid measurement status: {scope_key}")
        is_current = status == "CURRENT_AUTHORITATIVE"
        expected_record_keys = RECORD_KEYS | ({"comparison_gates"} if is_current else set())
        record = _exact_keys(record, expected_record_keys, f"measurement_scopes.{scope_key}")
        _require(record.get("scope_key") == scope_key, f"scope key mismatch: {scope_key}")
        _require(isinstance(record.get("authorized_use"), str) and record["authorized_use"], f"missing authorized use: {scope_key}")
        _require(isinstance(record.get("lifecycle"), str) and record["lifecycle"], f"missing lifecycle: {scope_key}")
        _require(
            isinstance(record.get("referenced_by"), list)
            and bool(record["referenced_by"])
            and all(isinstance(item, str) and item for item in record["referenced_by"]),
            f"invalid references: {scope_key}",
        )

        scope_keys = (
            {
                "base_index_sha256",
                "basis_sha256",
                "intervention",
                "positions_per_row",
                "prefix_identity",
                "prefix_payload_sha256",
                "reducer_semantics",
                "reducer_sha256",
                "row_ids",
                "scorer_semantics",
                "scorer_sha256",
                "suffix_identity",
                "suffix_layers",
                "suffix_shard_sha256",
                "support_width",
                "teacher_mode",
                "teacher_row_payloads",
                "teacher_sha256",
                "teacher_support_sha256",
                "window_manifest_sha256",
            }
            if is_current
            else {
                "base_index_sha256",
                "basis_sha256",
                "intervention",
                "positions_per_row",
                "prefix_identity",
                "reducer_semantics",
                "reducer_sha256",
                "row_ids",
                "scorer_semantics",
                "scorer_sha256",
                "suffix_identity",
                "support_width",
                "teacher_mode",
                "teacher_sha256",
                "window_manifest_sha256",
                "wrapper_sha256",
            }
        )
        scope = _exact_keys(record.get("scope"), scope_keys, f"scope: {scope_key}")
        _require(scope.get("base_index_sha256") == data["basis_sha256"], f"base/index mismatch: {scope_key}")
        _require(scope.get("basis_sha256") == data["basis_sha256"], f"basis mismatch: {scope_key}")
        row_ids_value = scope.get("row_ids")
        _require(
            isinstance(row_ids_value, list)
            and bool(row_ids_value)
            and len(row_ids_value) == len(set(row_ids_value))
            and all(isinstance(row_id, str) and row_id.isdigit() for row_id in row_ids_value),
            f"missing or invalid row bank: {scope_key}",
        )
        row_ids = cast(list[str], row_ids_value)
        positions_per_row = _strict_int(scope.get("positions_per_row"), f"positions_per_row: {scope_key}")
        support_width = _strict_int(scope.get("support_width"), f"support_width: {scope_key}")
        _require(positions_per_row > 0, f"invalid position cutoff: {scope_key}")
        _require(support_width > 0, f"invalid support width: {scope_key}")
        _require(isinstance(scope.get("scorer_semantics"), str) and scope["scorer_semantics"], f"missing scorer semantics: {scope_key}")
        _require(isinstance(scope.get("reducer_semantics"), str) and scope["reducer_semantics"], f"missing reducer semantics: {scope_key}")
        _sha256(scope.get("scorer_sha256"), f"scorer_sha256: {scope_key}")
        _sha256(scope.get("reducer_sha256"), f"reducer_sha256: {scope_key}")
        _sha256(scope.get("teacher_sha256"), f"teacher_sha256: {scope_key}")
        _identity(scope.get("prefix_identity"), f"prefix_identity: {scope_key}")
        _identity(scope.get("suffix_identity"), f"suffix_identity: {scope_key}")

        intervention_keys = (
            {"changed_tensors", "experts", "layer", "nontarget_tensors_changed", "projections", "roster_sha256"}
            if is_current
            else {"changed_tensors", "experts", "layer", "nontarget_source_identical", "projections"}
        )
        intervention = _exact_keys(scope.get("intervention"), intervention_keys, f"intervention: {scope_key}")
        _require(_is_int(intervention.get("changed_tensors")) and intervention["changed_tensors"] > 0, f"invalid changed tensors: {scope_key}")
        _require(_is_int(intervention.get("layer")) and intervention["layer"] >= 0, f"invalid intervention layer: {scope_key}")
        _require(
            isinstance(intervention.get("projections"), list)
            and bool(intervention["projections"])
            and len(intervention["projections"]) == len(set(intervention["projections"]))
            and all(isinstance(item, str) and item for item in intervention["projections"]),
            f"invalid intervention projections: {scope_key}",
        )
        if is_current:
            experts = _exact_keys(intervention.get("experts"), {"count", "first", "last"}, f"experts: {scope_key}")
            _require(
                all(_is_int(experts.get(field)) for field in ("count", "first", "last"))
                and experts["count"] == experts["last"] - experts["first"] + 1
                and experts["first"] >= 0,
                f"invalid intervention expert range: {scope_key}",
            )
            _require(
                intervention["changed_tensors"] == experts["count"] * len(intervention["projections"]),
                f"changed tensors do not close to roster: {scope_key}",
            )
            _require(intervention.get("nontarget_tensors_changed") == 0, f"nontarget tensors changed: {scope_key}")
            _sha256(intervention.get("roster_sha256"), f"roster_sha256: {scope_key}")
            for field in ("prefix_payload_sha256", "suffix_shard_sha256", "teacher_support_sha256"):
                _sha256(scope.get(field), f"{field}: {scope_key}")
            _require(scope["prefix_identity"].get("sha256") == scope["prefix_payload_sha256"], f"prefix identity mismatch: {scope_key}")
            _require(scope["suffix_identity"].get("sha256") == scope["suffix_shard_sha256"], f"suffix identity mismatch: {scope_key}")
            _require(scope["teacher_support_sha256"] == scope["teacher_sha256"], f"teacher support mismatch: {scope_key}")
            _require(
                isinstance(scope.get("suffix_layers"), list)
                and bool(scope["suffix_layers"])
                and all(_is_int(layer) and layer >= 0 for layer in scope["suffix_layers"]),
                f"invalid suffix layers: {scope_key}",
            )
            teacher_rows = _exact_keys(scope.get("teacher_row_payloads"), set(row_ids), f"teacher row payloads: {scope_key}")
            for row_id, sha256 in teacher_rows.items():
                _sha256(sha256, f"teacher row payload: {scope_key}/{row_id}")
        else:
            _require(
                isinstance(intervention.get("experts"), list)
                and bool(intervention["experts"])
                and intervention["experts"] == sorted(set(intervention["experts"]))
                and all(_is_int(expert) and expert >= 0 for expert in intervention["experts"]),
                f"invalid intervention experts: {scope_key}",
            )
            _require(
                intervention["changed_tensors"] == len(intervention["experts"]) * len(intervention["projections"]),
                f"changed tensors do not close to roster: {scope_key}",
            )
            _require(intervention.get("nontarget_source_identical") is True, f"nontarget source mismatch: {scope_key}")
            _sha256(scope.get("wrapper_sha256"), f"wrapper_sha256: {scope_key}")

        artifact_keys = (
            {"artifact_file_bytes", "artifact_kind", "artifact_sha256", "candidate_payload_sha256"}
            if is_current
            else {"artifact_kind", "artifact_sha256", "build_terminal_sha256", "candidate_payload_sha256", "codec_members"}
        )
        artifact = _exact_keys(record.get("artifact"), artifact_keys, f"artifact: {scope_key}")
        _sha256(artifact.get("artifact_sha256"), f"artifact_sha256: {scope_key}")
        _sha256(artifact.get("candidate_payload_sha256"), f"candidate_payload_sha256: {scope_key}")
        _require(isinstance(artifact.get("artifact_kind"), str) and artifact["artifact_kind"], f"missing artifact kind: {scope_key}")
        if is_current:
            _require(_is_int(artifact.get("artifact_file_bytes")) and artifact["artifact_file_bytes"] > 0, f"invalid artifact file bytes: {scope_key}")
        else:
            _sha256(artifact.get("build_terminal_sha256"), f"build_terminal_sha256: {scope_key}")
            codec_members = _exact_keys(artifact.get("codec_members"), {"E000_down", "E001_down"}, f"codec members: {scope_key}")
            for name, member in codec_members.items():
                member = _exact_keys(member, {"decoded_tensor_sha256", "packed_sha256"}, f"codec member: {scope_key}/{name}")
                _sha256(member.get("decoded_tensor_sha256"), f"decoded tensor: {scope_key}/{name}")
                _sha256(member.get("packed_sha256"), f"packed tensor: {scope_key}/{name}")

        evidence_keys = (
            {"direct_file_rehash", "manifest", "score_artifact", "terminal"}
            if is_current
            else {"direct_file_rehash", "manifest", "signal", "teacher_artifact_sha256", "terminal"}
        )
        evidence = _exact_keys(record.get("evidence"), evidence_keys, f"evidence: {scope_key}")
        _require(evidence.get("direct_file_rehash") is True, f"terminal not directly rehashed: {scope_key}")
        manifest = _exact_keys(evidence.get("manifest"), {"name", "sha256"}, f"manifest: {scope_key}")
        _require(isinstance(manifest.get("name"), str) and manifest["name"], f"missing manifest name: {scope_key}")
        _sha256(manifest.get("sha256"), f"window manifest: {scope_key}")
        _require(scope.get("window_manifest_sha256") == manifest["sha256"], f"window manifest mismatch: {scope_key}")
        terminal_keys = {"name", "relative_path", "sha256", "status"} | ({"decision"} if not is_current else set())
        terminal_value = evidence.get("terminal")
        if isinstance(terminal_value, dict):
            _relative_path(terminal_value.get("relative_path"), f"terminal path: {scope_key}")
        terminal = _exact_keys(terminal_value, terminal_keys, f"terminal: {scope_key}")
        _require(isinstance(terminal.get("name"), str) and terminal["name"], f"missing terminal name: {scope_key}")
        _sha256(terminal.get("sha256"), f"terminal.sha256: {scope_key}")
        _relative_path(terminal.get("relative_path"), f"terminal path: {scope_key}")
        _require(terminal.get("status") == "PASS", f"invalid terminal status: {scope_key}")
        if is_current:
            score_artifact = _exact_keys(evidence.get("score_artifact"), {"name", "sha256"}, f"score artifact: {scope_key}")
            _require(isinstance(score_artifact.get("name"), str) and score_artifact["name"], f"missing score artifact name: {scope_key}")
            _sha256(score_artifact.get("sha256"), f"score artifact: {scope_key}")
        else:
            _require(terminal.get("decision") in {"GREEN", "RED"}, f"invalid terminal decision: {scope_key}")
            _sha256(evidence.get("teacher_artifact_sha256"), f"teacher_artifact_sha256: {scope_key}")
            signal = _exact_keys(evidence.get("signal"), {"name", "sha256"}, f"signal: {scope_key}")
            _require(isinstance(signal.get("name"), str) and signal["name"], f"missing signal name: {scope_key}")
            _sha256(signal.get("sha256"), f"signal: {scope_key}")

        measurement = _exact_keys(record.get("measurement"), MEASUREMENT_KEYS, f"measurement: {scope_key}")
        matches = _strict_int(measurement.get("top1_matches"), f"Top-1 matches: {scope_key}")
        positions = _strict_int(measurement.get("top1_positions"), f"Top-1 positions: {scope_key}")
        rate = _finite_float(measurement.get("top1_rate"), f"Top-1 rate: {scope_key}", minimum=0.0)
        _require(positions > 0 and 0 <= matches <= positions, f"invalid Top-1 counts: {scope_key}")
        _require(rate == matches / positions, f"Top-1 rate does not close: {scope_key}")
        expected_positions = len(row_ids) * positions_per_row
        _require(positions == expected_positions, f"row-bank position count does not close: {scope_key}")
        mean_kld = measurement.get("mean_support_renormalized_kld")
        _require(isinstance(mean_kld, float) and math.isfinite(mean_kld) and mean_kld >= 0, f"missing finite nonnegative KLD: {scope_key}")
        _require(isinstance(measurement.get("kld_semantics"), str) and measurement["kld_semantics"], f"missing KLD semantics: {scope_key}")
        _require(isinstance(measurement.get("top1_semantics"), str) and measurement["top1_semantics"], f"missing Top-1 semantics: {scope_key}")
        wire_keys = (
            {"artifact_file_bpw", "denominator_weights", "numerator_bits", "selected_payload_bpw", "selected_tensor_payload_bytes"}
            if is_current
            else {"denominator_weights", "numerator_bits", "selected_payload_bpw"}
        )
        wire_value = measurement.get("wire")
        if isinstance(wire_value, dict):
            _require(_is_int(wire_value.get("numerator_bits")), f"invalid wire numerator: {scope_key}")
            _require(_is_int(wire_value.get("denominator_weights")), f"invalid wire denominator: {scope_key}")
        wire = _exact_keys(wire_value, wire_keys, f"wire: {scope_key}")
        numerator = _strict_int(wire.get("numerator_bits"), f"wire numerator: {scope_key}")
        denominator = _strict_int(wire.get("denominator_weights"), f"wire denominator: {scope_key}")
        _require(numerator > 0, f"invalid wire numerator: {scope_key}")
        _require(denominator > 0, f"invalid wire denominator: {scope_key}")
        selected_payload_bpw = _finite_float(wire.get("selected_payload_bpw"), f"wire bpw: {scope_key}", minimum=0.0)
        _require(selected_payload_bpw == numerator / denominator, f"wire bpw does not close: {scope_key}")
        if is_current:
            selected_bytes = _strict_int(wire.get("selected_tensor_payload_bytes"), f"selected payload bytes: {scope_key}")
            _require(selected_bytes > 0, f"invalid selected payload bytes: {scope_key}")
            _require(numerator == selected_bytes * 8, f"wire numerator does not close to selected payload bytes: {scope_key}")
            artifact_file_bpw = wire.get("artifact_file_bpw")
            _require(
                isinstance(artifact_file_bpw, float)
                and math.isfinite(artifact_file_bpw)
                and artifact_file_bpw == artifact["artifact_file_bytes"] * 8 / denominator,
                f"artifact file bpw does not close: {scope_key}",
            )

        binding = _exact_keys(record.get("reporting_binding"), REPORTING_BINDING_KEYS, f"reporting binding: {scope_key}")
        key_parts = scope_key.split("/")
        intervention_slug = _intervention_slug(intervention)
        expected_teacher_key = "eagerteacher" if scope["teacher_mode"] == "eagerteacher" else f"teacher{scope['teacher_sha256'][:8]}"
        expected_key_parts = [
            "ff0731",
            "exl3-k2",
            intervention_slug,
            f"rows{'-'.join(row_ids)}",
            f"positions{positions_per_row}",
            f"support{support_width}",
            expected_teacher_key,
            f"scorer{scope['scorer_sha256'][:8]}",
        ]
        _require(
            len(key_parts) == 8
            and key_parts[1] == "exl3-k2"
            and key_parts == expected_key_parts,
            f"invalid measurement key scope closure: {scope_key}",
        )
        expected_bank_positions = f"rows{'-'.join(row_ids)}:{positions}"
        expected_binding = {
            "artifact_sha256": record["artifact"]["artifact_sha256"],
            "bank_positions": expected_bank_positions,
            "intervention_scope": key_parts[2],
            "measurement_key": scope_key,
            "scorer_sha256": scope["scorer_sha256"],
            "support": support_width,
            "terminal_sha256": terminal["sha256"],
        }
        for field, expected in expected_binding.items():
            _require(binding.get(field) == expected, f"reporting binding mismatch for {field}: {scope_key}")

        row_evidence_value = record.get("per_row_evidence")
        _require(isinstance(row_evidence_value, dict), f"per-row evidence must be an object: {scope_key}")
        row_evidence = cast(dict[str, Any], row_evidence_value)
        if row_evidence.get("status") == "REAGGREGATED_EXACT_MATCH":
            row_evidence = _exact_keys(row_evidence, {"checks", "method", "rows", "status"}, f"per-row evidence: {scope_key}")
            checks = _exact_keys(
                row_evidence.get("checks"),
                {"aggregate_matches_expected", "aggregate_mean_kld_expected", "positions_expected"},
                f"per-row checks: {scope_key}",
            )
            _require(all(value is True for value in checks.values()), f"per-row checks are not all true: {scope_key}")
            _require(isinstance(row_evidence.get("method"), str) and row_evidence["method"], f"missing reaggregation method: {scope_key}")
            rows_value = row_evidence.get("rows", [])
            _require(isinstance(rows_value, list), f"rows must be a list: {scope_key}")
            rows = cast(list[dict[str, Any]], rows_value)
            for index, row in enumerate(rows):
                rows[index] = _exact_keys(
                    row,
                    {"candidate_payload_sha256", "mean_support_renormalized_kld", "positions", "row_id", "teacher_payload_sha256", "top1_matches", "top1_rate"},
                    f"per-row evidence: {scope_key}/{index}",
                )
            _require([row.get("row_id") for row in rows] == row_ids, f"row evidence order mismatch: {scope_key}")
            _require(sum(row["positions"] for row in rows) == positions, f"row evidence positions do not close: {scope_key}")
            _require(sum(row["top1_matches"] for row in rows) == matches, f"row Top-1 counts do not close: {scope_key}")
            for row in rows:
                row_matches = _strict_int(row.get("top1_matches"), f"row Top-1 matches: {scope_key}/{row.get('row_id')}")
                row_positions = _strict_int(row.get("positions"), f"row positions: {scope_key}/{row.get('row_id')}")
                _require(
                    row_positions == positions_per_row
                    and 0 <= row_matches <= row_positions,
                    f"invalid row Top-1 counts: {scope_key}/{row.get('row_id')}",
                )
                _require(row.get("top1_rate") == row_matches / row_positions, f"row Top-1 rate does not close: {scope_key}/{row.get('row_id')}")
                _require(
                    isinstance(row.get("mean_support_renormalized_kld"), float)
                    and math.isfinite(row["mean_support_renormalized_kld"])
                    and row["mean_support_renormalized_kld"] >= 0,
                    f"row KLD must be finite and nonnegative: {scope_key}/{row.get('row_id')}",
                )
                _sha256(row.get("candidate_payload_sha256"), f"candidate row payload: {scope_key}/{row.get('row_id')}")
                _sha256(row.get("teacher_payload_sha256"), f"teacher row payload: {scope_key}/{row.get('row_id')}")
            rebuilt_kld = math.fsum(row["mean_support_renormalized_kld"] * row["positions"] for row in rows) / positions
            _require(math.isclose(rebuilt_kld, measurement["mean_support_renormalized_kld"], rel_tol=0.0, abs_tol=1e-15), f"row KLD does not close: {scope_key}")
        else:
            row_evidence = _exact_keys(
                row_evidence,
                {"available_inputs", "missing_inputs", "reason", "status"},
                f"per-row evidence: {scope_key}",
            )
            _require(row_evidence.get("status") == "NOT_PERSISTED", f"unsupported row evidence status: {scope_key}")
            _require(
                isinstance(row_evidence.get("available_inputs"), list)
                and bool(row_evidence["available_inputs"])
                and isinstance(row_evidence.get("missing_inputs"), list)
                and bool(row_evidence["missing_inputs"])
                and isinstance(row_evidence.get("reason"), str)
                and bool(row_evidence["reason"]),
                f"missing row-evidence explanation: {scope_key}",
            )

        gates_value = record.get("comparison_gates", [])
        _require(isinstance(gates_value, list), f"comparison gates must be a list: {scope_key}")
        if is_current:
            for gate in gates_value:
                if isinstance(gate, dict):
                    _relative_path(gate.get("relative_path"), f"comparison gate path: {scope_key}")
            gate_inventory = [
                (gate.get("sha256"), gate.get("role"), gate.get("relative_path"))
                for gate in gates_value
                if isinstance(gate, dict)
            ]
            _require(
                len(gate_inventory) == len(gates_value)
                and len(gate_inventory) == len(set(gate_inventory))
                and set(gate_inventory) == EXPECTED_COMPARISON_GATES,
                f"comparison gate inventory mismatch: {scope_key}",
            )
        for gate in gates_value:
            gate_keys = {"decision", "expand_full_train8", "k2", "q2", "relative_path", "role", "sha256"}
            if "source_locator" in gate:
                gate_keys.add("source_locator")
            _relative_path(gate.get("relative_path"), f"comparison gate path: {scope_key}")
            gate = _exact_keys(gate, gate_keys, f"comparison gate: {scope_key}")
            _sha256(gate.get("sha256"), f"comparison gate: {scope_key}")
            _relative_path(gate.get("relative_path"), f"comparison gate path: {scope_key}")
            _require(gate.get("decision") in {"GREEN", "RED"}, f"invalid gate decision: {scope_key}")
            _require(gate.get("expand_full_train8") is False, f"invalid expansion decision: {scope_key}")
            _require(isinstance(gate.get("role"), str) and gate["role"], f"missing gate role: {scope_key}")
            if "source_locator" in gate:
                _safe_description(gate["source_locator"], f"source locator: {scope_key}")
            k2_keys = {"candidate_payload_sha256", "mean_support_renormalized_kld", "top1_matches", "top1_positions"}
            q2_keys = set(k2_keys)
            if "selected_payload_bpw" in gate["k2"]:
                k2_keys.add("selected_payload_bpw")
            if "full_wire_bpw" in gate["q2"]:
                q2_keys.add("full_wire_bpw")
            k2 = _exact_keys(gate.get("k2"), k2_keys, f"comparison gate K2: {scope_key}")
            q2 = _exact_keys(gate.get("q2"), q2_keys, f"comparison gate Q2: {scope_key}")
            for arm_name, arm in (("k2", k2), ("q2", q2)):
                arm_matches = _strict_int(arm.get("top1_matches"), f"gate Top-1 matches: {scope_key}/{arm_name}")
                arm_positions = _strict_int(arm.get("top1_positions"), f"gate Top-1 positions: {scope_key}/{arm_name}")
                arm_kld = arm.get("mean_support_renormalized_kld")
                _require(
                    arm_positions > 0 and 0 <= arm_matches <= arm_positions,
                    f"invalid gate Top-1: {scope_key}/{arm_name}",
                )
                _require(isinstance(arm_kld, float) and math.isfinite(arm_kld) and arm_kld >= 0, f"gate KLD must be nonnegative: {scope_key}/{arm_name}")
                _sha256(arm.get("candidate_payload_sha256"), f"gate candidate payload: {scope_key}/{arm_name}")
                for wire_field in ("selected_payload_bpw", "full_wire_bpw"):
                    if wire_field in arm:
                        wire_bpw = arm[wire_field]
                        _require(
                            isinstance(wire_bpw, float) and math.isfinite(wire_bpw) and wire_bpw > 0,
                            f"gate wire bpw must be positive: {scope_key}/{arm_name}",
                        )
            _require(k2.get("top1_matches") == matches and k2.get("top1_positions") == positions, f"gate Top-1 does not bind scope: {scope_key}")
            _require(k2.get("mean_support_renormalized_kld") == measurement["mean_support_renormalized_kld"], f"gate KLD does not bind scope: {scope_key}")
            _require(k2.get("candidate_payload_sha256") == artifact["candidate_payload_sha256"], f"gate payload does not bind scope: {scope_key}")
            _require(q2.get("top1_positions") == positions, f"gate denominator does not bind scope: {scope_key}")
            expected_decision = (
                "GREEN"
                if q2["top1_matches"] > k2["top1_matches"]
                or (
                    q2["top1_matches"] == k2["top1_matches"]
                    and q2["mean_support_renormalized_kld"] <= k2["mean_support_renormalized_kld"]
                )
                else "RED"
            )
            _require(gate.get("decision") == expected_decision, f"gate decision does not close: {scope_key}")
            if "selected_payload_bpw" in k2:
                _require(k2["selected_payload_bpw"] == selected_payload_bpw, f"gate wire does not bind scope: {scope_key}")

    quarantine_value = data.get("quarantine")
    _require(isinstance(quarantine_value, list) and bool(quarantine_value), "quarantine ledger must be nonempty")
    assert isinstance(quarantine_value, list)
    quarantine = quarantine_value
    ids = [item.get("id") for item in quarantine]
    _require(len(ids) == len(set(ids)), "duplicate quarantine id")
    quarantine_keys = {
        "terminal-digest-mismatch": {"claim", "claimed_sha256", "id", "observed_sha256", "resolution", "status", "type"},
        "gate-content-conflict": {"claimed", "gate_sha256", "id", "observed", "resolution", "status", "type"},
        "scope-substitution": {"claim", "id", "resolution", "status", "type"},
        "old-teacher-result": {"claim", "id", "resolution", "status", "type"},
        "mislabeled-input-reference": {"claim", "id", "resolution", "source_document_sha256", "status", "type"},
        "missing-scope": {"claim", "id", "resolution", "status", "type"},
    }
    for item in quarantine:
        item_type = item.get("type") if isinstance(item, dict) else None
        _require(item_type in quarantine_keys, "invalid quarantine type")
        assert isinstance(item_type, str)
        item = _exact_keys(item, quarantine_keys[item_type], f"quarantine.{item_type}")
        _require(item.get("status") == "QUARANTINED", "quarantine entry missing status")
        _require(
            isinstance(item.get("id"), str) and QUARANTINE_ID_RE.fullmatch(item["id"]) is not None,
            "invalid quarantine id",
        )
        _require(isinstance(item.get("resolution"), str) and item["resolution"], "missing quarantine resolution")
        for field in ("observed_sha256", "gate_sha256", "source_document_sha256"):
            if field in item:
                _sha256(item[field], f"quarantine.{item_type}.{field}")
        if item_type == "terminal-digest-mismatch":
            _require(
                isinstance(item.get("claimed_sha256"), str)
                and bool(item["claimed_sha256"])
                and item["claimed_sha256"] != item["observed_sha256"],
                "quarantine terminal-digest mismatch must preserve distinct claimed text",
            )
        elif item_type == "gate-content-conflict":
            claimed = _exact_keys(
                item["claimed"],
                {"decision", "k2_mean_kld", "k2_top1_rate", "q2_mean_kld", "q2_top1_rate"},
                "quarantine gate claimed",
            )
            observed = _exact_keys(
                item["observed"],
                {"decision", "k2_mean_kld", "k2_top1_matches", "k2_top1_positions", "q2_mean_kld", "q2_top1_matches", "q2_top1_positions"},
                "quarantine gate observed",
            )
            _require(claimed.get("decision") in {"GREEN", "RED"}, "invalid quarantine gate claimed decision")
            _require(observed.get("decision") in {"GREEN", "RED"}, "invalid quarantine gate observed decision")
            for field in ("k2_mean_kld", "q2_mean_kld"):
                _require(
                    type(claimed.get(field)) is float and math.isfinite(claimed[field]) and claimed[field] >= 0,
                    f"invalid quarantine gate claimed number: {field}",
                )
                _require(
                    type(observed.get(field)) is float and math.isfinite(observed[field]) and observed[field] >= 0,
                    f"invalid quarantine gate observed number: {field}",
                )
            for field in ("k2_top1_rate", "q2_top1_rate"):
                _require(
                    type(claimed.get(field)) is float and math.isfinite(claimed[field]) and 0 <= claimed[field] <= 1,
                    f"invalid quarantine gate claimed number: {field}",
                )
            for prefix in ("k2", "q2"):
                gate_matches = observed.get(f"{prefix}_top1_matches")
                gate_positions = observed.get(f"{prefix}_top1_positions")
                if (
                    type(gate_matches) is not int
                    or type(gate_positions) is not int
                    or gate_positions <= 0
                    or not 0 <= gate_matches <= gate_positions
                ):
                    raise ValidationError(f"invalid quarantine gate Top-1: {prefix}")
        elif item_type == "old-teacher-result":
            claim = _exact_keys(item["claim"], {"mean_kld", "top1_matches", "top1_positions"}, "quarantine old teacher claim")
            _require(
                type(claim.get("mean_kld")) is float and math.isfinite(claim["mean_kld"]) and claim["mean_kld"] >= 0,
                "invalid quarantine old-teacher KLD",
            )
            _require(
                _is_int(claim.get("top1_matches"))
                and _is_int(claim.get("top1_positions"))
                and claim["top1_positions"] > 0
                and 0 <= claim["top1_matches"] <= claim["top1_positions"],
                "invalid quarantine old-teacher Top-1",
            )

    missing_scopes = data.get("missing_scopes", [])
    _require(isinstance(missing_scopes, list), "missing_scopes must be a list")
    for item in missing_scopes:
        item = _exact_keys(item, {"family", "measurement_key", "reason", "status"}, "missing scope")
        _require(item.get("status") == "MISSING_NOT_A_MEASUREMENT", "invalid missing-scope status")
        _require(item.get("family") in {"EXL3 K2", "EXL3 K3"}, "invalid missing-scope family")
        _require(isinstance(item.get("measurement_key"), str) and "/exl3-k" in item["measurement_key"], "invalid missing-scope key")
        expected_tier = item["family"][-1]
        _require(f"/exl3-k{expected_tier}/" in item["measurement_key"], "missing-scope tier does not match family")
        _require(isinstance(item.get("reason"), str) and item["reason"], "missing missing-scope reason")
    missing_scope_inventory = {
        (item.get("family"), item.get("measurement_key")) for item in missing_scopes if isinstance(item, dict)
    }
    _require(
        len(missing_scopes) == len(missing_scope_inventory)
        and missing_scope_inventory == EXPECTED_MISSING_SCOPE_KEYS,
        "missing-scope inventory mismatch",
    )


def validate_report(text: str, scope_records: dict[str, Any]) -> None:
    failures: list[str] = []
    table_tiers: set[str] = set()
    table_headers: list[str] | None = None
    section_tiers: set[str] = set()
    text = HTML_COMMENT_RE.sub("", text)
    for line_number, line in enumerate(text.splitlines(), 1):
        is_table_line = "|" in line
        if not is_table_line:
            table_tiers = set()
            table_headers = None
        if MISSING_LINE_RE.fullmatch(line.strip()) is not None and MEASUREMENT_CLAIM_RE.search(line) is None:
            continue
        line_tiers = {match.group(1) for match in TIER_RE.finditer(line)}
        if re.match(r"^\s{0,3}#{1,6}\s+", line) is not None:
            section_tiers = line_tiers
        if is_table_line and line_tiers:
            table_tiers.update(line_tiers)
        if is_table_line:
            candidate_headers = _table_header_names(_markdown_cells(line))
            if candidate_headers is not None:
                table_headers = candidate_headers
        governed_tiers = line_tiers | section_tiers | (table_tiers if is_table_line else set())
        if not governed_tiers:
            continue
        numeric_text = line
        for known_key in scope_records:
            numeric_text = numeric_text.replace(f"`{known_key}`", "")
        numeric_text = FIELD_VALUE_RE.sub("", numeric_text)
        numeric_text = SHA256_SEARCH_RE.sub("", numeric_text)
        numeric_text = QUARANTINE_ID_RE.sub("", SHA_LABEL_RE.sub("", TIER_RE.sub("", FAMILY_RE.sub("", numeric_text))))
        numeric_text = METRIC_LABEL_RE.sub("", numeric_text)
        if NUMBER_RE.search(numeric_text) is None:
            continue
        key_matches = MEASUREMENT_KEY_RE.findall(line)
        if not key_matches:
            failures.append(f"line {line_number}: quantitative K2/K3 claim lacks measurement_key")
            continue
        if len(key_matches) != 1:
            failures.append(f"line {line_number}: quantitative K2/K3 claim has duplicate measurement_key")
            continue
        measurement_key = key_matches[0]
        record = scope_records.get(measurement_key)
        if record is None:
            failures.append(f"line {line_number}: quantitative K2/K3 claim uses unknown measurement_key")
            continue
        expected_tier = measurement_key.split("/")[1][-1]
        if any(tier != expected_tier for tier in governed_tiers):
            failures.append(f"line {line_number}: quantitative K2/K3 claim tier does not match measurement_key")
        binding = record["reporting_binding"]
        measurement = record["measurement"]
        fields = {
            "bank_positions": (BANK_POSITIONS_RE, binding["bank_positions"]),
            "intervention_scope": (INTERVENTION_SCOPE_RE, binding["intervention_scope"]),
            "support": (SUPPORT_RE, str(binding["support"])),
            "scorer_sha256": (SCORER_RE, binding["scorer_sha256"]),
            "terminal_sha256": (TERMINAL_RE, binding["terminal_sha256"]),
            "artifact_sha256": (ARTIFACT_RE, binding["artifact_sha256"]),
            "top1": (
                TOP1_BINDING_RE,
                f"{measurement['top1_matches']}/{measurement['top1_positions']}",
            ),
            "kld": (KLD_BINDING_RE, str(measurement["mean_support_renormalized_kld"])),
        }
        fields["wire_bpw"] = (WIRE_BPW_BINDING_RE, str(measurement["wire"]["selected_payload_bpw"]))
        for field, (pattern, expected) in fields.items():
            matches = pattern.findall(line)
            if field == "top1":
                matches = [f"{numerator.replace(',', '')}/{denominator.replace(',', '')}" for numerator, denominator in matches]
            if not matches:
                failures.append(f"line {line_number}: quantitative K2/K3 claim lacks {field}")
            elif len(matches) != 1:
                failures.append(f"line {line_number}: quantitative K2/K3 claim has duplicate {field}")
            elif matches[0] != expected:
                failures.append(f"line {line_number}: quantitative K2/K3 claim has mismatched {field}")

        claim_text = FIELD_VALUE_RE.sub("", line)
        line_hashes = set(SHA256_SEARCH_RE.findall(line))
        expected_role_hashes = {
            "terminal": binding["terminal_sha256"],
            "artifact": binding["artifact_sha256"],
            "scorer": binding["scorer_sha256"],
        }
        for role, pattern in VISIBLE_HASH_LABEL_PATTERNS.items():
            if any(value != expected_role_hashes[role] for value in pattern.findall(line)):
                failures.append(f"line {line_number}: visible {role} SHA-256 does not bind measurement")
        unknown_hashes = line_hashes - _sha256_values(record)
        if unknown_hashes:
            failures.append(f"line {line_number}: quantitative K2/K3 claim contains unbound SHA-256")
        claim_text = SHA256_SEARCH_RE.sub("", claim_text)
        for known_key in scope_records:
            claim_text = claim_text.replace(f"`{known_key}`", "")
        allowed_fractions = {(measurement["top1_matches"], measurement["top1_positions"])}
        allowed_percentages = {
            Decimal(measurement["top1_matches"]) * Decimal(100) / Decimal(measurement["top1_positions"])
        }
        allowed_decimals = {
            Decimal(str(measurement["mean_support_renormalized_kld"])),
            Decimal(str(measurement["top1_rate"])),
            Decimal(str(measurement["wire"]["selected_payload_bpw"])),
        }
        if "artifact_file_bpw" in measurement["wire"]:
            allowed_decimals.add(Decimal(str(measurement["wire"]["artifact_file_bpw"])))
        for gate in record.get("comparison_gates", []):
            if gate["sha256"] not in line_hashes:
                continue
            for arm in (gate["k2"], gate["q2"]):
                arm_matches = arm["top1_matches"]
                arm_positions = arm["top1_positions"]
                allowed_fractions.add((arm_matches, arm_positions))
                allowed_percentages.add(Decimal(arm_matches) * Decimal(100) / Decimal(arm_positions))
                allowed_decimals.add(Decimal(str(arm["mean_support_renormalized_kld"])))
                for wire_field in ("selected_payload_bpw", "full_wire_bpw"):
                    if wire_field in arm:
                        wire_bpw = arm[wire_field]
                        _require(
                            isinstance(wire_bpw, float) and math.isfinite(wire_bpw) and wire_bpw > 0,
                            f"gate wire bpw must be positive: {measurement_key}/{gate['sha256']}",
                        )
                        allowed_decimals.add(Decimal(str(wire_bpw)))

        visible_kld_values = re.findall(rf"\bKLD\b\s+[*_`]*({FLOAT_TEXT})", claim_text, re.IGNORECASE)
        for visible_kld in visible_kld_values:
            if _decimal(visible_kld) != Decimal(str(measurement["mean_support_renormalized_kld"])):
                failures.append(f"line {line_number}: visible KLD does not bind measurement")
        visible_bpw_values = re.findall(rf"({FLOAT_TEXT})\s*[*_`]*\s*bpw\b", claim_text, re.IGNORECASE)
        allowed_wire_values = {
            Decimal(str(measurement["wire"]["selected_payload_bpw"])),
            *(
                Decimal(str(measurement["wire"][field]))
                for field in ("artifact_file_bpw",)
                if field in measurement["wire"]
            ),
        }
        for visible_bpw in visible_bpw_values:
            if _decimal(visible_bpw) not in allowed_wire_values:
                failures.append(f"line {line_number}: visible wire bpw does not bind measurement")
        if is_table_line and table_headers is not None and TABLE_SEPARATOR_RE.search(line) is None:
            _validate_table_semantics(table_headers, _markdown_cells(line), record, line_number, failures)

        for numerator, denominator in FRACTION_RE.findall(claim_text):
            claimed_fraction = (int(numerator.replace(",", "")), int(denominator.replace(",", "")))
            if claimed_fraction not in allowed_fractions:
                failures.append(f"line {line_number}: mismatched claimed Top-1 fraction")
        without_percentages = PERCENT_RE.sub("", claim_text)
        for percentage in PERCENT_RE.findall(claim_text):
            if _decimal(percentage) not in allowed_percentages:
                failures.append(f"line {line_number}: mismatched claimed Top-1 percentage")
        without_fractions = FRACTION_RE.sub("", without_percentages)
        without_metric_noise = METRIC_LABEL_RE.sub("", TIER_RE.sub("", FAMILY_RE.sub("", without_fractions)))
        for decimal_text in DECIMAL_RE.findall(without_metric_noise):
            if _decimal(decimal_text) not in allowed_decimals:
                failures.append(f"line {line_number}: mismatched claimed KLD/wire value")
        without_decimals = DECIMAL_RE.sub("", without_metric_noise)
        allowed_integers = {
            measurement["top1_matches"],
            measurement["top1_positions"],
            record["scope"]["positions_per_row"],
            record["scope"]["support_width"],
            record["scope"]["intervention"]["changed_tensors"],
            record["scope"]["intervention"]["layer"],
            *(int(row_id) for row_id in record["scope"]["row_ids"]),
        }
        experts = record["scope"]["intervention"]["experts"]
        if isinstance(experts, list):
            allowed_integers.update(experts)
        else:
            allowed_integers.update(experts.values())
        allowed_integers.update(
            arm[field]
            for gate in record.get("comparison_gates", [])
            if gate["sha256"] in line_hashes
            for arm in (gate["k2"], gate["q2"])
            for field in ("top1_matches", "top1_positions")
        )
        integer_text = QUARANTINE_ID_RE.sub("", SHA_LABEL_RE.sub("", without_decimals))
        for integer in INTEGER_RE.findall(integer_text):
            if int(integer.replace(",", "")) not in allowed_integers:
                failures.append(f"line {line_number}: mismatched claimed integer")
    _require(not failures, "; ".join(failures))


def validate_files(ledger_path: Path, report_path: Path) -> None:
    raw_ledger = ledger_path.read_text()
    ledger = json.loads(
        raw_ledger,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_object_without_duplicate_keys,
    )
    validate_ledger(ledger)
    canonical = json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _require(raw_ledger == canonical, "ledger JSON is not canonical (sorted keys, two-space indent, final newline)")
    validate_report(report_path.read_text(), ledger["measurement_scopes"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the EXL3 K2 scope ledger and quantitative reporting claims.")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        validate_files(args.ledger, args.report)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        parser.exit(1, f"EXL3 K2 report validation failed: {exc}\n")
    print("EXL3 K2 report validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
