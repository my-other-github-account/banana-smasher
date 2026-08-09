from __future__ import annotations

import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "notes/exl3-k2-scope-ledger/EXL3_K2_SCOPE_LEDGER.json"
REPORT = ROOT / "notes/exl3-k2-scope-ledger/REPORT.md"
VALIDATOR = ROOT / "tools/validate_exl3_k2_report.py"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_exl3_k2_report", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_scope_ledger_and_report_pass_guard() -> None:
    validator = _validator()
    validator.validate_files(LEDGER, REPORT)


def _ledger() -> dict:
    return json.loads(LEDGER.read_text())


def _bound_claim(data: dict, scope_key: str) -> str:
    record = data["measurement_scopes"][scope_key]
    binding = record["reporting_binding"]
    measurement = record["measurement"]
    return (
        "EXL3 K2 Top-1 1968/2048; "
        f"measurement_key=`{scope_key}`; "
        f"top1=`{measurement['top1_matches']}/{measurement['top1_positions']}`; "
        f"kld=`{measurement['mean_support_renormalized_kld']}`; "
        f"wire_bpw=`{measurement['wire']['selected_payload_bpw']}`; "
        f"bank_positions=`{binding['bank_positions']}`; "
        f"intervention_scope=`{binding['intervention_scope']}`; "
        f"support=`{binding['support']}`; "
        f"scorer_sha256=`{binding['scorer_sha256']}`; "
        f"terminal_sha256=`{binding['terminal_sha256']}`; "
        f"artifact_sha256=`{binding['artifact_sha256']}`.\n"
    )


@pytest.mark.parametrize(
    "claim",
    [
        "EXL3 K2: 96.09375%\n",
        "EXL3 K3 Top-1 1/2\n",
        "K2 Top-1 1968/2048\n",
        "EXL3 K2 score 1968\n",
        "`K2` Top-1 1968/2048\n",
        "K2 Top-1 `1968/2048`\n",
        "| K3 | 1/2 |\n",
        "| K2 | 1968 |\n",
    ],
)
def test_bare_quantitative_k2_or_k3_claim_fails(claim: str) -> None:
    validator = _validator()
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(claim, _ledger()["measurement_scopes"])


def test_complete_scope_bound_claim_passes() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    validator.validate_report(_bound_claim(data, scope_key), data["measurement_scopes"])


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("top1=`1968/2048`", "top1=`1/2`", "mismatched top1"),
        ("kld=`0.018689766940723482`", "kld=`9.0`", "mismatched kld"),
        ("wire_bpw=`2.0117225646972656`", "wire_bpw=`9.0`", "mismatched wire_bpw"),
    ],
)
def test_claimed_metrics_must_match_measurement(old: str, new: str, message: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_report(_bound_claim(data, scope_key).replace(old, new), data["measurement_scopes"])


def test_k3_claim_cannot_bind_to_k2_measurement() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace("EXL3 K2", "EXL3 K3", 1)
    with pytest.raises(validator.ValidationError, match="tier does not match"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_explicit_missing_scope_is_not_treated_as_a_measurement() -> None:
    validator = _validator()
    validator.validate_report(
        "EXL3 K3 Exact64 over 65,536 positions is `MISSING_NOT_A_MEASUREMENT`.\n",
        _ledger()["measurement_scopes"],
    )


@pytest.mark.parametrize("prefix", ["fake-", "fake.", "fake:", "fake/"])
def test_prefixed_fake_measurement_key_is_not_accepted(prefix: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace("measurement_key=", f"{prefix}measurement_key=")
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_duplicate_reporting_binding_is_rejected() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).rstrip() + f" artifact_sha256=`{'0' * 64}`.\n"
    with pytest.raises(validator.ValidationError, match="duplicate artifact_sha256"):
        validator.validate_report(claim, data["measurement_scopes"])


@pytest.mark.parametrize(
    "field",
    [
        "measurement_key",
        "bank_positions",
        "intervention_scope",
        "support",
        "scorer_sha256",
        "terminal_sha256",
        "artifact_sha256",
        "top1",
        "kld",
        "wire_bpw",
    ],
)
def test_dot_prefixed_fake_field_is_not_accepted(field: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace(f"{field}=", f"fake.{field}=")
    with pytest.raises(validator.ValidationError, match=f"lacks {field}"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_unlabeled_metrics_cannot_contradict_bound_metrics() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace(
        "EXL3 K2 Top-1 1968/2048;",
        "EXL3 K2 Top-1 1/2; KLD 9.0; 9.0 bpw;",
    )
    with pytest.raises(validator.ValidationError, match="claimed Top-1"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_integer_only_metric_cannot_contradict_bound_metrics() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace(
        "EXL3 K2 Top-1 1968/2048;",
        "EXL3 K2 score 9999;",
    )
    with pytest.raises(validator.ValidationError, match="claimed integer"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_html_comment_bindings_do_not_satisfy_visible_claim() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    bound = _bound_claim(data, scope_key)
    hidden = bound[bound.index("measurement_key=") :].rstrip()
    claim = "EXL3 K2 Top-1 1968/2048 <!-- " + hidden + " -->\n"
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_k2_table_header_requires_bindings_on_numeric_rows() -> None:
    validator = _validator()
    report = "| Candidate | K2 Top-1 |\n|---|---:|\n| ordinary Q2 | 1968/2048 |\n"
    with pytest.raises(validator.ValidationError, match="line 3:.*lacks measurement_key"):
        validator.validate_report(report, _ledger()["measurement_scopes"])


def test_k2_table_without_leading_pipe_requires_bindings() -> None:
    validator = _validator()
    report = "Candidate | K2 Top-1\n--- | ---:\nordinary Q2 | 1968/2048\n"
    with pytest.raises(validator.ValidationError, match="line 3:.*lacks measurement_key"):
        validator.validate_report(report, _ledger()["measurement_scopes"])


def test_k2_section_heading_requires_bindings_on_numeric_prose() -> None:
    validator = _validator()
    report = "## EXL3 K2 results\nTop-1 1968/2048\n"
    with pytest.raises(validator.ValidationError, match="line 2:.*lacks measurement_key"):
        validator.validate_report(report, _ledger()["measurement_scopes"])


def test_multiline_html_comment_cannot_hide_reporting_bindings() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    bound = _bound_claim(data, scope_key).rstrip()
    claim, bindings = bound.split("; ", 1)
    report = f"{claim}; <!-- {bindings}\n-->\n"
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(report, data["measurement_scopes"])


def test_scope_key_in_table_row_names_k2_and_requires_bindings() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    report = f"| current | `{scope_key}` | 1968/2048 |\n"
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(report, data["measurement_scopes"])


def test_fully_inline_scope_key_and_metric_cannot_bypass_guard() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(f"`{scope_key} Top-1 1968/2048`\n", data["measurement_scopes"])


@pytest.mark.parametrize(
    "claim",
    [
        "EXL3 K2 Top-1 1968/2048; status is not MISSING_NOT_A_MEASUREMENT\n",
        "MISSING_NOT_A_MEASUREMENT, but EXL3 K3 measured 96%\n",
        "EXL3 K2 is MISSING_NOT_A_MEASUREMENT; Top-1 1968/2048\n",
        "EXL3 K2 remains MISSING_NOT_A_MEASUREMENT. Top-1 1968/2048\n",
        "| EXL3 K2 | remains MISSING_NOT_A_MEASUREMENT; | 1968/2048 |\n",
    ],
)
def test_missing_marker_cannot_hide_a_measurement(claim: str) -> None:
    validator = _validator()
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(claim, _ledger()["measurement_scopes"])


@pytest.mark.parametrize(
    "field",
    [
        "bank_positions",
        "intervention_scope",
        "support",
        "scorer_sha256",
        "terminal_sha256",
        "artifact_sha256",
    ],
)
def test_every_reporting_binding_is_required(field: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key)
    claim = claim.replace(f"{field}=", f"removed_{field}=")
    with pytest.raises(validator.ValidationError, match=f"lacks {field}"):
        validator.validate_report(claim, data["measurement_scopes"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("bank_positions", "wrong-bank:1"),
        ("intervention_scope", "wrong-intervention"),
        ("support", "4096"),
        ("scorer_sha256", "0" * 64),
        ("terminal_sha256", "0" * 64),
        ("artifact_sha256", "0" * 64),
    ],
)
def test_arbitrary_reporting_binding_is_rejected(field: str, replacement: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    expected = str(data["measurement_scopes"][scope_key]["reporting_binding"][field])
    claim = _bound_claim(data, scope_key).replace(f"{field}=`{expected}`", f"{field}=`{replacement}`")
    with pytest.raises(validator.ValidationError, match=f"mismatched {field}"):
        validator.validate_report(claim, data["measurement_scopes"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record["evidence"]["terminal"].pop("relative_path"), "terminal path"),
        (lambda record: record["measurement"]["wire"].pop("numerator_bits"), "wire numerator"),
        (lambda record: record["measurement"]["wire"].pop("denominator_weights"), "wire denominator"),
        (lambda record: record["reporting_binding"].__setitem__("bank_positions", "wrong-bank:1"), "bank_positions"),
        (lambda record: record["reporting_binding"].__setitem__("intervention_scope", "wrong-intervention"), "intervention_scope"),
    ],
)
def test_ledger_requires_scope_bound_reporting_evidence(mutation, message: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    mutation(data["measurement_scopes"][scope_key])
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_ledger(data)


def test_nonfinite_measurement_is_rejected() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["measurement"]["mean_support_renormalized_kld"] = float("nan")
    with pytest.raises(validator.ValidationError, match="finite KLD"):
        validator.validate_ledger(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda scope, measurement: scope.__setitem__("support_width", True),
        lambda scope, measurement: measurement.__setitem__("top1_matches", True),
        lambda scope, measurement: measurement["wire"].__setitem__("numerator_bits", True),
    ],
)
def test_boolean_is_not_accepted_as_integer(mutation) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    record = data["measurement_scopes"][scope_key]
    mutation(record["scope"], record["measurement"])
    with pytest.raises(validator.ValidationError):
        validator.validate_ledger(data)


def test_kld_must_be_nonnegative() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["measurement"]["mean_support_renormalized_kld"] = -0.1
    with pytest.raises(validator.ValidationError, match="nonnegative KLD"):
        validator.validate_ledger(data)


def test_wire_numerator_closes_to_selected_payload_bytes() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    wire = data["measurement_scopes"][scope_key]["measurement"]["wire"]
    wire["numerator_bits"] += 8
    wire["selected_payload_bpw"] = wire["numerator_bits"] / wire["denominator_weights"]
    with pytest.raises(validator.ValidationError, match="wire numerator does not close"):
        validator.validate_ledger(data)


def test_current_scope_and_quarantine_conflicts_are_explicit() -> None:
    data = _ledger()
    current = data["measurement_scopes"][data["decision"]["current_scope_key"]]
    assert current["measurement"]["top1_matches"] == 1968
    assert current["measurement"]["top1_positions"] == 2048
    assert current["measurement"]["mean_support_renormalized_kld"] == 0.018689766940723482
    assert current["per_row_evidence"]["status"] == "NOT_PERSISTED"

    quarantine = {entry["id"]: entry for entry in data["quarantine"]}
    assert quarantine["Q-001"]["claimed_sha256"] != quarantine["Q-001"]["observed_sha256"]
    assert quarantine["Q-002"]["claimed"]["decision"] == "GREEN"
    assert quarantine["Q-002"]["observed"]["decision"] == "RED"
    assert quarantine["Q-003"]["type"] == "scope-substitution"


def test_each_measurement_has_explicit_acceptance_identities_and_semantics() -> None:
    data = _ledger()
    for record in data["measurement_scopes"].values():
        scope = record["scope"]
        assert scope["base_index_sha256"] == data["basis_sha256"]
        assert len(scope["teacher_sha256"]) == 64
        assert scope["window_manifest_sha256"] == record["evidence"]["manifest"]["sha256"]
        assert scope["scorer_semantics"]
        assert scope["reducer_semantics"]
        assert record["status"] in {"CURRENT_AUTHORITATIVE", "HISTORICAL_AUTHENTICATED"}
        for name in ("prefix_identity", "suffix_identity"):
            identity = scope[name]
            assert identity["status"] in {"BOUND", "NOT_APPLICABLE"}
            assert identity.get("sha256") or identity.get("reason")


def test_fully_inline_code_measurement_cannot_bypass_guard() -> None:
    validator = _validator()
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report("`K2 Top-1 1968/2048`\n", _ledger()["measurement_scopes"])


def test_missing_status_cannot_exempt_a_measured_claim() -> None:
    validator = _validator()
    claim = "EXL3 K2 Top-1 1968/2048 is `MISSING_NOT_A_MEASUREMENT`.\n"
    with pytest.raises(validator.ValidationError, match="lacks measurement_key"):
        validator.validate_report(claim, _ledger()["measurement_scopes"])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_any_nonfinite_ledger_number_is_rejected(nonfinite: float) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["comparison_gates"][0]["q2"]["mean_support_renormalized_kld"] = nonfinite
    with pytest.raises(validator.ValidationError, match="nonfinite"):
        validator.validate_ledger(data)


def test_unknown_top_level_ledger_key_is_rejected() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    data["unknown_key"] = "unexpected"
    with pytest.raises(validator.ValidationError, match="top-level keys"):
        validator.validate_ledger(data)


def test_decision_scope_roles_cannot_be_swapped() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    data["decision"]["current_scope_key"], data["decision"]["historical_scope_key"] = (
        data["decision"]["historical_scope_key"],
        data["decision"]["current_scope_key"],
    )
    with pytest.raises(validator.ValidationError, match="decision scope role"):
        validator.validate_ledger(data)


def test_missing_scope_family_must_match_key_tier() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    data["missing_scopes"][0]["measurement_key"] = data["missing_scopes"][0]["measurement_key"].replace(
        "/exl3-k2/", "/exl3-k3/"
    )
    with pytest.raises(validator.ValidationError, match="missing-scope tier"):
        validator.validate_ledger(data)


def test_quarantine_numeric_types_are_strict() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    data["quarantine"][1]["observed"]["k2_top1_matches"] = True
    with pytest.raises(validator.ValidationError, match="quarantine gate Top-1"):
        validator.validate_ledger(data)


def test_gate_wire_bpw_must_be_positive() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["comparison_gates"][1]["q2"]["full_wire_bpw"] = -1.0
    with pytest.raises(validator.ValidationError, match="gate wire bpw"):
        validator.validate_ledger(data)


@pytest.mark.parametrize(
    "path",
    [
        ("decision",),
        ("record",),
        ("record", "artifact"),
        ("record", "scope"),
        ("record", "scope", "intervention"),
        ("record", "evidence"),
        ("record", "evidence", "terminal"),
        ("record", "measurement"),
        ("record", "measurement", "wire"),
        ("record", "reporting_binding"),
        ("record", "per_row_evidence"),
        ("record", "comparison_gates", 0),
        ("missing_scopes", 0),
        ("quarantine", 0),
    ],
)
def test_unknown_nested_ledger_key_is_rejected(path: tuple[str | int, ...]) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    node = data
    for part in path:
        if part == "record":
            node = data["measurement_scopes"][scope_key]
        else:
            node = node[part]
    node["unknown_key"] = "unexpected"
    with pytest.raises(validator.ValidationError, match="unexpected keys"):
        validator.validate_ledger(data)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../private/TERMINAL.json",
        r"C:\\private\\TERMINAL.json",
        "https:/private/TERMINAL.json",
        "./receipts/TERMINAL.json",
    ],
)
def test_terminal_locator_must_be_normalized_repo_safe_relative_path(relative_path: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["evidence"]["terminal"]["relative_path"] = relative_path
    with pytest.raises(validator.ValidationError, match="relative path"):
        validator.validate_ledger(data)


def test_terminal_status_and_gate_locator_are_required() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["evidence"]["terminal"]["status"] = "FAIL"
    with pytest.raises(validator.ValidationError, match="terminal status"):
        validator.validate_ledger(data)

    data = deepcopy(_ledger())
    data["measurement_scopes"][scope_key]["comparison_gates"][0].pop("relative_path")
    with pytest.raises(validator.ValidationError, match="comparison gate path"):
        validator.validate_ledger(data)


def test_per_row_rate_must_close_to_integer_counts() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["historical_scope_key"]
    data["measurement_scopes"][scope_key]["per_row_evidence"]["rows"][0]["top1_rate"] = 0.0
    with pytest.raises(validator.ValidationError, match="row Top-1 rate"):
        validator.validate_ledger(data)


def test_measurement_key_segments_must_close_to_scope() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    old_key = data["decision"]["current_scope_key"]
    new_key = old_key.replace("l034-experts000-255-fused13-down", "wrong")
    record = data["measurement_scopes"].pop(old_key)
    record["scope_key"] = new_key
    record["reporting_binding"]["measurement_key"] = new_key
    record["reporting_binding"]["intervention_scope"] = "wrong"
    data["measurement_scopes"][new_key] = record
    data["decision"]["current_scope_key"] = new_key
    with pytest.raises(validator.ValidationError, match="measurement key"):
        validator.validate_ledger(data)


def test_canonical_file_validation_rejects_json_nan(tmp_path: Path) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["comparison_gates"][0]["q2"]["mean_support_renormalized_kld"] = float("nan")
    ledger_path = tmp_path / "ledger.json"
    report_path = tmp_path / "report.md"
    ledger_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    report_path.write_text(REPORT.read_text())
    with pytest.raises(validator.ValidationError, match="nonfinite JSON constant"):
        validator.validate_files(ledger_path, report_path)


def test_canonical_file_validation_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    validator = _validator()
    ledger_path = tmp_path / "ledger.json"
    report_path = tmp_path / "report.md"
    raw = LEDGER.read_text().replace(
        '  "schema": "banana-smasher-exl3-k2-scope-ledger-v1"',
        '  "schema": "ambiguous",\n  "schema": "banana-smasher-exl3-k2-scope-ledger-v1"',
        1,
    )
    ledger_path.write_text(raw)
    report_path.write_text(REPORT.read_text())
    with pytest.raises(validator.ValidationError, match="duplicate JSON key"):
        validator.validate_files(ledger_path, report_path)


def test_negative_per_row_and_gate_kld_are_rejected() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    historical_key = data["decision"]["historical_scope_key"]
    data["measurement_scopes"][historical_key]["per_row_evidence"]["rows"][0][
        "mean_support_renormalized_kld"
    ] = -0.1
    with pytest.raises(validator.ValidationError, match="nonnegative"):
        validator.validate_ledger(data)

    data = deepcopy(_ledger())
    current_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][current_key]["comparison_gates"][0]["q2"][
        "mean_support_renormalized_kld"
    ] = -0.1
    with pytest.raises(validator.ValidationError, match="nonnegative"):
        validator.validate_ledger(data)


def test_artifact_file_bpw_closes_to_file_bytes() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    wire = data["measurement_scopes"][scope_key]["measurement"]["wire"]
    wire["artifact_file_bpw"] += 0.1
    with pytest.raises(validator.ValidationError, match="artifact file bpw"):
        validator.validate_ledger(data)


def test_visible_wrong_hash_cannot_be_camouflaged_by_correct_binding() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace("EXL3 K2 Top-1", f"EXL3 K2 {'0' * 64} Top-1", 1)
    with pytest.raises(validator.ValidationError, match="unbound SHA-256"):
        validator.validate_report(claim, data["measurement_scopes"])


@pytest.mark.parametrize(
    ("label", "wrong_hash_field"),
    [
        ("terminal SHA-256", "artifact_sha256"),
        ("artifact SHA-256", "terminal_sha256"),
        ("scorer SHA-256", "terminal_sha256"),
    ],
)
def test_visible_role_labeled_hash_must_match_its_binding(label: str, wrong_hash_field: str) -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    binding = data["measurement_scopes"][scope_key]["reporting_binding"]
    claim = _bound_claim(data, scope_key).replace(
        "EXL3 K2 Top-1",
        f"EXL3 K2 {label}=`{binding[wrong_hash_field]}`; Top-1",
        1,
    )
    with pytest.raises(validator.ValidationError, match=f"visible {label}"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_visible_kld_and_wire_values_cannot_be_swapped() -> None:
    validator = _validator()
    data = _ledger()
    scope_key = data["decision"]["current_scope_key"]
    claim = _bound_claim(data, scope_key).replace(
        "EXL3 K2 Top-1 1968/2048;",
        "EXL3 K2 Top-1 1968/2048; KLD 2.0117225646972656; 0.018689766940723482 bpw;",
    )
    with pytest.raises(validator.ValidationError, match="visible KLD"):
        validator.validate_report(claim, data["measurement_scopes"])


def test_gate_table_arms_and_decision_are_semantically_bound() -> None:
    validator = _validator()
    data = _ledger()
    report = REPORT.read_text()
    gate_sha = "476ee64e7e919bfaa851ccc8e0e1e3e760831dd547e7c9d1dfdc837b2126a0da"
    gate_line = next(line for line in report.splitlines() if gate_sha in line)
    header = "| Gate | Candidate | Candidate Top-1 | K2 Top-1 | Candidate KLD | K2 KLD | Decision |\n|---|---|---:|---:|---:|---:|---|\n"

    swapped = gate_line.replace("| 1,959/2,048 | 1,968/2,048 |", "| 1,968/2,048 | 1,959/2,048 |")
    with pytest.raises(validator.ValidationError, match="candidate Top-1"):
        validator.validate_report(header + swapped + "\n", data["measurement_scopes"])

    flipped = gate_line.replace("**RED**", "**GREEN**")
    with pytest.raises(validator.ValidationError, match="gate decision"):
        validator.validate_report(header + flipped + "\n", data["measurement_scopes"])


@pytest.mark.parametrize("relative_path", [".", "receipts/../private.json"])
def test_all_locator_fields_reject_dot_and_traversal(relative_path: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    data["measurement_scopes"][scope_key]["evidence"]["terminal"]["relative_path"] = relative_path
    with pytest.raises(validator.ValidationError, match="relative path"):
        validator.validate_ledger(data)

    data = deepcopy(_ledger())
    data["measurement_scopes"][scope_key]["comparison_gates"][1]["source_locator"] = relative_path
    with pytest.raises(validator.ValidationError, match="source locator"):
        validator.validate_ledger(data)


def test_measurement_family_tier_cannot_be_relabelled_wholesale() -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    old_key = data["decision"]["current_scope_key"]
    new_key = old_key.replace("/exl3-k2/", "/exl3-k3/")
    record = data["measurement_scopes"].pop(old_key)
    record["scope_key"] = new_key
    record["reporting_binding"]["measurement_key"] = new_key
    data["measurement_scopes"][new_key] = record
    data["decision"]["current_scope_key"] = new_key
    with pytest.raises(validator.ValidationError, match="measurement key"):
        validator.validate_ledger(data)


@pytest.mark.parametrize("status", ["CURRENT_AUTHORITATIVE", "HISTORICAL_AUTHENTICATED"])
def test_changed_tensor_count_closes_to_roster(status: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = next(key for key, record in data["measurement_scopes"].items() if record["status"] == status)
    data["measurement_scopes"][scope_key]["scope"]["intervention"]["changed_tensors"] += 1
    with pytest.raises(validator.ValidationError, match="changed tensors.*roster"):
        validator.validate_ledger(data)


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "false"])
def test_missing_scope_inventory_is_exact(mutation: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    if mutation == "omit":
        data["missing_scopes"].pop()
    elif mutation == "duplicate":
        data["missing_scopes"].append(deepcopy(data["missing_scopes"][0]))
    else:
        data["missing_scopes"][0]["measurement_key"] = data["decision"]["current_scope_key"]
    with pytest.raises(validator.ValidationError, match="missing-scope inventory"):
        validator.validate_ledger(data)


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "false"])
def test_comparison_gate_inventory_is_exact(mutation: str) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    scope_key = data["decision"]["current_scope_key"]
    gates = data["measurement_scopes"][scope_key]["comparison_gates"]
    if mutation == "omit":
        gates.pop()
    elif mutation == "duplicate":
        gates.append(deepcopy(gates[0]))
    else:
        gates[0]["role"] = "false-authority"
    with pytest.raises(validator.ValidationError, match="comparison gate inventory"):
        validator.validate_ledger(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["quarantine"][1]["claimed"].__setitem__("k2_top1_rate", True),
        lambda data: data["quarantine"][1]["claimed"].__setitem__("q2_top1_rate", 2.0),
        lambda data: data["quarantine"][1]["observed"].__setitem__("k2_mean_kld", "0.1"),
        lambda data: data["quarantine"][3]["claim"].__setitem__("mean_kld", True),
        lambda data: data["quarantine"][3]["claim"].__setitem__("top1_matches", -1),
    ],
)
def test_quarantine_numeric_values_are_typed_and_bounded(mutation) -> None:
    validator = _validator()
    data = deepcopy(_ledger())
    mutation(data)
    with pytest.raises(validator.ValidationError, match="quarantine"):
        validator.validate_ledger(data)
