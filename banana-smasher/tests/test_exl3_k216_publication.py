from __future__ import annotations

from decimal import Decimal, getcontext
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "Evals/README.md"
EVIDENCE = (
    ROOT
    / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/evidence"
    / "REAP-K216-EXL3-3.0/measurement.json"
)


def test_reap_k216_mmlu_row_is_complete_and_bound_to_public_evidence() -> None:
    getcontext().prec = 100
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["status"] == "PASS"
    assert evidence["variant"] == "REAP-K216-EXL3-3.0"
    assert evidence["correct"] == 409
    assert evidence["n"] == 500
    assert Decimal(evidence["mmlu_percent_exact"]) == Decimal("81.8")
    assert evidence["qrows_sha256"] == (
        "2125004934a28c60fa6e533ffefa4fb4f795a3ace3fa17ddbc29f3852c770eb0"
    )
    assert evidence["independent_recomputation"] == "PASS"
    bpw = Decimal(evidence["base_equivalent_comparison_bpw"])
    percent = Decimal(evidence["mmlu_percent_exact"])
    gb = Decimal(evidence["complete_artifact_decimal_gb"])
    assert Decimal(evidence["above_chance_mmlu_per_bpw_exact"]) == (
        percent - Decimal(25)
    ) / bpw
    assert Decimal(evidence["raw_mmlu_per_bpw_exact"]) == percent / bpw
    assert Decimal(evidence["above_chance_mmlu_per_gb_exact"]) == (
        percent - Decimal(25)
    ) / gb

    text = EVALS.read_text()
    assert (
        "0xSero REAP-K216 EXL3 3.0** | **84.81%** (55,584/65,536) | "
        "**0.386969** | **81.80%** (409/500) | **18.899** | **27.218** | "
        "**0.532**"
    ) in text
    assert "REAP-K216-EXL3-3.0/measurement.json" in text


def test_reap_k216_public_evidence_has_no_private_execution_identifiers() -> None:
    text = EVIDENCE.read_text().lower()
    for forbidden in ("/home/", "/users/", "task_id", "spark-", "dnola", "macmini"):
        assert forbidden not in text
