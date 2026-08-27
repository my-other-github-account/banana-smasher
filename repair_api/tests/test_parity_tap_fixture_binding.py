from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "control-receipts" / "t_cc497d90" / "PARITY_TAP_FIXTURE_BINDING.json"
FIXTURE = ROOT / "control-receipts" / "t_d2e95913" / "CANONICAL_PRE_WINDOW28_FIXTURE.json"
REQUIRED = (
    "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
    "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
)
ZERO_COUNTERS = (
    "timed_model_payload_reads", "fallback_calls", "reconstruction_calls",
    "reference_fwht_calls", "cpu_relay_bytes", "layer_streaming_calls",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_canonical_window28_fixture_is_bound_to_public_parity_tap() -> None:
    binding = json.loads(BINDING.read_text())
    fixture = json.loads(FIXTURE.read_text())

    assert binding["status"] == "PASS_BOUND"
    assert binding["public_api"] == {
        "method": "ResidentRepairAPI.parity_tap",
        "version": "v1",
        "checkpoint": "UPDATE_000",
        "window": 28,
        "modes": ["current", "sealed_reference"],
        "one_window_only": True,
        "diagnostic_only": True,
    }
    assert sha256(FIXTURE) == binding["canonical_fixture_contract"]["sha256"]
    assert fixture["status"] == "PASS_UNIQUE_MATCH"
    assert fixture["fixture"]["source_sha256"] == binding["true_answer_key"]["source_sha256"]
    assert fixture["fixture"]["selector"] == {
        "mapping_key": "q_lp_at_ref",
        "position_slice": [0, 1024],
        "support_slice": [0, 8192],
    }
    assert fixture["fixture"]["dtype"] == binding["true_answer_key"]["dtype"]
    assert fixture["fixture"]["shape"] == binding["true_answer_key"]["shape"]
    assert fixture["sealed_receipt"]["sha256"] == binding["sealed_row_receipt"]["sha256"]
    assert fixture["sealed_receipt"]["kld_mean"] == 0.13269903835046534
    assert binding["comparison_contract"]["ordered_tensors"] == list(REQUIRED)


def test_existing_public_receipts_close_schema_modes_reads_and_nonpromotion() -> None:
    binding = json.loads(BINDING.read_text())
    for mode, receipt in binding["existing_public_tap_receipts"].items():
        path = ROOT / receipt["path"]
        assert sha256(path) == receipt["sha256"]
        payload = json.loads(path.read_text())
        assert payload["schema"] == "banana-smasher-resident-parity-tap-v1"
        assert payload["status"] == "PASS"
        assert payload["quality_status"] == "DIAGNOSTIC_ONLY_UNPROMOTED"
        assert payload["public_api"]["method"] == "ResidentRepairAPI.parity_tap"
        assert payload["mode"] == mode
        assert set(payload["taps"]) == set(REQUIRED)
        for name in REQUIRED:
            assert set(payload["taps"][name]) == {"sha256", "dtype", "shape", "sample"}
        assert all(payload["runtime_counters"][name] == 0 for name in ZERO_COUNTERS)
        assert len(payload["runtime_counters"]["resident_ready"]) == 2
        assert not ({"target_score", "canary_status", "target_ladder"} & set(payload))
        # Canonical reserialization is byte-stable even though JSON object order is irrelevant.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert canonical == json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"), allow_nan=False)
