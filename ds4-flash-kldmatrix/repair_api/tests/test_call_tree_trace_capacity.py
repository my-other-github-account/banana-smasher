import json

import pytest
import torch

from repair_api.call_tree_trace import FullCallTreeTrace


def test_trace_keeps_more_than_500k_events_and_seals_terminal_at_data_cap(tmp_path) -> None:
    path = tmp_path / "call-tree.jsonl"
    trace = FullCallTreeTrace(
        torch.nn.Identity(),
        path,
        rail="observer_capacity_probe",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
        max_events=500_001,
    ).start()

    for _ in range(500_000):
        trace._event({"kind": "capacity_probe"})

    assert trace._count == 500_001
    with pytest.raises(RuntimeError, match="W28_CALL_TREE_EVENT_CAP_RED:500001"):
        trace._event({"kind": "must_not_be_written"})

    terminal = trace.stop(status="ERROR")
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert FullCallTreeTrace(
        torch.nn.Identity(),
        tmp_path / "default-cap.jsonl",
        rail="default_capacity_probe",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ).max_events == 1_000_000
    assert len(rows) == 500_002
    assert rows[-1] == {
        "event_count": 500_002,
        "kind": "footer",
        "ordinal": 500_001,
        "rail": "observer_capacity_probe",
        "status": "ERROR",
    }
    assert terminal["status"] == "ERROR"
    assert terminal["event_count"] == 500_002
    assert path.with_suffix(".jsonl.terminal.json").exists()
