from __future__ import annotations

import json
from pathlib import Path

import torch

from repair_api.call_tree_trace import (
    FullCallTreeTrace,
    _provider_dispatch_identity,
    _semantic_line_boundary,
)


def grouped_packed_projection(value: torch.Tensor) -> torch.Tensor:
    return value


class FakeExpert(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return grouped_packed_projection(value)


class FakeStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = {0: FakeExpert(), 1: FakeExpert()}


def test_static_w28_grouped_dispatch_lines_are_named() -> None:
    assert _semantic_line_boundary(
        "/installed/repair_api/assets/static_w28_fast_v7_expert_base.py",
        "_project",
        146,
    ) == ("grouped_mm_dispatch_input", ("x",))
    assert _semantic_line_boundary(
        "/installed/repair_api/assets/static_w28_fast_v7_expert_base.py",
        "_project",
        151,
    ) == ("grouped_mm_dispatch_output", ("value",))


def test_provider_dispatch_identity_binds_installed_expert_and_callable(tmp_path: Path) -> None:
    model = FakeStudent()
    identity = _provider_dispatch_identity(model)
    assert identity["status"] == "BOUND"
    assert identity["layer_count"] == 2
    assert identity["layers"] == [0, 1]
    assert len(identity["implementations"]) == 1
    implementation = identity["implementations"][0]
    assert implementation["expert_class"].endswith(".FakeExpert")
    assert implementation["dispatch_callable"].endswith(".grouped_packed_projection")
    assert implementation["dispatch_source_sha256"]

    trace_path = tmp_path / "trace.jsonl"
    with FullCallTreeTrace(
        model,
        trace_path,
        rail="test",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ):
        pass
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    event = next(row for row in rows if row["kind"] == "provider_dispatch_identity")
    assert event["status"] == "BOUND"
    assert event["implementations"] == identity["implementations"]
