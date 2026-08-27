from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from repair_api.call_tree_trace import (
    FullCallTreeTrace,
    _provider_dispatch_identity,
    _provider_rebound_project_code_bindings,
    _semantic_line_boundary,
)


def grouped_packed_projection(value: torch.Tensor) -> torch.Tensor:
    return value


class FakeExpert(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return grouped_packed_projection(value)


class WrappedFakeExpert(FakeExpert):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value)


class ProjectFakeExpert(torch.nn.Module):
    def _project(self, projection: str, x: torch.Tensor) -> torch.Tensor:
        return grouped_packed_projection(x)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._project("w2", value)


class WrappedProjectFakeExpert(ProjectFakeExpert):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value)


class ReboundProjectFakeExpert(ProjectFakeExpert):
    def _project(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super()._project(*args, **kwargs)


class FakeStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = {0: WrappedFakeExpert(), 1: WrappedFakeExpert()}


class ProjectFakeStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = {0: WrappedProjectFakeExpert()}


class ReboundProjectFakeStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = {0: ReboundProjectFakeExpert()}


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
    assert implementation["expert_class"].endswith(".WrappedFakeExpert")
    assert implementation["dispatch_owner_class"].endswith(".FakeExpert")
    assert implementation["dispatch_owner_method"] == "forward"
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


def test_provider_dispatch_identity_binds_project_method_under_wrapped_forward() -> None:
    model = ProjectFakeStudent()
    identity = _provider_dispatch_identity(model)
    implementation = identity["implementations"][0]
    assert implementation["expert_class"].endswith(".WrappedProjectFakeExpert")
    assert implementation["dispatch_owner_class"].endswith(".ProjectFakeExpert")
    assert implementation["dispatch_owner_method"] == "_project"
    assert implementation["dispatch_callable"].endswith(".grouped_packed_projection")
    assert implementation["dispatch_source_sha256"]


def test_provider_project_code_object_emits_call_and_return_boundaries(tmp_path: Path) -> None:
    model = ProjectFakeStudent()
    trace_path = tmp_path / "project-trace.jsonl"
    value = torch.ones(2, 3)
    with FullCallTreeTrace(
        model,
        trace_path,
        rail="test",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ):
        assert torch.equal(model.experts[0](value), value)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    grouped = [
        row for row in rows
        if str(row.get("semantic_boundary", "")).startswith("grouped_mm_dispatch_")
    ]
    assert [row["semantic_boundary"] for row in grouped] == [
        "grouped_mm_dispatch_input", "grouped_mm_dispatch_output",
    ]
    assert [row["projection"] for row in grouped] == ["w2", "w2"]
    assert all(
        row["code_binding"]["owner_class"].endswith(".ProjectFakeExpert")
        for row in grouped
    )


def test_rebound_project_code_emits_wrapper_and_super_identity(tmp_path: Path) -> None:
    model = ReboundProjectFakeStudent()
    bindings = _provider_rebound_project_code_bindings(model)
    assert len(bindings) == 1
    binding = next(iter(bindings.values()))
    assert binding["owner_class"].endswith(".ReboundProjectFakeExpert")
    assert binding["super_owner_class"].endswith(".ProjectFakeExpert")
    assert binding["super_callable"].endswith(".ProjectFakeExpert._project")

    trace_path = tmp_path / "rebound-project-trace.jsonl"
    value = torch.ones(2, 3)
    with FullCallTreeTrace(
        model,
        trace_path,
        rail="test",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ):
        assert torch.equal(model.experts[0](value), value)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rebound = [
        row for row in rows
        if str(row.get("semantic_boundary", "")).startswith("rebound_project_")
    ]
    assert [row["semantic_boundary"] for row in rebound] == [
        "rebound_project_input", "rebound_project_output",
    ]
    assert [row["projection"] for row in rebound] == ["w2", "w2"]
    assert rebound[0]["code_binding"]["super_firstlineno"] == ProjectFakeExpert._project.__code__.co_firstlineno
    assert rebound[0]["tensors"][0]["name"] == "x"
