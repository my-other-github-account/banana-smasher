from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI


PROVIDER_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"


class Immutable942cProjectionProvider:
    """Behavioral fixture for the immutable provider's projection call order."""

    def __init__(self) -> None:
        self.packed_w1 = torch.tensor([1])
        self.packed_w2 = torch.tensor([2])
        self.packed_w3 = torch.tensor([3])
        self.su_w1 = torch.tensor([11])
        self.su_w2 = torch.tensor([12])
        self.su_w3 = torch.tensor([13])
        self.sv_w1 = torch.tensor([21])
        self.sv_w2 = torch.tensor([22])
        self.sv_w3 = torch.tensor([23])
        self.plane_source = type("Plane", (), {"wire_lut": lambda _self: torch.tensor([31])})()

    def _project(
        self,
        projection: str,
        x: torch.Tensor,
        assignments: torch.Tensor,
        packed: torch.Tensor,
        lut_master: torch.Tensor,
        su: torch.Tensor,
        sv: torch.Tensor,
        *_args,
    ) -> torch.Tensor:
        del assignments, packed, lut_master, su, sv
        if projection == "w1":
            return torch.full_like(x, -1.0)
        if projection == "w3":
            return torch.full_like(x, 0.5)
        assert projection == "w2"
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        assignments = top_k_index.reshape(-1).to(torch.int64)
        routed = hidden_states.repeat_interleave(top_k_index.shape[1], dim=0)
        lut = self.plane_source.wire_lut()
        gate = self._project(
            "w1", routed, assignments, self.packed_w1, lut, self.su_w1, self.sv_w1
        )
        up = self._project(
            "w3", routed, assignments, self.packed_w3, lut, self.su_w3, self.sv_w3
        )
        product = torch.nn.functional.silu(gate) * up
        down = self._project(
            "w2", product, assignments, self.packed_w2, lut, self.su_w2, self.sv_w2
        )
        return (down * top_k_weights.reshape(-1, 1)).to(hidden_states.dtype)


def test_public_projection_wrapper_binds_combined_gate_up_and_product_exactly() -> None:
    config = {"basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"}
    bound = ResidentRepairAPI.bind_combined_gate_up_projection(
        config, provider_expert_sha256=PROVIDER_SHA256
    )

    assert config == {"basis_sha256": bound["basis_sha256"]}
    assert bound["resident_gate_up_projection"] == "combined_4096_bf16_f_linear_v1"
    assert bound["resident_gate_up_provider_sha256"] == PROVIDER_SHA256
    with pytest.raises(ArtifactError, match="requires provider 942c3074"):
        ResidentRepairAPI.bind_combined_gate_up_projection(
            config, provider_expert_sha256="0" * 64
        )

    from repair_api.modern_green_resident import _bind_sealed_gate_up_projection

    calls: list[tuple[torch.Tensor, ...]] = []

    def combined_projection(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args)
        x = args[0]
        return torch.ones_like(x), torch.full_like(x, 2.0)

    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    indices = torch.tensor([[0]], dtype=torch.int64)
    weights = torch.ones((1, 1), dtype=torch.bfloat16)
    baseline = Immutable942cProjectionProvider().forward(hidden, indices, weights)
    wrapped_type = _bind_sealed_gate_up_projection(
        Immutable942cProjectionProvider,
        bound,
        combined_projection=combined_projection,
    )
    observed = wrapped_type().forward(hidden, indices, weights)
    expected = (torch.nn.functional.silu(torch.ones_like(hidden)) * 2).to(torch.bfloat16)

    assert not torch.equal(baseline, expected)
    assert torch.equal(observed, expected)
    assert len(calls) == 1
    assert calls[0][2] is not calls[0][3]
    assert calls[0][2].item() == 1
    assert calls[0][3].item() == 3
    assert wrapped_type.__name__ == Immutable942cProjectionProvider.__name__


def test_product_runner_does_not_shadow_the_construction_time_projection_wrapper() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1] / "resident_full64_accept.py"
    ).read_text()
    binding = source.index("config = api.bind_combined_gate_up_projection(")
    construction = source.index("engine = ModernGreenResidentEngine(")

    assert binding < construction
    assert "projection_binding = _install_r20_full_weight_projection(engine)" not in source
    assert "projection_binding = engine.sealed_gate_up_runtime_witness(" in source[construction:]
    assert "require_activation=False" in source[construction:]


def test_combined_projection_uses_one_expert_local_bf16_linear() -> None:
    from repair_api.modern_green_resident import (
        _sealed_builder_combined_gate_up_projection,
    )

    torch.manual_seed(942)
    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    packed_w1 = torch.tensor([0, 1])
    packed_w3 = torch.tensor([2, 3])
    weights = [torch.randn(3, 2, dtype=torch.bfloat16) for _ in range(4)]
    su = torch.zeros((2, 3))
    sv = torch.zeros((2, 2))

    def full_weight_builder(packed, _lut, _su, _sv):
        return weights[int(packed.item())]

    gate, up = _sealed_builder_combined_gate_up_projection(
        hidden,
        assignments,
        packed_w1,
        packed_w3,
        torch.zeros(1),
        su,
        sv,
        su,
        sv,
        full_weight_builder=full_weight_builder,
    )
    expected = torch.empty((4, 4), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        combined_weight = torch.cat(
            (weights[expert].T.contiguous(), weights[expert + 2].T.contiguous()),
            dim=0,
        )
        expected[mask] = torch.nn.functional.linear(
            hidden[mask].contiguous(), combined_weight
        )
    expected_gate, expected_up = expected.chunk(2, dim=-1)

    assert torch.equal(gate, expected_gate)
    assert torch.equal(up, expected_up)


def test_down_projection_uses_expert_local_bf16_f_linear(monkeypatch) -> None:
    import repair_api.modern_green_resident as resident
    from repair_api.modern_green_resident import _sealed_builder_down_projection

    torch.manual_seed(106)
    hidden = torch.randn(5, 3, dtype=torch.bfloat16)
    assignments = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int64)
    packed_w2 = torch.tensor([0, 1])
    weights = [torch.randn(3, 4, dtype=torch.bfloat16) for _ in range(2)]
    su = torch.zeros((2, 3))
    sv = torch.zeros((2, 4))

    def full_weight_builder(packed, _lut, _su, _sv):
        return weights[int(packed.item())]

    captured = {}
    monkeypatch.setattr(
        resident, "_POST_W2_CHAIN_OBSERVER",
        lambda raw, exposure: captured.update(raw=raw, exposure=exposure),
        raising=False,
    )
    observed = _sealed_builder_down_projection(
        hidden, assignments, packed_w2, torch.zeros(1), su, sv,
        full_weight_builder=full_weight_builder,
    )
    expected = torch.empty((5, 4), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        expected[mask] = torch.nn.functional.linear(
            hidden[mask].contiguous(), weights[expert].T.contiguous()
        )

    # Attempt46 authenticated the native BF16 GEMM bytes and the public FP32
    # provider exposure independently.  Route assembly owns the later BF16 round.
    assert observed.dtype == torch.float32
    assert torch.equal(observed, expected.float())
    assert captured["raw"].dtype == torch.bfloat16
    assert torch.equal(captured["raw"], expected)
    assert captured["exposure"] is observed


def test_routed_return_rounds_weighted_rows_then_uses_reference_reshape_sum() -> None:
    from repair_api.modern_green_resident import _sealed_builder_accumulate_routes

    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    routed_output = torch.tensor(
        [[0.006927490234375], [0.578125], [-0.8125], [-0.94140625], [0.671875], [-0.134765625]],
        dtype=torch.float32,
    )
    top_k_index = torch.tensor([[5, 4, 3, 2, 1, 0]], dtype=torch.int64)
    top_k_weights = torch.tensor(
        [[0.3333, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16
    )

    observed = _sealed_builder_accumulate_routes(
        hidden, routed_output, top_k_index, top_k_weights
    )
    weighted = (
        routed_output.to(torch.bfloat16) * top_k_weights.reshape(-1, 1)
    )
    widened_then_rounded = (
        routed_output * top_k_weights.reshape(-1, 1)
    ).to(torch.bfloat16)
    expected = weighted.view(1, 6, 1).sum(dim=1).to(torch.bfloat16)

    # IEEE BF16 multiplication can round to the same bytes either way, so the
    # authenticated contract is the executed intermediate dtype, not merely the
    # final value.  Keep this focused source assertion beside the value check.
    import inspect
    source = inspect.getsource(_sealed_builder_accumulate_routes)
    assert "routed_output.to(hidden_states.dtype)" in source
    assert "routed_output\n        * top_k_weights" not in source
    assert torch.equal(weighted, widened_then_rounded)
    assert torch.equal(observed, expected)


def test_l000_w2_execution_writes_in_process_provider_and_bundle_header(
    monkeypatch, tmp_path, capsys
) -> None:
    from repair_api.modern_green_resident import _bind_sealed_gate_up_projection

    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {"basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"},
        provider_expert_sha256=PROVIDER_SHA256,
    )
    build_receipt = tmp_path / "BUNDLE_BUILD.json"
    build_receipt.write_text(json.dumps({
        "canonical_head": "d6f9c92",
        "canonical_resident_sha256": "a" * 64,
        "bundle_resident_sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
    }))
    header = tmp_path / "W2_RUNTIME_HEADER.json"
    monkeypatch.setenv("REQUIRE_W2_RUNTIME_HEADER", "1")
    monkeypatch.setenv("W2_RUNTIME_HEADER_PATH", str(header))
    monkeypatch.setenv("BUNDLE_BUILD_RECEIPT_PATH", str(build_receipt))

    def combined(x, *_args):
        return torch.ones_like(x), torch.ones_like(x)

    def down(x, *_args):
        return x.to(torch.bfloat16).float()

    wrapped = _bind_sealed_gate_up_projection(
        Immutable942cProjectionProvider,
        config,
        combined_projection=combined,
        down_projection=down,
    )()
    wrapped.L = 0
    wrapped.forward(
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.ones((1, 1), dtype=torch.bfloat16),
    )

    row = json.loads(header.read_text())
    assert row["status"] == "PASS_L000_W2_RUNTIME_HEADER"
    assert row["live_w2_input_dtype"] == "torch.bfloat16"
    assert row["live_w2_output_dtype"] == "torch.float32"
    assert row["executing_provider_module_file"] == str(Path(__file__).resolve())
    assert any(frame["co_filename"].endswith("modern_green_resident.py") for frame in row["w2_projection_frames"])
    assert row["bundle_build"]["canonical_resident_sha256"] == row["bundle_build"]["bundle_resident_sha256"]
    assert "PASS_L000_W2_RUNTIME_HEADER" in capsys.readouterr().out
