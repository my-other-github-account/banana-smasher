from __future__ import annotations

from pathlib import Path

import pytest
import torch

from banana_smasher.fwht import bounded_fwht, fwht_stats
from banana_smasher.kmajor_batch import (
    BatchedKMajorVQLinearFn,
    batched_kmajor_vjp_stats,
    reset_batched_kmajor_vjp,
)
from banana_smasher.kmajor_fused import (
    fused_codebook_vjp,
    fused_codebook_vjp_from_inputs,
)
from banana_smasher.kmajor_graph import (
    LayerProjectionKMajorFn,
    layer_graph_forward,
    layer_graph_vjp_stats,
    reset_layer_graph_vjp,
)
from banana_smasher.production_update import run_full_depth_update
from banana_smasher.token_sizing import MemoryBudget

GiB = 1024**3


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "aot_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
        "code_sha256": "6" * 64,
    }


def _budget() -> MemoryBudget:
    return MemoryBudget(
        available_bytes=16 * GiB,
        resident_frozen_bytes=1,
        trainable_bytes=1,
        optimizer_bytes=2,
        staging_bytes=1,
        calibrated_activation_bytes_per_token=1,
    )


def _dense(
    codebook: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    wire = codebook.detach()
    weight = wire[codes.long()].reshape(codes.shape[0], -1) * columns
    return weight.transpose(0, 1).contiguous()


class _ReferenceKMajor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, codebook, codes, scales, dense):
        ctx.save_for_backward(inputs, codes, scales, dense)
        ctx.codebook_shape = tuple(codebook.shape)
        return inputs @ dense

    @staticmethod
    def backward(ctx, grad_output):
        inputs, codes, scales, dense = ctx.saved_tensors
        grad_input = grad_output @ dense.T
        grad_weight = grad_output.T @ inputs
        code_dim = int(ctx.codebook_shape[1])
        columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        grouped = (grad_weight.float() * columns).reshape(
            codes.shape[0], codes.shape[1], code_dim
        )
        grad_codebook = torch.zeros(ctx.codebook_shape)
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, code_dim)
        )
        return grad_input, grad_codebook, None, None, None


def test_batched_kmajor_vjp_matches_same_work_reference() -> None:
    torch.manual_seed(7)
    experts, rows, width, code_dim, codes_count, batch = 2, 4, 32, 2, 8, 3
    reference_codebook = torch.randn(codes_count, code_dim, requires_grad=True)
    accelerated_codebook = reference_codebook.detach().clone().requires_grad_(True)
    inputs_reference = [torch.randn(batch, width, requires_grad=True) for _ in range(experts)]
    inputs_accelerated = [value.detach().clone().requires_grad_(True) for value in inputs_reference]
    codes = [
        torch.randint(0, codes_count, (rows, width // code_dim), dtype=torch.int32)
        for _ in range(experts)
    ]
    scales = [torch.full((rows, width // 32), 127, dtype=torch.uint8) for _ in range(experts)]
    gradients = [torch.randn(batch, rows) for _ in range(experts)]

    expected = [
        _ReferenceKMajor.apply(x, reference_codebook, c, s, _dense(reference_codebook, c, s))
        for x, c, s in zip(inputs_reference, codes, scales)
    ]
    sum((value * grad).sum() for value, grad in zip(expected, gradients)).backward()

    reset_batched_kmajor_vjp(batch_size=experts, allow_reference=True)
    observed = [
        BatchedKMajorVQLinearFn.apply(
            x, accelerated_codebook, c, s, _dense(accelerated_codebook, c, s)
        )
        for x, c, s in zip(inputs_accelerated, codes, scales)
    ]
    sum((value * grad).sum() for value, grad in zip(observed, gradients)).backward()

    torch.testing.assert_close(accelerated_codebook.grad, reference_codebook.grad)
    for left, right in zip(inputs_accelerated, inputs_reference):
        torch.testing.assert_close(left.grad, right.grad)
    assert batched_kmajor_vjp_stats() == {
        "forward_calls": 2,
        "backward_calls": 2,
        "batch_flushes": 1,
        "max_pending": 2,
        "batch_size": 2,
        "unique_groups": 1,
        "active_groups": 0,
        "reference_opt_in": True,
    }


def test_batched_kmajor_cpu_reference_requires_explicit_opt_in() -> None:
    codebook = torch.randn(8, 2, requires_grad=True)
    inputs = torch.randn(3, 32, requires_grad=True)
    codes = torch.randint(0, 8, (4, 16), dtype=torch.int32)
    scales = torch.full((4, 1), 127, dtype=torch.uint8)
    reset_batched_kmajor_vjp(batch_size=1)
    output = BatchedKMajorVQLinearFn.apply(
        inputs, codebook, codes, scales, _dense(codebook, codes, scales)
    )

    with pytest.raises(RuntimeError, match="explicit reference opt-in"):
        output.square().sum().backward()


def test_grouped_layer_graph_preserves_unbalanced_routing_and_vjp() -> None:
    torch.manual_seed(9)
    experts, tokens, top_k, hidden, code_dim, code_count = 4, 6, 2, 32, 2, 8
    route_index = torch.tensor(
        [[0, 1], [0, 1], [0, 2], [1, 2], [1, 2], [0, 2]], dtype=torch.int64
    )
    route_weights = torch.rand(tokens, top_k)
    inputs = torch.randn(tokens, hidden, requires_grad=True)
    codebook13 = torch.randn(code_count, code_dim, requires_grad=True)
    codebook2 = torch.randn(code_count, code_dim, requires_grad=True)
    codes13 = torch.randint(
        0, code_count, (experts, hidden * 2, hidden // code_dim), dtype=torch.int32
    )
    codes2 = torch.randint(
        0, code_count, (experts, hidden, hidden // code_dim), dtype=torch.int32
    )
    scales13 = torch.full((experts, hidden * 2, hidden // 32), 127, dtype=torch.uint8)
    scales2 = torch.full((experts, hidden, hidden // 32), 127, dtype=torch.uint8)
    payloads = {
        "13": {
            "codebook": codebook13,
            "codes": codes13,
            "scales": scales13,
            "dense": torch.stack(
                [_dense(codebook13, codes13[i], scales13[i]) for i in range(experts)]
            ),
        },
        "2": {
            "codebook": codebook2,
            "codes": codes2,
            "scales": scales2,
            "dense": torch.stack(
                [_dense(codebook2, codes2[i], scales2[i]) for i in range(experts)]
            ),
        },
    }

    reset_layer_graph_vjp(allow_reference=True)
    output = layer_graph_forward(inputs, route_index, route_weights, payloads, limit=10.0)
    output.square().mean().backward()

    assert output.shape == (tokens, hidden)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert codebook13.grad is not None and torch.isfinite(codebook13.grad).all()
    assert codebook2.grad is not None and torch.isfinite(codebook2.grad).all()
    stats = layer_graph_vjp_stats()
    assert stats["forward_calls"] == 2
    assert stats["backward_calls"] == 2
    assert stats["grouped_experts"] == experts * 2
    assert stats["max_nodes_per_projection"] == 1
    assert stats["reference_opt_in"] is True


def test_layer_graph_cpu_reference_requires_explicit_opt_in() -> None:
    codebook = torch.randn(8, 2, requires_grad=True)
    codes = torch.randint(0, 8, (1, 4, 16), dtype=torch.int32)
    scales = torch.full((1, 4, 1), 127, dtype=torch.uint8)
    dense = _dense(codebook, codes[0], scales[0]).unsqueeze(0)
    activations = torch.randn(1, 3, 32, requires_grad=True)
    reset_layer_graph_vjp()
    output = LayerProjectionKMajorFn.apply(
        activations, codebook, codes, scales, dense
    )

    with pytest.raises(RuntimeError, match="explicit reference opt-in"):
        output.square().sum().backward()


def test_bounded_fwht_depth_seam_has_bounded_scratch() -> None:
    values = torch.randn(2, 32)
    pointer = values.data_ptr()
    fwht_stats(reset=True)
    transformed = bounded_fwht(values, inplace=True)
    assert transformed.data_ptr() == pointer
    assert fwht_stats()["max_scratch_bytes"] == values.numel() * values.element_size() // 2


class TokenLayer(torch.nn.Module):
    def __init__(self, counters: dict[str, int]) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.counters = counters

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for name in self.counters:
            self.counters[name] += 1
        return hidden * self.scale


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_full_depth_update_uses_physical_tokens_at_every_depth(
    tmp_path: Path, depth: int
) -> None:
    tokens = 8
    counters = {
        "kmajor_batch": 0,
        "kmajor_fused": 0,
        "grouped_vjp": 0,
        "layer_graph": 0,
        "fwht": 0,
    }
    layers = torch.nn.ModuleList([TokenLayer(counters) for _ in range(depth)])
    frozen = torch.nn.Linear(1, 1, bias=False)
    frozen.requires_grad_(False)
    input_ids = torch.arange(tokens, dtype=torch.float32).reshape(1, -1)
    targets = torch.zeros((1, tokens, 1), dtype=torch.float32)
    mask = torch.ones((1, tokens), dtype=torch.bool)
    positions = torch.arange(tokens).reshape(1, -1)

    receipt = run_full_depth_update(
        layers=layers,
        frozen_modules=[frozen],
        input_ids=input_ids,
        teacher_targets=targets,
        teacher_mask=mask,
        positions=positions,
        requested_tokens=tokens,
        segments=1,
        batch_size=1,
        memory_budget=_budget(),
        encode=lambda segment: segment["input_ids"].unsqueeze(-1),
        loss_sum=lambda hidden, segment: (
            (hidden - segment["teacher_targets"])
            .masked_select(segment["teacher_mask"].unsqueeze(-1))
            .square()
            .sum()
        ),
        output=tmp_path / f"depth-{depth}.pt",
        identity=_identity(),
        peak_memory_bytes=123,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-4),
        backend_sentinels=lambda: counters,
    )

    production = receipt["production_runtime"]
    assert production["depth"] == depth
    assert production["depth_shapes"] == [[[1, tokens, 1]] * depth]
    assert production["trainable_parameter_tensors"] == depth
    assert production["frozen_parameter_tensors"] == 1
    assert production["warm_start_used"] is False
    assert production["backend_sentinels"] == {
        "fwht": depth,
        "grouped_vjp": depth,
        "kmajor_batch": depth,
        "kmajor_fused": depth,
        "layer_graph": depth,
    }
    assert receipt["physical_tokens"] == tokens
    assert receipt["observed_input_shape"] == [1, tokens]
    assert receipt["optimizer_steps"] == 1
    assert all(parameter.grad is None for parameter in frozen.parameters())


def test_full_depth_update_rejects_slow_or_reference_backend(tmp_path: Path) -> None:
    counters = {
        "kmajor_batch": 1,
        "kmajor_fused": 1,
        "grouped_vjp": 1,
        "fwht": 1,
    }
    layer = TokenLayer(counters)
    values = torch.ones((1, 2))
    with pytest.raises(ValueError, match="accelerated production backend"):
        run_full_depth_update(
            layers=[layer],
            frozen_modules=[],
            input_ids=values,
            teacher_targets=values.unsqueeze(-1),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(2).reshape(1, -1),
            requested_tokens=2,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            encode=lambda segment: segment["input_ids"].unsqueeze(-1),
            loss_sum=lambda hidden, segment: hidden.square().sum(),
            output=tmp_path / "forbidden.pt",
            identity=_identity(),
            peak_memory_bytes=0,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-4),
            backend_sentinels=lambda: counters,
        )


def test_batched_kmajor_refuses_an_undrained_tail() -> None:
    codebook = torch.randn(8, 2, requires_grad=True)
    codes = torch.randint(0, 8, (4, 16), dtype=torch.int32)
    scales = torch.full((4, 1), 127, dtype=torch.uint8)
    reset_batched_kmajor_vjp(batch_size=2, allow_reference=True)
    outputs = [
        BatchedKMajorVQLinearFn.apply(
            torch.randn(3, 32, requires_grad=True),
            codebook,
            codes,
            scales,
            _dense(codebook, codes, scales),
        )
        for _ in range(3)
    ]

    with pytest.raises(RuntimeError, match="undrained.*tail"):
        torch.stack([value.square().sum() for value in outputs]).sum().backward()


def test_fused_kmajor_rejects_untrusted_codes_and_layout_before_launch() -> None:
    grad_weight = torch.randn(4, 32)
    scales = torch.full((4, 1), 127, dtype=torch.uint8)
    out_of_range = torch.full((4, 16), 8, dtype=torch.int32)
    with pytest.raises(ValueError, match="code index"):
        fused_codebook_vjp(grad_weight, out_of_range, scales, 8, 2)

    strided = torch.zeros((4, 32), dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        fused_codebook_vjp(grad_weight, strided, scales, 8, 2)


def test_fused_input_vjp_requires_bfloat16_contract() -> None:
    with pytest.raises(ValueError, match="bfloat16"):
        fused_codebook_vjp_from_inputs(
            torch.randn(2, 4),
            torch.randn(2, 32),
            torch.zeros((4, 16), dtype=torch.int32),
            torch.full((4, 1), 127, dtype=torch.uint8),
            8,
            2,
        )


def test_full_depth_update_rejects_static_caller_sentinels(tmp_path: Path) -> None:
    counters = {
        name: 1
        for name in (
            "kmajor_batch",
            "kmajor_fused",
            "grouped_vjp",
            "layer_graph",
            "fwht",
        )
    }
    layer = torch.nn.Linear(1, 1, bias=False)
    values = torch.ones((1, 2))
    with pytest.raises(ValueError, match="observed path activity"):
        run_full_depth_update(
            layers=[layer],
            frozen_modules=[],
            input_ids=values,
            teacher_targets=values.unsqueeze(-1),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(2).reshape(1, -1),
            requested_tokens=2,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            encode=lambda segment: segment["input_ids"].unsqueeze(-1),
            loss_sum=lambda hidden, segment: hidden.square().sum(),
            output=tmp_path / "static.pt",
            identity=_identity(),
            peak_memory_bytes=0,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-4),
            backend_sentinels=lambda: counters,
        )


def test_full_depth_update_defaults_to_no_equivalence_claim(tmp_path: Path) -> None:
    counters = {
        name: 0
        for name in (
            "kmajor_batch",
            "kmajor_fused",
            "grouped_vjp",
            "layer_graph",
            "fwht",
        )
    }
    layer = TokenLayer(counters)
    values = torch.ones((1, 2))
    receipt = run_full_depth_update(
        layers=[layer],
        frozen_modules=[],
        input_ids=values,
        teacher_targets=values.unsqueeze(-1),
        teacher_mask=torch.ones_like(values, dtype=torch.bool),
        positions=torch.arange(2).reshape(1, -1),
        requested_tokens=2,
        segments=1,
        batch_size=1,
        memory_budget=_budget(),
        encode=lambda segment: segment["input_ids"].unsqueeze(-1),
        loss_sum=lambda hidden, segment: hidden.square().sum(),
        output=tmp_path / "honest.pt",
        identity=_identity(),
        peak_memory_bytes=0,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-4),
        backend_sentinels=lambda: counters,
    )
    assert receipt["semantic_parity"] == {
        "claim": "causal-segmented-no-equivalence-claim",
        "tested": False,
    }


def test_full_depth_update_rejects_frozen_buffer_mutation(tmp_path: Path) -> None:
    counters = {
        name: 0
        for name in (
            "kmajor_batch",
            "kmajor_fused",
            "grouped_vjp",
            "layer_graph",
            "fwht",
        )
    }
    layer = TokenLayer(counters)
    frozen = torch.nn.Module()
    frozen.register_buffer("frozen_value", torch.ones(1))
    values = torch.ones((1, 2))

    def encode(segment):
        frozen.frozen_value.add_(1)
        return segment["input_ids"].unsqueeze(-1)

    with pytest.raises(RuntimeError, match="frozen.*mutated"):
        run_full_depth_update(
            layers=[layer],
            frozen_modules=[frozen],
            input_ids=values,
            teacher_targets=values.unsqueeze(-1),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(2).reshape(1, -1),
            requested_tokens=2,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            encode=encode,
            loss_sum=lambda hidden, segment: hidden.square().sum(),
            output=tmp_path / "mutated.pt",
            identity=_identity(),
            peak_memory_bytes=0,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-4),
            backend_sentinels=lambda: counters,
        )
