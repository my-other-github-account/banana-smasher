from __future__ import annotations

import copy
import threading

import torch

_LOCK = threading.Lock()
_STATS: dict[str, int] = {
    "forward_calls": 0,
    "backward_calls": 0,
    "grouped_experts": 0,
    "max_nodes_per_projection": 0,
    "grad_weight_bmm_launches": 0,
    "reduction_kernel_launches": 0,
}
_ALLOW_REFERENCE = False


def reset_layer_graph_vjp(*, allow_reference: bool = False) -> None:
    global _ALLOW_REFERENCE
    if not isinstance(allow_reference, bool):
        raise TypeError("allow_reference must be a bool")
    with _LOCK:
        _ALLOW_REFERENCE = allow_reference
        for key in _STATS:
            _STATS[key] = 0


def layer_graph_vjp_stats() -> dict[str, int | bool]:
    with _LOCK:
        return {**copy.deepcopy(_STATS), "reference_opt_in": _ALLOW_REFERENCE}


def _balanced_activations(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack routed tokens into a padded expert-major slab without assuming balance."""
    if hidden_states.ndim != 2 or top_k_index.ndim != 2:
        raise ValueError("layer graph expects rank-2 hidden states and routing")
    if tuple(top_k_weights.shape) != tuple(top_k_index.shape):
        raise ValueError("routing index/weight shape drift")
    if top_k_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("routing indices must be integer tensors")
    if experts <= 0:
        raise ValueError("layer graph expert count must be positive")
    if top_k_index.numel() and (
        int(top_k_index.min()) < 0 or int(top_k_index.max()) >= experts
    ):
        raise ValueError("routing index is outside the expert range")

    tokens, top_k = map(int, top_k_index.shape)
    counts = torch.bincount(top_k_index.reshape(-1).long(), minlength=experts)
    routes = tokens * top_k
    routes_per_expert = int(counts.max()) if routes else 0
    if routes_per_expert <= 0:
        raise ValueError("layer graph requires at least one routed expert")
    slot_major_experts = top_k_index.transpose(0, 1).reshape(-1).long()
    order = torch.argsort(slot_major_experts, stable=True)
    sorted_expert = slot_major_experts[order]
    token = torch.arange(tokens, device=top_k_index.device).repeat(top_k)[order]
    slot = (
        torch.arange(top_k, device=top_k_index.device)
        .repeat_interleave(tokens)[order]
    )
    starts = torch.cumsum(counts, 0) - counts
    within_expert = torch.arange(routes, device=top_k_index.device) - torch.repeat_interleave(
        starts, counts
    )
    activations = torch.zeros(
        experts,
        routes_per_expert,
        hidden_states.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    ).index_put(
        (sorted_expert, within_expert), hidden_states.index_select(0, token)
    )
    return activations, token, slot, sorted_expert, within_expert


def _eager_grouped_codebook_vjp(
    grad_weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    codebook_shape: tuple[int, ...],
) -> torch.Tensor:
    experts = int(grad_weight.shape[0])
    code_dim = int(codebook_shape[1])
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    grouped = (grad_weight.float() * scale_columns).reshape(
        experts, codes.shape[1], codes.shape[2], code_dim
    )
    partial = torch.zeros(
        experts,
        *codebook_shape,
        dtype=torch.float32,
        device=grad_weight.device,
    )
    for expert in range(experts):
        partial[expert].index_add_(
            0,
            codes[expert].reshape(-1).long(),
            grouped[expert].reshape(-1, code_dim),
        )
    return partial.sum(dim=0)


class LayerProjectionKMajorFn(torch.autograd.Function):
    """One BMM/autograd node for one same-codebook expert projection."""

    @staticmethod
    def forward(ctx, activations, codebook, codes, scales, dense):
        if activations.ndim != 3 or dense.ndim != 3:
            raise ValueError("layer-projection K-major tensors must be rank 3")
        if int(activations.shape[0]) != int(dense.shape[0]):
            raise ValueError("expert cardinality drift")
        if int(activations.shape[2]) != int(dense.shape[1]):
            raise ValueError("activation/tile K dimension drift")
        if codes.requires_grad or scales.requires_grad or dense.requires_grad:
            raise ValueError("packed planes and detached K-major slab must stay frozen")
        if codes.dtype != torch.int32 or scales.dtype != torch.uint8:
            raise ValueError("K-major packed planes must be int32 codes and uint8 scales")
        activations = activations.contiguous()
        ctx.save_for_backward(activations, codes, scales, dense)
        ctx.codebook_shape = tuple(codebook.shape)
        result = torch.bmm(activations, dense)
        with _LOCK:
            _STATS["forward_calls"] += 1
            _STATS["grouped_experts"] += int(activations.shape[0])
            _STATS["max_nodes_per_projection"] = max(
                _STATS["max_nodes_per_projection"], 1
            )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        activations, codes, scales, dense = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if not grad_output.is_cuda and not _ALLOW_REFERENCE:
            raise RuntimeError(
                "CPU layer-graph VJP is a debug path requiring explicit reference opt-in"
            )
        grad_activations = torch.bmm(grad_output, dense.transpose(1, 2))
        grad_weight = torch.bmm(grad_output.transpose(1, 2), activations)
        with _LOCK:
            _STATS["grad_weight_bmm_launches"] += 1
        if grad_output.is_cuda:
            from .kmajor_fused import fused_grouped_codebook_vjp

            grad_codebook = fused_grouped_codebook_vjp(
                grad_weight,
                codes,
                scales,
                int(ctx.codebook_shape[0]),
                int(ctx.codebook_shape[1]),
            )
            with _LOCK:
                _STATS["reduction_kernel_launches"] += 1
        else:
            grad_codebook = _eager_grouped_codebook_vjp(
                grad_weight, codes, scales, ctx.codebook_shape
            )
        with _LOCK:
            _STATS["backward_calls"] += 1
        return grad_activations, grad_codebook, None, None, None


def layer_graph_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    payloads: dict[str, dict[str, torch.Tensor]],
    *,
    limit: float,
) -> torch.Tensor:
    """Run two grouped K-major projections with one node per layer projection."""
    required = {"13", "2"}
    if set(payloads) != required:
        raise ValueError(f"layer graph requires projection payloads {sorted(required)}")
    experts = int(payloads["13"]["dense"].shape[0])
    if int(payloads["2"]["dense"].shape[0]) != experts:
        raise ValueError("projection expert cardinality drift")
    activations, token, slot, sorted_expert, within_expert = _balanced_activations(
        hidden_states, top_k_index, top_k_weights, experts=experts
    )
    first = payloads["13"]
    projected = LayerProjectionKMajorFn.apply(
        activations,
        first["codebook"],
        first["codes"],
        first["scales"],
        first["dense"],
    )
    gate, up = projected.chunk(2, dim=-1)
    intermediate = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
        min=-limit, max=limit
    )
    second = payloads["2"]
    projected = LayerProjectionKMajorFn.apply(
        intermediate,
        second["codebook"],
        second["codes"],
        second["scales"],
        second["dense"],
    )
    weighted = projected[sorted_expert, within_expert] * top_k_weights[
        token, slot, None
    ]
    return torch.zeros_like(hidden_states).index_add(0, token, weighted)
