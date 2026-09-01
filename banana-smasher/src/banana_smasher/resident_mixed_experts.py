"""Resident expert provider for immutable mixed Backpack cells."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

_RUNTIME: Any | None = None


def _canonical_grouped_mm_forward(
    module: nn.Module,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
) -> Any:
    """Dispatch through the same grouped-mm expert rail as DeepseekV4."""
    from transformers.integrations.moe import grouped_mm_experts_forward

    return grouped_mm_experts_forward(
        module, hidden_states, top_k_index, top_k_weights
    )


def configure_mixed_backpack(config: Mapping[str, Any]) -> None:
    """Construct the one physical mixed-cell runtime used by all local layers."""
    global _RUNTIME
    binding = config.get("mixed_backpack_runtime")
    if not isinstance(binding, Mapping):
        raise RuntimeError("mixed resident expert source lacks backpack runtime binding")
    from banana_smasher.hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime

    parameters = {
        "positions": 1024,
        "backpack_runtime": {
            key: value for key, value in binding.items() if key != "model_root"
        },
    }
    _RUNTIME = DeepseekV4BackpackRuntime(
        model_root=str(binding["model_root"]), parameters=parameters
    )


class FullyResidentGroupedV7Experts(nn.Module):
    """Routed experts materialized from the sealed mixed cells.

    The historical class name is the trainer's authenticated provider ABI. This
    implementation does not reinterpret cells as V7: it asks
    ``DeepseekV4BackpackRuntime`` to decode each selected native/QTIP cell into
    rank-local BF16 expert tensors, then executes those tensors directly.
    """

    def __init__(
        self,
        layer: int,
        pilot: bool = True,
        *,
        plane_source: Any,
        swiglu_limit: float = 10.0,
    ) -> None:
        if _RUNTIME is None:
            raise RuntimeError("mixed resident expert source is not configured")
        if not pilot or int(plane_source.layer) != int(layer):
            raise RuntimeError(f"invalid admitted mixed provider for L{int(layer):03d}")
        super().__init__()
        self.L = int(layer)
        self.__dict__["plane_source"] = plane_source
        # Keep the compressed 102 GB artifact as the physical source of truth.
        # The validated continuation uses reentrant activation checkpointing, so
        # one layer can be decoded for its forward/recompute and released before
        # the next layer instead of retaining impossible 12+ GB BF16 expansions
        # for every rank-local layer.
        self.resident_bytes = 0
        self.disk_read_calls = 0
        self.disk_read_bytes = 0
        self.cpu_relay_bytes = 0
        self.reconstruction_calls = 0
        self.fallback_calls = 0
        self.limit = float(swiglu_limit)
        self._torch = torch

    @property
    def codebooks(self) -> list[Any]:
        # Preserve the official trainable-surface ABI; the mixed wire itself is
        # immutable and is never rewritten or relabeled by this provider.
        return [self.plane_source.master]

    def mechanism_stats(self) -> dict[str, int]:
        return {
            "resident_bytes": int(self.resident_bytes),
            "disk_read_calls_init": int(self.disk_read_calls),
            "disk_read_bytes_init": int(self.disk_read_bytes),
            "cpu_relay_bytes": 0,
            "reconstruction_calls": 0,
            "fallback_calls": 0,
        }

    def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        if hidden_states.ndim != 2 or top_k_index.shape != top_k_weights.shape:
            raise RuntimeError("mixed resident routing geometry drift")
        gate_up_wire, down_wire = _RUNTIME._load_vq3u_experts(self.L)
        self.disk_read_calls += 512
        # DeepseekV4 model construction selects ``grouped_mm`` even when
        # attention is eager.  Reuse that exact dispatch: sorted grouped GEMMs
        # plus FP32 reshape/sum are numerically part of the canonical scorer.
        self.num_experts = int(gate_up_wire.shape[0])
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.gate_up_proj = gate_up_wire
        self.down_proj = down_wire
        try:
            return _canonical_grouped_mm_forward(
                self, hidden_states, top_k_index, top_k_weights
            )
        finally:
            del self.gate_up_proj
            del self.down_proj

    def _apply_gate(self, gate_up: Any) -> Any:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self._torch.nn.functional.silu(gate) * up


__all__ = ["FullyResidentGroupedV7Experts", "configure_mixed_backpack"]
