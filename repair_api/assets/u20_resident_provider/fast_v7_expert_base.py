"""Fully resident, grouped official-K2 routed experts for all V7 layers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fast_k2_grouped import grouped_k2_stats, grouped_packed_projection

EXPERTS = 256
PACKED_BYTES = 2_097_152
PROJECTIONS = ("w1", "w2", "w3")
PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}


class FullyResidentGroupedV7Experts(nn.Module):
    """One routed layer whose complete immutable K2 wire stays on CUDA."""

    def __init__(self, layer: int, pilot: bool = True, *, plane_source: Any) -> None:
        super().__init__()
        self.L = int(layer)
        self.pilot = bool(pilot)
        if not self.pilot or int(plane_source.layer) != self.L:
            raise RuntimeError(f"invalid admitted PlaneSource for L{self.L:03d}")
        self.__dict__["plane_source"] = plane_source
        device = plane_source.master.device
        if device.type != "cuda":
            raise RuntimeError("fully resident V7 experts require CUDA")
        self.resident_bytes = 0
        self.disk_read_calls = 0
        self.disk_read_bytes = 0
        self.cpu_relay_bytes = 0
        self.reconstruction_calls = 0
        self.fallback_calls = 0
        self.trace_enabled = os.environ.get("BR_TRACE_TIMING", "0") == "1"
        self._trace_events: list[tuple[str, Any, Any]] = []
        self._trace_forward_calls = 0
        self._trace_route_rows = 0
        self._trace_unique_experts = 0
        self.act = F.silu
        self.limit = 10.0

        for projection in PROJECTIONS:
            m, k = PROJECTION_SHAPES[projection]
            packed = torch.empty(
                (EXPERTS, k // 16, m // 16, 32),
                dtype=torch.int16,
                device=device,
            )
            su = torch.empty((EXPERTS, k), dtype=torch.float16, device=device)
            sv = torch.empty((EXPERTS, m), dtype=torch.float16, device=device)
            expected = PACKED_BYTES + (k + m) * 2 + 4
            for expert in range(EXPERTS):
                path = plane_source.member_path(expert, projection)
                payload = Path(path).read_bytes()
                if len(payload) != expected:
                    raise RuntimeError(
                        f"L{self.L:03d} E{expert:03d}/{projection} byte geometry drift"
                    )
                packed_cpu = torch.from_numpy(
                    np.frombuffer(payload[:PACKED_BYTES], dtype="<i2")
                    .copy()
                    .reshape(k // 16, m // 16, 32)
                )
                su_cpu = torch.from_numpy(
                    np.frombuffer(
                        payload[PACKED_BYTES : PACKED_BYTES + k * 2], dtype="<f2"
                    ).copy()
                )
                sv_cpu = torch.from_numpy(
                    np.frombuffer(
                        payload[PACKED_BYTES + k * 2 : PACKED_BYTES + (k + m) * 2],
                        dtype="<f2",
                    ).copy()
                )
                packed[expert].copy_(packed_cpu)
                su[expert].copy_(su_cpu)
                sv[expert].copy_(sv_cpu)
                self.disk_read_calls += 1
                self.disk_read_bytes += len(payload)
            self.register_buffer(f"packed_{projection}", packed, persistent=False)
            self.register_buffer(f"su_{projection}", su, persistent=False)
            self.register_buffer(f"sv_{projection}", sv, persistent=False)
            self.resident_bytes += sum(
                value.numel() * value.element_size() for value in (packed, su, sv)
            )

    def reset_trace(self) -> None:
        self._trace_events.clear()
        self._trace_forward_calls = 0
        self._trace_route_rows = 0
        self._trace_unique_experts = 0

    def trace_snapshot(self, *, clear: bool = True) -> dict[str, Any]:
        if not self.trace_enabled:
            return {}
        projection_gpu_seconds = {name: 0.0 for name in PROJECTIONS}
        for projection, started, ended in self._trace_events:
            projection_gpu_seconds[projection] += float(started.elapsed_time(ended)) / 1000.0
        value = {
            "layer": self.L,
            "forward_calls": self._trace_forward_calls,
            "route_rows": self._trace_route_rows,
            "unique_expert_observations": self._trace_unique_experts,
            "projection_calls": len(self._trace_events),
            "projection_gpu_seconds": projection_gpu_seconds,
        }
        if clear:
            self.reset_trace()
        return value

    def _project(
        self,
        projection: str,
        x: torch.Tensor,
        assignments: torch.Tensor,
        packed: torch.Tensor,
        lut_master: torch.Tensor,
        su: torch.Tensor,
        sv: torch.Tensor,
    ) -> torch.Tensor:
        if not self.trace_enabled:
            return grouped_packed_projection(x, assignments, packed, lut_master, su, sv)
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        value = grouped_packed_projection(x, assignments, packed, lut_master, su, sv)
        ended.record()
        self._trace_events.append((projection, started, ended))
        return value

    @property
    def codebooks(self) -> list[torch.Tensor]:
        return [self.plane_source.master]

    def mechanism_stats(self) -> dict[str, int]:
        return {
            **grouped_k2_stats(),
            "resident_bytes": int(self.resident_bytes),
            "disk_read_calls_init": int(self.disk_read_calls),
            "disk_read_bytes_init": int(self.disk_read_bytes),
            "cpu_relay_bytes": int(self.cpu_relay_bytes),
            "reconstruction_calls": int(self.reconstruction_calls),
            "fallback_calls": int(self.fallback_calls),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve batch-4 math with one live expert workspace per mb2 group."""
        sealed_group_tokens = 2 * 2048
        if hidden_states.shape[0] > sealed_group_tokens:
            if hidden_states.shape[0] % sealed_group_tokens:
                raise RuntimeError("packed V7 batch must divide sealed group2 geometry")
            outputs = []
            for group in range(hidden_states.shape[0] // sealed_group_tokens):
                start = group * sealed_group_tokens
                stop = start + sealed_group_tokens
                outputs.append(self.forward(
                    hidden_states[start:stop],
                    top_k_index[start:stop],
                    top_k_weights[start:stop],
                ))
            return torch.cat(outputs, dim=0)
        if hidden_states.ndim != 2 or top_k_index.shape != top_k_weights.shape:
            raise RuntimeError("grouped V7 routing geometry drift")
        if top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("grouped V7 routing token count drift")
        token_index = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            .unsqueeze(1)
            .expand_as(top_k_index)
            .reshape(-1)
        )
        expert_index = top_k_index.reshape(-1).to(torch.int64)
        route_weight = top_k_weights.reshape(-1, 1).float()
        routed_hidden = hidden_states[token_index].contiguous()
        lut_master = self.plane_source.wire_lut().reshape(-1).contiguous()
        if self.trace_enabled:
            self._trace_forward_calls += 1
            self._trace_route_rows += int(routed_hidden.shape[0])
            self._trace_unique_experts += int(torch.unique(expert_index).numel())

        gate = self._project(
            "w1",
            routed_hidden,
            expert_index,
            self.packed_w1,
            lut_master,
            self.su_w1,
            self.sv_w1,
        ).clamp(max=self.limit)
        up = self._project(
            "w3",
            routed_hidden,
            expert_index,
            self.packed_w3,
            lut_master,
            self.su_w3,
            self.sv_w3,
        ).clamp(min=-self.limit, max=self.limit)
        activated = self.act(gate) * up
        routed_output = self._project(
            "w2",
            activated,
            expert_index,
            self.packed_w2,
            lut_master,
            self.su_w2,
            self.sv_w2,
        )
        routed_output = routed_output * route_weight
        final = torch.zeros_like(hidden_states, dtype=routed_output.dtype)
        final.index_add_(0, token_index, routed_output)
        self.cpu_relay_bytes += 0
        self.reconstruction_calls += 0
        return final.to(hidden_states.dtype)


__all__ = ["FullyResidentGroupedV7Experts"]
