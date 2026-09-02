"""Fully resident, grouped official-K2 routed experts for all V7 layers."""
from __future__ import annotations

import concurrent.futures
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


def _load_projection_payloads(
    paths: list[Path], *, m: int, k: int, packed_bytes: int = PACKED_BYTES
) -> tuple[Any, Any, Any, int, int]:
    expected = packed_bytes + (k + m) * 2 + 4
    packed_host = np.empty((len(paths), k // 16, m // 16, 32), dtype="<i2")
    su_host = np.empty((len(paths), k), dtype="<f2")
    sv_host = np.empty((len(paths), m), dtype="<f2")
    def load_one(item: tuple[int, Path]) -> int:
        expert, path = item
        payload = Path(path).read_bytes()
        if len(payload) != expected:
            raise RuntimeError(f"resident wire byte geometry drift: {path}")
        packed_host[expert] = np.frombuffer(
            payload[:packed_bytes], dtype="<i2"
        ).reshape(k // 16, m // 16, 32)
        su_host[expert] = np.frombuffer(
            payload[packed_bytes : packed_bytes + k * 2], dtype="<f2"
        )
        sv_host[expert] = np.frombuffer(
            payload[packed_bytes + k * 2 : packed_bytes + (k + m) * 2],
            dtype="<f2",
        )
        return len(payload)

    workers = min(8, max(1, len(paths)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        read_bytes = sum(pool.map(load_one, enumerate(paths)))
    return (
        torch.from_numpy(packed_host),
        torch.from_numpy(su_host),
        torch.from_numpy(sv_host),
        len(paths),
        read_bytes,
    )


class FullyResidentGroupedV7Experts(nn.Module):
    """One routed layer whose complete immutable K2 wire stays on CUDA."""

    def __init__(
        self,
        layer: int,
        pilot: bool = True,
        *,
        plane_source: Any,
        swiglu_limit: float,
    ) -> None:
        super().__init__()
        self.L = int(layer)
        self.pilot = bool(pilot)
        if not self.pilot or int(plane_source.layer) != self.L:
            raise RuntimeError(f"invalid admitted PlaneSource for L{self.L:03d}")
        self.__dict__["plane_source"] = plane_source
        self.limit = float(swiglu_limit)
        if not np.isfinite(self.limit) or self.limit <= 0:
            raise RuntimeError("fully resident V7 expert SwiGLU limit is invalid")
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

        # Candidate-K storage sharing is injected by the public resident scorer.
        # The provider returns CUDA-IPC views whose allocating rank remains alive
        # for the complete score. Installing those exact storages here avoids a
        # second packed-wire materialization in the consuming rank.
        storage_provider = globals().get("_resident_storage_provider")
        if callable(storage_provider):
            shared = storage_provider(self.L, plane_source)
            if not isinstance(shared, dict):
                raise RuntimeError(f"L{self.L:03d} brokered resident storage is not a mapping")
            shared_master = shared.get("master")
            projections = shared.get("projections")
            if (
                not isinstance(projections, dict)
                or set(projections) != set(PROJECTIONS)
                or not isinstance(shared_master, torch.Tensor)
                or tuple(shared_master.shape) != (1024,)
                or shared_master.dtype != torch.float32
                or shared_master.device != device
            ):
                raise RuntimeError(f"L{self.L:03d} brokered resident storage identity drift")
            plane_source.master = nn.Parameter(shared_master, requires_grad=True)
            for projection in PROJECTIONS:
                values = projections[projection]
                if not isinstance(values, (tuple, list)) or len(values) != 3:
                    raise RuntimeError(f"L{self.L:03d}/{projection} brokered storage closure drift")
                packed, su, sv = values
                m, k = PROJECTION_SHAPES[projection]
                expected = (
                    ((EXPERTS, k // 16, m // 16, 32), torch.int16),
                    ((EXPERTS, k), torch.float16),
                    ((EXPERTS, m), torch.float16),
                )
                for tensor, (shape, dtype) in zip((packed, su, sv), expected):
                    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                        raise RuntimeError(f"L{self.L:03d}/{projection} brokered shape drift")
                    if tensor.dtype != dtype or tensor.device != device:
                        raise RuntimeError(f"L{self.L:03d}/{projection} brokered tensor identity drift")
                self.register_buffer(f"packed_{projection}", packed, persistent=False)
                self.register_buffer(f"su_{projection}", su, persistent=False)
                self.register_buffer(f"sv_{projection}", sv, persistent=False)
                self.resident_bytes += sum(
                    value.numel() * value.element_size() for value in (packed, su, sv)
                )
            return

        for projection in PROJECTIONS:
            m, k = PROJECTION_SHAPES[projection]
            paths = [
                Path(plane_source.member_path(expert, projection))
                for expert in range(EXPERTS)
            ]
            packed_cpu, su_cpu, sv_cpu, read_calls, read_bytes = _load_projection_payloads(
                paths, m=m, k=k
            )
            packed = packed_cpu.to(device=device)
            su = su_cpu.to(device=device)
            sv = sv_cpu.to(device=device)
            self.disk_read_calls += read_calls
            self.disk_read_bytes += read_bytes
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

        # The sealed DeepseekV4Experts path runs every BF16 linear before its
        # clamp/SwiGLU seam. The grouped kernel accumulates in FP32, so round
        # each projection back to the public layer dtype at the same boundary.
        gate = self._project(
            "w1",
            routed_hidden,
            expert_index,
            self.packed_w1,
            lut_master,
            self.su_w1,
            self.sv_w1,
        )
        up = self._project(
            "w3",
            routed_hidden,
            expert_index,
            self.packed_w3,
            lut_master,
            self.su_w3,
            self.sv_w3,
        )
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
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
        route_shape = tuple(int(value) for value in top_k_index.shape)
        routed_output = routed_output.reshape(
            route_shape[0], route_shape[1], hidden_states.shape[1]
        )
        expert_order = torch.argsort(top_k_index, dim=1, stable=True)
        ordered_output = torch.gather(
            routed_output,
            1,
            expert_order.unsqueeze(-1).expand_as(routed_output),
        )
        final = torch.zeros_like(hidden_states)
        for route_slot in range(route_shape[1]):
            final = (final + ordered_output[:, route_slot]).to(hidden_states.dtype)
        self.cpu_relay_bytes += 0
        self.reconstruction_calls += 0
        return final


__all__ = ["FullyResidentGroupedV7Experts"]
