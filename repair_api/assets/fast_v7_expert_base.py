"""Fully resident, grouped official-K2 routed experts for all V7 layers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fast_k2_grouped import (
    grouped_k2_stats,
    grouped_packed_projection,
    grouped_sealed_gate_up_projection,
)

EXPERTS = 256
PACKED_BYTES = 2_097_152
PROJECTIONS = ("w1", "w2", "w3")
PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}


def _read_wire_payload(path: Path) -> bytes:
    """Read one immutable wire and release its clean source-cache pages."""
    with path.open("rb") as stream:
        payload = stream.read()
        advise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        if advise is not None and dontneed is not None:
            try:
                advise(stream.fileno(), 0, 0, dontneed)
            except OSError as exc:
                raise RuntimeError(f"wire source page-cache eviction failed: {path}: {exc}") from exc
    return payload


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
                payload = _read_wire_payload(Path(path))
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
        started = None
        ended = None
        if self.trace_enabled:
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
        value = grouped_packed_projection(x, assignments, packed, lut_master, su, sv)
        if os.environ.get("FAST_K2_SEALED_PROJECTION_BF16", "0") == "1":
            # RUN1698 exposes every BF16 GEMM result to JointV7ExpertBase as
            # FP32. Preserve that boundary without reconstructing full weights.
            value = value.to(torch.bfloat16).float()
        if started is not None and ended is not None:
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
        authentic_trace = getattr(self, "_authentic_expert_trace", None)
        if authentic_trace is not None:
            slot_index = (
                torch.arange(top_k_index.shape[1], device=top_k_index.device)
                .unsqueeze(0).expand_as(top_k_index).reshape(-1)
            )
            authentic_trace["route_key"] = torch.stack(
                (expert_index, slot_index, token_index), dim=1,
            ).detach()
            authentic_trace["route_weight"] = top_k_weights.reshape(-1).detach()
        route_weight = top_k_weights.reshape(-1, 1)
        routed_hidden = hidden_states[token_index].contiguous()
        lut_master = self.plane_source.wire_lut().reshape(-1).contiguous()
        if self.trace_enabled:
            self._trace_forward_calls += 1
            self._trace_route_rows += int(routed_hidden.shape[0])
            self._trace_unique_experts += int(torch.unique(expert_index).numel())

        # The sealed DeepseekV4Experts path runs every BF16 linear before its
        # clamp/SwiGLU seam. The grouped kernel accumulates in FP32, so round
        # each projection back to the public layer dtype at the same boundary.
        if os.environ.get("FAST_K2_SEALED_FULL_WEIGHT_BF16", "0") == "1":
            gate, up = grouped_sealed_gate_up_projection(
                routed_hidden,
                expert_index,
                self.packed_w1,
                self.packed_w3,
                lut_master,
                self.su_w1,
                self.sv_w1,
                self.su_w3,
                self.sv_w3,
            )
        else:
            gate = self._project(
                "w1", routed_hidden, expert_index, self.packed_w1,
                lut_master, self.su_w1, self.sv_w1,
            )
            up = self._project(
                "w3", routed_hidden, expert_index, self.packed_w3,
                lut_master, self.su_w3, self.sv_w3,
            )
        if os.environ.get("FAST_K2_SEALED_NO_SWIGLU_CLAMP", "0") != "1":
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        if authentic_trace is not None:
            authentic_trace["gate"] = gate.detach()
            authentic_trace["up"] = up.detach()
        activated = self.act(gate) * up
        if authentic_trace is not None:
            authentic_trace["activated"] = activated.detach()
        routed_output = self._project(
            "w2",
            activated,
            expert_index,
            self.packed_w2,
            lut_master,
            self.su_w2,
            self.sv_w2,
        )
        if authentic_trace is not None:
            authentic_trace["w2_down"] = routed_output.detach()
        # Match the selected grouped_mm provider's return operator: round each
        # route weight to the BF16 projection dtype before the single token-major
        # accumulation, then cast the completed routed tensor to the layer dtype.
        routed_output = routed_output * route_weight.to(dtype=routed_output.dtype)
        if authentic_trace is not None:
            authentic_trace["weighted_routed_output"] = routed_output.detach()
        # Match transformers/integrations/moe.py:470-474 exactly: restore the
        # original token/slot geometry and use its FP32-accumulating reduction.
        # The all-active census proved every weighted row exact and localized
        # the first difference to resident BF16 index_add_ versus this sum.
        final = routed_output.view(
            hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[-1]
        ).sum(dim=1)
        self.cpu_relay_bytes += 0
        self.reconstruction_calls += 0
        return final.to(hidden_states.dtype)


__all__ = ["FullyResidentGroupedV7Experts"]
