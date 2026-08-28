"""Fully resident, grouped official-K2 routed experts for all V7 layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .grouped_k2 import grouped_k2_stats, grouped_packed_projection

EXPERTS = 256
PACKED_BYTES = 2_097_152
PROJECTIONS = ("w1", "w2", "w3")
PROJECTION_SHAPES = {
    # (output_features, input_features): packed is [tiles_k, tiles_m, 32].
    "w1": (4096, 2048),
    "w2": (2048, 4096),
    "w3": (4096, 2048),
}
_PLANE_COMPONENTS = ("SU", "SV")


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

    @property
    def codebooks(self) -> list[torch.Tensor]:
        return [self.plane_source.master]

    def promote_l028_su_sv(self) -> list[tuple[str, nn.Parameter]]:
        """Promote only L028 SU/SV wire views to named FP32 masters."""
        if self.L != 28:
            raise RuntimeError("expert-plane expansion is restricted to L028")
        if hasattr(self, "_expert_plane_rows"):
            return list(self._expert_plane_rows)
        rows: list[tuple[str, nn.Parameter]] = []
        for projection in PROJECTIONS:
            for component in _PLANE_COMPONENTS:
                buffer_name = f"{component.lower()}_{projection}"
                wire = getattr(self, buffer_name)
                if wire.dtype != torch.float16 or wire.ndim != 2 or wire.shape[0] != EXPERTS:
                    raise RuntimeError(f"L028 {projection}/{component} wire geometry drift")
                delattr(self, buffer_name)
                masters = nn.ParameterList(
                    [nn.Parameter(row.detach().float().clone()) for row in wire]
                )
                setattr(self, f"{component.lower()}_master_{projection}", masters)
                for expert, parameter in enumerate(masters):
                    rows.append(
                        (
                            f"model.layers.28.mlp.experts.E{expert:03d}.{projection}.{component}",
                            parameter,
                        )
                    )
        self._expert_plane_rows = rows
        if len(rows) != 1536 or sum(parameter.numel() for _name, parameter in rows) != 4_718_592:
            raise RuntimeError("L028 SU/SV promoted roster coverage drift")
        return list(rows)

    def expert_plane_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if not hasattr(self, "_expert_plane_rows"):
            raise RuntimeError("L028 SU/SV masters have not been promoted")
        return list(self._expert_plane_rows)

    def expert_plane_wire_view(self, projection: str, component: str) -> torch.Tensor:
        if projection not in PROJECTIONS or component not in _PLANE_COMPONENTS:
            raise RuntimeError("expert-plane wire view must be one of L028 SU/SV w1/w2/w3")
        masters = getattr(self, f"{component.lower()}_master_{projection}", None)
        if masters is None:
            return getattr(self, f"{component.lower()}_{projection}")
        return torch.stack(tuple(masters), dim=0).to(torch.float16)

    def expert_plane_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.expert_plane_parameters()
        }

    def load_expert_plane_state(self, state: Any) -> None:
        rows = self.expert_plane_parameters()
        if not isinstance(state, dict) or set(state) != {name for name, _parameter in rows}:
            raise RuntimeError("L028 SU/SV checkpoint roster drift")
        with torch.no_grad():
            for name, parameter in rows:
                value = state[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"L028 SU/SV checkpoint shape drift: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=torch.float32))

    def _projection_plane(self, projection: str, component: str) -> torch.Tensor:
        if hasattr(self, f"{component.lower()}_master_{projection}"):
            return self.expert_plane_wire_view(projection, component)
        return getattr(self, f"{component.lower()}_{projection}")

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

        gate = grouped_packed_projection(
            routed_hidden,
            expert_index,
            self.packed_w1,
            lut_master,
            self._projection_plane("w1", "SU"),
            self._projection_plane("w1", "SV"),
        ).clamp(max=self.limit)
        up = grouped_packed_projection(
            routed_hidden,
            expert_index,
            self.packed_w3,
            lut_master,
            self._projection_plane("w3", "SU"),
            self._projection_plane("w3", "SV"),
        ).clamp(min=-self.limit, max=self.limit)
        activated = self.act(gate) * up
        routed_output = grouped_packed_projection(
            activated,
            expert_index,
            self.packed_w2,
            lut_master,
            self._projection_plane("w2", "SU"),
            self._projection_plane("w2", "SV"),
        )
        routed_output = routed_output * route_weight
        final = torch.zeros_like(hidden_states, dtype=routed_output.dtype)
        final.index_add_(0, token_index, routed_output)
        self.cpu_relay_bytes += 0
        self.reconstruction_calls += 0
        return final.to(hidden_states.dtype)


__all__ = ["FullyResidentGroupedV7Experts"]
