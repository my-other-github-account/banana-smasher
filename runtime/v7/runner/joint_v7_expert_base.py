#!/usr/bin/env python3
"""Exact QTIP2 V7 compact-wire experts using official packed4 qtip_k2 decode.

Replaces in-graph Qtip2PhysicalLayer quadratic [512,2] TLUT + Wscale/FWHT
order with official 1D fp16[1024] MUL1 decode_k2_matrix + inverse_transform.
No extra Wscale. PlaneSource FP32 master is cast to fp16[1024] on every
projection forward so the LUT seam stays live.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qtip_v7_repair import _parse_v7_member

PUBLIC_SRC = Path(os.environ["BANANA_SMASHER_PUBLIC_SRC"]).resolve()
if str(PUBLIC_SRC) not in os.sys.path:
    os.sys.path.insert(0, str(PUBLIC_SRC))
from banana_smasher import qtip_k2 as official_k2  # noqa: E402

EXPERTS = 256
PROJECTIONS = ("w1", "w2", "w3")
_V7_PACKED_BYTES = 2_097_152
_V7_PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}


def official_parent_lut(source: Any) -> torch.Tensor:
    """1D fp16[1024] view of the live PlaneSource master. No [512,2] reshape."""
    lut = source.master.to(dtype=torch.float16).reshape(-1).contiguous()
    if tuple(lut.shape) != (1024,) or lut.dtype != torch.float16:
        raise RuntimeError(f"official parent LUT geometry drift {tuple(lut.shape)} {lut.dtype}")
    return lut


def packed_k2_from_member(path: Path, projection: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Parse compact V7 member as official packed int16[k/16, m/16, 32] + SU/SV."""
    m, k = _V7_PROJECTION_SHAPES[projection]
    payload = Path(path).read_bytes()
    expected = _V7_PACKED_BYTES + (k + m) * 2 + 4
    if len(payload) != expected:
        raise RuntimeError(f"QTIP V7 member byte geometry drift {path}: {len(payload)} != {expected}")
    packed = torch.from_numpy(
        np.frombuffer(payload[:_V7_PACKED_BYTES], dtype="<i2").copy().reshape(k // 16, m // 16, 32)
    ).to(device=device)
    su = torch.from_numpy(
        np.frombuffer(payload[_V7_PACKED_BYTES:_V7_PACKED_BYTES + k * 2], dtype="<f2").copy()
    ).to(device=device, dtype=torch.float32)
    sv = torch.from_numpy(
        np.frombuffer(
            payload[_V7_PACKED_BYTES + k * 2:_V7_PACKED_BYTES + (k + m) * 2],
            dtype="<f2",
        ).copy()
    ).to(device=device, dtype=torch.float32)
    return {"packed": packed, "su": su, "sv": sv, "m": m, "k": k}


class OfficialQtipK2PhysicalLayer(nn.Module):
    """Official packed4 qtip_k2.decode_k2_matrix + inverse_transform capsule."""

    def __init__(self, *, parent_lut: torch.Tensor, packed: torch.Tensor, su: torch.Tensor, sv: torch.Tensor) -> None:
        super().__init__()
        self.parent_lut = parent_lut
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("su", su, persistent=False)
        self.register_buffer("sv", sv, persistent=False)
        self.checkpoint_depth = False
        self.source_schema = "banana-smasher-qtip-v7-official-k2-v1"

    @property
    def codebooks(self) -> list[torch.Tensor]:
        return [self.parent_lut]

    def _weight(self) -> torch.Tensor:
        decoded = official_k2.decode_k2_matrix(self.packed, self.parent_lut)
        # official rail: inverse_transform(...).T.contiguous().to(bfloat16)
        return official_k2.inverse_transform(decoded, self.su, self.sv).T.contiguous().to(torch.bfloat16)

    def _forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise ValueError("official QTIP2 layer expects [1, tokens, K]")
        weight = self._weight()
        return torch.matmul(hidden.to(torch.bfloat16), weight.transpose(0, 1)).float()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.checkpoint_depth:
            return self._forward(hidden)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._forward, hidden, use_reentrant=False)


class JointV7ExpertBase(nn.Module):
    """Drop-in TrainableExperts using official qtip_k2 physical decoding."""

    def __init__(self, layer: int, pilot: bool = True, *, plane_source: Any) -> None:
        super().__init__()
        self.L = int(layer)
        self.pilot = bool(pilot)
        if not self.pilot or int(plane_source.layer) != self.L:
            raise RuntimeError(f"invalid admitted PlaneSource for L{self.L:03d}")
        self.__dict__["plane_source"] = plane_source
        self.act = F.silu

    def _projection(self, expert: int, projection: str) -> OfficialQtipK2PhysicalLayer:
        source = self.plane_source
        path = source.member_path(expert, projection)
        source.wire_lut()  # accounting only; official numerical path remains source.master
        device = source.master.device
        parsed = packed_k2_from_member(path, projection, device)
        # Keep parse_v7_member as a geometry check against the admitted contract.
        admitted = _parse_v7_member(path, projection=projection)
        if list(map(int, admitted["shape"])) != [parsed["m"], parsed["k"]]:
            raise RuntimeError(f"L{self.L:03d} admitted vs official packed shape drift")
        return OfficialQtipK2PhysicalLayer(
            parent_lut=official_parent_lut(source),
            packed=parsed["packed"],
            su=parsed["su"],
            sv=parsed["sv"],
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=EXPERTS).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert in hit:
            top_k_pos, token_idx = torch.where(mask[expert])
            hidden = hidden_states[token_idx]
            w1 = self._projection(expert, "w1")
            w3 = self._projection(expert, "w3")
            gate = w1(hidden.unsqueeze(0)).squeeze(0)
            up = w3(hidden.unsqueeze(0)).squeeze(0)
            activated = self.act(gate) * up
            del gate, up, w1, w3
            w2 = self._projection(expert, "w2")
            current = w2(activated.unsqueeze(0)).squeeze(0)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
            del activated, current, w2
        return final


__all__ = ["JointV7ExpertBase", "OfficialQtipK2PhysicalLayer", "official_parent_lut"]
