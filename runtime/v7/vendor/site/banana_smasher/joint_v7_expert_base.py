"""Persistent exact QTIP2 V7 compact-wire experts for joint repair.

Immutable member payloads are parsed and decoded once per process and retained
on their target device. Only ``PlaneSource.wire_lut()`` is differentiable.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from banana_smasher.fwht import fwht_stats
from banana_smasher.qtip_v7_repair import _parse_v7_member
from banana_smasher.update_backends.physical_bundle import (
    Qtip2PhysicalLayer,
    _FrozenQtip2Payload,
    _qtip2_decode_states,
)

EXPERTS = 256
PROJECTIONS = ("w1", "w2", "w3")


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


class _BoundedFrozenPayloadCache:
    """Process-wide pinned hit-set cache bounded by exact decoded bytes."""

    def __init__(self) -> None:
        self.max_bytes = int(os.environ.get("JOINT_V7_FROZEN_CACHE_BYTES", str(2 << 30)))
        if self.max_bytes < 0:
            raise RuntimeError("JOINT_V7_FROZEN_CACHE_BYTES must be non-negative")
        self._items: OrderedDict[
            tuple[str, int, int, int, int, str, str],
            tuple[_FrozenQtip2Payload, int],
        ] = OrderedDict()
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.capacity_skips = 0
        self.parse_calls = 0
        self.decode_calls = 0
        self.materialization_calls = 0
        self.payload_bytes_read = 0

    def stats(self) -> dict[str, int]:
        return {
            "max_bytes": self.max_bytes,
            "resident_bytes": self.resident_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "entries": len(self._items),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "capacity_skips": self.capacity_skips,
            "parse_calls": self.parse_calls,
            "decode_calls": self.decode_calls,
            "materialization_calls": self.materialization_calls,
            "payload_bytes_read": self.payload_bytes_read,
        }

    def get_or_load(
        self,
        path: Path,
        *,
        projection: str,
        device: torch.device,
    ) -> _FrozenQtip2Payload:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        key = (
            str(resolved),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            projection,
            str(device),
        )
        cached = self._items.get(key)
        if cached is not None:
            self.hits += 1
            return cached[0]

        self.misses += 1
        self.parse_calls += 1
        self.payload_bytes_read += int(stat.st_size)
        payload = _parse_v7_member(resolved, projection=projection)
        m, k = map(int, payload["shape"])
        trellis = payload["trellis"].to(
            device=device, dtype=torch.uint16
        ).contiguous()
        self.decode_calls += 1
        states = _qtip2_decode_states(trellis, m=m, k=k)
        frozen = _FrozenQtip2Payload(
            {
                "states": states,
                "su": payload["SU"].to(device=device, dtype=torch.float32),
                "sv": payload["SV"].to(device=device, dtype=torch.float32),
                "wscale": payload["Wscale"].to(device=device, dtype=torch.float32),
            }
        )
        self.materialization_calls += 1
        item_bytes = sum(_tensor_bytes(value) for value in frozen.buffers())
        if item_bytes > self.max_bytes:
            self.capacity_skips += 1
            return frozen
        while self._items and self.resident_bytes + item_bytes > self.max_bytes:
            _old_key, (_old_payload, old_bytes) = self._items.popitem(last=False)
            self.resident_bytes -= old_bytes
            self.evictions += 1
        self._items[key] = (frozen, item_bytes)
        self.resident_bytes += item_bytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return frozen


_FROZEN_PAYLOAD_CACHE = _BoundedFrozenPayloadCache()


def frozen_payload_cache_stats() -> dict[str, int]:
    """Return exact engagement and residency counters for durable receipts."""
    return _FROZEN_PAYLOAD_CACHE.stats()


def joint_v7_fast_path_stats() -> dict[str, Any]:
    """Return cache and FWHT counters used to prove production engagement."""
    return {
        "frozen_payload_cache": frozen_payload_cache_stats(),
        "fwht": fwht_stats(),
    }


class JointV7ExpertBase(nn.Module):
    """Drop-in TrainableExperts using persistent public QTIP2 decoding."""

    def __init__(self, layer: int, pilot: bool = True, *, plane_source: Any) -> None:
        super().__init__()
        self.L = int(layer)
        self.pilot = bool(pilot)
        if not self.pilot or int(plane_source.layer) != self.L:
            raise RuntimeError(f"invalid admitted PlaneSource for L{self.L:03d}")
        self.__dict__["plane_source"] = plane_source
        self.fwht_backend = os.environ.get("JOINT_V7_FWHT_BACKEND", "quack")
        if self.fwht_backend != "quack":
            raise RuntimeError(
                "joint V7 production repair requires fused Quack FWHT; "
                f"got {self.fwht_backend!r}"
            )
        self.limit = 10.0
        self.act = F.silu

    def preload_frozen_payloads(
        self, experts: list[int] | tuple[int, ...]
    ) -> dict[str, int]:
        """Warm one exact routed hit set before its objective traversal."""
        source = self.plane_source
        device = source.master.device
        for expert in experts:
            expert = int(expert)
            if not 0 <= expert < EXPERTS:
                raise RuntimeError(f"invalid V7 expert index {expert}")
            for projection in PROJECTIONS:
                _FROZEN_PAYLOAD_CACHE.get_or_load(
                    source.member_path(expert, projection),
                    projection=projection,
                    device=device,
                )
        return frozen_payload_cache_stats()

    def _projection(self, expert: int, projection: str) -> Qtip2PhysicalLayer:
        source = self.plane_source
        frozen = _FROZEN_PAYLOAD_CACHE.get_or_load(
            source.member_path(expert, projection),
            projection=projection,
            device=source.master.device,
        )
        tlut = source.wire_lut()
        if tuple(tlut.shape) != (512, 2) or tlut.dtype != torch.float32:
            raise RuntimeError(f"L{self.L:03d} FP16-wire TLUT geometry drift")
        return Qtip2PhysicalLayer(
            tlut=tlut,
            frozen=frozen,
            source_schema="banana-smasher-qtip-v7-public-unit-v1",
            fwht_backend=self.fwht_backend,
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=EXPERTS).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        self.preload_frozen_payloads(hit)
        for expert in hit:
            top_k_pos, token_idx = torch.where(mask[expert])
            hidden = hidden_states[token_idx]
            w1 = self._projection(expert, "w1")
            w3 = self._projection(expert, "w3")
            gate = w1(hidden.unsqueeze(0)).squeeze(0).clamp(max=self.limit)
            up = w3(hidden.unsqueeze(0)).squeeze(0).clamp(
                min=-self.limit, max=self.limit
            )
            activated = self.act(gate) * up
            del gate, up, w1, w3
            w2 = self._projection(expert, "w2")
            current = w2(activated.unsqueeze(0)).squeeze(0)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
            del activated, current, w2
        return final


__all__ = [
    "JointV7ExpertBase",
    "frozen_payload_cache_stats",
    "joint_v7_fast_path_stats",
]
