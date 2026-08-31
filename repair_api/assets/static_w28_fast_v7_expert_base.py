"""Fully resident, grouped official-K2 routed experts for all V7 layers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
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
    grouped_route_metadata,
)

EXPERTS = 256
PACKED_BYTES = 2_097_152
PROJECTIONS = ("w1", "w2", "w3")
PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}

# The resident builder installs layers serially and each blocking H2D migration
# completes before the next layer starts. Reuse one pageable packed-wire arena
# across those layers; explicit multi-GiB host registration stalls the bounded
# cold load, while blocking H2D already provides the required staging boundary.
_PACKED_HOST_ARENA: torch.Tensor | None = None


def _shared_packed_host_arena(shape: tuple[int, int]) -> torch.Tensor:
    global _PACKED_HOST_ARENA
    if _PACKED_HOST_ARENA is None:
        _PACKED_HOST_ARENA = torch.empty(shape, dtype=torch.int16, pin_memory=False)
    elif tuple(_PACKED_HOST_ARENA.shape) != shape:
        raise RuntimeError("shared packed host arena geometry drift")
    return _PACKED_HOST_ARENA


def _managed_packed_allocation(
    shape: tuple[int, ...], *, device: torch.device
) -> tuple[int, Any, np.ndarray]:
    """Allocate one coherent managed wire and expose its CPU fill view."""
    elements = int(np.prod(shape, dtype=np.int64))
    nbytes = elements * 2
    cudart = ctypes.CDLL("/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.12")
    malloc_managed = cudart.cudaMallocManaged
    malloc_managed.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
    ]
    malloc_managed.restype = ctypes.c_int
    pointer = ctypes.c_void_p()
    result = malloc_managed(ctypes.byref(pointer), nbytes, 1)
    if result != 0 or not pointer.value:
        raise RuntimeError(f"cudaMallocManaged refused packed wire: {result}")
    class CudaMemLocation(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]
    prefetch = cudart.cudaMemPrefetchAsync
    prefetch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        CudaMemLocation,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    prefetch.restype = ctypes.c_int
    result = prefetch(pointer.value, nbytes, CudaMemLocation(2, 0), 0, None)
    if result != 0:
        raise RuntimeError(f"cudaMemPrefetchAsync refused CPU fill: {result}")
    torch.cuda.synchronize(device=device)
    owner = (ctypes.c_uint8 * nbytes).from_address(pointer.value)
    array = np.ctypeslib.as_array(owner).view("<i2").reshape(shape)
    return pointer.value, owner, array


def _managed_packed_tensor(
    pointer: int, owner: Any, shape: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    """Expose a CPU-populated managed wire as a CUDA tensor."""
    nbytes = int(np.prod(shape, dtype=np.int64)) * 2
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    metadata = {
        "size": shape,
        "stride": tuple(stride),
        "dtype": torch.int16,
        "device": device,
        "storage_offset": 0,
    }
    tensor = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        metadata, storage
    )
    setattr(tensor, "_managed_cpu_owner", owner)
    return tensor


def _load_projection_payloads_into(
    paths: list[Path], packed: np.ndarray, *, m: int, k: int,
    packed_bytes: int = PACKED_BYTES, pin_memory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Read one projection into an admitted managed view and pinned scales."""
    su_tensor = torch.empty((len(paths), k), dtype=torch.float16, pin_memory=pin_memory)
    sv_tensor = torch.empty((len(paths), m), dtype=torch.float16, pin_memory=pin_memory)
    su = su_tensor.numpy()
    sv = sv_tensor.numpy()
    expected = packed_bytes + (k + m) * 2 + 4
    read_bytes = 0
    # The verified cache is one local file per expert/projection. Serial 2-MiB
    # reads leave the NVMe queue at depth one and made the immutable 34-GiB
    # rank shard exceed the cold-start bound. Keep expert order deterministic,
    # but admit a bounded queue so only this local input mechanic changes.
    workers = min(16, len(paths))
    def read_one(item: tuple[int, Path]) -> int:
        expert, path = item
        trailer = bytearray(4)
        fd = os.open(path, os.O_RDONLY)
        try:
            count = os.preadv(
                fd,
                [
                    memoryview(packed[expert]).cast("B"),
                    memoryview(su[expert]).cast("B"),
                    memoryview(sv[expert]).cast("B"),
                    trailer,
                ],
                0,
            )
        finally:
            os.close(fd)
        # The final four bytes are authenticated opaque wire payload, not a
        # format magic.  Exact total geometry is the invariant.
        if count != expected:
            raise RuntimeError(f"wire byte geometry drift: {path}")
        return count

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="w28-wire-read") as pool:
        for count in pool.map(read_one, enumerate(paths)):
            read_bytes += count
    return su_tensor, sv_tensor, len(paths), read_bytes


def _load_projection_payloads(
    paths: list[Path], *, m: int, k: int, packed_bytes: int = PACKED_BYTES,
    pin_memory: bool = True, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Read one projection into coherent managed state for focused admission."""
    packed_shape = (len(paths), k // 16, m // 16, 32)
    packed_pointer, packed_owner, packed = _managed_packed_allocation(
        packed_shape, device=device,
    )
    su_tensor, sv_tensor, read_calls, read_bytes = _load_projection_payloads_into(
        paths, packed, m=m, k=k, packed_bytes=packed_bytes,
        pin_memory=pin_memory,
    )
    packed_tensor = _managed_packed_tensor(
        packed_pointer, packed_owner, packed_shape, device=device
    )
    return (
        packed_tensor,
        su_tensor,
        sv_tensor,
        read_calls,
        read_bytes,
    )


def _transfer_projection_payloads(
    packed: torch.Tensor,
    su_cpu: torch.Tensor,
    sv_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Migrate one exact host-filled projection into ordinary CUDA storage."""
    if packed.device.type != "cpu" or not packed.is_contiguous():
        raise RuntimeError("host-filled packed wire device/geometry drift")
    packed_cuda = packed.to(device=device)
    su = su_cpu.to(device=device)
    sv = sv_cpu.to(device=device)
    return packed_cuda, su, sv


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
        self.routed_return_reduction = "source_eager_expert_major_index_add"
        self.trace_enabled = os.environ.get("BR_TRACE_TIMING", "0") == "1"
        self._trace_events: list[tuple[str, Any, Any]] = []
        self._trace_forward_calls = 0
        self._trace_route_rows = 0
        self._trace_unique_experts = 0
        self.act = F.silu

        # Fill one bounded pinned host arena directly from the local verified
        # wires, then migrate each projection into ordinary CUDA storage.  This
        # avoids the pathological per-page managed-memory write fault observed
        # when preadv targeted a 1.5-GiB cudaMallocManaged arena.
        projection_elements = (
            EXPERTS * PROJECTION_SHAPES["w1"][0]
            * PROJECTION_SHAPES["w1"][1] * 32 // 256
        )
        arena_shape = (len(PROJECTIONS), projection_elements)
        arena_cpu_tensor = _shared_packed_host_arena(arena_shape)
        # Keep one pageable relay alive while each blocking H2D copy leaves
        # independent ordinary CUDA storage behind.
        self._packed_host_arena_owner = arena_cpu_tensor
        arena_cpu = arena_cpu_tensor.numpy()
        loaded = {}
        for projection_index, projection in enumerate(PROJECTIONS):
            m, k = PROJECTION_SHAPES[projection]
            packed_shape = (EXPERTS, k // 16, m // 16, 32)
            paths = [
                plane_source.member_path(expert, projection)
                for expert in range(EXPERTS)
            ]
            su_cpu, sv_cpu, read_calls, read_bytes = _load_projection_payloads_into(
                paths,
                arena_cpu[projection_index].reshape(packed_shape),
                m=m,
                k=k,
            )
            loaded[projection] = (
                projection_index, packed_shape, su_cpu, sv_cpu,
                read_calls, read_bytes,
            )
        arena = arena_cpu_tensor
        for projection in PROJECTIONS:
            (
                projection_index, packed_shape, su_cpu, sv_cpu,
                read_calls, read_bytes,
            ) = loaded[projection]
            packed_tensor = arena[projection_index].reshape(packed_shape)
            packed, su, sv = _transfer_projection_payloads(
                packed_tensor, su_cpu, sv_cpu, device=device,
            )
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
        route_rows_per_sample: int,
        route_metadata: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        if not self.trace_enabled:
            value = grouped_packed_projection(
                x, assignments, packed, lut_master, su, sv,
                route_rows_per_sample=route_rows_per_sample,
                route_metadata=route_metadata,
            )
            if os.environ.get("FAST_K2_SEALED_PROJECTION_BF16", "0") == "1":
                rounded = value.to(torch.bfloat16).float()
                tap_path = os.environ.get("PACKED_BOUNDARY_TAP_PATH")
                if tap_path and self.L == 0 and projection == "w1" and not getattr(self, "_packed_tap_done", False):
                    self._packed_tap_done = True
                    raw_cpu = value.detach().to(torch.bfloat16).cpu().contiguous()
                    rounded_cpu = rounded.detach().to(torch.bfloat16).cpu().contiguous()
                    delta = (value.detach().float() - rounded.detach().float()).abs()
                    row = {
                        "schema": "banana-smasher-packed-first-boundary-tap-v1",
                        "status": "PASS_FIRST_DIVERGENCE_L000_W1_PROJECTION_BF16_ROUND",
                        "layer": self.L,
                        "projection": projection,
                        "input_shape": list(x.shape),
                        "raw_bf16_sha256": hashlib.sha256(raw_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        "rounded_bf16_sha256": hashlib.sha256(rounded_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        "fp32_mismatch_count": int(torch.count_nonzero(delta).item()),
                        "fp32_max_abs_delta": float(delta.max().item()),
                        "raw_sample": value.detach().reshape(-1)[:8].float().cpu().tolist(),
                        "rounded_sample": rounded.detach().reshape(-1)[:8].float().cpu().tolist(),
                    }
                    path = Path(tap_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    raw = (json.dumps(row, indent=2, sort_keys=True) + "\n").encode()
                    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
                    with temporary.open("wb") as stream:
                        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, path)
                value = rounded
            return value
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        value = grouped_packed_projection(
            x, assignments, packed, lut_master, su, sv,
            route_rows_per_sample=route_rows_per_sample,
            route_metadata=route_metadata,
        )
        if os.environ.get("FAST_K2_SEALED_PROJECTION_BF16", "0") == "1":
            value = value.to(torch.bfloat16).float()
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
        """Preserve sealed group2 arithmetic while overlapping only expert work."""
        sealed_group_tokens = 2 * 2048
        if hidden_states.shape[0] > sealed_group_tokens:
            if hidden_states.shape[0] % sealed_group_tokens:
                raise RuntimeError("packed V7 batch must divide sealed group2 geometry")
            launch_stream = torch.cuda.current_stream(device=hidden_states.device)
            streams = [
                torch.cuda.Stream(device=hidden_states.device)
                for _ in range(hidden_states.shape[0] // sealed_group_tokens)
            ]
            outputs = []
            for group, stream in enumerate(streams):
                start = group * sealed_group_tokens
                stop = start + sealed_group_tokens
                stream.wait_stream(launch_stream)
                with torch.cuda.stream(stream):
                    outputs.append(self.forward(
                        hidden_states[start:stop],
                        top_k_index[start:stop],
                        top_k_weights[start:stop],
                    ))
            for stream in streams:
                launch_stream.wait_stream(stream)
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
        if hidden_states.shape[0] % 2048:
            raise RuntimeError(
                "packed V7 rows must retain the padded 2048-token window geometry"
            )
        route_rows_per_sample = 2048 * int(top_k_index.shape[1])
        lut_master = self.plane_source.wire_lut().reshape(-1).contiguous()
        route_metadata = grouped_route_metadata(
            expert_index, EXPERTS, input_tensor=routed_hidden
        )
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
            route_rows_per_sample,
            route_metadata,
        )
        up = self._project(
            "w3",
            routed_hidden,
            expert_index,
            self.packed_w3,
            lut_master,
            self.su_w3,
            self.sv_w3,
            route_rows_per_sample,
            route_metadata,
        )
        activated = self.act(gate) * up
        routed_output = self._project(
            "w2",
            activated,
            expert_index,
            self.packed_w2,
            lut_master,
            self.su_w2,
            self.sv_w2,
            route_rows_per_sample,
            route_metadata,
        )
        routed_output = (
            routed_output * route_weight
        ).to(hidden_states.dtype)
        # DeepseekV4Experts' source/eager forward iterates experts in ascending
        # order and performs one BF16 index_add_ per expert.  The resident
        # provider is installed in place of that decorated module, so changing
        # model.config._experts_implementation cannot select this reduction for
        # it.  Preserve the source dispatch here at the provider's real return
        # boundary instead of silently using the grouped token-local sum.
        final = torch.zeros_like(hidden_states)
        for expert_idx in torch.unique(expert_index, sorted=True):
            selected = expert_index == expert_idx
            final.index_add_(
                0,
                token_index[selected],
                routed_output[selected].to(final.dtype),
            )
        self.cpu_relay_bytes += 0
        self.reconstruction_calls += 0
        return final


__all__ = ["FullyResidentGroupedV7Experts"]
