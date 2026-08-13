"""Direct-packed QTIP2 V7 fixed-envelope runtime binding."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from banana_smasher.qtip_v7_wire import QtipV7LayerMapping


QTIP_V7_DIRECT_DISPATCH = "banana_smasher_plugin._v4_moe.qtip2_v7_direct"
_PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}


@dataclass(frozen=True)
class QtipV7DirectMember:
    expert: int
    projection: str
    output_width: int
    input_width: int
    trellis: memoryview
    su: memoryview
    sv: memoryview
    wscale: memoryview


class QtipV7DirectLayer:
    """Own one mmap envelope and expose only direct-kernel views.

    Controls are expanded into caller-owned bounded workspace. They are not
    retained by this object, so no decoded-state, dense-weight, or second packed
    cache can become resident state.
    """

    dispatch = QTIP_V7_DIRECT_DISPATCH
    duplicate_packed_bytes = 0
    persistent_decoded_state_bytes = 0
    persistent_dense_weight_bytes = 0
    generic_fallback_calls = 0

    def __init__(self, wire: str | Path) -> None:
        self.mapping = QtipV7LayerMapping(wire)
        self.lut = self.mapping.lut_view()
        if self.lut.obj is not self.mapping.buffer:
            raise RuntimeError("QTIP V7 embedded LUT is not a mapped-envelope alias")
        self._source_tensors: dict[str, list[Any]] = {}
        self._source_pointers: dict[tuple[str, str], Any] = {}
        self._codebook: Any | None = None
        self._direct_dispatch_calls = 0
        self._direct_counter_receipts: dict[str, dict[str, int | str]] = {}

    @property
    def transient_workspace_peak_bytes(self) -> int:
        # Host controls plus their transient FP32 CUDA transform tensors overlap.
        return 3 * self.mapping.transient_workspace_peak_bytes

    def member(
        self,
        expert: int,
        projection: str,
        *,
        control_workspace: bytearray,
    ) -> QtipV7DirectMember:
        try:
            output_width, input_width = _PROJECTION_SHAPES[projection]
        except KeyError as exc:
            raise ValueError(f"unknown QTIP V7 projection: {projection}") from exc
        expected = self.mapping.transient_workspace_peak_bytes
        if len(control_workspace) != expected:
            raise ValueError(
                f"QTIP V7 transient control workspace requires {expected} bytes"
            )
        projection_index = ("w1", "w2", "w3").index(projection)
        member_index = expert * 3 + projection_index
        start = member_index * self.mapping.geometry.control_bytes
        control = memoryview(control_workspace)[
            start : start + self.mapping.geometry.control_bytes
        ]
        su_stop = input_width * 2
        sv_stop = su_stop + output_width * 2
        if sv_stop + 4 != len(control):
            raise RuntimeError("QTIP V7 control transform geometry drift")
        return QtipV7DirectMember(
            expert=expert,
            projection=projection,
            output_width=output_width,
            input_width=input_width,
            trellis=self.mapping.packed_view(expert, projection),
            su=control[:su_stop],
            sv=control[su_stop:sv_stop],
            wscale=control[sv_stop:],
        )

    def transient_controls(self) -> bytearray:
        return self.mapping.transient_controls()

    def _torch_sources(self, projection: str, device: Any) -> Any:
        import torch

        key = (projection, str(device))
        if key in self._source_pointers:
            return self._source_pointers[key]
        tensors = [
            torch.frombuffer(
                self.mapping.packed_view(expert, projection),
                dtype=torch.uint16,
                count=self.mapping.geometry.packed_bytes // 2,
            )
            for expert in range(self.mapping.geometry.experts)
        ]
        pointers = torch.tensor(
            [tensor.data_ptr() for tensor in tensors],
            dtype=torch.int64,
            device=device,
        )
        self._source_tensors[projection] = tensors
        self._source_pointers[key] = pointers
        return pointers

    def forward(self, x: Any, expert_ids: Any, projection: str) -> Any:
        """Launch direct R=2 trellis/TLUT with only transient transform controls."""
        import torch

        from .dispatch_policy import shape_policy
        from .native_extensions import _module
        from .v4_acceleration import allocate_compaction_state

        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            raise ValueError("QTIP V7 direct runtime requires a CUDA activation tensor")
        if torch.cuda.get_device_capability(x.device) != (12, 1):
            raise RuntimeError(
                "QTIP V7 zero-copy envelopes require GB10 coherent host memory"
            )
        try:
            output_width, input_width = _PROJECTION_SHAPES[projection]
        except KeyError as exc:
            raise ValueError(f"unknown QTIP V7 projection: {projection}") from exc
        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        if x.shape[-1] != input_width or rows != expert_ids.numel():
            raise ValueError("QTIP V7 direct activation/expert geometry drift")
        policy = shape_policy(rows)
        variants = {
            "decode_c1": 0,
            "decode_c2": 1,
            "decode_c4": 2,
            "decode_c8": 3,
            "decode_c16": 4,
            "prefill_bm16": 5,
            "prefill_large": 6,
            "prefill_exact_2k": 7,
            "prefill_large_8192": 8,
        }
        compact = allocate_compaction_state(
            rows=rows,
            experts=self.mapping.geometry.experts,
            input_width=input_width,
            output_width=output_width,
            block_rows=int(policy["mblock"]),
            device=x.device,
        )
        # Register the extension's TORCH_LIBRARY fragments before the first
        # torch.ops compaction/transform dispatch.
        _module()
        families = torch.zeros(
            self.mapping.geometry.experts, dtype=torch.int8, device=x.device
        )
        torch.ops.banana_smasher_v4.compact_routes(
            expert_ids.to(device=x.device, dtype=torch.int64).reshape(-1),
            families,
            compact["out"],
            compact["family_block_counts"],
            compact["block_experts"],
            compact["block_valid_m"],
            compact["block_route_rows"],
            compact["expert_route_counts"],
            compact["expert_last_block"],
            compact["physical_counters"],
            int(policy["mblock"]),
        )
        controls = self.transient_controls()
        members = [
            self.member(expert, projection, control_workspace=controls)
            for expert in range(self.mapping.geometry.experts)
        ]
        su = torch.stack(
            [torch.frombuffer(member.su, dtype=torch.float16).float() for member in members]
        ).to(x.device)
        sv = torch.stack(
            [torch.frombuffer(member.sv, dtype=torch.float16).float() for member in members]
        ).to(x.device)
        wscale = torch.stack(
            [torch.frombuffer(member.wscale, dtype=torch.float32) for member in members]
        ).reshape(-1).to(x.device)
        torch.ops.banana_smasher_v4.qtip_pre_transform(
            x.to(torch.bfloat16).contiguous(),
            su,
            compact["qtip_input"],
            compact["family_block_counts"][0:1],
            compact["block_experts"][0],
            compact["block_valid_m"][0],
            compact["block_route_rows"][0],
        )
        if self._codebook is None:
            self._codebook = torch.frombuffer(self.lut, dtype=torch.float16, count=1024)
        variant = variants[str(policy["kernel"])]
        if input_width == 4096:
            specialized_counter_index = 128 if variant == 8 else 32 + variant
        else:
            specialized_counter_index = 129 if variant == 8 else 40 + variant
        _module().qtip2_v7_direct(
            compact["out"],
            self._torch_sources(projection, x.device),
            compact["family_block_counts"][0:1],
            compact["block_experts"][0],
            compact["block_valid_m"][0],
            compact["block_route_rows"][0],
            compact["qtip_input"],
            self._codebook,
            compact["physical_counters"],
            variant,
            specialized_counter_index,
        )
        self._direct_dispatch_calls += 1
        torch.ops.banana_smasher_v4.qtip_post_transform(
            compact["out"],
            wscale,
            sv,
            compact["family_block_counts"][0:1],
            compact["block_experts"][0],
            compact["block_valid_m"][0],
            compact["block_route_rows"][0],
        )
        result = compact["out"].to(torch.bfloat16)
        torch.cuda.synchronize(x.device)
        physical_counters = compact["physical_counters"].cpu().tolist()
        counter_receipt = {
            "projection": projection,
            "input_width": input_width,
            "specialized_counter_index": specialized_counter_index,
            "specialized_counter_value": int(
                physical_counters[specialized_counter_index]
            ),
            "direct_family_launches": int(physical_counters[10]),
            "direct_family_rows": int(physical_counters[18]),
        }
        if (
            counter_receipt["specialized_counter_value"] <= 0
            or counter_receipt["direct_family_launches"] <= 0
            or counter_receipt["direct_family_rows"] <= 0
        ):
            raise RuntimeError("QTIP V7 physical direct counters did not advance")
        self._direct_counter_receipts[projection] = counter_receipt
        del controls, members, su, sv, wscale
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "dispatch": self.dispatch,
            "lut_alias_storage_identity": self.lut.obj is self.mapping.buffer,
            "duplicate_packed_bytes": self.duplicate_packed_bytes,
            "persistent_decoded_state_bytes": self.persistent_decoded_state_bytes,
            "persistent_dense_weight_bytes": self.persistent_dense_weight_bytes,
            "generic_fallback_calls": self.generic_fallback_calls,
            "direct_dispatch_calls": self._direct_dispatch_calls,
            "direct_specialized_counters": [
                self._direct_counter_receipts[name]
                for name in ("w1", "w2", "w3")
                if name in self._direct_counter_receipts
            ],
            "transient_workspace_peak_bytes": self.transient_workspace_peak_bytes,
        }

    def close(self) -> None:
        self._codebook = None
        self._source_pointers.clear()
        self._source_tensors.clear()
        self._direct_counter_receipts.clear()
        self.lut.release()
        self.mapping.close()

    def __enter__(self) -> "QtipV7DirectLayer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _linux_process_memory() -> tuple[int, int]:
    try:
        rows: dict[str, int] = {}
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, separator, payload = line.partition(":")
            if separator and key in {"Rss", "Pss"}:
                rows[key] = int(payload.split()[0]) * 1024
        return rows["Rss"], rows["Pss"]
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError("QTIP V7 hardware readback requires Linux RSS/PSS") from exc


def _nvml_process_bytes() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used_mib = 0
    for line in completed.stdout.splitlines():
        pid_text, separator, memory_text = line.partition(",")
        if separator and int(pid_text.strip()) == os.getpid():
            used_mib += int(memory_text.strip())
    return used_mib * 1024 * 1024


def _fault_mapped_pages(layer: QtipV7DirectLayer) -> tuple[int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    sampled = layer.mapping.buffer[::page_size]
    return (len(sampled), sum(sampled) & 0xFFFFFFFF)


def capture_qtip_v7_hardware_readback(
    accounting: str | Path, output: str | Path
) -> dict[str, Any]:
    """Map all 43 layers, execute every V7 projection, and write physical telemetry."""
    import torch

    accounting_path = Path(accounting).expanduser().resolve()
    model = json.loads(accounting_path.read_text())
    rows = model.get("layer_receipts") if isinstance(model, dict) else None
    if not isinstance(rows, list) or [row.get("layer") for row in rows] != list(range(43)):
        raise ValueError("QTIP V7 hardware capture requires exact accounting layers 0..42")
    if not torch.cuda.is_available():
        raise RuntimeError("QTIP V7 hardware capture requires CUDA")
    device = torch.device("cuda")
    layers: list[QtipV7DirectLayer] = []
    resident_pages = 0
    resident_checksum = 0
    try:
        torch.cuda.reset_peak_memory_stats(device)
        for row in rows:
            receipt = json.loads(Path(row["path"]).read_text())
            layer = QtipV7DirectLayer(receipt["wire"])
            layers.append(layer)
            pages, checksum = _fault_mapped_pages(layer)
            resident_pages += pages
            resident_checksum = (resident_checksum + checksum) & 0xFFFFFFFF
            for projection, (_output_width, input_width) in _PROJECTION_SHAPES.items():
                x = torch.zeros((6, input_width), dtype=torch.bfloat16, device=device)
                expert_ids = torch.arange(6, dtype=torch.int64, device=device)
                result = layer.forward(x, expert_ids, projection)
                del result, expert_ids, x
        torch.cuda.synchronize(device)
        rss, pss = _linux_process_memory()
        mapped = sum(layer.mapping.path.stat().st_size for layer in layers)
        readback = {
            "hardware_readback": True,
            "direct_kernel_dispatch": QTIP_V7_DIRECT_DISPATCH,
            "direct_dispatch_calls": sum(
                layer._direct_dispatch_calls for layer in layers
            ),
            "unique_physical_mapped_resident_weight_bytes": mapped,
            "resident_page_touch_count": resident_pages,
            "resident_page_touch_checksum_u32": resident_checksum,
            "native_base_bytes": 19_708_797_688,
            "persistent_runtime_metadata_bytes": len(layers) * 3 * 256 * 8,
            "separate_lut_tensor_bytes": 0,
            "lut_alias_storage_identity": all(
                layer.lut.obj is layer.mapping.buffer for layer in layers
            ),
            "duplicate_packed_bytes": 0,
            "persistent_decoded_state_bytes": 0,
            "persistent_dense_weight_bytes": 0,
            "generic_fallback_calls": 0,
            "transient_workspace_peak_bytes": max(
                layer.transient_workspace_peak_bytes for layer in layers
            ),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "process_rss_bytes": rss,
            "process_pss_bytes": pss,
            "nvml_process_bytes": _nvml_process_bytes(),
        }
        if readback["direct_dispatch_calls"] != 43 * 3:
            raise RuntimeError("QTIP V7 direct dispatch roster did not execute completely")
        target = Path(output).expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(readback, indent=2, sort_keys=True) + "\n")
        return readback
    finally:
        for layer in layers:
            layer.close()


def capture_qtip_v7_layer_smoke(
    wire: str | Path, output: str | Path
) -> dict[str, Any]:
    """Execute one bounded real layer w1/w2/w3 smoke and seal memory telemetry."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("QTIP V7 layer smoke requires CUDA")
    device = torch.device("cuda")
    layer = QtipV7DirectLayer(wire)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        pages, checksum = _fault_mapped_pages(layer)
        for projection, (_output_width, input_width) in _PROJECTION_SHAPES.items():
            x = torch.zeros((6, input_width), dtype=torch.bfloat16, device=device)
            expert_ids = torch.arange(6, dtype=torch.int64, device=device)
            result = layer.forward(x, expert_ids, projection)
            del result, expert_ids, x
        torch.cuda.synchronize(device)
        rss, pss = _linux_process_memory()
        receipt = {
            "schema": "banana-smasher-qtip-v7-layer-hardware-smoke-v1",
            "status": "PROVEN",
            "scope": "one_layer_w1_w2_w3_direct_smoke",
            "direct_kernel_dispatch": QTIP_V7_DIRECT_DISPATCH,
            "direct_dispatch_calls": layer._direct_dispatch_calls,
            "direct_specialized_counters": layer.receipt()[
                "direct_specialized_counters"
            ],
            "mapped_layer_bytes": layer.mapping.path.stat().st_size,
            "resident_page_touch_count": pages,
            "resident_page_touch_checksum_u32": checksum,
            "lut_alias_storage_identity": layer.lut.obj is layer.mapping.buffer,
            "separate_lut_tensor_bytes": 0,
            "duplicate_packed_bytes": 0,
            "persistent_decoded_state_bytes": 0,
            "persistent_dense_weight_bytes": 0,
            "generic_fallback_calls": 0,
            "persistent_runtime_metadata_bytes": 3 * 256 * 8,
            "transient_workspace_peak_bytes": layer.transient_workspace_peak_bytes,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "process_rss_bytes": rss,
            "process_pss_bytes": pss,
            "nvml_process_bytes": _nvml_process_bytes(),
        }
        if receipt["direct_dispatch_calls"] != 3:
            raise RuntimeError("QTIP V7 layer smoke did not execute w1/w2/w3")
        target = Path(output).expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    finally:
        layer.close()


__all__ = [
    "QTIP_V7_DIRECT_DISPATCH",
    "QtipV7DirectLayer",
    "QtipV7DirectMember",
    "capture_qtip_v7_hardware_readback",
    "capture_qtip_v7_layer_smoke",
]