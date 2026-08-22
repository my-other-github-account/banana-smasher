from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .d4_wire import decode_d4_expert
from .hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime
from .loader import PackLoader
from .qtip25_native_v4 import decode_native_v4_torch, native_v4_geometry
from .qtip_v7_routes import _load_qtip2_v7_member_roster, load_qtip2_v7_wire


def _available_materialization_bytes(
    torch: Any,
    device: str,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> int:
    """Use reclaimable host memory for CUDA unified-memory GB10 devices."""

    free, _ = torch.cuda.mem_get_info()
    if "GB10" not in torch.cuda.get_device_properties(device).name:
        return int(free)
    try:
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return max(int(free), int(line.split()[1]) * 1024)
    except OSError:
        pass
    return int(free)


def _decode_mxfp4_e2m1(torch: Any, packed: Any, scale: Any) -> Any:
    """Decode native packed MXFP4/E8M0 weights into their logical matrix."""

    raw = packed.view(torch.uint8)
    indices = torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(raw.shape[0], -1)
    lut = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
        device=packed.device,
    )
    values = lut[indices.long()]
    scales = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
    scales = scales.repeat_interleave(32, dim=1)[:, : values.shape[1]]
    if scales.shape != values.shape:
        raise ValueError(
            f"native MXFP4/E8M0 geometry mismatch: values={tuple(values.shape)} "
            f"scales={tuple(scales.shape)}"
        )
    return (values * scales).to(torch.bfloat16)


def _decode_compressed(
    torch: Any,
    L: int,
    S: int,
    R: int,
    V: int,
    m: int,
    k: int,
    compressed: Any,
    expanded_lut: Any,
) -> Any:
    """Public QTIP bitshift decode, matching Cornell-RelaxML/qtip."""

    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)
    if compressed.shape != (R * m * k // 16,):
        raise ValueError("QTIP compressed tensor shape mismatch")
    block_size = 16 * 16
    bits_per_block = R * block_size
    compressed = (
        compressed.view(torch.uint8)
        .reshape(m // 32, k // 32, block_size // 8, 2, 2, R)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, bits_per_block // 16, 2)
        .flip((-1,))
        .view(torch.uint16)
        .reshape(m // 16, k // 16, bits_per_block // 16)
    )
    blocked = compressed.reshape(R * m * k // bits_per_block, bits_per_block // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = (
        torch.cat((blocked_roll, blocked), dim=-1)
        .reshape(blocked.shape[0], -1)
        .contiguous()
        .view(torch.uint32)
    )
    expanded32 = (
        blocked32.reshape(*blocked32.shape, 1)
        .expand(*blocked32.shape, 16)
        .view(torch.int32)
    )
    shifts = torch.arange(0, 16, dtype=torch.int32, device=blocked.device).reshape(
        1, 1, -1
    )
    shifts = shifts.expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - L :: R << V],
        (1 << L) - 1,
    )
    mma_swizzled = expanded_lut[indices]
    return (
        mma_swizzled.reshape(m // 16, k // 16, 16, 16)
        .reshape(m // 16, k // 16, 8, 4, 2, 2, 2)
        .permute(0, -2, 2, 1, -3, 3, -1)
        .reshape(m, k)
    )


def _fwht(torch: Any, value: Any) -> Any:
    n = value.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT requires power-of-two last dimension, got {n}")
    if value.is_cuda:
        hadamard_transform = importlib.import_module(
            "quack.hadamard"
        ).hadamard_transform
        return hadamard_transform(value.contiguous(), scale=1 / math.sqrt(n))
    result = value.contiguous()
    width = 1
    while width < n:
        pair = result.reshape(*result.shape[:-1], n // (2 * width), 2, width)
        left, right = pair[..., 0, :], pair[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            *result.shape[:-1], n
        )
        width *= 2
    return result / (n**0.5)


class DeepseekV4BackpackRuntime(DeepseekV4D4Runtime):
    """Layerwise DeepSeek-V4 runtime for a virtual mixed Backpack assignment."""

    API_VERSION = 1

    def __init__(self, *, model_root: str | Path, parameters: dict[str, Any]) -> None:
        super().__init__(model_root=model_root, parameters=parameters)
        binding = parameters.get("backpack_runtime")
        required = {
            "basis_sha256",
            "virtual_manifest",
            "materialization_index",
        }
        qtip_groups = {
            "qtip2": {"qtip2_root_map"},
            "qtip3": {"qtip3_root_map"},
        }
        v7_group = {
            "qtip2_v7_root_map",
            "qtip2_v7_shared_lut",
            "qtip2_v7_member_roster",
        }
        allowed = required | v7_group | set().union(*qtip_groups.values())
        if (
            not isinstance(binding, Mapping)
            or not required.issubset(binding)
            or set(binding) - allowed
        ):
            raise ValueError(
                f"backpack_runtime requires {sorted(required)} and only declared source groups"
            )
        self.basis_sha256 = str(binding["basis_sha256"])
        self.virtual_manifest_path = Path(str(binding["virtual_manifest"])).resolve()
        self.materialization_index_path = Path(
            str(binding["materialization_index"])
        ).resolve()
        manifest = json.loads(self.virtual_manifest_path.read_text())
        if manifest.get("basis_sha256") != self.basis_sha256:
            raise ValueError("virtual Backpack basis mismatch")
        index_binding = manifest.get("materialization_index")
        if not isinstance(index_binding, Mapping):
            raise ValueError(
                "virtual Backpack materialization index binding is missing"
            )
        index_sha = hashlib.sha256(
            self.materialization_index_path.read_bytes()
        ).hexdigest()
        if index_sha != index_binding.get("sha256"):
            raise ValueError("virtual Backpack materialization index SHA-256 mismatch")
        self.rows_by_layer: dict[int, list[dict[str, Any]]] = {}
        for line in self.materialization_index_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            layer = int(row["layer"])
            self.rows_by_layer.setdefault(layer, []).append(row)
        if set(self.rows_by_layer) != set(range(43)) or any(
            len(rows) != 512 for rows in self.rows_by_layer.values()
        ):
            raise ValueError("virtual Backpack index must cover 43x256x2 cells")
        selected_source_keys = {
            str(row["source_key"])
            for rows in self.rows_by_layer.values()
            for row in rows
        }
        for source_key, fields in qtip_groups.items():
            if (source_key in selected_source_keys) != fields.issubset(binding):
                raise ValueError(
                    f"backpack_runtime {source_key} binding/selection mismatch"
                )
        if ("qtip2_v7" in selected_source_keys) != v7_group.issubset(binding):
            raise ValueError("backpack_runtime qtip2_v7 binding/selection mismatch")
        source_bindings = manifest.get("source_bindings")
        if not isinstance(source_bindings, Mapping):
            raise ValueError("virtual Backpack source bindings are missing")
        self.d4_loaders: dict[str, PackLoader] = {}
        for source_key in sorted(
            selected_source_keys.intersection({"d4_k2048", "d4_k4096"})
        ):
            source = source_bindings.get(source_key)
            if (
                not isinstance(source, Mapping)
                or source.get("basis_sha256") != self.basis_sha256
            ):
                raise ValueError(f"{source_key} source binding identity mismatch")
            root = Path(str(source.get("root", ""))).resolve()
            relative = Path(str(source.get("identity", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{source_key} source identity is unsafe")
            identity = root / relative
            if not identity.is_file() or hashlib.sha256(
                identity.read_bytes()
            ).hexdigest() != source.get("identity_sha256"):
                raise ValueError(f"{source_key} source identity drift")
            # The immutable source binding above names the already-sealed pack
            # manifest.  Do not re-hash a ~100 GB pack at exact64 startup.
            loader = PackLoader(root, verify=False)
            expected_prefix = f"layers.0.truevq_d4.{source_key}."
            if not any(
                name.startswith(expected_prefix) for name in loader.tensor_index
            ):
                raise ValueError(f"{source_key} pack lacks its declared D4 tensors")
            self.d4_loaders[source_key] = loader
            self._record_path(identity)
        self.root_maps: dict[str, dict[str, str]] = {}
        for source_key in ("qtip2", "qtip3"):
            if source_key not in selected_source_keys:
                continue
            path = Path(str(binding[f"{source_key}_root_map"])).resolve()
            root_map = json.loads(path.read_text())
            if (
                root_map.get("status") != "PASS"
                or root_map.get("basis_sha256") != self.basis_sha256
                or root_map.get("tier") != source_key
            ):
                raise ValueError(f"{source_key} root-map identity mismatch")
            layer_roots = root_map.get("layer_roots")
            if not isinstance(layer_roots, Mapping) or set(layer_roots) != {
                str(layer) for layer in range(43)
            }:
                raise ValueError(f"{source_key} root-map layer coverage mismatch")
            self.root_maps[source_key] = {
                str(layer): str(root) for layer, root in layer_roots.items()
            }
            self._record_path(path)
        self.qtip2_v7_shared_lut_path: Path | None = None
        self.qtip2_v7_roster_members: dict[tuple[int, int, str], tuple[Path, str]] = {}
        if v7_group.issubset(binding):
            path = Path(str(binding["qtip2_v7_root_map"])).resolve()
            root_map = json.loads(path.read_text())
            if (
                root_map.get("status") != "PASS"
                or root_map.get("basis_sha256") != self.basis_sha256
                or root_map.get("tier") != "qtip2_v7"
            ):
                raise ValueError("qtip2_v7 root-map identity mismatch")
            layer_roots = root_map.get("layer_roots")
            if not isinstance(layer_roots, Mapping) or set(layer_roots) != {
                str(layer) for layer in range(43)
            }:
                raise ValueError("qtip2_v7 root-map layer coverage mismatch")
            lut = Path(str(binding["qtip2_v7_shared_lut"])).resolve()
            expected_lut_sha = root_map.get("shared_lut_sha256")
            if (
                not lut.is_file()
                or lut.stat().st_size != 2048
                or hashlib.sha256(lut.read_bytes()).hexdigest() != expected_lut_sha
            ):
                raise ValueError("qtip2_v7 shared LUT identity mismatch")
            self.root_maps["qtip2_v7"] = {
                str(layer): str(root) for layer, root in layer_roots.items()
            }
            self.qtip2_v7_shared_lut_path = lut
            self.qtip2_v7_roster_members = _load_qtip2_v7_member_roster(
                binding["qtip2_v7_member_roster"],
                expected_basis_sha256=self.basis_sha256,
                expected_roster_sha256=str(root_map.get("selected_wire_roster_sha256")),
            )
            self._record_path(path)
            self._record_path(lut)
            self._record_path(Path(str(binding["qtip2_v7_member_roster"])).resolve())
        self._record_path(self.virtual_manifest_path)
        self._record_path(self.materialization_index_path)

    def _decode_native_qtip3_payloads(
        self, payloads: list[Mapping[str, Any]]
    ) -> list[Any]:
        """Decode and transform one same-shape QTIP3 cell batch on-device."""

        if not payloads:
            return []
        torch = self.torch
        device = self.device
        rows, columns = [int(value) for value in payloads[0]["shape"]]
        tlut_cpu = payloads[0]["tlut"].float()
        packed_rows = []
        for payload in payloads:
            if [int(value) for value in payload["shape"]] != [rows, columns]:
                raise ValueError("QTIP3 decode batch shape mismatch")
            if not torch.equal(payload["tlut"].float(), tlut_cpu):
                raise ValueError("QTIP3 decode batch TLUT mismatch")
            packed_rows.append(payload["trellis"].reshape(rows, -1))
        packed = torch.cat(packed_rows, dim=0).to(device)
        tlut = tlut_cpu.to(device)
        raw = decode_native_v4_torch(
            packed,
            torch.ones(packed.shape[0], dtype=torch.float32, device=device),
            positions=columns,
            tlut=tlut,
            geometry=native_v4_geometry(3.0),
        ).reshape(len(payloads), rows, columns)
        scale = torch.stack([payload["Wscale"].reshape(()) for payload in payloads]).to(
            device
        )
        sv = torch.stack([payload["SV"].float() for payload in payloads]).to(device)
        su = torch.stack([payload["SU"].float() for payload in payloads]).to(device)
        decoded = raw * scale[:, None, None]
        decoded = _fwht(torch, decoded.transpose(-1, -2)).transpose(-1, -2)
        decoded = decoded * sv[:, :, None]
        decoded = _fwht(torch, decoded) * su[:, None, :]
        return list(decoded.to(torch.bfloat16).unbind(0))

    def _load_verified_native_qtip3_payload(
        self, layer: int, expert: int, projection: str
    ) -> Mapping[str, Any]:
        root = Path(self.root_maps["qtip3"][str(layer)])
        unit_root = root / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
        receipt_path = unit_root / "QTIP_SOLVE_RECEIPT.json"
        artifact_path = unit_root / "QTIP_UNIT.pt"
        receipt = json.loads(receipt_path.read_text())
        basis = receipt.get("basis_gate")
        if (
            receipt.get("schema") != "banana-smasher-qtip-solve-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("layer") != layer
            or receipt.get("expert") != expert
            or receipt.get("projection") != projection
            or not isinstance(basis, Mapping)
            or basis.get("index_sha256") != self.basis_sha256
        ):
            raise ValueError(f"QTIP unit receipt identity mismatch: {unit_root}")
        self._record_path(receipt_path)
        payload = self._load_qtip_payload(
            receipt=receipt,
            artifact_path=artifact_path,
            source_key="qtip3",
        )
        geometry = payload.get("geometry")
        expected_geometry = native_v4_geometry(3.0).as_mapping()
        if (
            payload.get("schema") != "banana-smasher-qtip3-native-v6-unit-v1"
            or not isinstance(geometry, Mapping)
            or any(geometry.get(key) != value for key, value in expected_geometry.items())
        ):
            raise ValueError(f"QTIP3 unit payload identity mismatch: {artifact_path}")
        rows, columns = [int(value) for value in payload["shape"]]
        expected_shape = (4096, 2048) if projection == "down" else (4096, 4096)
        if (rows, columns) != expected_shape:
            raise ValueError(f"QTIP3 unit shape mismatch: {artifact_path}")
        return payload

    def _decode_qtip(
        self, source_key: str, layer: int, expert: int, projection: str
    ) -> Any:
        torch = self.torch
        root = Path(self.root_maps[source_key][str(layer)])
        unit_root = root / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
        receipt_path = unit_root / "QTIP_SOLVE_RECEIPT.json"
        artifact_path = unit_root / "QTIP_UNIT.pt"
        receipt = json.loads(receipt_path.read_text())
        basis = receipt.get("basis_gate")
        if (
            receipt.get("schema") != "banana-smasher-qtip-solve-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("layer") != layer
            or receipt.get("expert") != expert
            or receipt.get("projection") != projection
            or not isinstance(basis, Mapping)
            or basis.get("index_sha256") != self.basis_sha256
        ):
            raise ValueError(f"QTIP unit receipt identity mismatch: {unit_root}")
        self._record_path(receipt_path)
        payload = self._load_qtip_payload(
            receipt=receipt,
            artifact_path=artifact_path,
            source_key=source_key,
        )
        geometry = payload.get("geometry")
        expected_k = 2 if source_key == "qtip2" else 3
        legacy_geometry = (
            isinstance(geometry, Mapping)
            and int(geometry.get("L", -1)) == 16
            and int(geometry.get("K", -1)) == expected_k
            and int(geometry.get("V", -1)) == 2
            and int(geometry.get("tlut_bits", -1)) == 9
            and geometry.get("decode_mode") == "quantlut_sym"
        )
        native_qtip3_geometry = (
            source_key == "qtip3"
            and isinstance(geometry, Mapping)
            and int(geometry.get("L", -1)) == 16
            and int(geometry.get("B", -1)) == 12
            and int(geometry.get("V", -1)) == 4
            and int(geometry.get("rate_num", -1)) == 3
            and int(geometry.get("rate_den", -1)) == 1
            and int(geometry.get("tlut_bits", -1)) == 9
            and geometry.get("decode_mode") == "paired_quantlut_sym"
        )
        expected_schemas = {
            "banana-smasher-qtip2-public-unit-v1",
            "ds4-qtip-hyb-bounded36-unit-v1",
            "banana-smasher-qtip-unit-v1",
            "banana-smasher-qtip3-native-v6-unit-v1",
        }
        if payload.get("schema") not in expected_schemas or not (
            legacy_geometry or native_qtip3_geometry
        ):
            raise ValueError(f"QTIP unit payload identity mismatch: {artifact_path}")
        rows, columns = [int(value) for value in payload["shape"]]
        expected_shape = (4096, 2048) if projection == "down" else (4096, 4096)
        if (rows, columns) != expected_shape:
            raise ValueError(f"QTIP unit shape mismatch: {artifact_path}")
        device = self.device
        tlut = payload["tlut"].float().to(device)
        if native_qtip3_geometry:
            packed = payload["trellis"].to(device)
            raw = decode_native_v4_torch(
                packed,
                torch.ones(rows, dtype=torch.float32, device=device),
                positions=columns,
                tlut=tlut,
                geometry=native_v4_geometry(3.0),
            )
        else:
            index = torch.arange(1 << 16, device=device)
            quadratic = (index + 1) * index
            sign_flip = 1 - ((quadratic >> 15) & 1) * 2
            lut_index = (quadratic >> 6) & ((1 << 9) - 1)
            expanded = tlut[lut_index]
            expanded[:, 0] *= sign_flip
            raw = _decode_compressed(
                torch,
                16,
                9,
                expected_k,
                1,
                rows,
                columns,
                payload["trellis"].to(device).reshape(-1),
                expanded,
            )
        decoded = raw * payload["Wscale"].to(device)
        decoded = _fwht(torch, decoded.T).T * payload["SV"].float().to(device)[:, None]
        decoded = _fwht(torch, decoded) * payload["SU"].float().to(device)
        return decoded.to(torch.bfloat16)

    def _load_qtip_payload(
        self,
        *,
        receipt: Mapping[str, Any],
        artifact_path: Path,
        source_key: str,
    ) -> dict[str, Any]:
        """Load a monolithic unit or compose its closure-bound split wire."""

        torch = self.torch
        split = receipt.get("closure_split_payload")
        if split is None:
            self._record_path(artifact_path)
            return torch.load(
                artifact_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        required = {
            "closure_receipt_sha256",
            "control_path",
            "control_sha256",
            "codes_path",
            "codes_sha256",
            "source_host",
        }
        if not isinstance(split, Mapping) or set(split) != required:
            raise ValueError(f"QTIP split payload binding mismatch: {artifact_path}")
        for key in ("closure_receipt_sha256", "control_sha256", "codes_sha256"):
            value = split[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"QTIP split payload {key} is invalid: {artifact_path}")
        control_path = Path(str(split["control_path"])).resolve()
        codes_path = Path(str(split["codes_path"])).resolve()
        if not control_path.is_file() or not codes_path.is_file():
            raise FileNotFoundError(f"QTIP split payload is incomplete: {artifact_path}")
        self._record_path(control_path)
        self._record_path(codes_path)
        control = torch.load(
            control_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(control, Mapping):
            raise ValueError(f"QTIP split control is not a mapping: {control_path}")
        shape = control.get("shape")
        geometry = control.get("geometry")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"QTIP split control geometry mismatch: {control_path}")
        expected_k = 2 if source_key == "qtip2" else 3
        rows, columns = (int(value) for value in shape)
        expected_codes_bytes = expected_k * rows * columns // 8
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
        if (
            codes.dtype != np.uint8
            or not codes.flags.c_contiguous
            or codes.nbytes != expected_codes_bytes
        ):
            raise ValueError(f"QTIP split codes geometry mismatch: {codes_path}")
        if isinstance(geometry, Mapping) and source_key != "qtip3":
            return {
                **control,
                "schema": "banana-smasher-qtip-unit-v1",
                "geometry": {**geometry, "K": expected_k},
                # Avoid aliasing a read-only numpy mmap; each compact member is
                # released with its decoded layer state.
                "trellis": torch.from_numpy(np.array(codes, copy=True)),
            }
        if source_key != "qtip3":
            raise ValueError(f"QTIP split control geometry mismatch: {control_path}")

        # Native-v6 QTIP3 deliberately keeps the compact codes separate from
        # the small solve control.  Bind the sibling public producer receipt
        # and its shared TLUT before composing the runtime payload.
        cell_receipt_path = codes_path.parent / "CELL_RECEIPT.json"
        cell_receipt = json.loads(cell_receipt_path.read_text())
        cell_geometry = cell_receipt.get("geometry")
        artifacts = cell_receipt.get("artifacts")
        tlut_binding = cell_receipt.get("tlut")
        if (
            cell_receipt.get("schema") != "banana-smasher-qtip-native-v4-cell-v1"
            or cell_receipt.get("status") != "PASS"
            or cell_receipt.get("provider") != "qtip-native-v6@3.00"
            or cell_receipt.get("codec_version") != "v6"
            or cell_receipt.get("basis_sha256") != self.basis_sha256
            or not isinstance(cell_geometry, Mapping)
            or not isinstance(artifacts, Mapping)
            or not isinstance(tlut_binding, Mapping)
            or artifacts.get("codes", {}).get("sha256") != split["codes_sha256"]
            or cell_receipt.get("control", {}).get("sha256") != split["control_sha256"]
        ):
            raise ValueError(f"QTIP3 split producer identity mismatch: {cell_receipt_path}")
        expected_geometry = native_v4_geometry(3.0).as_mapping()
        if any(cell_geometry.get(key) != value for key, value in expected_geometry.items()):
            raise ValueError(f"QTIP3 split producer geometry mismatch: {cell_receipt_path}")
        tlut_path = codes_path.parents[3] / "inputs" / "qtip_tlut.npy"
        if (
            not tlut_path.is_file()
            or hashlib.sha256(tlut_path.read_bytes()).hexdigest()
            != tlut_binding.get("sha256")
        ):
            raise ValueError(f"QTIP3 split shared TLUT identity mismatch: {tlut_path}")
        tlut = np.load(tlut_path, mmap_mode="r", allow_pickle=False)
        if (
            tlut.dtype != np.float32
            or tlut.shape != (512, 2)
            or hashlib.sha256(tlut.tobytes(order="C")).hexdigest()
            != tlut_binding.get("tensor_sha256")
        ):
            raise ValueError(f"QTIP3 split shared TLUT tensor mismatch: {tlut_path}")
        self._record_path(cell_receipt_path)
        self._record_path(tlut_path)
        return {
            **control,
            "schema": "banana-smasher-qtip3-native-v6-unit-v1",
            "geometry": dict(cell_geometry),
            "tlut": torch.from_numpy(np.array(tlut, copy=True)),
            "trellis": torch.from_numpy(
                np.array(codes, copy=True).reshape(rows, expected_codes_bytes // rows)
            ),
        }

    def _decode_qtip2_v7_part(
        self, layer: int, expert: int, wire_projection: str
    ) -> Any:
        """Decode one current raw V7 member through the established QTIP math."""

        if self.qtip2_v7_shared_lut_path is None:
            raise ValueError("qtip2_v7 source selected without a shared LUT binding")
        roster_key = (layer, expert, wire_projection)
        try:
            member, member_sha256 = self.qtip2_v7_roster_members[roster_key]
        except KeyError as exc:
            raise ValueError(
                "qtip2_v7 artifact roster has no unique member for "
                f"layer={layer} expert={expert} projection={wire_projection}"
            ) from exc
        if hashlib.sha256(member.read_bytes()).hexdigest() != member_sha256:
            raise ValueError(
                "qtip2_v7 artifact roster member SHA-256 drift for "
                f"layer={layer} expert={expert} projection={wire_projection}"
            )
        payload = load_qtip2_v7_wire(member, projection=wire_projection)
        torch = self.torch
        device = self.device
        packed = (
            torch.from_numpy(np.array(payload["packed"], copy=True))
            .to(device)
            .reshape(-1)
        )
        su = torch.from_numpy(np.array(payload["SU"], copy=True)).float().to(device)
        sv = torch.from_numpy(np.array(payload["SV"], copy=True)).float().to(device)
        scale = (
            torch.from_numpy(np.array(payload["Wscale"], copy=True)).float().to(device)
        )
        lut_values = np.fromfile(self.qtip2_v7_shared_lut_path, dtype="<f2")
        if lut_values.shape != (1024,):
            raise ValueError("qtip2_v7 shared LUT must be float16[1024]")
        tlut = torch.from_numpy(lut_values.copy()).reshape(512, 2).float().to(device)
        index = torch.arange(1 << 16, device=device)
        quadratic = (index + 1) * index
        sign_flip = 1 - ((quadratic >> 15) & 1) * 2
        lut_index = (quadratic >> 6) & ((1 << 9) - 1)
        expanded = tlut[lut_index]
        expanded[:, 0] *= sign_flip
        rows, columns = payload["weight_shape"]
        raw = _decode_compressed(torch, 16, 9, 2, 1, rows, columns, packed, expanded)
        decoded = raw * scale
        decoded = _fwht(torch, decoded.T).T * sv[:, None]
        decoded = _fwht(torch, decoded) * su
        return decoded.to(torch.bfloat16)

    def _decode_qtip2_v7(self, layer: int, expert: int, projection: str) -> Any:
        if projection == "down":
            return self._decode_qtip2_v7_part(layer, expert, "w2")
        if projection != "fused13":
            raise ValueError(f"unsupported qtip2_v7 logical projection: {projection}")
        gate = self._decode_qtip2_v7_part(layer, expert, "w1")
        up = self._decode_qtip2_v7_part(layer, expert, "w3")
        result = self.torch.cat((gate, up), dim=0)
        del gate, up
        return result

    def _native(self, layer: int, expert: int, projection: str) -> Any:
        prefix = f"layers.{layer}.ffn.experts.{expert}."

        def decode(name: str) -> Any:
            weight = self._get_tensor(prefix + name + ".weight").to(self.device)
            scale = self._get_tensor(prefix + name + ".scale").to(self.device)
            return _decode_mxfp4_e2m1(self.torch, weight, scale)

        if projection == "down":
            return decode("w2")
        gate = decode("w1")
        up = decode("w3")
        result = self.torch.cat((gate, up), dim=0)
        del gate, up
        return result

    def _decode_d4(
        self,
        source_key: str,
        layer: int,
        expert: int,
        projection: str,
        layer_view: Any,
    ) -> Any:
        """Decode one selected fixed-D4 cell from its verified uniform pack."""

        bits = 11 if source_key == "d4_k2048" else 12
        codebook_size = 1 << bits
        rows, columns = (4096, 2048) if projection == "down" else (4096, 4096)
        prefix = f"layers.{layer}.truevq_d4.{source_key}.{projection}."
        expert_ids = np.asarray(
            layer_view.get(prefix + "expert_ids"), dtype=np.int64
        ).reshape(-1)
        positions = np.flatnonzero(expert_ids == expert)
        if positions.size != 1:
            raise ValueError(
                f"{source_key} expert partition mismatch: layer={layer} "
                f"expert={expert} projection={projection} matches={positions.size}"
            )
        position = int(positions[0])
        code_bytes = rows * columns // 4 * bits // 8
        scale_bytes = rows * columns // 32
        packed_codes = np.asarray(
            layer_view.get(prefix + "codes"), dtype=np.uint8
        ).reshape(-1)
        packed_scales = np.asarray(
            layer_view.get(prefix + "scales"), dtype=np.uint8
        ).reshape(-1)
        codebook = np.asarray(layer_view.get(prefix + "codebooks"), dtype=np.float16)
        if codebook.shape != (codebook_size, 4):
            raise ValueError(
                f"{source_key} codebook shape mismatch: layer={layer} "
                f"projection={projection} shape={codebook.shape}"
            )
        code_start = position * code_bytes
        scale_start = position * scale_bytes
        code_slice = packed_codes[code_start : code_start + code_bytes]
        scale_slice = packed_scales[scale_start : scale_start + scale_bytes]
        if code_slice.size != code_bytes or scale_slice.size != scale_bytes:
            raise ValueError(
                f"{source_key} expert payload is truncated: layer={layer} "
                f"expert={expert} projection={projection}"
            )
        return decode_d4_expert(
            code_slice,
            scale_slice,
            codebook,
            bits=bits,
            rows=rows,
            columns=columns,
            torch=self.torch,
            device=self.device,
        )

    @contextmanager
    def terminal_stage(self):
        """Score Top-8192 support and full-vocabulary argmax without Python lists."""

        torch = self.torch
        self._begin_stage()
        model = self.model
        model.model.norm.weight = torch.nn.Parameter(
            self._get_tensor("norm.weight").to(self.device).to(torch.bfloat16),
            requires_grad=False,
        )
        model.model.hc_head.hc_fn = torch.nn.Parameter(
            self._get_tensor("hc_head_fn").to(self.device).to(torch.float32),
            requires_grad=False,
        )
        model.model.hc_head.hc_base = torch.nn.Parameter(
            self._get_tensor("hc_head_base").to(self.device).to(torch.float32),
            requires_grad=False,
        )
        model.model.hc_head.hc_scale = torch.nn.Parameter(
            self._get_tensor("hc_head_scale").to(self.device).to(torch.float32),
            requires_grad=False,
        )
        model.lm_head.weight = torch.nn.Parameter(
            self._get_tensor("head.weight").to(self.device).to(torch.bfloat16),
            requires_grad=False,
        )
        resources = [model.model.norm, model.model.hc_head, model.lm_head]
        self._resident_now()

        def _score(
            activation: Any,
            support_token_ids: Any,
            *,
            window_id: object,
        ) -> dict[str, Any]:
            del window_id
            support_token_ids = torch.as_tensor(
                support_token_ids,
                dtype=torch.long,
                device=self.device,
            )
            pairs: list[Any] = []
            top1: list[Any] = []
            with torch.no_grad():
                hidden = model.model.norm(
                    model.model.hc_head(activation.hidden.unsqueeze(0))
                ).squeeze(0)
                for start in range(0, hidden.shape[0], 128):
                    logits = model.lm_head(
                        hidden[start : start + 128].to(torch.bfloat16)
                    ).float()
                    support = support_token_ids[start : start + logits.shape[0]]
                    pairs.append(logits.gather(1, support).to(torch.float16).cpu())
                    top1.append(logits.argmax(-1).to(torch.int32).cpu())
                    del logits, support
            self._resident_now()
            return {
                "q_lp_at_ref": torch.cat(pairs),
                "q_argmax": torch.cat(top1),
            }

        try:
            yield _score
        finally:
            while resources:
                self._dematerialize(resources.pop())
            self._release()
            self._stage_active = False

    def _load_vq3u_experts(self, layer: int) -> tuple[Any, Any]:
        torch = self.torch
        required = (256 * 4096 * 4096 + 256 * 4096 * 2048) * 2
        free = _available_materialization_bytes(torch, self.device)
        if free - (4 << 30) < required:
            raise RuntimeError(
                f"layer {layer}: insufficient CUDA memory for mixed Backpack materialization: "
                f"free={free}, required_plus_guard={required + (4 << 30)}"
            )
        gate_up = torch.empty(256, 4096, 4096, dtype=torch.bfloat16, device=self.device)
        down = torch.empty(256, 4096, 2048, dtype=torch.bfloat16, device=self.device)
        rows = sorted(
            self.rows_by_layer[layer],
            key=lambda row: (int(row["expert"]), str(row["projection"])),
        )
        seen: set[tuple[int, str]] = set()
        selected_d4_tiers = {
            str(row["source_key"])
            for row in rows
            if str(row["source_key"]).startswith("d4_k")
        }
        qtip3_values: dict[tuple[int, str], Any] = {}
        for projection in ("down", "fused13"):
            qtip3_rows = [
                row
                for row in rows
                if row["source_key"] == "qtip3" and row["projection"] == projection
            ]
            for start in range(0, len(qtip3_rows), 32):
                batch_rows = qtip3_rows[start : start + 32]
                payloads = [
                    self._load_verified_native_qtip3_payload(
                        layer, int(row["expert"]), projection
                    )
                    for row in batch_rows
                ]
                values = self._decode_native_qtip3_payloads(payloads)
                for row, value in zip(batch_rows, values, strict=True):
                    qtip3_values[(int(row["expert"]), projection)] = value
        with ExitStack() as stack:
            d4_views = {
                tier: stack.enter_context(
                    self.d4_loaders[tier].open_layer(layer, framework="np")
                )
                for tier in selected_d4_tiers
            }
            for position, row in enumerate(rows, 1):
                expert = int(row["expert"])
                projection = str(row["projection"])
                source_key = str(row["source_key"])
                key = (expert, projection)
                if key in seen or source_key not in {
                    "native_mxfp4",
                    "qtip2",
                    "qtip3",
                    "qtip2_v7",
                    "d4_k2048",
                    "d4_k4096",
                }:
                    raise ValueError(f"invalid mixed Backpack cell row: {row}")
                seen.add(key)
                if source_key == "native_mxfp4":
                    value = self._native(layer, expert, projection)
                elif source_key == "qtip2":
                    value = self._decode_qtip(source_key, layer, expert, projection)
                elif source_key == "qtip3":
                    value = qtip3_values.pop(key)
                elif source_key == "qtip2_v7":
                    value = self._decode_qtip2_v7(layer, expert, projection)
                else:
                    value = self._decode_d4(
                        source_key,
                        layer,
                        expert,
                        projection,
                        d4_views[source_key],
                    )
                destination = down if projection == "down" else gate_up
                if value.shape != destination[expert].shape:
                    raise ValueError(
                        f"mixed Backpack cell shape mismatch: cell={row['cell_id']} "
                        f"source={source_key} value={tuple(value.shape)} "
                        f"destination={tuple(destination[expert].shape)}"
                    )
                destination[expert].copy_(value)
                del value
                if position % 64 == 0:
                    print(
                        f"BACKPACK_LAYER_PROGRESS layer={layer} cells={position}/512",
                        flush=True,
                    )
                    self.synchronize()
        if qtip3_values:
            raise ValueError(f"layer {layer}: unused QTIP3 decode batch cells")
        if len(seen) != 512:
            raise ValueError(f"layer {layer}: mixed Backpack cell coverage mismatch")
        gc.collect()
        return gate_up, down
