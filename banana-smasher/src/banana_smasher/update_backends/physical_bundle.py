from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import torch

from ..fwht import bounded_fwht, fwht_stats
from ..kmajor_graph import (
    layer_graph_forward,
    layer_graph_vjp_stats,
    reset_layer_graph_vjp,
)

_BUNDLE_SCHEMA = "banana-smasher-physical-repair-bundle-v1"
_QTIP2_SCHEMAS = {
    "banana-smasher-qtip2-public-unit-v1",
    "banana-smasher-qtip-unit-v1",
    "banana-smasher-qtip-v7-public-unit-v1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tensor(document: dict[str, Any], name: str) -> torch.Tensor:
    value = document.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"physical repair bundle requires tensor {name}")
    return value


def _dense_kmajor(
    codebook: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    code_dim = int(codebook.shape[1])
    values = codebook.detach()[codes.long()].reshape(
        *codes.shape[:-1], int(codes.shape[-1]) * code_dim
    )
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    if tuple(values.shape) != tuple(scale_columns.shape):
        raise ValueError(
            "physical repair projection code/scale geometry mismatch: "
            f"values={tuple(values.shape)} scales={tuple(scale_columns.shape)}"
        )
    return (values.float() * scale_columns).transpose(-1, -2).contiguous()


class _FrozenLayerPayload(torch.nn.Module):
    def __init__(
        self,
        *,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        projections: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.register_buffer("top_k_index", top_k_index)
        self.register_buffer("top_k_weights", top_k_weights)
        for name in ("13", "2"):
            projection = projections[name]
            self.register_buffer(f"codes_{name}", projection["codes"])
            self.register_buffer(f"scales_{name}", projection["scales"])
            self.register_buffer(f"dense_{name}", projection["dense"])

    def projection(self, name: str, codebook: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "codebook": codebook,
            "codes": getattr(self, f"codes_{name}"),
            "scales": getattr(self, f"scales_{name}"),
            "dense": getattr(self, f"dense_{name}"),
        }


class PhysicalRepairLayer(torch.nn.Module):
    """Standalone grouped K-major repair capsule with frozen QTIP geometry."""

    def __init__(
        self,
        *,
        codebook_13: torch.Tensor,
        codebook_2: torch.Tensor,
        frozen: _FrozenLayerPayload,
        clamp_limit: float,
    ) -> None:
        super().__init__()
        self.codebook_13 = torch.nn.Parameter(codebook_13.float().contiguous())
        self.codebook_2 = torch.nn.Parameter(codebook_2.float().contiguous())
        self.frozen = frozen
        self.clamp_limit = float(clamp_limit)
        self.checkpoint_depth = False

    @property
    def codebooks(self) -> list[torch.nn.Parameter]:
        return [self.codebook_13, self.codebook_2]

    def _forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise ValueError("physical repair layer expects [1, tokens, hidden]")
        transformed = bounded_fwht(hidden.float(), normalize=True)
        flat = transformed.reshape(-1, int(transformed.shape[-1]))
        payloads = {
            "13": self.frozen.projection("13", self.codebook_13),
            "2": self.frozen.projection("2", self.codebook_2),
        }
        tokens = int(flat.shape[0])
        repaired = layer_graph_forward(
            flat,
            self.frozen.top_k_index[:tokens],
            self.frozen.top_k_weights[:tokens],
            payloads,
            limit=self.clamp_limit,
        ).reshape_as(transformed)
        return bounded_fwht(repaired, normalize=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.checkpoint_depth:
            return self._forward(hidden)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._forward, hidden, use_reentrant=False)


def _qtip2_decode_states(trellis: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    """Decode QTIP2 wire once into compact immutable 16-bit trellis states."""
    if m % 32 or k % 32:
        raise ValueError("QTIP2 repair capsule requires M and K divisible by 32")
    if trellis.dtype != torch.uint16:
        raise ValueError("QTIP2 trellis requires canonical torch.uint16 wire")
    compressed = trellis.contiguous()
    expected = 2 * m * k // 16
    if tuple(compressed.shape) != (expected,):
        compressed = compressed.reshape(-1)
    if int(compressed.numel()) != expected:
        raise ValueError(
            f"QTIP2 trellis size drift {compressed.numel()} != {expected}"
        )
    block_size = 16 * 16
    bits_per_block = 2 * block_size
    compressed = (
        compressed.view(torch.uint8)
        .reshape(m // 32, k // 32, block_size // 8, 2, 2, 2)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, bits_per_block // 16, 2)
        .flip((-1,))
        .contiguous()
        .view(torch.uint16)
        .reshape(m // 16, k // 16, bits_per_block // 16)
    )
    blocked = compressed.reshape(2 * m * k // bits_per_block, bits_per_block // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = (
        torch.cat((blocked_roll, blocked), dim=-1)
        .reshape(blocked.shape[0], -1)
        .contiguous()
        .view(torch.uint32)
    )
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(
        *blocked32.shape, 16
    ).view(torch.int32)
    shifts = torch.arange(
        16, dtype=torch.int32, device=blocked.device
    ).reshape(1, 1, -1).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    state = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 0::4], (1 << 16) - 1
    )
    return state.to(torch.uint16).contiguous()


class _FrozenQtip2Payload(torch.nn.Module):
    def __init__(self, mapping: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name, value in mapping.items():
            self.register_buffer(name, value)


class Qtip2PhysicalLayer(torch.nn.Module):
    """Exact QTIP2 quantlut/FWHT capsule with only TLUT trainable."""

    def __init__(
        self,
        *,
        tlut: torch.Tensor,
        frozen: _FrozenQtip2Payload,
        source_schema: str,
    ) -> None:
        super().__init__()
        self.tlut = (
            tlut
            if isinstance(tlut, torch.nn.Parameter)
            else torch.nn.Parameter(tlut.float().contiguous())
        )
        self.frozen = frozen
        self.source_schema = source_schema
        self.checkpoint_depth = False

    @property
    def codebooks(self) -> list[torch.nn.Parameter]:
        return [self.tlut]

    def _weight(self) -> torch.Tensor:
        state = self.frozen.states.to(torch.int32)
        quadratic = (state + 1) * state
        lut_index = (quadratic >> 6) & ((1 << 9) - 1)
        raw = self.tlut[lut_index]
        sign = (1 - ((quadratic >> 15) & 1) * 2).to(raw.dtype)
        raw = torch.stack((raw[..., 0] * sign, raw[..., 1]), dim=-1)
        m = int(self.frozen.sv.numel())
        k = int(self.frozen.su.numel())
        raw = (
            raw.reshape(m // 16, k // 16, 16, 16)
            .reshape(m // 16, k // 16, 8, 4, 2, 2, 2)
            .permute(0, -2, 2, 1, -3, 3, -1)
            .reshape(m, k)
        )
        q = raw * self.frozen.wscale
        q = bounded_fwht(q.transpose(0, 1), normalize=True).transpose(0, 1)
        q = q * self.frozen.sv[:, None]
        q = bounded_fwht(q, normalize=True) * self.frozen.su
        return q

    def _forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise ValueError("QTIP2 repair layer expects [1, tokens, K]")
        weight = self._weight()
        return torch.matmul(hidden.float(), weight.transpose(0, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.checkpoint_depth:
            return self._forward(hidden)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._forward, hidden, use_reentrant=False)


class PhysicalBundleRuntime:
    """Data-only physical runtime; no external module or mission-code imports."""

    def __init__(self, request: dict[str, Any], context: dict[str, Any]) -> None:
        bundle_value = request.get("bundle")
        expected_sha = request.get("bundle_sha256")
        if not isinstance(bundle_value, str) or not isinstance(expected_sha, str):
            raise ValueError("physical repair request requires bundle and bundle_sha256")
        self.bundle_path = Path(bundle_value).expanduser().resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError(self.bundle_path)
        observed_sha = _sha256_file(self.bundle_path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                "physical repair bundle identity mismatch: "
                f"expected={expected_sha} observed={observed_sha}"
            )
        self.bundle_sha256 = observed_sha
        self.context = context
        self.device = torch.device(
            request.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("physical repair request requires unavailable CUDA")
        self.document = torch.load(
            self.bundle_path, map_location="cpu", weights_only=False
        )
        if not isinstance(self.document, dict) or self.document.get("schema") != _BUNDLE_SCHEMA:
            raise ValueError(f"physical repair bundle schema must be {_BUNDLE_SCHEMA!r}")
        self.packed_indices: list[torch.Tensor] = []
        self.layers: list[torch.nn.Module] = []
        self._staged: dict[str, Any] | None = None
        self._source_retired = False

    def authenticate_aot(self) -> dict[str, object]:
        identity = self.context.get("identity", {})
        aot_sha256 = identity.get("aot_sha256")
        if not isinstance(aot_sha256, str) or len(aot_sha256) != 64:
            raise RuntimeError("physical repair requires sealed AOT identity")
        return {
            "status": "PASS_AUTHENTICATED_AOT",
            "sha256": aot_sha256,
            "runtime": "installed-banana-smasher-layer-graph",
        }

    def decode_packed_indices(self) -> dict[str, object]:
        if self.packed_indices:
            raise RuntimeError("physical repair packed indices already decoded")
        layers = self.document.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("physical repair bundle requires non-empty layers")
        decoded_bytes = 0
        for layer_index, layer in enumerate(layers):
            if isinstance(layer, dict) and layer.get("schema") in _QTIP2_SCHEMAS:
                shape = layer.get("shape")
                geometry = layer.get("geometry")
                if (
                    not isinstance(shape, (list, tuple))
                    or len(shape) != 2
                    or not isinstance(geometry, dict)
                ):
                    raise ValueError(
                        f"physical QTIP2 layer {layer_index} requires shape and geometry"
                    )
                expected_geometry = {
                    "L": 16,
                    "V": 2,
                    "tlut_bits": 9,
                    "decode_mode": "quantlut_sym",
                    "td_x": 16,
                    "td_y": 16,
                }
                rate = geometry.get("K")
                if (
                    isinstance(rate, bool)
                    or not isinstance(rate, int)
                    or rate <= 0
                    or any(geometry.get(name) != value for name, value in expected_geometry.items())
                ):
                    raise ValueError(
                        f"physical QTIP layer {layer_index} geometry is not rate-parameterized L16/V2"
                    )
                if rate != 2:
                    raise ValueError(
                        f"installed canonical consumer does not support QTIP rate {rate}"
                    )
                m, k = (int(shape[0]), int(shape[1]))
                source_trellis = _require_tensor(layer, "trellis")
                if source_trellis.dtype != torch.uint16:
                    raise ValueError(
                        "physical QTIP2 trellis requires canonical torch.uint16 wire"
                    )
                trellis = source_trellis.to(device=self.device).contiguous()
                states = _qtip2_decode_states(trellis, m=m, k=k)
                trellis.requires_grad_(False)
                states.requires_grad_(False)
                layer["decoded_trellis"] = trellis
                layer["decoded_states"] = states
                self.packed_indices.extend((trellis, states))
                decoded_bytes += sum(
                    value.numel() * value.element_size()
                    for value in (trellis, states)
                )
                continue
            projections = layer.get("projections") if isinstance(layer, dict) else None
            if not isinstance(projections, dict) or set(projections) != {"13", "2"}:
                raise ValueError(
                    f"physical repair layer {layer_index} requires projections 13 and 2"
                )
            for projection_name in ("13", "2"):
                packed = _require_tensor(projections[projection_name], "packed_codes")
                if packed.dtype not in (torch.int8, torch.int16, torch.int32):
                    raise ValueError("physical repair packed codes require integer wire")
                decoded = packed.to(device=self.device, dtype=torch.int32).contiguous()
                decoded.requires_grad_(False)
                projections[projection_name]["codes"] = decoded
                self.packed_indices.append(decoded)
                decoded_bytes += decoded.numel() * decoded.element_size()
        return {
            "status": "PASS_DECODED_ONCE",
            "tensors": self.packed_indices,
            "decoded_bytes": decoded_bytes,
            "packed_wire_preserved": True,
        }

    def build_persistent_layer_layouts(
        self, packed_indices: list[torch.Tensor]
    ) -> dict[str, object]:
        if [id(value) for value in packed_indices] != [
            id(value) for value in self.packed_indices
        ] or self.layers:
            raise RuntimeError("physical repair persistent layout state drift")
        reset_layer_graph_vjp(allow_reference=self.device.type != "cuda")
        fwht_stats(reset=True)
        shared_qtip_v7_luts: dict[int, torch.nn.Parameter] = {}
        for layer_index, layer_document in enumerate(self.document["layers"]):
            if layer_document.get("schema") in _QTIP2_SCHEMAS:
                m, k = (int(value) for value in layer_document["shape"])
                su = _require_tensor(layer_document, "SU")
                sv = _require_tensor(layer_document, "SV")
                tlut = _require_tensor(layer_document, "tlut")
                wscale = _require_tensor(layer_document, "Wscale")
                if tuple(su.shape) != (k,) or tuple(sv.shape) != (m,):
                    raise ValueError(
                        f"physical QTIP2 layer {layer_index} sign-plane shape mismatch"
                    )
                if tuple(tlut.shape) != (512, 2) or wscale.numel() != 1:
                    raise ValueError(
                        f"physical QTIP2 layer {layer_index} TLUT/scale shape mismatch"
                    )
                frozen = _FrozenQtip2Payload(
                    {
                        "trellis": layer_document["decoded_trellis"],
                        "states": layer_document["decoded_states"],
                        "su": su.to(self.device, dtype=torch.float32),
                        "sv": sv.to(self.device, dtype=torch.float32),
                        "wscale": wscale.to(self.device, dtype=torch.float32),
                    }
                )
                source_schema = str(layer_document["schema"])
                layer_lut = int(layer_document.get("layer_lut", layer_index))
                shared_tlut: torch.Tensor = tlut.to(self.device)
                if source_schema == "banana-smasher-qtip-v7-public-unit-v1":
                    shared_tlut = shared_qtip_v7_luts.setdefault(
                        layer_lut,
                        torch.nn.Parameter(shared_tlut.float().contiguous()),
                    )
                qtip2_layer = Qtip2PhysicalLayer(
                    tlut=shared_tlut,
                    frozen=frozen,
                    source_schema=source_schema,
                )
                reconstructed = layer_document.get("reconstructed_weight")
                if isinstance(reconstructed, torch.Tensor):
                    with torch.no_grad():
                        decoded = qtip2_layer._weight().half().cpu()
                    if not torch.equal(
                        decoded.view(torch.int16),
                        reconstructed.half().cpu().view(torch.int16),
                    ):
                        raise ValueError(
                            f"physical QTIP2 layer {layer_index} packed decode mismatch"
                        )
                self.layers.append(qtip2_layer)
                continue
            projections: dict[str, dict[str, torch.Tensor]] = {}
            codebooks: dict[str, torch.Tensor] = {}
            for name in ("13", "2"):
                source = layer_document["projections"][name]
                codebook = _require_tensor(source, "codebook").to(self.device).float()
                codes = source["codes"]
                scales = _require_tensor(source, "scales").to(
                    self.device, dtype=torch.uint8
                ).contiguous()
                scales.requires_grad_(False)
                projections[name] = {
                    "codes": codes,
                    "scales": scales,
                    "dense": _dense_kmajor(codebook, codes, scales),
                }
                codebooks[name] = codebook
            frozen = _FrozenLayerPayload(
                top_k_index=_require_tensor(layer_document, "top_k_index").to(
                    self.device, dtype=torch.int32
                ),
                top_k_weights=_require_tensor(layer_document, "top_k_weights").to(
                    self.device, dtype=torch.float32
                ),
                projections=projections,
            )
            self.layers.append(
                PhysicalRepairLayer(
                    codebook_13=codebooks["13"],
                    codebook_2=codebooks["2"],
                    frozen=frozen,
                    clamp_limit=float(layer_document.get("clamp_limit", 7.0)),
                )
            )
        return {
            "status": "PASS_PERSISTENT_LAYOUTS",
            "layers": len(self.layers),
            "persistent": True,
            "forward_io_operations": 0,
        }

    def stage_inputs(self, *, largest_first: bool) -> dict[str, object]:
        if not largest_first or self._staged is not None:
            raise RuntimeError("physical repair staging must run exactly once largest-first")
        names = (
            "teacher_targets",
            "activation_inputs",
            "input_ids",
            "teacher_mask",
            "positions",
        )
        values = {name: _require_tensor(self.document, name) for name in names}
        order = sorted(
            names,
            key=lambda name: values[name].numel() * values[name].element_size(),
            reverse=True,
        )
        staged: dict[str, Any] = {
            name: values[name].to(self.device).contiguous() for name in order
        }
        stage_order_nbytes = [
            values[name].numel() * values[name].element_size() for name in order
        ]
        self.document = None
        gc.collect()
        self._source_retired = True
        self._staged = staged
        return {
            "status": "PASS_STAGED_LARGEST_FIRST",
            "stage_order_nbytes": stage_order_nbytes,
            "source_retired": True,
            **staged,
        }

    def configure_depth_checkpointing(self, *, required: bool) -> dict[str, object]:
        if not required:
            raise ValueError("physical repair requires depth checkpointing")
        for layer in self.layers:
            layer.checkpoint_depth = True
        return {
            "status": "PASS_DEPTH_CHECKPOINTING",
            "depth_groups": len(self.layers),
        }

    def update_bundle(self) -> dict[str, object]:
        if not self.layers or self._staged is None or not self._source_retired:
            raise RuntimeError("physical repair runtime was not fully initialized")
        optimizer = self.document_optimizer
        codebooks: list[torch.Tensor] = []
        seen: set[int] = set()
        for layer in self.layers:
            layer_codebooks = getattr(layer, "codebooks")
            for value in layer_codebooks:
                if id(value) not in seen:
                    seen.add(id(value))
                    codebooks.append(value)
        return {
            "layers": self.layers,
            "codebooks": codebooks,
            "frozen_modules": [layer.frozen for layer in self.layers],
            "encode": self._encode,
            "loss_sum": self._loss_sum,
            "optimizer_factory": lambda parameters: torch.optim.SGD(
                parameters, lr=optimizer["learning_rate"]
            ),
            "reset_backend_sentinels": self.reset_backend_sentinels,
            "backend_sentinels": self.backend_sentinels,
            "allow_reference": all(
                isinstance(layer, Qtip2PhysicalLayer) for layer in self.layers
            ),
            "peak_memory_bytes": self.peak_memory_bytes,
            "synchronize": self.synchronize,
        }

    @property
    def document_optimizer(self) -> dict[str, float]:
        # The source document is retired after staging, so retain only this tiny
        # immutable scalar contract on first access.
        if not hasattr(self, "_optimizer"):
            raise RuntimeError("physical repair optimizer contract was not retained")
        return self._optimizer

    def retain_optimizer_contract(self) -> None:
        value = self.document.get("optimizer", {})
        if value.get("name") != "sgd" or not isinstance(
            value.get("learning_rate"), (int, float)
        ):
            raise ValueError("physical repair bundle requires SGD learning_rate")
        self._optimizer = {"learning_rate": float(value["learning_rate"])}

    @staticmethod
    def _loss_sum(hidden: torch.Tensor, segment: dict[str, Any]) -> torch.Tensor:
        selected = (hidden - segment["teacher_targets"].float()).masked_select(
            segment["teacher_mask"].unsqueeze(-1)
        )
        return selected.square().sum()

    def _encode(self, segment: dict[str, Any]) -> torch.Tensor:
        assert self._staged is not None
        positions = segment["positions"].reshape(-1).long()
        return self._staged["activation_inputs"].index_select(1, positions)

    def backend_sentinels(self) -> dict[str, int]:
        graph = layer_graph_vjp_stats()
        fwht = fwht_stats()
        backward = int(graph["backward_calls"])
        return {
            "kmajor_batch": int(graph["grouped_experts"]),
            "kmajor_fused": max(int(graph["reduction_kernel_launches"]), backward),
            "grouped_vjp": backward,
            "layer_graph": int(graph["forward_calls"]),
            "fwht": int(fwht["calls"]),
        }

    def reset_backend_sentinels(self) -> None:
        reset_layer_graph_vjp(allow_reference=self.device.type != "cuda")
        fwht_stats(reset=True)

    def peak_memory_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
