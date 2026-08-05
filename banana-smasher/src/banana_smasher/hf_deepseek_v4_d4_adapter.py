from __future__ import annotations

import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def _full_vocab_support_logprob(logits: Any, support_token_ids: Any) -> tuple[Any, Any]:
    """Gather fp16 full-softmax logprob and int32 full-vocabulary argmax."""

    import torch

    logprob = torch.log_softmax(logits.float(), dim=-1)
    return (
        logprob.gather(1, support_token_ids.long()).to(torch.float16),
        logprob.argmax(-1).to(torch.int32),
    )


def _unpack_le_values(
    packed: np.ndarray, *, bits: int, value_count: int
) -> np.ndarray:
    """Decode a little-endian fixed-width bitstream without a compatibility file."""

    source = np.asarray(packed, dtype=np.uint8)
    padding_bits = source.size * 8 - value_count * bits
    if (
        source.ndim != 1
        or bits not in {11, 12}
        or value_count < 1
        or padding_bits < 0
        or padding_bits > 7
    ):
        raise ValueError(
            f"le{bits} payload size does not match {value_count} values: "
            f"got {source.shape}"
        )
    offsets = np.arange(value_count, dtype=np.uint64) * bits
    byte_offsets = (offsets >> 3).astype(np.int64)
    shifts = (offsets & 7).astype(np.uint32)
    padded = np.pad(source, (0, 2))
    words = (
        padded[byte_offsets].astype(np.uint32)
        | (padded[byte_offsets + 1].astype(np.uint32) << 8)
        | (padded[byte_offsets + 2].astype(np.uint32) << 16)
    )
    return ((words >> shifts) & ((1 << bits) - 1)).astype(np.uint16)


def _unpack_le12_values(packed: np.ndarray) -> np.ndarray:
    """Decode little-endian packed 12-bit pairs without materializing a layer file."""

    source = np.asarray(packed, dtype=np.uint8)
    if source.ndim != 1 or source.size % 3:
        raise ValueError(
            "le12 payload must be a one-dimensional multiple of 3 bytes, "
            f"got {source.shape}"
        )
    return _unpack_le_values(source, bits=12, value_count=source.size * 2 // 3)


def _open_bs_pack_projection(
    model_root: Path,
    manifest: dict[str, Any],
    *,
    layer: int,
    projection: str,
    experts: int,
    rows: int,
    columns: int,
    k: int,
    d: int,
    scale_group_size: int = 32,
) -> dict[str, Any]:
    """Open one bound bs-pack D4 projection as mmap-backed per-expert arrays."""

    if projection not in ("fused13", "down"):
        raise ValueError(f"unsupported D4 projection {projection!r}")
    if rows <= 0 or columns <= 0 or experts <= 0 or d <= 0 or k <= 0:
        raise ValueError("D4 packed dimensions must be positive")
    if columns % d or columns % scale_group_size:
        raise ValueError(
            f"D4 packed columns {columns} must divide by d={d} "
            f"and scale group={scale_group_size}"
        )
    prefix = f"layers.{layer}.truevq_d4.d4_k{k}.{projection}."
    tensor_index = manifest.get("tensor_index")
    if not isinstance(tensor_index, dict):
        raise RuntimeError("bs-pack manifest has no tensor_index object")
    code_bits = 11 if k == 2048 else 12
    codes_per_expert = rows * (columns // d)
    code_bytes_per_expert = (codes_per_expert * code_bits + 7) // 8
    expected = {
        "codebooks": ("fp16", "<f2", k * d * 2),
        "codes": (
            f"le{code_bits}",
            "|u1",
            experts * code_bytes_per_expert,
        ),
        "scales": (
            "e8m0",
            "|u1",
            experts * rows * (columns // scale_group_size),
        ),
    }
    opened: dict[str, Any] = {}
    root = model_root.resolve()
    for role, (encoding, dtype, expected_bytes) in expected.items():
        entry = tensor_index.get(prefix + role)
        if not isinstance(entry, dict):
            raise RuntimeError(f"bs-pack manifest missing {prefix + role}")
        if (
            entry.get("encoding") != encoding
            or entry.get("dtype") != dtype
            or int(entry.get("subtier", -1)) != k
            or int(entry.get("data_bytes", -1)) != expected_bytes
        ):
            raise RuntimeError(
                f"bs-pack member schema mismatch for {prefix + role}: {entry}"
            )
        path = (root / str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"bs-pack member escapes model root: {path}") from exc
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"bs-pack member size mismatch for {path}: expected {expected_bytes}"
            )
        if role == "codebooks":
            opened["codebook"] = np.memmap(
                path, dtype="<f2", mode="r", shape=(k, d)
            )
        elif role == "codes":
            opened["codes"] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(experts, code_bytes_per_expert),
            )
            opened["code_bits"] = code_bits
        else:
            opened["scales"] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(experts, rows, columns // scale_group_size),
            )
        opened[f"{role}_path"] = path
    return opened


def _decode_bs_pack_expert(
    torch: Any,
    *,
    codebook: np.ndarray,
    packed_codes: np.ndarray,
    scales: np.ndarray,
    rows: int,
    columns: int,
    d: int,
    scale_group_size: int,
    device: str,
    dtype: Any,
    code_bits: int = 12,
) -> Any:
    """Reconstruct one dense expert directly from mmap-backed bs-pack members."""

    expected_codes = rows * (columns // d)
    decoded = _unpack_le_values(
        packed_codes, bits=code_bits, value_count=expected_codes
    )
    if decoded.size != expected_codes:
        raise RuntimeError(
            f"packed expert code count mismatch: {decoded.size} != {expected_codes}"
        )
    codes_tensor = torch.tensor(decoded.astype(np.int64, copy=False), device=device)
    codebook_tensor = torch.tensor(np.asarray(codebook), device=device).float()
    scales_tensor = torch.tensor(np.asarray(scales), device=device).float()
    scale_columns = torch.exp2(scales_tensor - 127.0).repeat_interleave(
        scale_group_size, dim=1
    )
    if tuple(scale_columns.shape) != (rows, columns):
        raise RuntimeError(
            "packed expert scale shape mismatch: "
            f"{tuple(scale_columns.shape)} != {(rows, columns)}"
        )
    return (
        codebook_tensor[codes_tensor].reshape(rows, columns) * scale_columns
    ).to(dtype)


class _Activation:
    def __init__(self, hidden: Any, input_ids: Any) -> None:
        self.hidden = hidden
        self.input_ids = input_ids


class DeepseekV4D4Runtime:
    """HF DeepSeek-V4 runtime that keeps only one D4 transformer layer resident."""

    API_VERSION = 1

    def __init__(self, *, model_root: str | Path, parameters: dict[str, Any]) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        self.device = os.environ.get("BANANA_SMASHER_DEVICE", "cuda")
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DeepSeek-V4 D4 layerwise runtime requires CUDA")
        self.model_root = Path(model_root).expanduser().resolve()
        planes_value = os.environ.get("BANANA_SMASHER_D4_PLANES_DIR")
        self.planes_dir = (
            Path(planes_value).expanduser().resolve()
            if planes_value
            else self.model_root / "planes"
        )
        self.positions = int(parameters["positions"])
        self.config = AutoConfig.from_pretrained(self.model_root, trust_remote_code=False)
        index_path = self.model_root / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(
                self.config,
                attn_implementation="eager",
            )
        self.model.eval()
        self._bytes_read = 0
        self._counted_paths: set[Path] = {index_path}
        self._bytes_read += index_path.stat().st_size
        self._d4_k = 4096
        self._d4_d = 4
        self._d4_scale_group_size = 32
        self._base_allocated = int(torch.cuda.memory_allocated())
        self._peak_resident = 0
        self._stage_active = False

    def _record_path(self, path: Path) -> None:
        if path not in self._counted_paths:
            self._counted_paths.add(path)
            self._bytes_read += path.stat().st_size

    def _get_tensor(self, name: str) -> Any:
        from safetensors import safe_open

        path = self.model_root / self.weight_map[name]
        self._record_path(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def _resident_now(self) -> int:
        value = max(0, int(self.torch.cuda.memory_allocated()) - self._base_allocated)
        self._peak_resident = max(self._peak_resident, value)
        return value if self._stage_active else 0

    def _begin_stage(self) -> None:
        if self._stage_active:
            raise RuntimeError("DeepSeek-V4 D4 runtime stages cannot overlap")
        self._stage_active = True

    def _release(self) -> None:
        gc.collect()
        self.synchronize()
        self.torch.cuda.empty_cache()
        self._resident_now()

    def _fp8(self, weight_name: str, scale_name: str) -> Any:
        torch = self.torch
        weight = self._get_tensor(weight_name).to(self.device)
        scale = self._get_tensor(scale_name).to(self.device)
        scale = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
        rows, columns = weight.shape
        scale = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :columns]
        return (weight.to(torch.float32) * scale).to(torch.bfloat16)

    def _ensure_materialization_memory(self, required: int, layer: int) -> None:
        free, _ = self.torch.cuda.mem_get_info()
        if free - (4 << 30) < required:
            raise RuntimeError(
                f"layer {layer}: insufficient CUDA memory for D4 materialization: "
                f"free={free}, required_plus_guard={required + (4 << 30)}"
            )

    def _load_vq3u_experts(self, layer: int) -> tuple[Any, Any]:
        path = self.planes_dir / f"vq3u_layer_{layer:03d}.pt"
        if path.is_file():
            return self._load_vq3u_pt_experts(layer)
        return self._load_bs_pack_experts(layer)

    def _load_vq3u_pt_experts(self, layer: int) -> tuple[Any, Any]:
        torch = self.torch
        path = self.planes_dir / f"vq3u_layer_{layer:03d}.pt"
        self._record_path(path)
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        meta = data.get("meta", {})
        k = int(meta.get("k", int(data["cb13"].shape[0])))
        d = int(meta.get("d", int(data["cb13"].shape[1])))
        if k not in (2048, 4096, 8192) or d != 4:
            raise RuntimeError(f"layer {layer}: unsupported VQ3U metadata {meta}")
        if data["codes13"].dtype != torch.int16 or data["codes2"].dtype != torch.int16:
            raise RuntimeError(f"layer {layer}: VQ3U codes must use int16 storage")

        required = (256 * 4096 * 4096 + 256 * 4096 * 2048) * 2
        self._ensure_materialization_memory(required, layer)
        gate_up = torch.empty(
            256, 4096, 4096, dtype=torch.bfloat16, device=self.device
        )
        down = torch.empty(
            256, 4096, 2048, dtype=torch.bfloat16, device=self.device
        )
        cb13 = data["cb13"].to(self.device).float()
        cb2 = data["cb2"].to(self.device).float()
        for expert_id in range(256):
            for key, destination in (("13", gate_up), ("2", down)):
                codes = data[f"codes{key}"][expert_id].to(self.device)
                scales = data[f"sc{key}"][expert_id].to(self.device)
                codebook = cb13 if key == "13" else cb2
                scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(
                    32, dim=1
                )
                destination[expert_id] = (
                    codebook[codes.long()].reshape(codes.shape[0], -1) * scale_columns
                ).to(torch.bfloat16)
                del codes, scales, scale_columns
        del cb13, cb2, data
        return gate_up, down

    def _load_bs_pack_experts(self, layer: int) -> tuple[Any, Any]:
        torch = self.torch
        manifest_path = self.model_root / "BANANA_PACK_MANIFEST.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"layer {layer}: neither compatibility plane nor bound bs-pack "
                "manifest exists"
            )
        self._record_path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        tensor_index = manifest.get("tensor_index")
        if not isinstance(tensor_index, dict):
            raise RuntimeError("bs-pack manifest has no tensor_index object")
        layer_prefix = f"layers.{layer}.truevq_d4.d4_k"
        packed_tiers = {
            int(key[len(layer_prefix) :].split(".", 1)[0])
            for key in tensor_index
            if key.startswith(layer_prefix) and ".codebooks" in key
        }
        if len(packed_tiers) != 1:
            raise RuntimeError(
                f"layer {layer}: expected one bound D4 subtier, got {sorted(packed_tiers)}"
            )
        d4_k = packed_tiers.pop()
        experts = int(self.config.n_routed_experts)
        rows = int(self.config.hidden_size)
        intermediate = int(self.config.moe_intermediate_size)
        gate_columns = intermediate * 2
        down_columns = intermediate
        geometry = {"fused13": gate_columns, "down": down_columns}
        projections: dict[str, dict[str, Any]] = {}
        for projection, columns in geometry.items():
            opened = _open_bs_pack_projection(
                self.model_root,
                manifest,
                layer=layer,
                projection=projection,
                experts=experts,
                rows=rows,
                columns=columns,
                k=d4_k,
                d=self._d4_d,
                scale_group_size=self._d4_scale_group_size,
            )
            for role in ("codebooks", "codes", "scales"):
                self._record_path(Path(opened[f"{role}_path"]))
            projections[projection] = opened

        required = experts * rows * (gate_columns + down_columns) * 2
        self._ensure_materialization_memory(required, layer)
        gate_up = torch.empty(
            experts,
            rows,
            gate_columns,
            dtype=torch.bfloat16,
            device=self.device,
        )
        down = torch.empty(
            experts,
            rows,
            down_columns,
            dtype=torch.bfloat16,
            device=self.device,
        )
        for expert_id in range(experts):
            for projection, destination, columns in (
                ("fused13", gate_up, gate_columns),
                ("down", down, down_columns),
            ):
                opened = projections[projection]
                destination[expert_id] = _decode_bs_pack_expert(
                    torch,
                    codebook=opened["codebook"],
                    packed_codes=opened["codes"][expert_id],
                    scales=opened["scales"][expert_id],
                    rows=rows,
                    columns=columns,
                    d=self._d4_d,
                    scale_group_size=self._d4_scale_group_size,
                    device=self.device,
                    dtype=torch.bfloat16,
                    code_bits=opened["code_bits"],
                )
        del projections
        return gate_up, down

    def _build_layer_state(self, layer: int) -> dict[str, Any]:
        torch = self.torch
        prefix = f"layers.{layer}."
        keys = [key for key in self.weight_map if key.startswith(prefix)]
        consumed: set[str] = set()
        state: dict[str, Any] = {}

        def tensor(name: str) -> Any:
            key = prefix + name
            consumed.add(key)
            return self._get_tensor(key)

        def has(name: str) -> bool:
            return prefix + name in self.weight_map

        def fp8(name: str) -> Any:
            weight_name = prefix + name + ".weight"
            scale_name = prefix + name + ".scale"
            consumed.update((weight_name, scale_name))
            return self._fp8(weight_name, scale_name)

        def bf16(name: str) -> Any:
            return tensor(name).to(self.device).to(torch.bfloat16)

        def float32(name: str) -> Any:
            return tensor(name).to(self.device).to(torch.float32)

        state["self_attn.q_a_proj.weight"] = fp8("attn.wq_a")
        state["self_attn.q_b_proj.weight"] = fp8("attn.wq_b")
        state["self_attn.kv_proj.weight"] = fp8("attn.wkv")
        state["self_attn.o_a_proj.weight"] = fp8("attn.wo_a")
        state["self_attn.o_b_proj.weight"] = fp8("attn.wo_b")
        state["self_attn.sinks"] = float32("attn.attn_sink")
        state["self_attn.q_a_norm.weight"] = bf16("attn.q_norm.weight")
        state["self_attn.kv_norm.weight"] = bf16("attn.kv_norm.weight")
        state["input_layernorm.weight"] = bf16("attn_norm.weight")
        state["post_attention_layernorm.weight"] = bf16("ffn_norm.weight")

        if has("attn.compressor.wkv.weight"):
            state["self_attn.compressor.position_bias"] = float32(
                "attn.compressor.ape"
            )
            state["self_attn.compressor.kv_norm.weight"] = bf16(
                "attn.compressor.norm.weight"
            )
            state["self_attn.compressor.kv_proj.weight"] = bf16(
                "attn.compressor.wkv.weight"
            )
            state["self_attn.compressor.gate_proj.weight"] = bf16(
                "attn.compressor.wgate.weight"
            )
        if has("attn.indexer.wq_b.weight"):
            indexer = "self_attn.compressor.indexer."
            state[indexer + "position_bias"] = float32(
                "attn.indexer.compressor.ape"
            )
            state[indexer + "kv_norm.weight"] = bf16(
                "attn.indexer.compressor.norm.weight"
            )
            state[indexer + "kv_proj.weight"] = bf16(
                "attn.indexer.compressor.wkv.weight"
            )
            state[indexer + "gate_proj.weight"] = bf16(
                "attn.indexer.compressor.wgate.weight"
            )
            state[indexer + "q_b_proj.weight"] = fp8("attn.indexer.wq_b")
            state[indexer + "scorer.weights_proj.weight"] = bf16(
                "attn.indexer.weights_proj.weight"
            )

        state["mlp.gate.weight"] = bf16("ffn.gate.weight")
        if has("ffn.gate.tid2eid"):
            state["mlp.gate.tid2eid"] = tensor("ffn.gate.tid2eid").to(
                self.device
            )
        if has("ffn.gate.bias"):
            state["mlp.gate.e_score_correction_bias"] = float32("ffn.gate.bias")
        state["attn_hc.fn"] = float32("hc_attn_fn")
        state["attn_hc.base"] = float32("hc_attn_base")
        state["attn_hc.scale"] = float32("hc_attn_scale")
        state["ffn_hc.fn"] = float32("hc_ffn_fn")
        state["ffn_hc.base"] = float32("hc_ffn_base")
        state["ffn_hc.scale"] = float32("hc_ffn_scale")
        state["mlp.shared_experts.gate_proj.weight"] = fp8(
            "ffn.shared_experts.w1"
        )
        state["mlp.shared_experts.up_proj.weight"] = fp8(
            "ffn.shared_experts.w3"
        )
        state["mlp.shared_experts.down_proj.weight"] = fp8(
            "ffn.shared_experts.w2"
        )
        gate_up, down = self._load_vq3u_experts(layer)
        state["mlp.experts.gate_up_proj"] = gate_up
        state["mlp.experts.down_proj"] = down
        consumed.update(key for key in keys if ".ffn.experts." in key)
        missed = set(keys) - consumed
        if missed:
            raise RuntimeError(
                f"layer {layer}: unconsumed checkpoint keys: {sorted(missed)[:8]}"
            )
        return state

    def _materialize_layer(self, layer: int, state: dict[str, Any]) -> Any:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding,
        )

        block = self.model.model.layers[layer]
        _, unexpected = block.load_state_dict(state, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"layer {layer}: unexpected state keys {unexpected[:8]}")
        for name, module in list(block.named_modules()):
            if isinstance(module, DeepseekV4RotaryEmbedding):
                parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
                setattr(
                    parent,
                    name.rsplit(".", 1)[-1],
                    DeepseekV4RotaryEmbedding(self.config).to(self.device),
                )
        meta = [name for name, value in block.named_parameters() if value.is_meta]
        meta += [name for name, value in block.named_buffers() if value.is_meta]
        if meta:
            raise RuntimeError(f"layer {layer}: tensors remain on meta device: {meta[:8]}")
        return block

    def _dematerialize(self, module: Any) -> None:
        torch = self.torch
        for child in module.modules():
            for name, parameter in list(child._parameters.items()):
                if parameter is not None:
                    child._parameters[name] = torch.nn.Parameter(
                        torch.empty(
                            parameter.shape, device="meta", dtype=parameter.dtype
                        ),
                        requires_grad=False,
                    )
            for name, buffer in list(child._buffers.items()):
                if buffer is not None:
                    child._buffers[name] = torch.empty(
                        buffer.shape, device="meta", dtype=buffer.dtype
                    )

    @contextmanager
    def initial_stage(self):
        torch = self.torch
        self._begin_stage()
        resources = [
            self._get_tensor("embed.weight").to(self.device).to(torch.bfloat16)
        ]

        def embed(token_ids: list[int], *, window_id: object) -> _Activation:
            del window_id
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            hidden = torch.nn.functional.embedding(ids, resources[0])
            hidden = hidden.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
            self._resident_now()
            return _Activation(hidden, ids)

        try:
            yield embed
        finally:
            resources.clear()
            self._release()
            self._stage_active = False

    @contextmanager
    def layer_stage(self, layer: int):
        torch = self.torch
        self._begin_stage()
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding,
        )

        state = self._build_layer_state(layer)
        resources = [
            self._materialize_layer(layer, state),
            DeepseekV4RotaryEmbedding(self.config).to(self.device),
        ]
        state.clear()
        self._resident_now()

        def forward(activation: _Activation, *, window_id: object) -> _Activation:
            del window_id
            block, rotary = resources
            hidden = activation.hidden.unsqueeze(0)
            input_ids = activation.input_ids.unsqueeze(0)
            position_ids = torch.arange(
                hidden.shape[1], device=self.device
            ).unsqueeze(0)
            base_hidden = hidden[:, :, 0, :]
            position_embeddings = {
                "main": rotary(
                    base_hidden, position_ids=position_ids, layer_type="main"
                ),
                "compress": rotary(
                    base_hidden, position_ids=position_ids, layer_type="compress"
                ),
            }
            cache = DynamicCache(config=self.config)
            attention_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=base_hidden,
                attention_mask=None,
                past_key_values=cache,
                position_ids=position_ids,
            )
            with torch.no_grad():
                output = block(
                    hidden,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    past_key_values=cache,
                )
            self._resident_now()
            return _Activation(output.squeeze(0), activation.input_ids)

        try:
            yield forward
        finally:
            while resources:
                self._dematerialize(resources.pop())
            self._release()
            self._stage_active = False

    @contextmanager
    def terminal_stage(self):
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

        def score(
            activation: _Activation,
            support_token_ids: Any,
            *,
            window_id: object,
        ) -> dict[str, Any]:
            del window_id
            with torch.no_grad():
                hidden = model.model.norm(
                    model.model.hc_head(activation.hidden.unsqueeze(0))
                ).squeeze(0)
                hidden = hidden[: len(support_token_ids)]
                gathered: list[Any] = []
                argmax: list[Any] = []
                for start in range(0, hidden.shape[0], 128):
                    logits = model.lm_head(
                        hidden[start : start + 128].to(torch.bfloat16)
                    ).float()
                    support = torch.as_tensor(
                        support_token_ids[start : start + logits.shape[0]],
                        dtype=torch.long,
                        device=self.device,
                    )
                    q_lp, q_argmax = _full_vocab_support_logprob(logits, support)
                    gathered.append(q_lp.cpu())
                    argmax.append(q_argmax.cpu())
                    del logits, support
            self._resident_now()
            return {
                "q_lp_at_ref": torch.cat(gathered),
                "q_argmax": torch.cat(argmax),
            }

        try:
            yield score
        finally:
            while resources:
                self._dematerialize(resources.pop())
            self._release()
            self._stage_active = False

    def export_activation(self, activation: _Activation) -> np.ndarray:
        torch = self.torch
        hidden = activation.hidden.detach().to("cpu").contiguous().view(torch.uint16)
        ids = activation.input_ids.detach().to("cpu").to(torch.int64)
        low = (ids & 0xFFFF).to(torch.uint16)
        high = ((ids >> 16) & 0xFFFF).to(torch.uint16)
        return np.concatenate(
            (hidden.numpy().reshape(-1), low.numpy(), high.numpy())
        )

    def import_activation(self, activation: np.ndarray) -> _Activation:
        torch = self.torch
        packed = np.asarray(activation, dtype=np.uint16).reshape(-1)
        token_count = self.positions
        hidden_count = token_count * int(self.config.hc_mult) * int(
            self.config.hidden_size
        )
        if packed.size != hidden_count + 2 * token_count:
            raise ValueError(
                f"invalid packed activation size {packed.size}; expected "
                f"{hidden_count + 2 * token_count}"
            )
        hidden = (
            torch.from_numpy(packed[:hidden_count].copy())
            .view(torch.bfloat16)
            .reshape(token_count, self.config.hc_mult, self.config.hidden_size)
            .to(self.device)
        )
        low = torch.from_numpy(packed[hidden_count : hidden_count + token_count].copy()).to(
            torch.int64
        )
        high = torch.from_numpy(packed[hidden_count + token_count :].copy()).to(
            torch.int64
        )
        input_ids = (low | (high << 16)).to(self.device)
        self._resident_now()
        return _Activation(hidden, input_ids)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def resident_bytes(self) -> int:
        return self._resident_now()

    def peak_resident_bytes(self) -> int:
        self._resident_now()
        return self._peak_resident

    def bytes_read(self) -> int:
        return self._bytes_read
