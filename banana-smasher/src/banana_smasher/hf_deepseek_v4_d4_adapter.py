from __future__ import annotations

import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


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

    def _load_vq3u_experts(self, layer: int) -> tuple[Any, Any]:
        torch = self.torch
        path = self.planes_dir / f"vq3u_layer_{layer:03d}.pt"
        self._record_path(path)
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        meta = data.get("meta", {})
        k = int(meta.get("k", int(data["cb13"].shape[0])))
        d = int(meta.get("d", int(data["cb13"].shape[1])))
        if k not in (4096, 8192) or d != 4:
            raise RuntimeError(f"layer {layer}: unsupported VQ3U metadata {meta}")
        if data["codes13"].dtype != torch.int16 or data["codes2"].dtype != torch.int16:
            raise RuntimeError(f"layer {layer}: VQ3U codes must use int16 storage")

        required = (256 * 4096 * 4096 + 256 * 4096 * 2048) * 2
        free, _ = torch.cuda.mem_get_info()
        if free - (4 << 30) < required:
            raise RuntimeError(
                f"layer {layer}: insufficient CUDA memory for D4 materialization: "
                f"free={free}, required_plus_guard={required + (4 << 30)}"
            )
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
            support_token_ids: list[list[int]],
            *,
            window_id: object,
        ) -> dict[str, Any]:
            del window_id
            with torch.no_grad():
                hidden = model.model.norm(
                    model.model.hc_head(activation.hidden.unsqueeze(0))
                ).squeeze(0)
                pairs: list[list[float]] = []
                top1: list[int] = []
                for start in range(0, hidden.shape[0], 128):
                    logits = model.lm_head(
                        hidden[start : start + 128].to(torch.bfloat16)
                    ).float()
                    support = torch.tensor(
                        support_token_ids[start : start + logits.shape[0]],
                        dtype=torch.long,
                        device=self.device,
                    )
                    pairs.extend(logits.gather(1, support).cpu().tolist())
                    top1.extend(logits.argmax(-1).cpu().tolist())
                    del logits, support
            self._resident_now()
            return {"logits": pairs, "top1_token_ids": top1}

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
