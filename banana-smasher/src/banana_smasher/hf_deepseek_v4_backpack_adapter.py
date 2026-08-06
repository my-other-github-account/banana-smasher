from __future__ import annotations

import gc
import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


from .hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime


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
    shifts = torch.arange(0, 16, dtype=torch.int32, device=blocked.device).reshape(1, 1, -1)
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
        self._cache_drop_paths: set[Path] = set()
        super().__init__(model_root=model_root, parameters=parameters)
        binding = parameters.get("backpack_runtime")
        required = {
            "basis_sha256",
            "virtual_manifest",
            "materialization_index",
            "qtip2_root_map",
            "qtip3_root_map",
        }
        if not isinstance(binding, Mapping) or set(binding) != required:
            raise ValueError(f"backpack_runtime requires {sorted(required)}")
        self.basis_sha256 = str(binding["basis_sha256"])
        self.virtual_manifest_path = Path(str(binding["virtual_manifest"])).resolve()
        self.materialization_index_path = Path(str(binding["materialization_index"])).resolve()
        manifest = json.loads(self.virtual_manifest_path.read_text())
        if manifest.get("basis_sha256") != self.basis_sha256:
            raise ValueError("virtual Backpack basis mismatch")
        index_binding = manifest.get("materialization_index")
        if not isinstance(index_binding, Mapping):
            raise ValueError("virtual Backpack materialization index binding is missing")
        import hashlib

        index_sha = hashlib.sha256(self.materialization_index_path.read_bytes()).hexdigest()
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
        self.root_maps: dict[str, dict[str, str]] = {}
        for source_key in ("qtip2", "qtip3"):
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
        self._record_path(self.virtual_manifest_path)
        self._record_path(self.materialization_index_path)

    def _record_path(self, path: Path) -> None:
        super()._record_path(path)
        self._cache_drop_paths.add(path)

    def _drop_recorded_file_cache(self) -> None:
        """Release clean pages from prior expert slices before the CUDA guard check."""

        self.synchronize()
        self.torch.cuda.empty_cache()
        gc.collect()
        try:
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim")
        except (AttributeError, OSError):
            malloc_trim = None
        if malloc_trim is not None:
            malloc_trim(0)
        advise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        paths, self._cache_drop_paths = self._cache_drop_paths, set()
        if advise is None or dontneed is None:
            return
        for path in paths:
            try:
                descriptor = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                advise(descriptor, 0, 0, dontneed)
            except OSError:
                pass
            finally:
                os.close(descriptor)

    def _decode_qtip(self, source_key: str, layer: int, expert: int, projection: str) -> Any:
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
        self._record_path(artifact_path)
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        geometry = payload.get("geometry")
        expected_k = 2 if source_key == "qtip2" else 3
        expected_schema = (
            {"banana-smasher-qtip2-public-unit-v1", "banana-smasher-qtip-unit-v1"}
            if source_key == "qtip2"
            else {"ds4-qtip-hyb-bounded36-unit-v1", "banana-smasher-qtip-unit-v1"}
        )
        if (
            payload.get("schema") not in expected_schema
            or not isinstance(geometry, Mapping)
            or int(geometry.get("L", -1)) != 16
            or int(geometry.get("K", -1)) != expected_k
            or int(geometry.get("V", -1)) != 2
            or int(geometry.get("tlut_bits", -1)) != 9
            or geometry.get("decode_mode") != "quantlut_sym"
        ):
            raise ValueError(f"QTIP unit payload identity mismatch: {artifact_path}")
        rows, columns = [int(value) for value in payload["shape"]]
        expected_shape = (4096, 2048) if projection == "down" else (4096, 4096)
        if (rows, columns) != expected_shape:
            raise ValueError(f"QTIP unit shape mismatch: {artifact_path}")
        device = self.device
        tlut = payload["tlut"].float().to(device)
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

    def _phase_state(self, layer: int, phase: str) -> dict[str, Any]:
        """Build only the attention or MLP half of one layer for bounded residency."""

        torch = self.torch
        prefix = f"layers.{layer}."

        def tensor(name: str) -> Any:
            return self._get_tensor(prefix + name)

        def has(name: str) -> bool:
            return prefix + name in self.weight_map

        def fp8(name: str) -> Any:
            return self._fp8(prefix + name + ".weight", prefix + name + ".scale")

        def bf16(name: str) -> Any:
            return tensor(name).to(self.device).to(torch.bfloat16)

        def float32(name: str) -> Any:
            return tensor(name).to(self.device).to(torch.float32)

        state: dict[str, Any] = {}
        if phase == "attention":
            state["self_attn.q_a_proj.weight"] = fp8("attn.wq_a")
            state["self_attn.q_b_proj.weight"] = fp8("attn.wq_b")
            state["self_attn.kv_proj.weight"] = fp8("attn.wkv")
            state["self_attn.o_a_proj.weight"] = fp8("attn.wo_a")
            state["self_attn.o_b_proj.weight"] = fp8("attn.wo_b")
            state["self_attn.sinks"] = float32("attn.attn_sink")
            state["self_attn.q_a_norm.weight"] = bf16("attn.q_norm.weight")
            state["self_attn.kv_norm.weight"] = bf16("attn.kv_norm.weight")
            state["input_layernorm.weight"] = bf16("attn_norm.weight")
            state["attn_hc.fn"] = float32("hc_attn_fn")
            state["attn_hc.base"] = float32("hc_attn_base")
            state["attn_hc.scale"] = float32("hc_attn_scale")
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
            return state
        if phase not in {"mlp", "mlp-static"}:
            raise ValueError(f"unsupported Backpack layer phase: {phase}")
        state["post_attention_layernorm.weight"] = bf16("ffn_norm.weight")
        state["mlp.gate.weight"] = bf16("ffn.gate.weight")
        if has("ffn.gate.tid2eid"):
            state["mlp.gate.tid2eid"] = tensor("ffn.gate.tid2eid").to(self.device)
        if has("ffn.gate.bias"):
            state["mlp.gate.e_score_correction_bias"] = float32("ffn.gate.bias")
        state["ffn_hc.fn"] = float32("hc_ffn_fn")
        state["ffn_hc.base"] = float32("hc_ffn_base")
        state["ffn_hc.scale"] = float32("hc_ffn_scale")
        state["mlp.shared_experts.gate_proj.weight"] = fp8("ffn.shared_experts.w1")
        state["mlp.shared_experts.up_proj.weight"] = fp8("ffn.shared_experts.w3")
        state["mlp.shared_experts.down_proj.weight"] = fp8("ffn.shared_experts.w2")
        if phase == "mlp":
            gate_up, down = self._load_vq3u_experts(layer)
            state["mlp.experts.gate_up_proj"] = gate_up
            state["mlp.experts.down_proj"] = down
        return state

    def _materialize_phase(self, layer: int, phase: str) -> Any:
        state = self._phase_state(layer, phase)
        block = self.model.model.layers[layer]
        _, unexpected = block.load_state_dict(state, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"layer {layer} {phase}: unexpected state keys {unexpected[:8]}")
        state.clear()
        return block

    @contextmanager
    def attention_stage(self, layer: int):
        """Run the attention half for all windows without routed experts resident."""

        torch = self.torch
        self._begin_stage()
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding,
        )

        block = self._materialize_phase(layer, "attention")
        for name, module in list(block.named_modules()):
            if isinstance(module, DeepseekV4RotaryEmbedding):
                parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
                setattr(
                    parent,
                    name.rsplit(".", 1)[-1],
                    DeepseekV4RotaryEmbedding(self.config).to(self.device),
                )
        rotary = DeepseekV4RotaryEmbedding(self.config).to(self.device)
        self._resident_now()

        def forward(activation: Any, *, window_id: object) -> Any:
            del window_id
            hidden = activation.hidden.unsqueeze(0)
            input_ids = activation.input_ids.unsqueeze(0)
            position_ids = torch.arange(hidden.shape[1], device=self.device).unsqueeze(0)
            base_hidden = hidden[:, :, 0, :]
            position_embeddings = {
                "main": rotary(base_hidden, position_ids=position_ids, layer_type="main"),
                "compress": rotary(base_hidden, position_ids=position_ids, layer_type="compress"),
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
                dtype = hidden.dtype
                post, comb, collapsed = block.attn_hc(hidden)
                attn_output, _ = block.self_attn(
                    block.input_layernorm(collapsed),
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    past_key_values=cache,
                )
                output = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2)
                output = output + torch.matmul(
                    comb.to(dtype).transpose(-1, -2), hidden
                )
            self._resident_now()
            return type(activation)(output.squeeze(0), activation.input_ids)

        try:
            yield forward
        finally:
            self._dematerialize(block)
            self._dematerialize(rotary)
            self._release()
            self._stage_active = False

    @contextmanager
    def mlp_stage(self, layer: int):
        """Run the MLP half after attention weights have been dematerialized."""

        self._begin_stage()
        block = self._materialize_phase(layer, "mlp")
        self._resident_now()

        def forward(activation: Any, *, window_id: object) -> Any:
            del window_id
            hidden = activation.hidden.unsqueeze(0)
            input_ids = activation.input_ids.unsqueeze(0)
            with self.torch.no_grad():
                dtype = hidden.dtype
                post, comb, collapsed = block.ffn_hc(hidden)
                mlp_output = block.mlp(
                    block.post_attention_layernorm(collapsed), input_ids=input_ids
                )
                output = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2)
                output = output + self.torch.matmul(
                    comb.to(dtype).transpose(-1, -2), hidden
                )
            self._resident_now()
            return type(activation)(output.squeeze(0), activation.input_ids)

        try:
            yield forward
        finally:
            self._dematerialize(block)
            self._release()
            self._stage_active = False

    @contextmanager
    def mlp_chunk_stage(self, layer: int, expert_start: int, expert_stop: int):
        """Run one ordered routed-expert chunk with a bounded CUDA footprint."""

        if not (0 <= expert_start < expert_stop <= 256):
            raise ValueError("invalid Backpack expert chunk")
        torch = self.torch
        self._begin_stage()
        block = self._materialize_phase(layer, "mlp-static")
        expert_weights = list(
            self._load_vq3u_expert_chunk(layer, expert_start, expert_stop)
        )
        self._resident_now()

        def forward(
            activation: Any,
            *,
            window_id: object,
            accumulated: Any | None = None,
            include_shared_residual: bool = False,
        ) -> Any:
            del window_id
            hidden = activation.hidden.unsqueeze(0)
            input_ids = activation.input_ids.unsqueeze(0)
            if accumulated is not None and not torch.equal(
                activation.input_ids, accumulated.input_ids
            ):
                raise ValueError("Backpack MLP chunk input-id mismatch")
            with torch.no_grad():
                dtype = hidden.dtype
                post, comb, collapsed = block.ffn_hc(hidden)
                normalized = block.post_attention_layernorm(collapsed)
                batch, sequence, hidden_dim = normalized.shape
                flat = normalized.view(-1, hidden_dim)
                if block.mlp.is_hash:
                    _, weights, indices = block.mlp.gate(normalized, input_ids)
                else:
                    _, weights, indices = block.mlp.gate(normalized)
                routed = torch.zeros_like(flat)
                gate_up, down = expert_weights
                for expert in range(expert_start, expert_stop):
                    token_idx, top_k_pos = torch.where(indices == expert)
                    if token_idx.numel() == 0:
                        continue
                    current = torch.nn.functional.linear(
                        flat[token_idx], gate_up[expert - expert_start]
                    )
                    current = block.mlp.experts._apply_gate(current)
                    current = torch.nn.functional.linear(
                        current, down[expert - expert_start]
                    ) * weights[token_idx, top_k_pos, None]
                    routed.index_add_(0, token_idx, current.to(routed.dtype))
                mlp_output = routed.view(batch, sequence, hidden_dim)
                contribution = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2)
                if include_shared_residual:
                    shared = block.mlp.shared_experts(normalized)
                    contribution = contribution + (
                        post.to(dtype).unsqueeze(-1) * shared.unsqueeze(-2)
                    )
                    contribution = contribution + torch.matmul(
                        comb.to(dtype).transpose(-1, -2), hidden
                    )
                if accumulated is not None:
                    contribution = contribution + accumulated.hidden.unsqueeze(0)
            self._resident_now()
            return type(activation)(contribution.squeeze(0), activation.input_ids)

        try:
            yield forward
        finally:
            expert_weights.clear()
            self._dematerialize(block)
            self._release()
            self._stage_active = False

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

        def score(
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
            yield score
        finally:
            while resources:
                self._dematerialize(resources.pop())
            self._release()
            self._stage_active = False

    def _load_vq3u_experts(self, layer: int) -> tuple[Any, Any]:
        torch = self.torch
        required = (256 * 4096 * 4096 + 256 * 4096 * 2048) * 2
        free, _ = torch.cuda.mem_get_info()
        if free - (4 << 30) < required:
            raise RuntimeError(
                f"layer {layer}: insufficient CUDA memory for mixed Backpack materialization: "
                f"free={free}, required_plus_guard={required + (4 << 30)}"
            )
        gate_up = torch.empty(256, 4096, 4096, dtype=torch.bfloat16, device=self.device)
        down = torch.empty(256, 4096, 2048, dtype=torch.bfloat16, device=self.device)
        rows = sorted(
            self.rows_by_layer[layer], key=lambda row: (int(row["expert"]), str(row["projection"]))
        )
        seen: set[tuple[int, str]] = set()
        for position, row in enumerate(rows, 1):
            expert = int(row["expert"])
            projection = str(row["projection"])
            source_key = str(row["source_key"])
            key = (expert, projection)
            if key in seen or source_key not in {"native_mxfp4", "qtip2", "qtip3"}:
                raise ValueError(f"invalid mixed Backpack cell row: {row}")
            seen.add(key)
            if source_key == "native_mxfp4":
                value = self._native(layer, expert, projection)
            else:
                value = self._decode_qtip(source_key, layer, expert, projection)
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
        if len(seen) != 512:
            raise ValueError(f"layer {layer}: mixed Backpack cell coverage mismatch")
        gc.collect()
        return gate_up, down

    def _load_vq3u_expert_chunk(
        self, layer: int, expert_start: int, expert_stop: int
    ) -> tuple[Any, Any]:
        """Decode an ordered expert slice while preserving the 4 GiB guard."""

        torch = self.torch
        count = expert_stop - expert_start
        required = (count * 4096 * 4096 + count * 4096 * 2048) * 2
        self._drop_recorded_file_cache()
        free, _ = torch.cuda.mem_get_info()
        if free - (4 << 30) < required:
            raise RuntimeError(
                f"layer {layer}: insufficient CUDA memory for mixed Backpack expert chunk "
                f"[{expert_start},{expert_stop}): free={free}, "
                f"required_plus_guard={required + (4 << 30)}"
            )
        gate_up = torch.empty(
            count, 4096, 4096, dtype=torch.bfloat16, device=self.device
        )
        down = torch.empty(
            count, 4096, 2048, dtype=torch.bfloat16, device=self.device
        )
        rows = sorted(
            (
                row
                for row in self.rows_by_layer[layer]
                if expert_start <= int(row["expert"]) < expert_stop
            ),
            key=lambda row: (int(row["expert"]), str(row["projection"])),
        )
        seen: set[tuple[int, str]] = set()
        for position, row in enumerate(rows, 1):
            expert = int(row["expert"])
            projection = str(row["projection"])
            source_key = str(row["source_key"])
            key = (expert, projection)
            if key in seen or source_key not in {"native_mxfp4", "qtip2", "qtip3"}:
                raise ValueError(f"invalid mixed Backpack cell row: {row}")
            seen.add(key)
            if source_key == "native_mxfp4":
                value = self._native(layer, expert, projection)
            else:
                value = self._decode_qtip(source_key, layer, expert, projection)
            destination = down if projection == "down" else gate_up
            destination_index = expert - expert_start
            if value.shape != destination[destination_index].shape:
                raise ValueError(
                    f"mixed Backpack cell shape mismatch: cell={row['cell_id']} "
                    f"source={source_key} value={tuple(value.shape)} "
                    f"destination={tuple(destination[destination_index].shape)}"
                )
            destination[destination_index].copy_(value)
            del value
            if position % 64 == 0:
                print(
                    f"BACKPACK_LAYER_CHUNK_PROGRESS layer={layer} "
                    f"experts={expert_start}:{expert_stop} cells={position}/{count * 2}",
                    flush=True,
                )
                self.synchronize()
        if len(seen) != count * 2:
            raise ValueError(
                f"layer {layer}: mixed Backpack expert chunk coverage mismatch"
            )
        gc.collect()
        return gate_up, down
