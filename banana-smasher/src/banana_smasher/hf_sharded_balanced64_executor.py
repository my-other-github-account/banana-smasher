"""Package-owned layer-streamed HF executor for BALANCED64.

The executor binds the installed native Transformers implementation selected by
nested HF config semantics.  It never loads model code from the model tree or a
network endpoint.  Text windows are executed a layer at a time; vision tensors
remain a separate, unexecuted inventory for the fixed integer-token benchmark.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .qtip1 import EncodedQtip, QtipGeometry, decode_qtip, gaussian_tlut


_MIN_TRANSFORMERS = (5, 16, 0)
_TRANSFORMERS_COMMIT = "b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309"
_TOKENIZERS_VERSION = "0.23.1"
_Q2_GEOMETRY = {"L": 16, "K": 2, "V": 2, "tlut_bits": 9, "decode_mode": "quantlut_sym"}
_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def _version_tuple(text: str) -> tuple[int, ...]:
    values = []
    for part in text.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        values.append(int(digits))
    return tuple(values)


def _require_torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("sharded HF BALANCED64 execution requires torch") from exc


def require_hf_runtime() -> tuple[Any, Any]:
    """Return torch/transformers or fail with the exact public dependency gate."""

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "sharded HF BALANCED64 execution requires torch and transformers>=5.16; "
            "install the documented solve runtime"
        ) from exc
    version = getattr(transformers, "__version__", "0")
    if _version_tuple(version) < _MIN_TRANSFORMERS:
        raise RuntimeError(
            f"sharded HF BALANCED64 execution requires transformers>=5.16; found {version}"
        )
    try:
        transformers_distribution = importlib_metadata.distribution("transformers")
        tokenizers_distribution = importlib_metadata.distribution("tokenizers")
        direct_url = json.loads(transformers_distribution.read_text("direct_url.json") or "null")
        vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, Mapping) else None
        commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
        repository_url = direct_url.get("url") if isinstance(direct_url, Mapping) else None
        vcs = vcs_info.get("vcs") if isinstance(vcs_info, Mapping) else None
    except (importlib_metadata.PackageNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("sharded HF native runtime identity is not readable") from exc
    if (
        repository_url != "https://github.com/huggingface/transformers.git"
        or vcs != "git"
        or commit_id != _TRANSFORMERS_COMMIT
        or tokenizers_distribution.version != _TOKENIZERS_VERSION
    ):
        raise RuntimeError(
            "sharded HF native runtime identity drift: "
            f"transformers_commit={commit_id} tokenizers={tokenizers_distribution.version}"
        )
    return torch, transformers


def _subject_source(subject: Mapping[str, Any]) -> Mapping[str, Any] | None:
    source = subject.get("source")
    if isinstance(source, Mapping):
        return source
    return subject


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def _hybrid_semantics(config: Mapping[str, Any]) -> bool:
    text = _text_config(config)
    if text is None:
        return False
    layers = text.get("num_hidden_layers")
    routed = text.get("n_routed_experts")
    hc_mult = text.get("hc_mult")
    layer_types = text.get("layer_types")
    mlp_types = text.get("mlp_layer_types")
    vocab = text.get("vocab_size")
    architectures = config.get("architectures")
    return (
        isinstance(architectures, list)
        and len(architectures) == 1
        and isinstance(architectures[0], str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", architectures[0]) is not None
        and isinstance(layers, int)
        and not isinstance(layers, bool)
        and layers > 0
        and isinstance(routed, int)
        and not isinstance(routed, bool)
        and routed > 0
        and isinstance(hc_mult, int)
        and not isinstance(hc_mult, bool)
        and hc_mult > 0
        and isinstance(vocab, int)
        and not isinstance(vocab, bool)
        and vocab >= 8192
        and isinstance(layer_types, list)
        and len(layer_types) == layers
        and set(layer_types).issubset({"linear_attention", "deepseek_sparse_attention"})
        and "linear_attention" in layer_types
        and "deepseek_sparse_attention" in layer_types
        and isinstance(mlp_types, list)
        and len(mlp_types) == layers
        and set(mlp_types).issubset({"dense", "sparse"})
        and "sparse" in mlp_types
    )


class PackageHFShardedExecutor:
    """Factory entry point for the package-owned layer-streamed executor."""

    @staticmethod
    def supports(*, subject: Mapping[str, Any], role: str) -> bool:
        if role not in {"teacher", "candidate_pre"}:
            return False
        source = _subject_source(subject)
        root_value = source.get("model_root") if isinstance(source, Mapping) else None
        if not isinstance(root_value, str):
            return False
        try:
            config = json.loads(
                (Path(root_value).expanduser().resolve() / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(config, Mapping) and _hybrid_semantics(config)

    def __new__(cls, **kwargs):
        return LayerStreamedHFSession(**kwargs)


class ArtifactTensorStore:
    """Exact routed-Q2/native-rest tensor loader for one admitted artifact."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.root = Path(artifact["artifact_root"]).expanduser().resolve()
        self.routed = {row["name"]: row for row in artifact["routed_tensors"]}
        self.native = {row["name"]: row for row in artifact["native_tensors"]}
        self.payload_reads = 0
        self.model_reads = 0

    def names(self) -> set[str]:
        return set(self.routed) | set(self.native)

    def _load_array(self, binding: Mapping[str, Any]) -> np.ndarray:
        path = (self.root / binding["path"]).resolve()
        self.payload_reads += 1
        return np.load(path, allow_pickle=False)

    @staticmethod
    def _torch_from_numpy(array: np.ndarray):
        torch = _require_torch()
        if array.dtype == np.dtype("<u2"):
            return torch.from_numpy(array.copy()).view(torch.bfloat16)
        return torch.from_numpy(np.ascontiguousarray(array))

    def tensor(self, name: str):
        if name in self.routed:
            row = self.routed[name]
            geometry = QtipGeometry.from_mapping(row["wire"]["geometry"])
            packed = self._load_array(row["wire"]["trellis"])
            scales = self._load_array(row["wire"]["scales"])
            shape = tuple(int(value) for value in row["shape"])
            encoded = EncodedQtip(
                geometry=geometry,
                shape=shape,
                states=np.empty((0, 0), dtype=np.int32),
                packed=np.asarray(packed, dtype=np.uint16),
                scales=np.asarray(scales, dtype=np.float32),
            )
            tlut = gaussian_tlut(bits=geometry.tlut_bits, columns=geometry.V)
            return self._torch_from_numpy(decode_qtip(encoded, tlut=tlut))
        if name not in self.native:
            raise KeyError(name)
        row = self.native[name]
        dtype_name = row.get("dtype")
        path = (self.root / row["path"]).resolve()
        self.payload_reads += 1
        raw = path.read_bytes()
        shape = tuple(row["shape"])
        if dtype_name == "F8_E4M3":
            encoded = np.frombuffer(raw, dtype=np.uint8)
            bits = np.arange(256, dtype=np.uint16)
            exponent = (bits >> 3) & 0xF
            mantissa = bits & 0x7
            magnitude = np.where(
                exponent == 0,
                np.ldexp(mantissa.astype(np.float32) / 8.0, -6),
                np.ldexp(
                    1.0 + mantissa.astype(np.float32) / 8.0,
                    exponent.astype(int) - 7,
                ),
            ).astype(np.float32)
            magnitude[(exponent == 15) & (mantissa == 7)] = np.nan
            lookup = np.where(bits & 0x80, -magnitude, magnitude).astype(np.float32)
            array = lookup[encoded].reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError(f"exact native FP8 tensor contains non-finite values: {name}")
            return self._torch_from_numpy(array)
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported exact native tensor dtype: {dtype_name}")
        array = np.frombuffer(raw, dtype=_DTYPES[dtype_name]).reshape(shape)
        return self._torch_from_numpy(array.copy())

    def load_many(self, names: Sequence[str]) -> dict[str, Any]:
        return {name: self.tensor(name) for name in names}


class SourceTensorStore:
    """Immutable local safetensors loader grouped by shard working set."""

    def __init__(self, source: Mapping[str, Any]) -> None:
        self.root = Path(source["model_root"]).expanduser().resolve()
        index = json.loads((self.root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map = dict(index["weight_map"])
        self.payload_reads = 0
        self.model_reads = 0

    def names(self) -> set[str]:
        return set(self.weight_map)

    def load_many(self, names: Sequence[str]) -> dict[str, Any]:
        from safetensors import safe_open

        grouped: dict[str, list[str]] = {}
        for name in names:
            grouped.setdefault(self.weight_map[name], []).append(name)
        tensors: dict[str, Any] = {}
        for shard, shard_names in grouped.items():
            self.model_reads += 1
            with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    tensors[name] = handle.get_tensor(name)
        return tensors

    def tensor(self, name: str):
        return self.load_many([name])[name]


def top_support(hidden, weight, *, support_token_ids, support: int = 8192) -> dict[str, np.ndarray]:
    """Project terminal hidden states on full vocab or identical teacher support."""

    torch = _require_torch()
    with torch.no_grad():
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).transpose(0, 1)
        top1 = logits.argmax(dim=-1)
        if support_token_ids is None:
            support_logits, token_ids = torch.topk(logits, support, dim=-1, largest=True, sorted=True)
        else:
            token_ids = torch.as_tensor(support_token_ids, dtype=torch.long, device=logits.device)
            support_logits = logits.gather(1, token_ids)
    return {
        "support_token_ids": token_ids.cpu().numpy().astype(np.int32),
        "support_logits": support_logits.cpu().numpy().astype(np.float32),
        "top1_token_ids": top1.cpu().numpy().astype(np.int32),
    }


class LayerStreamedHFSession:
    """Execute all fixed windows through each native HF text layer exactly once."""

    def __init__(self, *, subject, role, suite_lock, corpus_rows) -> None:
        self.torch, self.transformers = require_hf_runtime()
        self.subject = subject
        self.role = role
        self.suite_lock = suite_lock
        self.corpus_rows = corpus_rows
        source = _subject_source(subject)
        if not isinstance(source, Mapping):
            raise ValueError("package HF executor requires admitted source identity")
        self.source = source
        self.store = SourceTensorStore(source) if role == "teacher" else ArtifactTensorStore(subject)
        self.device = os.environ.get("BANANA_SMASHER_DEVICE", "cuda")
        if self.device != "cuda" or not self.torch.cuda.is_available():
            raise RuntimeError("package HF layer-streamed executor requires CUDA")
        self._ready = False
        self._working_set_loads = 0
        self._outputs: list[dict[str, np.ndarray]] | None = None
        self._model = self._meta_model()

    def _meta_model(self):
        root = Path(self.source["model_root"])
        config = self.transformers.AutoConfig.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        architectures = getattr(config, "architectures", None)
        if (
            not isinstance(architectures, list)
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", architectures[0]) is None
        ):
            raise RuntimeError("HF config must declare exactly one installed architecture")
        model_class = getattr(self.transformers, architectures[0], None)
        pretrained_base = getattr(self.transformers, "PreTrainedModel", None)
        if (
            not isinstance(model_class, type)
            or not isinstance(pretrained_base, type)
            or not issubclass(model_class, pretrained_base)
        ):
            raise RuntimeError(
                f"installed transformers does not provide declared architecture: {architectures[0]}"
            )
        setattr(config, "_attn_implementation", "eager")
        with self.torch.device("meta"):
            model = model_class(config)
        model.eval()
        return model

    def _language_model(self):
        outer = getattr(self._model, "model", None)
        language = getattr(outer, "language_model", None)
        if language is None:
            language = outer
        layers = getattr(language, "layers", None)
        if language is None or layers is None:
            raise RuntimeError("native HF model does not expose a layer-streamable text model")
        return language

    def _prefix_for(self, module: Any) -> str:
        for name, candidate in self._model.named_modules():
            if candidate is module:
                return f"{name}." if name else ""
        raise RuntimeError("cannot bind native HF module to checkpoint prefix")

    @staticmethod
    def _scaled_weight(value: Any, scale: Any | None) -> Any:
        if scale is None:
            return value
        if value.ndim != 2:
            raise RuntimeError("FP8 inverse-scale binding requires a matrix weight")
        value = value.to(dtype=scale.dtype if scale.is_floating_point() else value.dtype).float()
        scale = scale.float()
        if scale.ndim == 0:
            return value * scale
        if scale.shape == value.shape:
            return value * scale
        if scale.ndim == 1:
            if scale.shape[0] == value.shape[0]:
                return value * scale[:, None]
            if scale.shape[0] == value.shape[1]:
                return value * scale[None, :]
        if scale.ndim == 2:
            row_block = math.ceil(value.shape[0] / scale.shape[0])
            column_block = math.ceil(value.shape[1] / scale.shape[1])
            expanded = scale.repeat_interleave(row_block, 0).repeat_interleave(column_block, 1)
            return value * expanded[: value.shape[0], : value.shape[1]]
        raise RuntimeError(
            f"unsupported FP8 inverse-scale geometry: weight={tuple(value.shape)} "
            f"scale={tuple(scale.shape)}"
        )

    @staticmethod
    def _expert_binding(full_name: str, available: set[str], experts: int):
        if full_name.endswith("experts.gate_up_proj"):
            base = full_name.removesuffix(".gate_up_proj")
            projections = ("gate_proj", "up_proj")
            kind = "gate_up"
        elif full_name.endswith("experts.down_proj"):
            base = full_name.removesuffix(".down_proj")
            projections = ("down_proj",)
            kind = "down"
        else:
            return None
        pattern = re.compile(rf"{re.escape(base)}\.(\d+)\.{projections[0]}\.weight\Z")
        observed = sorted(
            int(match.group(1))
            for name in available
            if (match := pattern.fullmatch(name)) is not None
        )
        if observed != list(range(experts)):
            return None
        weights = [
            f"{base}.{expert}.{projection}.weight"
            for expert in observed
            for projection in projections
        ]
        dependencies = []
        for weight in weights:
            dependencies.append(weight)
            scale = f"{weight}_scale_inv"
            if scale in available:
                dependencies.append(scale)
        return kind, observed, projections, dependencies

    @staticmethod
    def _semantic_binding(full_name: str, available: set[str]):
        if full_name.endswith(".self_attn.conv1d.weight"):
            base = full_name.removesuffix("conv1d.weight")
            weights = [f"{base}{axis}_conv1d.weight" for axis in ("q", "k", "v")]
            if all(name in available for name in weights):
                dependencies = []
                for weight in weights:
                    dependencies.append(weight)
                    scale = f"{weight}_scale_inv"
                    if scale in available:
                        dependencies.append(scale)
                return "concat", weights, dependencies
        if ".self_attn.forget_gate." in full_name:
            alias = full_name.replace(".self_attn.forget_gate.", ".self_attn.")
        else:
            match = re.fullmatch(r"(.+)\.(attn_hc|ffn_hc)\.(fn|base|scale)", full_name)
            if match is None:
                return None
            family = "attn" if match.group(2) == "attn_hc" else "ffn"
            alias = f"{match.group(1)}.hc_{family}_{match.group(3)}"
        if alias not in available:
            return None
        scale = f"{alias}_scale_inv"
        return "alias", alias, scale if scale in available else None

    def _materialize(self, module: Any) -> None:
        prefix = self._prefix_for(module)
        local_state = module.state_dict()
        if not local_state:
            return
        available = self.store.names()
        bindings = {}
        dependencies: list[str] = []
        for local_name, target in local_state.items():
            full_name = prefix + local_name
            if full_name in available:
                scale = f"{full_name}_scale_inv"
                binding = ("direct", full_name, scale if scale in available else None)
                dependencies.extend(name for name in binding[1:] if name is not None)
            else:
                binding = self._expert_binding(full_name, available, int(target.shape[0]))
                if binding is None:
                    binding = self._semantic_binding(full_name, available)
                if binding is None:
                    raise RuntimeError(
                        f"checkpoint is missing working-set tensor: prefix={prefix} "
                        f"missing={full_name}"
                    )
                if binding[0] in {"alias", "direct"}:
                    dependencies.extend(name for name in binding[1:] if name is not None)
                else:
                    dependencies.extend(binding[3] if binding[0] in {"gate_up", "down"} else binding[2])
            bindings[local_name] = binding
        loaded = self.store.load_many(list(dict.fromkeys(dependencies)))
        state = {}
        for local_name, target in local_state.items():
            binding = bindings[local_name]
            if binding[0] in {"direct", "alias"}:
                _, weight_name, scale_name = binding
                value = self._scaled_weight(
                    loaded[weight_name], None if scale_name is None else loaded[scale_name]
                )
            elif binding[0] == "concat":
                _, weight_names, _ = binding
                pieces = [
                    self._scaled_weight(
                        loaded[weight_name], loaded.get(f"{weight_name}_scale_inv")
                    )
                    for weight_name in weight_names
                ]
                value = self.torch.cat(pieces, dim=0)
                if (
                    value.ndim + 1 == target.ndim
                    and target.ndim == 3
                    and target.shape[1] == 1
                ):
                    value = value.unsqueeze(1)
            else:
                kind, observed, projections, _ = binding
                base = (prefix + local_name).rsplit(".", 1)[0]
                expert_tensors = []
                for expert in observed:
                    pieces = []
                    for projection in projections:
                        weight_name = f"{base}.{expert}.{projection}.weight"
                        scale_name = f"{weight_name}_scale_inv"
                        pieces.append(
                            self._scaled_weight(
                                loaded[weight_name], loaded.get(scale_name)
                            )
                        )
                    expert_tensors.append(
                        pieces[0] if kind == "down" else self.torch.cat(pieces, dim=0)
                    )
                value = self.torch.stack(expert_tensors, dim=0)
            full_name = prefix + local_name
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"native HF working-set shape mismatch: {full_name} "
                    f"checkpoint={tuple(value.shape)} module={tuple(target.shape)}"
                )
            if value.is_floating_point() and target.is_floating_point():
                value = value.to(dtype=target.dtype)
            state[local_name] = value.to(self.device)
        missing, unexpected = module.load_state_dict(state, strict=True, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"native HF working-set state mismatch: prefix={prefix} "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        meta = [name for name, value in module.state_dict().items() if value.is_meta]
        if meta:
            raise RuntimeError(f"native HF working set retains meta tensors: {prefix}{meta[:8]}")
        self._working_set_loads += 1

    def _dematerialize(self, module: Any) -> None:
        torch = self.torch
        for child in module.modules():
            for name, parameter in list(child._parameters.items()):
                if parameter is not None:
                    child._parameters[name] = torch.nn.Parameter(
                        torch.empty(parameter.shape, dtype=parameter.dtype, device="meta"),
                        requires_grad=False,
                    )
            for name, buffer in list(child._buffers.items()):
                if buffer is not None and buffer.is_floating_point():
                    child._buffers[name] = torch.empty(
                        buffer.shape, dtype=buffer.dtype, device="meta"
                    )
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _run(self, teacher_supports: Sequence[np.ndarray] | None) -> list[dict[str, np.ndarray]]:
        torch = self.torch
        language = self._language_model()
        tokens = [
            torch.tensor(row["token_ids"][:1024], dtype=torch.long, device=self.device).unsqueeze(0)
            for row in self.corpus_rows
        ]
        self._materialize(language.embed_tokens)
        with torch.no_grad():
            activations = [
                language.embed_tokens(ids)
                .unsqueeze(2)
                .expand(-1, -1, int(language.config.hc_mult), -1)
                .contiguous()
                .cpu()
                for ids in tokens
            ]
        self._dematerialize(language.embed_tokens)
        previous_topk: list[Any | None] = [None] * len(tokens)
        for layer_index, layer in enumerate(language.layers[: language.config.num_hidden_layers]):
            self._materialize(layer)
            next_activations = []
            next_topk = []
            with torch.no_grad():
                for ids, activation, prior in zip(tokens, activations, previous_topk, strict=True):
                    hidden = activation.to(self.device)
                    positions = torch.arange(hidden.shape[1], device=self.device).unsqueeze(0)
                    mask = torch.ones((1, hidden.shape[1]), dtype=torch.bool, device=self.device)
                    output, topk = layer(
                        hidden,
                        attention_mask=mask,
                        position_ids=positions,
                        position_embeddings=None,
                        input_ids=ids,
                        past_key_values=None,
                        prev_topk_indices=None if prior is None else prior.to(self.device),
                        use_cache=False,
                    )
                    next_activations.append(output.cpu())
                    next_topk.append(None if topk is None else topk.cpu())
            activations = next_activations
            previous_topk = next_topk
            self._dematerialize(layer)
        terminal = [language.hc_head, language.norm, self._model.lm_head]
        for module in terminal:
            self._materialize(module)
        outputs = []
        with torch.no_grad():
            for slot, activation in enumerate(activations):
                hidden = language.norm(language.hc_head(activation.to(self.device))).squeeze(0)
                support_ids = None if teacher_supports is None else teacher_supports[slot]
                outputs.append(
                    top_support(
                        hidden,
                        self._model.lm_head.weight,
                        support_token_ids=support_ids,
                        support=8192,
                    )
                )
        for module in reversed(terminal):
            self._dematerialize(module)
        return outputs

    def teacher_rows(self) -> list[dict[str, np.ndarray]]:
        if self.role != "teacher":
            raise RuntimeError("teacher_rows called for candidate executor")
        rows = self._run(None)
        for row in rows:
            row["position_map"] = np.arange(1024, dtype=np.uint16)
        self._outputs = rows
        return rows

    def candidate_rows(self, teachers: Sequence[Mapping[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        if self.role != "candidate_pre":
            raise RuntimeError("candidate_rows called for teacher executor")
        support_ids = [
            np.asarray(row["support_token_ids"])[np.asarray(row["position_map"], dtype=np.int64)]
            for row in teachers
        ]
        rows = self._run(support_ids)
        for row in rows:
            row.pop("support_token_ids", None)
            row["position_map"] = np.arange(1024, dtype=np.uint16)
        self._outputs = rows
        return rows

    def finish_setup(self) -> None:
        if self._outputs is None or len(self._outputs) != 64:
            raise RuntimeError("package HF executor setup is incomplete")
        self.torch.cuda.synchronize()
        self._ready = True

    def resident_ready(self) -> bool:
        return self._ready

    def counters(self) -> dict[str, int]:
        counters = {
            "setup_model_reads": int(self.store.model_reads),
            "setup_payload_reads": int(self.store.payload_reads),
            "working_set_loads": self._working_set_loads,
            "fallback": 0,
            "relay": 0,
            "reconstruction": 0,
            "streaming": 0,
        }
        if self.role == "candidate_pre":
            counters.update(timed_payload_reads=0, timed_model_reads=0)
        return counters
