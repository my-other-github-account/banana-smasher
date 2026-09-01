from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import os
import re
import tempfile
import time
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_identity import ArtifactIdentity
from .hf_balanced64 import POSITIONS_PER_WINDOW, SUPPORT, _suite_lock
from .hf_moe import open_hf_moe_uniform
from .hf_sharded_balanced64_executor import ArtifactTensorStore, LayerStreamedHFSession
from .hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime
from .qtip1 import EncodedQtip, QtipGeometry, decode_qtip, gaussian_tlut
from .resident_training import (
    ParameterDescriptor,
    ResidentModelAdapter,
    ResidentTrainer,
    ResidentTrainingPlan,
)

HF_UNIFORM_ARTIFACT_MODE = "hf-uniform-q2-native-rest-v1"
HF_UNIFORM_PROVIDER_FACTORY = "banana_smasher.hf_uniform_physical_provider:open_provider"
HF_UNIFORM_CHECKPOINT_FORMAT = "banana-smasher-hf-uniform-resident-checkpoint-v1"
HF_UNIFORM_MODEL_LAYER_COUNT = 45
HF_UNIFORM_PARAMETER_FAMILY = "routed_q2_scales"
HF_UNIFORM_MICROBATCH = 4
HF_UNIFORM_CANONICAL_UPDATES = 45
_UPDATE_RE = re.compile(r"UPDATE_(\d+)\Z")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _callable(reference: object, field: str):
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError(f"{field} must be a module:callable reference")
    module_name, attribute = reference.split(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load {field}={reference!r}") from exc
    if not callable(value):
        raise ValueError(f"{field}={reference!r} is not callable")
    return value


def _optional_callable(reference: object, field: str):
    if reference is None:
        return None
    if callable(reference):
        return reference
    return _callable(reference, field)


def _json_object(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return loaded


def _layer_split(value: Any, *, model_layer_count: int) -> dict[int, tuple[int, int]]:
    if not isinstance(value, Mapping):
        raise ValueError("HF-uniform resident provider requires exact two-rank layer_split")
    try:
        result = {int(rank): tuple(int(item) for item in bounds) for rank, bounds in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("HF-uniform resident provider requires exact two-rank layer_split") from exc
    if sorted(result) != [0, 1]:
        raise ValueError("HF-uniform resident provider requires ranks 0 and 1")
    covered: list[int] = []
    last = -1
    for rank in (0, 1):
        start, stop = result[rank]
        if start < 0 or stop < start or start != last + 1:
            raise ValueError("HF-uniform resident provider requires contiguous rank layer ranges")
        covered.extend(range(start, stop + 1))
        last = stop
    if covered != list(range(model_layer_count)):
        raise ValueError(
            f"HF-uniform resident provider requires exact 0..{model_layer_count - 1} coverage"
        )
    if model_layer_count == HF_UNIFORM_MODEL_LAYER_COUNT and result != {
        0: (0, 23),
        1: (24, 44),
    }:
        raise ValueError(
            "HF-uniform resident provider requires exact repair45 rank layer ranges "
            "[0,23]/[24,44]"
        )
    return result


def _parse_update(checkpoint: str) -> int:
    match = _UPDATE_RE.fullmatch(checkpoint)
    if match is None:
        raise ValueError(f"unsupported HF-uniform checkpoint name: {checkpoint!r}")
    return int(match.group(1))


def _tensor_layer(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"HF-uniform tensor name does not encode a layer id: {name}")
    return int(match.group(1))


def _binding_dependencies(binding: tuple[Any, ...]) -> tuple[str, ...]:
    kind = binding[0]
    if kind in {"direct", "alias"}:
        return tuple(
            str(name)
            for name in binding[1:]
            if isinstance(name, str)
        )
    if kind == "concat":
        return tuple(str(name) for name in binding[2])
    if kind in {"gate_up", "down"}:
        return tuple(str(name) for name in binding[3])
    if kind == "override":
        return ()
    raise ValueError(f"unsupported HF-uniform binding kind: {kind!r}")


def _compose_binding_value(
    session: LayerStreamedHFSession,
    *,
    full_name: str,
    target: Any,
    binding: tuple[Any, ...],
    loaded: Mapping[str, Any],
):
    kind = binding[0]
    if kind in {"direct", "alias"}:
        _, weight_name, scale_name = binding
        value = session._scaled_weight(  # pyright: ignore[reportPrivateUsage]
            loaded[weight_name],
            None if scale_name is None else loaded[scale_name],
        )
    elif kind == "concat":
        _, weight_names, _dependencies = binding
        value = session.torch.cat(
            [
                session._scaled_weight(  # pyright: ignore[reportPrivateUsage]
                    loaded[weight_name], loaded.get(f"{weight_name}_scale_inv")
                )
                for weight_name in weight_names
            ],
            dim=0,
        )
        if (
            value.ndim + 1 == target.ndim
            and target.ndim == 3
            and target.shape[1] == 1
        ):
            value = value.unsqueeze(1)
    elif kind in {"gate_up", "down"}:
        _kind, observed, projections, _dependencies = binding
        base = full_name.rsplit(".", 1)[0]
        pieces = []
        for expert in observed:
            projections_for_expert = []
            for projection in projections:
                weight_name = f"{base}.{expert}.{projection}.weight"
                scale_name = f"{weight_name}_scale_inv"
                projections_for_expert.append(
                    session._scaled_weight(  # pyright: ignore[reportPrivateUsage]
                        loaded[weight_name], loaded.get(scale_name)
                    )
                )
            pieces.append(
                projections_for_expert[0]
                if kind == "down"
                else session.torch.cat(projections_for_expert, dim=0)
            )
        value = session.torch.stack(pieces, dim=0)
    else:
        raise ValueError(f"unsupported HF-uniform binding kind: {kind!r}")
    return value


def _selected_windows(
    windows: Sequence[Mapping[str, Any]],
    corpus_rows: Mapping[int, Mapping[str, Any]],
    teacher_rows: Mapping[int, Mapping[str, np.ndarray]],
    ordinals: Sequence[int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    selected_windows = list(windows)
    if ordinals is not None:
        required = {int(value) for value in ordinals}
        selected_windows = [row for row in selected_windows if int(row["ordinal"]) in required]
        if len(selected_windows) != len(required):
            raise ValueError("HF-uniform resident provider score windows must exist in the suite lock")
    selected_corpus = [dict(corpus_rows[int(row["ordinal"])]) for row in selected_windows]
    selected_teachers = [dict(teacher_rows[int(row["ordinal"])]) for row in selected_windows]
    return selected_windows, selected_corpus, selected_teachers


def _load_bound_inputs(
    *,
    suite_lock_path: Path,
    teacher_capture_path: Path,
    corpus_path: Path,
    basis_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, np.ndarray]],
]:
    runtime = ShardedHFBalanced64Runtime()
    lock = _suite_lock(suite_lock_path)
    teacher_capture = _json_object(teacher_capture_path, "HF-uniform teacher capture")
    rows = teacher_capture.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("HF-uniform resident provider requires a complete teacher capture")
    windows = list(lock["windows"])
    corpus_rows, input_policy = runtime._read_corpus(
        corpus_path,
        lock,
        model_index_sha256=basis_sha256,
    )
    teachers = [
        runtime._load_teacher_row(row, expected, input_policy=input_policy)
        for row, expected in zip(rows, windows, strict=True)
    ]
    return (
        lock,
        windows,
        {int(row["ordinal"]): dict(row) for row in corpus_rows},
        {int(window["ordinal"]): teacher for window, teacher in zip(windows, teachers, strict=True)},
    )


def _validate_runtime_counters(value: Mapping[str, Any], *, timed: bool) -> dict[str, int]:
    required = {
        "setup_model_reads",
        "setup_payload_reads",
        "working_set_loads",
        "fallback",
        "relay",
        "reconstruction",
        "streaming",
    }
    if timed:
        required.update(("timed_payload_reads", "timed_model_reads"))
    if not required.issubset(value):
        raise ValueError("HF-uniform runtime counters are incomplete")
    counters: dict[str, int] = {}
    for name in sorted(required):
        counter = value[name]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError(f"HF-uniform runtime counter is invalid: {name}")
        counters[name] = counter
    for name in ("fallback", "relay", "reconstruction", "streaming"):
        if counters[name] != 0:
            raise ValueError(f"HF-uniform runtime counter is nonzero: {name}")
    if timed and (counters["timed_payload_reads"] or counters["timed_model_reads"]):
        raise ValueError("HF-uniform runtime performed reads inside the timed region")
    return counters


class _HFUniformTrainableSession(LayerStreamedHFSession):
    """Layer-streamed HF session with optional live module-parameter overrides."""

    def __init__(self, *, combined_overrides: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        self.combined_overrides = dict(combined_overrides or {})
        super().__init__(**kwargs)

    def _materialize(self, module: Any) -> None:
        prefix = self._prefix_for(module)
        local_state = module.state_dict()
        if not local_state:
            return
        available = self.store.names()
        bindings: dict[str, tuple[Any, ...]] = {}
        dependencies: list[str] = []
        for local_name, target in local_state.items():
            full_name = prefix + local_name
            if full_name in self.combined_overrides:
                bindings[local_name] = ("override", full_name)
                continue
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
                dependencies.extend(_binding_dependencies(binding))
            bindings[local_name] = binding
        loaded = self.store.load_many(list(dict.fromkeys(dependencies)))
        state = {}
        for local_name, target in local_state.items():
            binding = bindings[local_name]
            kind = binding[0]
            if kind == "override":
                value = self.combined_overrides[str(binding[1])]
                if value.dtype != target.dtype or value.device.type != self.device.split(":", 1)[0]:
                    raise RuntimeError(
                        f"trainable HF override dtype/device mismatch: {binding[1]} "
                        f"override={value.dtype}/{value.device} target={target.dtype}/{self.device}"
                    )
            else:
                value = _compose_binding_value(
                    self,
                    full_name=prefix + local_name,
                    target=target,
                    binding=binding,
                    loaded=loaded,
                )
                if value.is_floating_point() and target.is_floating_point():
                    value = value.to(dtype=target.dtype)
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"native HF working-set shape mismatch: {prefix + local_name} "
                    f"checkpoint={tuple(value.shape)} module={tuple(target.shape)}"
                )
            state[local_name] = value.to(self.device)
        # ``load_state_dict(assign=True)`` wraps composed tensors in fresh leaf
        # Parameters and severs their graph from the resident Q2 row scales.
        # Install the exact state entries directly so routed decoded weights keep
        # their MulBackward link to the only trainable artifact members.
        for local_name, value in state.items():
            parent_name, separator, leaf_name = local_name.rpartition(".")
            parent = module.get_submodule(parent_name) if separator else module
            if leaf_name in parent._parameters:  # pyright: ignore[reportPrivateUsage]
                parent._parameters[leaf_name] = value  # pyright: ignore[reportPrivateUsage]
            elif leaf_name in parent._buffers:  # pyright: ignore[reportPrivateUsage]
                parent._buffers[leaf_name] = value  # pyright: ignore[reportPrivateUsage]
            else:
                raise RuntimeError(
                    f"native HF working-set state has no destination: {prefix + local_name}"
                )
        meta = [name for name, value in module.state_dict().items() if value.is_meta]
        if meta:
            raise RuntimeError(f"native HF working set retains meta tensors: {prefix}{meta[:8]}")
        self._working_set_loads += 1

    def teacher_kl_loss(self, teachers: Sequence[Mapping[str, np.ndarray]]):
        torch = self.torch
        language = self._language_model()
        tokens = [
            torch.tensor(row["token_ids"][:1024], dtype=torch.long, device=self.device).unsqueeze(0)
            for row in self.corpus_rows
        ]
        self._materialize(language.embed_tokens)
        activations = [
            language.embed_tokens(ids)
            .unsqueeze(2)
            .expand(-1, -1, int(language.config.hc_mult), -1)
            .contiguous()
            for ids in tokens
        ]
        self._dematerialize(language.embed_tokens)
        previous_topk: list[Any | None] = [None] * len(tokens)
        for layer in language.layers[: language.config.num_hidden_layers]:
            self._materialize(layer)
            next_activations = []
            next_topk = []
            for ids, activation, prior in zip(tokens, activations, previous_topk, strict=True):
                positions = torch.arange(activation.shape[1], device=self.device).unsqueeze(0)
                mask = torch.ones((1, activation.shape[1]), dtype=torch.bool, device=self.device)
                output, topk = layer(
                    activation,
                    attention_mask=mask,
                    position_ids=positions,
                    position_embeddings=None,
                    input_ids=ids,
                    past_key_values=None,
                    prev_topk_indices=prior,
                    use_cache=False,
                )
                next_activations.append(output)
                next_topk.append(topk)
            activations = next_activations
            previous_topk = next_topk
            self._dematerialize(layer)
        terminal = [language.hc_head, language.norm, self._model.lm_head]
        for module in terminal:
            self._materialize(module)
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        positions = 0
        try:
            for activation, teacher in zip(activations, teachers, strict=True):
                hidden = language.norm(language.hc_head(activation)).squeeze(0)
                teacher_map = np.asarray(teacher["position_map"], dtype=np.int64)
                teacher_ids = np.asarray(teacher["support_token_ids"])[teacher_map]
                teacher_logits = np.asarray(teacher["support_logits"], dtype=np.float32)[teacher_map]
                support_ids = torch.as_tensor(teacher_ids, dtype=torch.long, device=self.device)
                support_logits = torch.as_tensor(
                    teacher_logits, dtype=torch.float32, device=self.device
                )
                positions += int(support_ids.shape[0])
                for start in range(0, hidden.shape[0], 128):
                    stop = start + 128
                    logits = self._model.lm_head(hidden[start:stop].to(torch.bfloat16)).float()
                    candidate = logits.gather(1, support_ids[start:stop])
                    teacher_chunk = support_logits[start:stop]
                    teacher_logprob = teacher_chunk - torch.logsumexp(
                        teacher_chunk, dim=-1, keepdim=True
                    )
                    candidate_logprob = candidate - torch.logsumexp(
                        candidate, dim=-1, keepdim=True
                    )
                    loss = loss + (
                        teacher_logprob.exp() * (teacher_logprob - candidate_logprob)
                    ).sum()
            if positions <= 0 or not bool(torch.isfinite(loss).item()):
                raise RuntimeError("HF-uniform teacher KL loss is non-finite")
            return loss / positions
        finally:
            for module in reversed(terminal):
                self._dematerialize(module)


class _TrainableArtifactTensorStore(ArtifactTensorStore):
    """Compose immutable Q2 trellises with live per-row artifact scales."""

    def __init__(self, artifact: Mapping[str, Any], *, tensor_overrides: Mapping[str, Any]) -> None:
        super().__init__(artifact)
        self.tensor_overrides = dict(tensor_overrides)

    def tensor(self, name: str):
        if name in self.tensor_overrides and name in self.routed:
            row = self.routed[name]
            geometry = QtipGeometry.from_mapping(row["wire"]["geometry"])
            packed = self._load_array(row["wire"]["trellis"])
            raw_shape = tuple(int(value) for value in row["shape"])
            if len(raw_shape) != 2:
                raise ValueError(f"invalid HF-uniform routed Q2 shape: {raw_shape}")
            shape = (raw_shape[0], raw_shape[1])
            unit = decode_qtip(
                EncodedQtip(
                    geometry=geometry,
                    shape=shape,
                    states=np.empty((0, 0), dtype=np.int32),
                    packed=np.asarray(packed, dtype=np.uint16),
                    scales=np.ones((shape[0],), dtype=np.float32),
                ),
                tlut=gaussian_tlut(bits=geometry.tlut_bits, columns=geometry.V),
            )
            scale = self.tensor_overrides[name]
            unit_tensor = self._torch_from_numpy(unit).to(
                device=scale.device, dtype=scale.dtype
            )
            return unit_tensor * scale[:, None]
        return super().tensor(name)

    def load_many(self, names: Sequence[str]) -> dict[str, Any]:
        return {name: self.tensor(name) for name in names}


class _HFUniformLayerStreamedBackend:
    """Resident backend that trains only routed-Q2 per-row artifact scales."""

    def __init__(
        self,
        *,
        plan: ResidentTrainingPlan,
        artifact: Mapping[str, Any],
        basis_sha256: str,
        rank: int,
        layer_split: Mapping[int, tuple[int, int]],
        suite_lock_path: str | Path,
        teacher_capture_path: str | Path,
        corpus_path: str | Path,
    ) -> None:
        self.plan = plan
        self.artifact = artifact
        self.basis_sha256 = str(basis_sha256)
        self.rank = int(rank)
        self.layer_split = {int(key): tuple(value) for key, value in layer_split.items()}
        (
            self.suite_lock,
            self.windows,
            self.corpus_rows_by_ordinal,
            self.teacher_rows_by_ordinal,
        ) = _load_bound_inputs(
            suite_lock_path=Path(suite_lock_path).expanduser().resolve(),
            teacher_capture_path=Path(teacher_capture_path).expanduser().resolve(),
            corpus_path=Path(corpus_path).expanduser().resolve(),
            basis_sha256=self.basis_sha256,
        )
        initial_corpus = [self.corpus_rows_by_ordinal[int(row["ordinal"])] for row in self.windows]
        self.session = _HFUniformTrainableSession(
            subject=artifact,
            role="candidate_pre",
            suite_lock=self.suite_lock,
            corpus_rows=initial_corpus,
        )
        self.store = _TrainableArtifactTensorStore(artifact, tensor_overrides={})
        self.session.store = self.store
        self._parameters: dict[str, Any] = {}
        self._descriptors: tuple[ParameterDescriptor, ...] = ()
        self._build_trainables()

    def _selected_routed_names(self) -> set[str]:
        start, stop = self.layer_split[self.rank]
        selected = set()
        for row in self.artifact["routed_tensors"]:
            name = str(row["name"])
            layer = _tensor_layer(name)
            if start <= layer <= stop:
                selected.add(name)
        return selected

    def _build_trainables(self) -> None:
        torch = self.session.torch
        selected_routed = self._selected_routed_names()
        if not selected_routed:
            raise ValueError("HF-uniform resident provider found no routed trainable tensors")
        descriptors: list[ParameterDescriptor] = []
        stable_ids: set[str] = set()
        for row in sorted(self.artifact["routed_tensors"], key=lambda value: str(value["name"])):
            name = str(row["name"])
            if name not in selected_routed:
                continue
            scale_binding = row["wire"]["scales"]
            scale_path = str(scale_binding["path"])
            stable_id = f"artifact-scale:{scale_path}"
            if stable_id in stable_ids:
                raise ValueError(f"duplicate HF-uniform artifact scale member: {scale_path}")
            scales = np.asarray(self.store._load_array(scale_binding), dtype=np.float32)
            shape = tuple(int(value) for value in row["shape"])
            if scales.shape != (shape[0],) or not np.isfinite(scales).all():
                raise ValueError(f"invalid HF-uniform Q2 per-row scale array: {scale_path}")
            parameter = torch.nn.Parameter(
                torch.from_numpy(scales.copy()).to(self.session.device), requires_grad=True
            )
            descriptor = ParameterDescriptor(name, HF_UNIFORM_PARAMETER_FAMILY, stable_id)
            self._parameters[name] = parameter
            stable_ids.add(stable_id)
            descriptors.append(descriptor)
        if not descriptors:
            raise ValueError("HF-uniform resident provider found no Q2 scale arrays to train")
        self._descriptors = tuple(sorted(descriptors, key=lambda row: row.stable_id))
        self.store.tensor_overrides = dict(self._parameters)
        self.session.combined_overrides = {}

    def resident_parameters(self):
        return [
            (descriptor, self._parameters[descriptor.name])
            for descriptor in self._descriptors
        ]

    def residency_metadata(self) -> Mapping[str, Any]:
        resident_bytes = sum(
            int(parameter.numel() * parameter.element_size())
            for parameter in self._parameters.values()
        )
        return {
            "resident_bytes": resident_bytes,
            "payload_disk_reads": int(self.store.payload_reads),
            "model_disk_reads": int(self.store.model_reads),
            "trainable_count": len(self._descriptors),
        }

    def _select_batch(
        self, windows: Sequence[int]
    ) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
        corpus = [self.corpus_rows_by_ordinal[int(window)] for window in windows]
        teachers = [self.teacher_rows_by_ordinal[int(window)] for window in windows]
        return corpus, teachers

    def candidate_rows(
        self,
        teachers: Sequence[Mapping[str, np.ndarray]],
        corpus_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, np.ndarray]]:
        previous = self.session.corpus_rows
        self.session.corpus_rows = list(corpus_rows)
        try:
            return list(self.session.candidate_rows(teachers))
        finally:
            self.session.corpus_rows = previous

    def loss_for_windows(self, windows: Sequence[int], *, tokens: int) -> Any:
        del tokens
        corpus_rows, teachers = self._select_batch(windows)
        previous = self.session.corpus_rows
        self.session.corpus_rows = corpus_rows
        try:
            return self.session.teacher_kl_loss(teachers)
        finally:
            self.session.corpus_rows = previous

    def post_optimizer_step(self, names: Sequence[str]) -> None:
        del names

    def trainable_state_dict(self) -> Mapping[str, Any]:
        return {
            descriptor.stable_id: self._parameters[descriptor.name].detach().cpu().clone()
            for descriptor in self._descriptors
        }

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {descriptor.stable_id for descriptor in self._descriptors}
        if set(state) != expected:
            raise ValueError("HF-uniform checkpoint trainable IDs do not match resident parameters")
        with self.session.torch.no_grad():
            for descriptor in self._descriptors:
                parameter = self._parameters[descriptor.name]
                source = self.session.torch.as_tensor(
                    state[descriptor.stable_id],
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter.copy_(source)
            self.post_optimizer_step([descriptor.name for descriptor in self._descriptors])

    def parameter_digests(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for descriptor in self._descriptors:
            tensor = self._parameters[descriptor.name].detach().cpu().contiguous()
            if tensor.dtype == self.session.torch.bfloat16:
                array = tensor.view(self.session.torch.uint16).numpy()
            else:
                array = tensor.numpy()
            digest = hashlib.sha256()
            digest.update(str(tensor.dtype).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
            result[descriptor.stable_id] = digest.hexdigest()
        return result

    def runtime_counters(self) -> dict[str, int]:
        return dict(self.session.counters())


class HFUniformResidentAdapter(ResidentModelAdapter):
    adapter_name = "hf-uniform-resident"
    adapter_version = "1"

    def __init__(
        self,
        *,
        model_source: str | None,
        artifact_root: str,
        artifact_identity_sha256: str,
        basis_sha256: str,
        rank: int,
        layer_split: Mapping[str, Sequence[int]] | Mapping[int, tuple[int, int]],
        suite_lock_path: str,
        teacher_capture_path: str,
        corpus_path: str,
        backend_factory: str | None = None,
        **_: Any,
    ) -> None:
        self.model_source = model_source
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_identity_sha256 = str(artifact_identity_sha256)
        self.basis_sha256 = str(basis_sha256)
        self.rank = int(rank)
        self.layer_split = _layer_split(
            layer_split, model_layer_count=HF_UNIFORM_MODEL_LAYER_COUNT
        )
        self.suite_lock_path = str(suite_lock_path)
        self.teacher_capture_path = str(teacher_capture_path)
        self.corpus_path = str(corpus_path)
        self.backend_factory = backend_factory
        self.plan: ResidentTrainingPlan | None = None
        self.backend: Any | None = None
        self._torch: Any | None = None
        self._descriptors: tuple[ParameterDescriptor, ...] = ()
        self._parameters: dict[str, Any] = {}
        self._selected: dict[str, tuple[ParameterDescriptor, ...]] = {}
        self._optimizer_group_descriptors: list[tuple[ParameterDescriptor, ...]] = []
        self._stable_to_name: dict[str, str] = {}
        self.optimizer: Any | None = None
        self._model_index_sha256: str | None = None
        self._last_learning_rates: dict[str, float] = {}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "model_source": self.model_source,
            "model_index_sha256": self._model_index_sha256,
            "parameter_ids": [row.stable_id for row in self._descriptors],
        }

    def stage(self, plan: ResidentTrainingPlan) -> Mapping[str, Any]:
        if self.backend is not None:
            return dict(self.backend.residency_metadata())
        import torch

        self.plan = plan
        self._torch = torch
        artifact = open_hf_moe_uniform(self.artifact_root)
        source = artifact.get("source", {})
        model_root = Path(str(source.get("model_root", ""))).expanduser().resolve()
        index = model_root / "model.safetensors.index.json"
        observed = _sha256(index)
        if observed != self.basis_sha256:
            raise ValueError(
                f"HF-uniform resident basis mismatch: {observed} != {self.basis_sha256}"
            )
        self._model_index_sha256 = observed
        factory = (
            _HFUniformLayerStreamedBackend
            if self.backend_factory is None
            else _callable(self.backend_factory, "backend_factory")
        )
        backend = factory(
            plan=plan,
            artifact=artifact,
            basis_sha256=self.basis_sha256,
            rank=self.rank,
            layer_split=self.layer_split,
            suite_lock_path=self.suite_lock_path,
            teacher_capture_path=self.teacher_capture_path,
            corpus_path=self.corpus_path,
        )
        rows = tuple(backend.resident_parameters())
        descriptors: list[ParameterDescriptor] = []
        parameters: dict[str, Any] = {}
        stable_to_name: dict[str, str] = {}
        for descriptor, parameter in rows:
            if not isinstance(descriptor, ParameterDescriptor):
                raise TypeError("HF-uniform backend must yield ParameterDescriptor/tensor pairs")
            if descriptor.name in parameters or descriptor.stable_id in stable_to_name:
                raise ValueError(f"duplicate HF-uniform parameter identity: {descriptor.stable_id}")
            descriptors.append(descriptor)
            parameters[descriptor.name] = parameter
            stable_to_name[descriptor.stable_id] = descriptor.name
        self.backend = backend
        self._descriptors = tuple(descriptors)
        self._parameters = parameters
        self._stable_to_name = stable_to_name
        return dict(backend.residency_metadata())

    def parameters(self) -> Sequence[ParameterDescriptor]:
        return self._descriptors

    def configure_parameter_groups(
        self, groups: Mapping[str, tuple[ParameterDescriptor, ...]]
    ) -> None:
        if self.plan is None or self._torch is None:
            raise RuntimeError("HF-uniform adapter is not staged")
        for parameter in self._parameters.values():
            parameter.requires_grad_(False)
        optimizer_groups: list[dict[str, Any]] = []
        descriptor_groups: list[tuple[ParameterDescriptor, ...]] = []
        plans = {row.name: row for row in self.plan.parameter_groups}
        for name, descriptors in groups.items():
            if not descriptors:
                continue
            for descriptor in descriptors:
                self._parameters[descriptor.name].requires_grad_(True)
            group_plan = plans[name]
            optimizer_groups.append(
                {
                    "params": [self._parameters[row.name] for row in descriptors],
                    "lr": group_plan.lr,
                    "group_name": name,
                    **dict(group_plan.options),
                }
            )
            descriptor_groups.append(descriptors)
        if not optimizer_groups:
            raise ValueError("HF-uniform selectors produced no optimizer groups")
        optimizer_options = dict(self.plan.optimizer_options)
        optimizer_options.setdefault("foreach", False)
        optimizer_name = self.plan.optimizer.lower()
        if optimizer_name == "adam":
            optimizer_class = self._torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_class = self._torch.optim.AdamW
        else:
            raise ValueError(f"HF-uniform adapter does not support optimizer {optimizer_name!r}")
        self.optimizer = optimizer_class(optimizer_groups, **optimizer_options)
        self._selected = dict(groups)
        self._optimizer_group_descriptors = descriptor_groups

    def _synchronize(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("HF-uniform optimizer is not configured")
        self.optimizer.zero_grad(set_to_none=True)

    def train_microbatch(
        self, windows: tuple[int, ...], *, tokens: int, loss_scale: float
    ) -> Mapping[str, float]:
        if self.backend is None:
            raise RuntimeError("HF-uniform backend is not resident")
        self._synchronize()
        forward_started = time.perf_counter()
        loss = self.backend.loss_for_windows(windows, tokens=tokens)
        self._synchronize()
        forward_seconds = time.perf_counter() - forward_started
        backward_started = time.perf_counter()
        (loss * loss_scale).backward()
        self._synchronize()
        backward_seconds = time.perf_counter() - backward_started
        return {
            "loss": float(loss.detach().cpu()),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "comm_seconds": 0.0,
        }

    def optimizer_step(self, learning_rates: dict[str, float]) -> float:
        if self.optimizer is None:
            raise RuntimeError("HF-uniform optimizer is not configured")
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rates[str(group["group_name"])]
        self._synchronize()
        started = time.perf_counter()
        self.optimizer.step()
        self._synchronize()
        selected_names = [
            descriptor.name
            for descriptors in self._selected.values()
            for descriptor in descriptors
        ]
        post_step = getattr(self.backend, "post_optimizer_step", None)
        if callable(post_step):
            post_step(selected_names)
        self._last_learning_rates = dict(learning_rates)
        return time.perf_counter() - started

    def trainable_state_dict(self) -> Mapping[str, Any]:
        if self.backend is None:
            return {}
        return dict(self.backend.trainable_state_dict())

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.backend is None:
            raise RuntimeError("HF-uniform backend is not resident")
        self.backend.load_trainable_state_dict(state)

    def optimizer_state_dict(self) -> Mapping[str, Any]:
        if self.optimizer is None:
            return {}
        raw = self.optimizer.state_dict()
        id_to_stable: dict[int, str] = {}
        groups: list[dict[str, Any]] = []
        for group, descriptors in zip(raw["param_groups"], self._optimizer_group_descriptors):
            parameter_ids = list(group["params"])
            if len(parameter_ids) != len(descriptors):
                raise RuntimeError("HF-uniform optimizer parameter-group identity drift")
            stable_ids = [row.stable_id for row in descriptors]
            id_to_stable.update(zip(parameter_ids, stable_ids))
            groups.append({**group, "params": stable_ids})
        state = {id_to_stable[key]: value for key, value in raw["state"].items()}
        return {"state": state, "param_groups": groups}

    def _tree_to_torch(self, value: Any) -> Any:
        import numpy as np

        if isinstance(value, np.ndarray):
            return self._torch.from_numpy(value)
        if isinstance(value, Mapping):
            return {key: self._tree_to_torch(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._tree_to_torch(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._tree_to_torch(item) for item in value)
        return value

    def load_optimizer_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.optimizer is None:
            if state:
                raise RuntimeError("HF-uniform optimizer is not configured")
            return
        current = self.optimizer.state_dict()
        stable_to_current: dict[str, int] = {}
        rebuilt_groups: list[dict[str, Any]] = []
        saved_groups = list(state.get("param_groups", ()))
        if len(saved_groups) != len(self._optimizer_group_descriptors):
            raise ValueError("HF-uniform checkpoint optimizer group count drift")
        for current_group, saved_group, descriptors in zip(
            current["param_groups"], saved_groups, self._optimizer_group_descriptors
        ):
            stable_ids = [row.stable_id for row in descriptors]
            if list(saved_group["params"]) != stable_ids:
                raise ValueError("HF-uniform sparse optimizer parameter IDs drift")
            stable_to_current.update(zip(stable_ids, current_group["params"]))
            rebuilt_groups.append({**saved_group, "params": list(current_group["params"])})
        saved_state = state.get("state", {})
        unknown = set(saved_state) - set(stable_to_current)
        if unknown:
            raise ValueError(
                f"HF-uniform optimizer state has unknown parameter IDs: {sorted(unknown)}"
            )
        raw = {
            "state": {
                stable_to_current[stable_id]: self._tree_to_torch(value)
                for stable_id, value in saved_state.items()
            },
            "param_groups": rebuilt_groups,
        }
        self.optimizer.load_state_dict(raw)

    def scheduler_state_dict(self) -> Mapping[str, Any]:
        return {"last_learning_rates": dict(self._last_learning_rates)}

    def load_scheduler_state_dict(self, state: Mapping[str, Any]) -> None:
        self._last_learning_rates = {
            str(name): float(value)
            for name, value in state.get("last_learning_rates", {}).items()
        }

    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(row.stable_id for row in self._descriptors)

    def parameter_digests(self) -> dict[str, str]:
        if self.backend is None:
            return {}
        method = getattr(self.backend, "parameter_digests", None)
        if callable(method):
            return dict(method())
        return {}


def open_hf_uniform_resident_adapter(**kwargs: Any) -> HFUniformResidentAdapter:
    return HFUniformResidentAdapter(**kwargs)


class HFUniformPhysicalProvider:
    """Physical resident provider for routed-Q2/native-rest HF artifacts."""

    physical_hf_uniform_provider = True

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        identity_sha256: str,
        basis_sha256: str,
        checkpoint: str,
        checkpoint_sha256: str,
        rank: int,
        run_root: str | Path,
        config: Mapping[str, Any],
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.identity = ArtifactIdentity.load(self.artifact_root)
        self.artifact = open_hf_moe_uniform(self.artifact_root)
        self.identity_sha256 = str(identity_sha256)
        self.basis_sha256 = str(basis_sha256)
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.rank = int(rank)
        self.run_root = Path(run_root).expanduser().resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.model_layer_count = int(self.config.get("model_layer_count", HF_UNIFORM_MODEL_LAYER_COUNT))
        self.layer_split = _layer_split(
            self.config.get("layer_split"), model_layer_count=self.model_layer_count
        )
        if self.rank not in self.layer_split:
            raise ValueError("HF-uniform resident provider rank must be 0 or 1")
        if self.model_layer_count != HF_UNIFORM_MODEL_LAYER_COUNT:
            raise ValueError("HF-uniform resident provider requires exact 45-layer geometry")
        self._executor_factory = _optional_callable(
            self.config.get("executor_factory"), "executor_factory"
        )
        self._backend_factory = self.config.get("hf_uniform_backend_factory")
        self._loaded_inputs: tuple[
            dict[str, Any],
            list[dict[str, Any]],
            dict[int, dict[str, Any]],
            dict[int, dict[str, np.ndarray]],
        ] | None = None
        self._pre_mean_kld: float | None = None
        self._trainer: ResidentTrainer | None = None
        self._adapter: HFUniformResidentAdapter | None = None
        self._checkpoint_path: Path | None = None
        self._loaded_checkpoint_path: Path | None = None
        self._validate_binding()

    def _validate_binding(self) -> None:
        layers = [row.get("layer") for row in self.identity.composition]
        geometry = self.artifact.get("geometry", {})
        source = self.artifact.get("source", {})
        if self.identity.sha256 != self.identity_sha256:
            raise ValueError("HF-uniform resident provider artifact identity mismatch")
        if self.identity.basis_sha256 != self.basis_sha256:
            raise ValueError("HF-uniform resident provider basis identity mismatch")
        if self.artifact.get("intent") != {
            "tier": "q2",
            "scope": "routed_only",
            "native_rest": True,
        }:
            raise ValueError("HF-uniform resident provider requires routed-Q2/native-rest artifact intent")
        if (
            source.get("model_index_sha256") != self.basis_sha256
            or geometry.get("expected_model_layers") != self.model_layer_count
            or layers != list(range(self.model_layer_count))
        ):
            raise ValueError("HF-uniform resident provider source/artifact/geometry mismatch")

    def _load_inputs(self):
        if self._loaded_inputs is None:
            self._loaded_inputs = _load_bound_inputs(
                suite_lock_path=Path(str(self.config["suite_lock_path"])).expanduser().resolve(),
                teacher_capture_path=Path(str(self.config["teacher_capture_path"])).expanduser().resolve(),
                corpus_path=Path(str(self.config["corpus_path"])).expanduser().resolve(),
                basis_sha256=self.basis_sha256,
            )
        return self._loaded_inputs

    def _runtime(self) -> ShardedHFBalanced64Runtime:
        return ShardedHFBalanced64Runtime(executor_factory=self._executor_factory)

    def _build_training_plan(self) -> ResidentTrainingPlan:
        _lock, windows, _corpus_rows, _teacher_rows = self._load_inputs()
        microbatch = int(self.config.get("training_microbatch", HF_UNIFORM_MICROBATCH))
        if microbatch <= 0 or len(windows) % microbatch:
            raise ValueError("HF-uniform resident provider requires a 64-window canonical microbatch schedule")
        model_root = Path(str(self.artifact["source"]["model_root"])).expanduser().resolve()
        config = {
            "model_source": None,
            "model_adapter": (
                "banana_smasher.hf_uniform_physical_provider:open_hf_uniform_resident_adapter"
            ),
            "model_root": str(model_root),
            "payload_root": str(self.artifact_root),
            "input_checkpoint": None,
            "run_root": str((self.run_root / "resident-training").resolve()),
            "topology": {
                "world_size": 2,
                "rank": self.rank,
                "layer_split": [list(self.layer_split[index]) for index in (0, 1)],
            },
            "windows": [int(row["ordinal"]) for row in windows],
            "tokens_per_window": POSITIONS_PER_WINDOW,
            "microbatch": microbatch,
            "gradient_accumulation": len(windows) // microbatch,
            "updates": int(self.config.get("canonical_updates", HF_UNIFORM_CANONICAL_UPDATES)),
            "optimizer": str(self.config.get("optimizer", "adam")),
            "optimizer_options": {"foreach": False, **dict(self.config.get("optimizer_options", {}))},
            "parameter_groups": [
                {
                    "name": HF_UNIFORM_PARAMETER_FAMILY,
                    "lr": float(self.config.get("learning_rate", 1.0e-3)),
                    "warmup_updates": int(self.config.get("warmup_updates", 0)),
                    "families": [HF_UNIFORM_PARAMETER_FAMILY],
                    "include": ["*"],
                }
            ],
            "adapter_options": {
                "artifact_root": str(self.artifact_root),
                "artifact_identity_sha256": self.identity.sha256,
                "basis_sha256": self.basis_sha256,
                "rank": self.rank,
                "layer_split": {str(key): list(value) for key, value in self.layer_split.items()},
                "suite_lock_path": str(self.config["suite_lock_path"]),
                "teacher_capture_path": str(self.config["teacher_capture_path"]),
                "corpus_path": str(self.config["corpus_path"]),
                "backend_factory": self._backend_factory,
            },
        }
        return ResidentTrainingPlan.from_dict(config)

    def _ensure_trainer(self) -> ResidentTrainer:
        if self._trainer is None or self._adapter is None:
            plan = self._build_training_plan()
            adapter = HFUniformResidentAdapter(
                model_source=plan.model_source,
                **dict(plan.adapter_options),
            )
            trainer = ResidentTrainer(plan, adapter=adapter)
            trainer.initialize()
            self._trainer = trainer
            self._adapter = adapter
        if self._checkpoint_path is not None and self._loaded_checkpoint_path != self._checkpoint_path:
            self._trainer.load_checkpoint(self._checkpoint_path)
            self._loaded_checkpoint_path = self._checkpoint_path
        return self._trainer

    def trainable_parameter_ids(self) -> tuple[str, ...]:
        self._ensure_trainer()
        assert self._adapter is not None
        return self._adapter.parameter_ids()

    def trainable_parameter_digests(self) -> dict[str, str]:
        self._ensure_trainer()
        assert self._adapter is not None
        return self._adapter.parameter_digests()

    def _candidate_rows_for_score(
        self,
        *,
        lock: Mapping[str, Any],
        windows: Sequence[Mapping[str, Any]],
        corpus_rows: Sequence[Mapping[str, Any]],
        teachers: Sequence[Mapping[str, np.ndarray]],
        subset: bool,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, int]]:
        runtime = self._runtime()
        if self._adapter is not None and self._adapter.backend is not None:
            method = getattr(self._adapter.backend, "candidate_rows", None)
            if not callable(method):
                raise ValueError("HF-uniform resident backend does not expose score candidate_rows")
            raw_candidates = method(teachers, corpus_rows)
            candidates = [runtime._candidate_arrays(value) for value in raw_candidates]
            counters = getattr(self._adapter.backend, "runtime_counters", None)
            if not callable(counters):
                raise ValueError("HF-uniform resident backend does not expose runtime counters")
            return candidates, _validate_runtime_counters(dict(counters()), timed=True)
        session = runtime._session(
            subject=self.artifact,
            role="candidate_pre",
            suite_lock=lock,
            corpus_rows=corpus_rows,
        )
        candidate_rows = getattr(session, "candidate_rows", None)
        candidate_window = getattr(session, "candidate_window", None)
        if callable(candidate_rows):
            raw_candidates = candidate_rows(teachers)
            if not isinstance(raw_candidates, list) or len(raw_candidates) != len(windows):
                raise ValueError("HF-uniform resident provider bulk executor returned the wrong row count")
        elif callable(candidate_window):
            raw_candidates = [
                candidate_window(
                    window=expected,
                    token_ids=corpus_row["token_ids"],
                    support_token_ids=teacher["support_token_ids"],
                    position_map=teacher["position_map"],
                )
                for expected, corpus_row, teacher in zip(
                    windows, corpus_rows, teachers, strict=True
                )
            ]
        else:
            raise ValueError("HF-uniform resident provider executor lacks candidate execution")
        if not subset:
            finish_setup = getattr(session, "finish_setup", None)
            if callable(finish_setup):
                finish_setup()
            resident_ready = getattr(session, "resident_ready", None)
            if callable(resident_ready) and resident_ready() is not True:
                raise ValueError("HF-uniform resident provider executor was not resident-ready")
        return [runtime._candidate_arrays(value) for value in raw_candidates], runtime._counters(
            session, timed=True
        )

    def _score_summary(self, *, ordinals: Sequence[int] | None = None) -> dict[str, Any]:
        lock, windows, corpus_rows_by_ordinal, teacher_rows_by_ordinal = self._load_inputs()
        windows, corpus_rows, teachers = _selected_windows(
            windows,
            corpus_rows_by_ordinal,
            teacher_rows_by_ordinal,
            ordinals,
        )
        started = time.perf_counter()
        candidates, counters = self._candidate_rows_for_score(
            lock=lock,
            windows=windows,
            corpus_rows=corpus_rows,
            teachers=teachers,
            subset=ordinals is not None,
        )
        elapsed = time.perf_counter() - started
        runtime = self._runtime()
        all_values: list[float] = []
        top1_matches = 0
        for teacher, candidate in zip(teachers, candidates, strict=True):
            all_values.extend(runtime._kld_values(teacher, candidate))
            top1_matches += int(
                np.count_nonzero(teacher["top1_token_ids"] == candidate["top1_token_ids"])
            )
        positions = len(windows) * POSITIONS_PER_WINDOW
        return {
            "mean_kld": fsum(all_values) / positions,
            "top1_matches": top1_matches,
            "positions": positions,
            "timed_wall_seconds": elapsed,
            "runtime_counters": {
                "windows": len(windows),
                "checkpoint_loads_during_score": 0,
                "candidate_file_reads_during_score": 0,
                "setup_model_reads": int(counters["setup_model_reads"]),
                "setup_payload_reads": int(counters["setup_payload_reads"]),
                "working_set_loads": int(counters["working_set_loads"]),
            },
            "support": SUPPORT,
            "execution_mode": "resident_model_in_memory",
            "rank_layer_range": list(self.layer_split[self.rank]),
        }

    def score(self, phase: str) -> Mapping[str, Any]:
        result = self._score_summary()
        result["checkpoint"] = self.checkpoint
        if phase == "pre":
            self._pre_mean_kld = float(result["mean_kld"])
        return result

    def score_probe(self, windows: Sequence[int]) -> Mapping[str, Any]:
        result = self._score_summary(ordinals=tuple(int(value) for value in windows))
        result["checkpoint"] = self.checkpoint
        return result

    def train(self, updates: int) -> Mapping[str, Any]:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("HF-uniform resident provider requires a positive update count")
        if self._pre_mean_kld is None:
            raise ValueError("HF-uniform resident provider requires a measured PRE before training")
        trainer = self._ensure_trainer()
        before = self.trainable_parameter_digests()
        steps = [trainer.train_step() for _ in range(updates)]
        checkpoint_update = _parse_update(self.checkpoint) + updates
        checkpoint = f"UPDATE_{checkpoint_update:03d}"
        checkpoint_path = trainer.save_checkpoint(
            self.run_root / "checkpoints" / f"{checkpoint}.safetensors"
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        after = self.trainable_parameter_digests()
        changed = [
            parameter_id
            for parameter_id, digest in after.items()
            if before.get(parameter_id) != digest
        ]
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_path = checkpoint_path
        self._loaded_checkpoint_path = checkpoint_path
        return {
            "updates": updates,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_path": str(checkpoint_path),
            "rank_layer_range": list(self.layer_split[self.rank]),
            "objective": "teacher_kl",
            "trainable_parameter_ids": list(after),
            "changed_parameter_ids": changed,
            "step_losses": [float(step.loss) for step in steps],
        }

    def restore_pre_score(self, pre: Mapping[str, Any]) -> None:
        self._pre_mean_kld = float(pre["mean_kld"])
        checkpoint = pre.get("checkpoint")
        if isinstance(checkpoint, str):
            self.checkpoint = checkpoint

    def restore_training(
        self, pre: Mapping[str, Any], training: Mapping[str, Any]
    ) -> None:
        self._pre_mean_kld = float(pre["mean_kld"])
        checkpoint_path = Path(str(training["checkpoint_path"])).expanduser().resolve()
        if _sha256(checkpoint_path) != str(training["checkpoint_sha256"]):
            raise ValueError("HF-uniform resident checkpoint bytes do not match the training receipt")
        self._checkpoint_path = checkpoint_path
        self._loaded_checkpoint_path = None
        self._ensure_trainer()
        self.checkpoint = str(training["checkpoint"])
        self.checkpoint_sha256 = str(training["checkpoint_sha256"])


def open_provider(**kwargs: Any) -> HFUniformPhysicalProvider:
    return HFUniformPhysicalProvider(**kwargs)


__all__ = [
    "HF_UNIFORM_ARTIFACT_MODE",
    "HF_UNIFORM_CHECKPOINT_FORMAT",
    "HF_UNIFORM_MODEL_LAYER_COUNT",
    "HFUniformPhysicalProvider",
    "open_hf_uniform_resident_adapter",
    "open_provider",
]
