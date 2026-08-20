from __future__ import annotations

import fnmatch
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
from abc import ABC
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


RESIDENT_TRAINING_SCHEMA = "banana-smasher-resident-training-v1"
RESIDENT_EXECUTION_RAIL = "resident-in-memory"
LEGACY_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "execution_mode",
        "input_checkpoint",
        "training_mode",
        "offline",
        "replay",
        "staged_files",
        "subprocess",
        "reload_per_step",
    }
)


@dataclass(frozen=True)
class ParameterDescriptor:
    """A named model parameter and its adapter-stable checkpoint identity."""

    name: str
    family: str
    parameter_id: str | None = None

    @property
    def stable_id(self) -> str:
        return self.name if self.parameter_id is None else self.parameter_id


@dataclass(frozen=True)
class ParameterGroupPlan:
    """Optimizer settings and explicit selectors for one parameter group."""

    name: str
    lr: float
    warmup_updates: int = 0
    families: tuple[str, ...] = ()
    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterGroupPlan":
        options = dict(value.get("options", {}))
        reserved = sorted({"params", "lr", "group_name"} & set(options))
        if reserved:
            raise ValueError(
                f"parameter-group options contain reserved optimizer keys: {reserved}"
            )
        return cls(
            name=str(value["name"]),
            lr=float(value["lr"]),
            warmup_updates=int(value.get("warmup_updates", 0)),
            families=tuple(map(str, value.get("families", ()))),
            include=tuple(map(str, value.get("include", ("*",)))),
            exclude=tuple(map(str, value.get("exclude", ()))),
            options=options,
        )

    def matches(self, parameter: ParameterDescriptor) -> bool:
        family_matches = not self.families or parameter.family in self.families
        included = any(
            fnmatch.fnmatchcase(parameter.name, pattern) for pattern in self.include
        )
        excluded = any(
            fnmatch.fnmatchcase(parameter.name, pattern) for pattern in self.exclude
        )
        return family_matches and included and not excluded


@dataclass(frozen=True)
class TrainingTopology:
    world_size: int = 1
    rank: int = 0
    layer_split: tuple[tuple[int, int], ...] = ((0, 0),)
    master_addr: str | None = None
    master_port: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TrainingTopology":
        raw = dict(value or {})
        world_size = int(raw.get("world_size", 1))
        split_raw = raw.get("layer_split", [[0, 0]])
        split = tuple((int(row[0]), int(row[1])) for row in split_raw)
        result = cls(
            world_size=world_size,
            rank=int(raw.get("rank", 0)),
            layer_split=split,
            master_addr=(
                None if raw.get("master_addr") is None else str(raw["master_addr"])
            ),
            master_port=(
                None if raw.get("master_port") is None else int(raw["master_port"])
            ),
        )
        if result.world_size < 1 or not 0 <= result.rank < result.world_size:
            raise ValueError("topology rank must be within world_size")
        if len(result.layer_split) != result.world_size:
            raise ValueError("topology layer_split must contain one range per rank")
        if any(first < 0 or last < first for first, last in result.layer_split):
            raise ValueError("topology layer ranges must be nonnegative and ordered")
        return result

    @property
    def layers_for_rank(self) -> tuple[int, int]:
        return self.layer_split[self.rank]


@dataclass(frozen=True)
class ResidentTrainingPlan:
    """Serializable public configuration for one fully resident trainer process."""

    model_source: str | None
    model_adapter: str | None
    model_root: Path
    payload_root: Path
    run_root: Path
    topology: TrainingTopology
    windows: tuple[int, ...]
    tokens_per_window: int
    microbatch: int
    gradient_accumulation: int
    updates: int
    optimizer: str
    optimizer_options: Mapping[str, Any]
    parameter_groups: tuple[ParameterGroupPlan, ...]
    adapter_options: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResidentTrainingPlan":
        schema = value.get("schema", RESIDENT_TRAINING_SCHEMA)
        if schema != RESIDENT_TRAINING_SCHEMA:
            raise ValueError(
                f"training schema must be {RESIDENT_TRAINING_SCHEMA!r}, got {schema!r}"
            )
        execution_rail = value.get("execution_rail", RESIDENT_EXECUTION_RAIL)
        if execution_rail != RESIDENT_EXECUTION_RAIL:
            raise ValueError(
                "training execution_rail must be resident-in-memory; offline, replay, "
                "staged-file, subprocess, and reload-per-step modes are not public"
            )
        forbidden = sorted(LEGACY_TRAINING_CONFIG_FIELDS & set(value))
        if forbidden:
            raise ValueError(
                "legacy training configuration fields are not public: "
                + ", ".join(forbidden)
            )
        model_source = value.get("model_source")
        model_adapter = value.get("model_adapter")
        if model_source is None and model_adapter is None:
            raise ValueError("training plan requires model_source or model_adapter")
        for path_key in ("model_root", "payload_root"):
            raw_path = value.get(path_key)
            if raw_path is not None and "://" in str(raw_path):
                raise ValueError(f"{path_key} must be a local filesystem path")
        groups = tuple(
            ParameterGroupPlan.from_dict(row)
            for row in value.get("parameter_groups", ())
        )
        if not groups:
            raise ValueError("training plan requires at least one parameter group")
        result = cls(
            model_source=None if model_source is None else str(model_source),
            model_adapter=None if model_adapter is None else str(model_adapter),
            model_root=Path(value["model_root"]),
            payload_root=Path(value["payload_root"]),
            run_root=Path(value["run_root"]),
            topology=TrainingTopology.from_dict(value.get("topology")),
            windows=tuple(map(int, value.get("windows", ()))),
            tokens_per_window=int(value.get("tokens_per_window", 0)),
            microbatch=int(value.get("microbatch", 1)),
            gradient_accumulation=int(value.get("gradient_accumulation", 1)),
            updates=int(value.get("updates", 1)),
            optimizer=str(value.get("optimizer", "adam")),
            optimizer_options=dict(value.get("optimizer_options", {})),
            parameter_groups=groups,
            adapter_options=dict(value.get("adapter_options", {})),
        )
        if not result.windows or any(window < 0 for window in result.windows):
            raise ValueError("training windows must contain nonnegative indices")
        for name in (
            "tokens_per_window",
            "microbatch",
            "gradient_accumulation",
            "updates",
        ):
            if getattr(result, name) < 1:
                raise ValueError(f"{name} must be positive")
        if any(group.lr <= 0 or group.warmup_updates < 0 for group in groups):
            raise ValueError(
                "parameter-group lr must be positive and warmup nonnegative"
            )
        names = [group.name for group in groups]
        if len(set(names)) != len(names):
            raise ValueError("parameter-group names must be unique")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "ResidentTrainingPlan":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("training config JSON root must be an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDENT_TRAINING_SCHEMA,
            "execution_rail": RESIDENT_EXECUTION_RAIL,
            "model_source": self.model_source,
            "model_adapter": self.model_adapter,
            "model_root": str(self.model_root),
            "payload_root": str(self.payload_root),
            "run_root": str(self.run_root),
            "topology": {
                "world_size": self.topology.world_size,
                "rank": self.topology.rank,
                "layer_split": [list(row) for row in self.topology.layer_split],
                "master_addr": self.topology.master_addr,
                "master_port": self.topology.master_port,
            },
            "windows": list(self.windows),
            "tokens_per_window": self.tokens_per_window,
            "microbatch": self.microbatch,
            "gradient_accumulation": self.gradient_accumulation,
            "updates": self.updates,
            "optimizer": self.optimizer,
            "optimizer_options": dict(self.optimizer_options),
            "parameter_groups": [asdict(group) for group in self.parameter_groups],
            "adapter_options": dict(self.adapter_options),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def select_parameter_groups(
    parameters: Iterable[ParameterDescriptor],
    groups: Iterable[ParameterGroupPlan],
) -> dict[str, tuple[ParameterDescriptor, ...]]:
    """Resolve include/exclude/family selectors and reject optimizer overlap."""

    group_rows = tuple(groups)
    selected: dict[str, list[ParameterDescriptor]] = {
        group.name: [] for group in group_rows
    }
    owners: dict[str, str] = {}
    for parameter in sorted(parameters, key=lambda item: item.name):
        for group in group_rows:
            if not group.matches(parameter):
                continue
            if parameter.name in owners:
                raise ValueError(
                    f"parameter {parameter.name!r} selected by multiple parameter groups: "
                    f"{owners[parameter.name]!r}, {group.name!r}"
                )
            owners[parameter.name] = group.name
            selected[group.name].append(parameter)
    return {name: tuple(rows) for name, rows in selected.items()}


REMOTE_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "davfs",
    "fuse",
    "fuseblk",
    "glusterfs",
    "lustre",
    "nfs",
    "nfs4",
    "smb3",
    "sshfs",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_local_compute_path(path: str | Path, label: str) -> dict[str, str]:
    """Resolve one compute input and reject known network/FUSE mounts."""

    if "://" in str(path):
        raise ValueError(f"{label} must be a local filesystem path")
    resolved = Path(path).expanduser().resolve(strict=True)
    fstype = "local"
    source = "local"
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.is_file():
        candidates: list[tuple[int, str, str]] = []
        for line in mountinfo.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            try:
                separator = fields.index("-")
            except ValueError:
                continue
            mountpoint = fields[4].replace("\\040", " ")
            try:
                resolved.relative_to(mountpoint)
            except ValueError:
                continue
            candidates.append(
                (len(mountpoint), fields[separator + 1], fields[separator + 2])
            )
        if not candidates:
            raise ValueError(f"{label} has no local mount identity: {resolved}")
        _length, fstype, source = max(candidates)
    elif sys.platform == "darwin" and resolved.is_relative_to("/Volumes"):
        raise ValueError(f"{label} must not use a mounted compute input: {resolved}")
    if fstype in REMOTE_FILESYSTEMS or fstype.startswith("fuse."):
        raise ValueError(
            f"{label} must be copied local before compute: {resolved} is {fstype} {source}"
        )
    return {"path": str(resolved), "fstype": fstype, "source": source}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _encode_state(value: Any, tensors: dict[str, Any], path: str = "state") -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = np.array(value, copy=True, order="C")
        return {"kind": "tensor", "key": key}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _encode_state(value.detach().cpu().numpy(), tensors, path)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                [
                    _encode_state(key, tensors, f"{path}.key"),
                    _encode_state(item, tensors, f"{path}.{key}"),
                ]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _encode_state(item, tensors, f"{path}.{index}")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _encode_state(item, tensors, f"{path}.{index}")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TypeError(
        f"checkpoint state at {path} has unsupported type {type(value).__name__}"
    )


def _decode_state(node: Mapping[str, Any], tensors: Mapping[str, Any]) -> Any:
    kind = node["kind"]
    if kind == "tensor":
        return tensors[str(node["key"])]
    if kind == "mapping":
        return {
            _decode_state(key, tensors): _decode_state(value, tensors)
            for key, value in node["items"]
        }
    if kind == "tuple":
        return tuple(_decode_state(item, tensors) for item in node["items"])
    if kind == "list":
        return [_decode_state(item, tensors) for item in node["items"]]
    if kind == "path":
        return Path(str(node["value"]))
    if kind == "scalar":
        return node.get("value")
    raise ValueError(f"unknown checkpoint state node kind: {kind!r}")


def _checkpoint_document(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    checkpoint = Path(path).resolve(strict=True)
    with safe_open(checkpoint, framework="np") as handle:
        metadata = handle.metadata()
        raw = metadata.get("banana_smasher_checkpoint") if metadata else None
        if raw is None:
            raise ValueError("checkpoint is missing Banana Smasher metadata")
        document = json.loads(raw)
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    if document.get("format") != "banana-smasher-resident-checkpoint-v1":
        raise ValueError("unsupported resident checkpoint format")
    return document, tensors


def _checkpoint_info(path: str | Path) -> dict[str, Any]:
    document, _tensors = _checkpoint_document(path)
    return {key: value for key, value in document.items() if key != "state_structure"}


def _read_train_status(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root) / "TRAIN_STATUS.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "banana-smasher-resident-training-status-v1"
    ):
        raise ValueError(f"not a resident training status file: {path}")
    return value


def load_resident_adapter(plan: ResidentTrainingPlan) -> "ResidentModelAdapter":
    specification = plan.model_adapter or plan.model_source
    if specification is None:
        raise ValueError("training plan has no model adapter")
    if specification == "official-k2-packed":
        from .official_k2_resident import OfficialK2PackedResidentAdapter

        return OfficialK2PackedResidentAdapter(
            model_source=plan.model_source, **dict(plan.adapter_options)
        )
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError(
            "model_adapter must be 'official-k2-packed' or 'module:object'"
        )
    candidate = getattr(importlib.import_module(module_name), attribute)
    signature = inspect.signature(candidate)
    accepts_kwargs = any(
        row.kind is inspect.Parameter.VAR_KEYWORD
        for row in signature.parameters.values()
    )
    supplied = dict(plan.adapter_options)
    if accepts_kwargs or "model_source" in signature.parameters:
        supplied["model_source"] = plan.model_source
    if accepts_kwargs or "plan" in signature.parameters:
        supplied["plan"] = plan
    adapter = candidate(**supplied)
    if not isinstance(adapter, ResidentModelAdapter):
        raise TypeError(
            f"model adapter {specification!r} did not return ResidentModelAdapter"
        )
    return adapter


class ResidentModelAdapter(ABC):
    """Provider boundary for a model family that can remain staged in one process."""

    adapter_name = "resident-adapter"
    adapter_version = "1"

    def metadata(self) -> dict[str, Any]:
        return {"name": self.adapter_name, "version": self.adapter_version}

    def stage(self, plan: ResidentTrainingPlan) -> Mapping[str, Any]:
        raise NotImplementedError

    def parameters(self) -> Iterable[ParameterDescriptor]:
        raise NotImplementedError

    def configure_parameter_groups(
        self, groups: Mapping[str, tuple[ParameterDescriptor, ...]]
    ) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    def train_microbatch(
        self, windows: tuple[int, ...], *, tokens: int, loss_scale: float
    ) -> Mapping[str, float]:
        raise NotImplementedError

    def optimizer_step(self, learning_rates: dict[str, float]) -> float:
        raise NotImplementedError

    def trainable_state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def optimizer_state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_optimizer_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"{self.adapter_name} does not accept optimizer state")

    def scheduler_state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_scheduler_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"{self.adapter_name} does not accept scheduler state")

    def deploy_export(self, checkpoint: Path, destination: Path) -> Mapping[str, Any]:
        raise NotImplementedError(f"{self.adapter_name} has no deploy/export hook")


@dataclass(frozen=True)
class TrainStepTiming:
    update: int
    loss: float
    tokens: int
    forward_seconds: float
    backward_seconds: float
    comm_seconds: float
    optimizer_seconds: float
    total_seconds: float

    @property
    def phase_seconds(self) -> dict[str, float]:
        return {
            "forward": self.forward_seconds,
            "backward": self.backward_seconds,
            "communication": self.comm_seconds,
            "optimizer": self.optimizer_seconds,
            "update_total": self.total_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "phase_seconds": self.phase_seconds}


class _ResidentTrainer:
    """Stages an adapter once, then executes repeated in-memory training updates."""

    def __init__(
        self,
        plan: ResidentTrainingPlan,
        *,
        adapter: ResidentModelAdapter,
        phase_observer: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.plan = plan
        self.adapter = adapter
        self.is_resident = False
        self.update = 0
        self.residency: dict[str, Any] = {}
        self.parameter_groups: dict[str, tuple[ParameterDescriptor, ...]] = {}
        self.status_path = self.plan.run_root / "TRAIN_STATUS.json"
        self._phase_observer = phase_observer

    @property
    def model_instance(self) -> ResidentModelAdapter:
        """The one adapter/model object retained for this process lifetime."""

        return self.adapter

    def _emit_phase(self, phase: str, **fields: Any) -> None:
        if self._phase_observer is not None:
            self._phase_observer(
                phase,
                {
                    "phase": phase,
                    "update": self.update,
                    "resident": self.is_resident,
                    **fields,
                },
            )

    def initialize(self) -> dict[str, Any]:
        if self.is_resident:
            return dict(self.residency)
        locality = {
            "model_root": require_local_compute_path(
                self.plan.model_root, "model root"
            ),
            "payload_root": require_local_compute_path(
                self.plan.payload_root, "payload root"
            ),
        }
        self.plan.run_root.mkdir(parents=True, exist_ok=True)
        staged_plan = replace(
            self.plan,
            model_root=Path(locality["model_root"]["path"]),
            payload_root=Path(locality["payload_root"]["path"]),
        )
        started = time.perf_counter()
        staged = dict(self.adapter.stage(staged_plan))
        self.parameter_groups = select_parameter_groups(
            self.adapter.parameters(), self.plan.parameter_groups
        )
        if not any(self.parameter_groups.values()):
            raise ValueError(
                "parameter selectors did not select any trainable parameters"
            )
        self.adapter.configure_parameter_groups(self.parameter_groups)
        self.residency = {**staged, "locality": locality}
        self.is_resident = True
        self._write_status("resident_ready")
        self._emit_phase("resident_load", elapsed_seconds=time.perf_counter() - started)
        return dict(self.residency)

    def _learning_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for group in self.plan.parameter_groups:
            multiplier = (
                1.0
                if group.warmup_updates == 0
                else min((self.update + 1) / group.warmup_updates, 1.0)
            )
            rates[group.name] = group.lr * multiplier
        return rates

    def train_step(self) -> TrainStepTiming:
        if not self.is_resident:
            self.initialize()
        batches = tuple(
            self.plan.windows[index : index + self.plan.microbatch]
            for index in range(0, len(self.plan.windows), self.plan.microbatch)
        )
        if any(len(batch) != self.plan.microbatch for batch in batches):
            raise ValueError("training windows must divide evenly into microbatches")
        if len(batches) != self.plan.gradient_accumulation:
            raise ValueError(
                "windows/microbatch must equal gradient_accumulation for one update"
            )
        self.adapter.zero_grad()
        started = time.perf_counter()
        forward = backward = comm = loss = 0.0
        scale = 1.0 / len(batches)
        for batch in batches:
            row = self.adapter.train_microbatch(
                batch,
                tokens=len(batch) * self.plan.tokens_per_window,
                loss_scale=scale,
            )
            loss += float(row.get("loss", 0.0)) * scale
            forward += float(row.get("forward_seconds", 0.0))
            backward += float(row.get("backward_seconds", 0.0))
            comm += float(row.get("comm_seconds", 0.0))
        optimizer_seconds = float(self.adapter.optimizer_step(self._learning_rates()))
        result = TrainStepTiming(
            update=self.update,
            loss=loss,
            tokens=len(self.plan.windows) * self.plan.tokens_per_window,
            forward_seconds=forward,
            backward_seconds=backward,
            comm_seconds=comm,
            optimizer_seconds=optimizer_seconds,
            total_seconds=time.perf_counter() - started,
        )
        self.update += 1
        self._write_status("training", last_step=result.to_dict())
        for phase, elapsed in result.phase_seconds.items():
            self._emit_phase(phase, elapsed_seconds=elapsed)
        return result

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        if not self.is_resident:
            self.initialize()
        started = time.perf_counter()
        from safetensors.numpy import save_file
        import numpy as np

        checkpoint = (
            self.plan.run_root / "checkpoints" / f"UPDATE_{self.update:08d}.safetensors"
            if path is None
            else Path(path)
        )
        checkpoint = checkpoint.resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trainables": dict(self.adapter.trainable_state_dict()),
            # Optimizer state is intentionally sparse: selected parameters remain in
            # param_groups even when Adam has not created a state row for them.
            "optimizer": dict(self.adapter.optimizer_state_dict()),
            "scheduler": dict(self.adapter.scheduler_state_dict()),
        }
        tensors: dict[str, Any] = {
            "__checkpoint_marker__": np.asarray([self.update], dtype=np.int64)
        }
        structure = _encode_state(state, tensors)
        group_ids = {
            name: [parameter.stable_id for parameter in parameters]
            for name, parameters in self.parameter_groups.items()
        }
        document = {
            "format": "banana-smasher-resident-checkpoint-v1",
            "next_update": self.update,
            "config": self.plan.to_dict(),
            "adapter": self.adapter.metadata(),
            "parameter_groups": group_ids,
            "trainable_count": sum(len(values) for values in group_ids.values()),
            "state_structure": structure,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint.name}.", suffix=".safetensors", dir=checkpoint.parent
        )
        os.close(descriptor)
        Path(temporary_name).unlink()
        try:
            save_file(
                tensors,
                temporary_name,
                metadata={
                    "banana_smasher_checkpoint": json.dumps(
                        document, sort_keys=True, separators=(",", ":")
                    )
                },
            )
            with open(temporary_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_name, checkpoint)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        digest = _sha256_file(checkpoint)
        _atomic_json(
            checkpoint.parent / "LATEST.json",
            {
                "format": document["format"],
                "checkpoint": checkpoint.name,
                "sha256": digest,
                "next_update": self.update,
            },
        )
        self._write_status(
            "checkpoint_saved", checkpoint=str(checkpoint), checkpoint_sha256=digest
        )
        self._emit_phase(
            "checkpoint_save",
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=str(checkpoint),
        )
        return checkpoint

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        if not self.is_resident:
            self.initialize()
        started = time.perf_counter()
        resident_model = self.model_instance
        local_checkpoint = Path(
            require_local_compute_path(path, "resume checkpoint")["path"]
        )
        document, tensors = _checkpoint_document(local_checkpoint)
        saved_config = document.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("checkpoint is missing resident training config metadata")
        current_config = json.loads(self.plan.to_json())
        immutable_resume_fields = (
            "model_source",
            "model_adapter",
            "model_root",
            "payload_root",
            "topology",
            "windows",
            "tokens_per_window",
            "microbatch",
            "gradient_accumulation",
            "optimizer",
            "optimizer_options",
            "parameter_groups",
            "adapter_options",
        )
        for key in immutable_resume_fields:
            if saved_config.get(key) != current_config.get(key):
                raise ValueError(
                    f"resume config mismatch for {key}: "
                    f"{saved_config.get(key)!r} != {current_config.get(key)!r}"
                )
        adapter = document.get("adapter", {})
        current_adapter = self.adapter.metadata()
        for key in ("name", "version", "model_index_sha256"):
            saved = adapter.get(key)
            if saved is not None and saved != current_adapter.get(key):
                raise ValueError(
                    f"checkpoint adapter {key} mismatch: {saved!r} != "
                    f"{current_adapter.get(key)!r}"
                )
        current_groups = {
            name: [parameter.stable_id for parameter in parameters]
            for name, parameters in self.parameter_groups.items()
        }
        if document.get("parameter_groups") != current_groups:
            raise ValueError(
                "checkpoint stable parameter IDs do not match this adapter"
            )
        state = _decode_state(document["state_structure"], tensors)
        self.adapter.load_trainable_state_dict(state["trainables"])
        self.adapter.load_optimizer_state_dict(state["optimizer"])
        self.adapter.load_scheduler_state_dict(state["scheduler"])
        self.update = int(document["next_update"])
        if self.model_instance is not resident_model:  # pragma: no cover
            raise RuntimeError("checkpoint hot-swap reconstructed the resident model")
        self._write_status("checkpoint_loaded", checkpoint=str(local_checkpoint))
        self._emit_phase(
            "checkpoint_hot_swap",
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=str(local_checkpoint),
        )
        return {
            key: value for key, value in document.items() if key != "state_structure"
        }

    def deploy_export(
        self, destination: str | Path, *, checkpoint: str | Path | None = None
    ) -> dict[str, Any]:
        checkpoint_path = (
            self.save_checkpoint() if checkpoint is None else Path(checkpoint)
        )
        output = Path(destination).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        return dict(self.adapter.deploy_export(checkpoint_path.resolve(), output))

    def _write_status(self, phase: str, **fields: Any) -> None:
        _atomic_json(
            self.status_path,
            {
                "schema": "banana-smasher-resident-training-status-v1",
                "phase": phase,
                "resident": self.is_resident,
                "update": self.update,
                "adapter": self.adapter.metadata(),
                "run_root": str(self.plan.run_root.resolve()),
                **fields,
            },
        )


class ResidentTrainingSession:
    """Sole public training path: API continuation on one resident model.

    Checkpoints are recovery/output artifacts. Loading one hot-swaps state into
    the existing adapter; it never constructs a replacement model or starts a
    subprocess.
    """

    def __init__(self, trainer: _ResidentTrainer) -> None:
        self._trainer = trainer

    @classmethod
    def open(
        cls,
        plan: ResidentTrainingPlan,
        *,
        adapter: ResidentModelAdapter | None = None,
        phase_observer: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> "ResidentTrainingSession":
        selected = load_resident_adapter(plan) if adapter is None else adapter
        trainer = _ResidentTrainer(
            plan, adapter=selected, phase_observer=phase_observer
        )
        trainer.initialize()
        return cls(trainer)

    @property
    def model_instance(self) -> ResidentModelAdapter:
        return self._trainer.model_instance

    @property
    def update(self) -> int:
        return self._trainer.update

    def hot_swap_checkpoint(self, checkpoint: str | Path) -> dict[str, Any]:
        return self._trainer.load_checkpoint(checkpoint)

    def continue_updates(
        self,
        count: int,
        *,
        checkpoint_every: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("resident continuation count must be a positive integer")
        if checkpoint_every is not None and (
            isinstance(checkpoint_every, bool)
            or not isinstance(checkpoint_every, int)
            or checkpoint_every < 1
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        model = self.model_instance
        steps: list[dict[str, Any]] = []
        checkpoints: list[str] = []
        for index in range(count):
            steps.append(self._trainer.train_step().to_dict())
            if checkpoint_every is not None and (index + 1) % checkpoint_every == 0:
                checkpoints.append(str(self._trainer.save_checkpoint()))
        if self.model_instance is not model:  # pragma: no cover
            raise RuntimeError("resident continuation reconstructed the model")
        return {
            "schema": RESIDENT_TRAINING_SCHEMA,
            "execution_rail": RESIDENT_EXECUTION_RAIL,
            "updates_completed": count,
            "next_update": self.update,
            "steps": steps,
            "checkpoints": checkpoints,
        }

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        return self._trainer.save_checkpoint(path)


__all__ = [
    "RESIDENT_EXECUTION_RAIL",
    "RESIDENT_TRAINING_SCHEMA",
    "ParameterDescriptor",
    "ParameterGroupPlan",
    "ResidentModelAdapter",
    "ResidentTrainingSession",
    "ResidentTrainingPlan",
    "TrainStepTiming",
    "TrainingTopology",
    "load_resident_adapter",
    "require_local_compute_path",
    "select_parameter_groups",
]
