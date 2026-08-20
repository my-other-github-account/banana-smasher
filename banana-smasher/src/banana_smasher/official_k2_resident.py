from __future__ import annotations

import hashlib
import importlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .resident_training import (
    ParameterDescriptor,
    ResidentModelAdapter,
    ResidentTrainingPlan,
)


def _load_symbol(specification: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("model_source must use 'module:factory' spelling")
    return getattr(importlib.import_module(module_name), attribute)


class OfficialK2PackedResidentAdapter(ResidentModelAdapter):
    """Concrete adapter for the proven fully resident grouped official-K2 path.

    ``model_source`` is a provider factory. The factory receives the public plan,
    this adapter, and the packaged ``FullyResidentGroupedV7Experts`` and
    ``grouped_packed_projection`` primitives. It returns a staged backend exposing
    ``resident_parameters()``, ``loss_for_windows()``, and residency metadata.
    This keeps checkpoint/model-family loading outside the trainer while retaining
    the exact packed resident kernels and sparse Adam semantics.
    """

    adapter_name = "official-k2-packed"
    adapter_version = "1"

    def __init__(
        self,
        *,
        model_source: str | None,
        backend_factory: Callable[..., Any] | None = None,
        expected_model_index_sha256: str | None = None,
        **options: Any,
    ) -> None:
        self.model_source = model_source
        self.backend_factory = backend_factory
        self.expected_model_index_sha256 = expected_model_index_sha256
        self.options = dict(options)
        self.plan: ResidentTrainingPlan | None = None
        self.backend: Any | None = None
        self._torch: Any | None = None
        self._descriptors: tuple[ParameterDescriptor, ...] = ()
        self._parameters: dict[str, Any] = {}
        self._stable_to_name: dict[str, str] = {}
        self._selected: dict[str, tuple[ParameterDescriptor, ...]] = {}
        self._optimizer_group_descriptors: list[tuple[ParameterDescriptor, ...]] = []
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

        from .grouped_k2 import grouped_packed_projection
        from .resident_v7_experts import FullyResidentGroupedV7Experts

        self.plan = plan
        self._torch = torch
        index = plan.model_root / "model.safetensors.index.json"
        observed = hashlib.sha256(index.read_bytes()).hexdigest()
        expected = self.expected_model_index_sha256
        if expected is None:
            raise ValueError(
                "official-k2-packed requires expected_model_index_sha256 in adapter_options"
            )
        if observed != expected.lower():
            raise ValueError(
                f"official-K2 model index SHA-256 mismatch: {observed} != {expected.lower()}"
            )
        self._model_index_sha256 = observed
        factory = self.backend_factory
        if factory is None:
            if self.model_source is None:
                raise ValueError("official-k2-packed requires a model_source provider factory")
            factory = _load_symbol(self.model_source)
        self.backend = factory(
            plan=plan,
            adapter=self,
            expert_factory=FullyResidentGroupedV7Experts,
            grouped_projection=grouped_packed_projection,
            **self.options,
        )
        rows = tuple(self.backend.resident_parameters())
        descriptors: list[ParameterDescriptor] = []
        parameters: dict[str, Any] = {}
        stable_to_name: dict[str, str] = {}
        for descriptor, parameter in rows:
            if not isinstance(descriptor, ParameterDescriptor):
                raise TypeError("resident_parameters must yield ParameterDescriptor/tensor pairs")
            if descriptor.name in parameters or descriptor.stable_id in stable_to_name:
                raise ValueError(f"duplicate official-K2 parameter identity: {descriptor.stable_id}")
            descriptors.append(descriptor)
            parameters[descriptor.name] = parameter
            stable_to_name[descriptor.stable_id] = descriptor.name
        self._descriptors = tuple(descriptors)
        self._parameters = parameters
        self._stable_to_name = stable_to_name
        return dict(self.backend.residency_metadata())

    def parameters(self) -> Iterable[ParameterDescriptor]:
        return self._descriptors

    def configure_parameter_groups(
        self, groups: Mapping[str, tuple[ParameterDescriptor, ...]]
    ) -> None:
        if self.plan is None or self._torch is None:
            raise RuntimeError("official-K2 adapter is not staged")
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
            raise ValueError("official-K2 selectors produced no optimizer groups")
        optimizer_options = dict(self.plan.optimizer_options)
        optimizer_options.setdefault("foreach", False)
        optimizer_name = self.plan.optimizer.lower()
        if optimizer_name == "adam":
            optimizer_class = self._torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_class = self._torch.optim.AdamW
        else:
            raise ValueError(f"official-K2 adapter does not support optimizer {optimizer_name!r}")
        self.optimizer = optimizer_class(optimizer_groups, **optimizer_options)
        self._selected = dict(groups)
        self._optimizer_group_descriptors = descriptor_groups

    def _synchronize(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("official-K2 optimizer is not configured")
        self.optimizer.zero_grad(set_to_none=True)

    def train_microbatch(
        self, windows: tuple[int, ...], *, tokens: int, loss_scale: float
    ) -> Mapping[str, float]:
        if self.backend is None:
            raise RuntimeError("official-K2 backend is not resident")
        self._synchronize()
        forward_started = time.perf_counter()
        value = self.backend.loss_for_windows(windows, tokens=tokens)
        if isinstance(value, tuple):
            loss, comm_seconds = value
        else:
            loss, comm_seconds = value, 0.0
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
            "comm_seconds": float(comm_seconds),
        }

    def optimizer_step(self, learning_rates: dict[str, float]) -> float:
        if self.optimizer is None:
            raise RuntimeError("official-K2 optimizer is not configured")
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rates[str(group["group_name"])]
        self._synchronize()
        started = time.perf_counter()
        self.optimizer.step()
        self._synchronize()
        self._last_learning_rates = dict(learning_rates)
        return time.perf_counter() - started

    def trainable_state_dict(self) -> Mapping[str, Any]:
        return {
            descriptor.stable_id: self._parameters[descriptor.name].detach().cpu().clone()
            for descriptors in self._selected.values()
            for descriptor in descriptors
        }

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._torch is None:
            raise RuntimeError("official-K2 adapter is not staged")
        expected = {
            descriptor.stable_id
            for descriptors in self._selected.values()
            for descriptor in descriptors
        }
        if set(state) != expected:
            raise ValueError("official-K2 checkpoint trainable IDs do not match selection")
        with self._torch.no_grad():
            for stable_id, value in state.items():
                parameter = self._parameters[self._stable_to_name[stable_id]]
                source = self._torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
                parameter.copy_(source)

    def optimizer_state_dict(self) -> Mapping[str, Any]:
        if self.optimizer is None:
            return {}
        raw = self.optimizer.state_dict()
        id_to_stable: dict[int, str] = {}
        groups: list[dict[str, Any]] = []
        for group, descriptors in zip(raw["param_groups"], self._optimizer_group_descriptors):
            parameter_ids = list(group["params"])
            if len(parameter_ids) != len(descriptors):
                raise RuntimeError("official-K2 optimizer parameter-group identity drift")
            stable_ids = [row.stable_id for row in descriptors]
            id_to_stable.update(zip(parameter_ids, stable_ids))
            groups.append({**group, "params": stable_ids})
        # Adam state is sparse by design: no-gradient selected parameters remain in
        # param_groups and simply do not have an entry in this mapping.
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
                raise RuntimeError("official-K2 optimizer is not configured")
            return
        current = self.optimizer.state_dict()
        stable_to_current: dict[str, int] = {}
        rebuilt_groups: list[dict[str, Any]] = []
        saved_groups = list(state.get("param_groups", ()))
        if len(saved_groups) != len(self._optimizer_group_descriptors):
            raise ValueError("official-K2 checkpoint optimizer group count drift")
        for current_group, saved_group, descriptors in zip(
            current["param_groups"], saved_groups, self._optimizer_group_descriptors
        ):
            stable_ids = [row.stable_id for row in descriptors]
            if list(saved_group["params"]) != stable_ids:
                raise ValueError("official-K2 sparse optimizer parameter IDs drift")
            stable_to_current.update(zip(stable_ids, current_group["params"]))
            rebuilt_groups.append({**saved_group, "params": list(current_group["params"])})
        saved_state = state.get("state", {})
        unknown = set(saved_state) - set(stable_to_current)
        if unknown:
            raise ValueError(f"official-K2 optimizer state has unknown parameter IDs: {sorted(unknown)}")
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

    def deploy_export(self, checkpoint: Path, destination: Path) -> Mapping[str, Any]:
        if self.backend is None or not hasattr(self.backend, "deploy_export"):
            raise NotImplementedError(
                "official-K2 provider must expose deploy_export for repair/export integration"
            )
        return self.backend.deploy_export(checkpoint, destination)


__all__ = ["OfficialK2PackedResidentAdapter"]
