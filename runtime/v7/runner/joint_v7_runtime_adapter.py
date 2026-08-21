#!/usr/bin/env python3
"""Exact PlaneSource seam for the proven joint whole-model TRAIN runtime."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LAYERS = 43
CANONICAL_BATCH = 4
DEFAULT_BASE = Path(
    "/home/dnola/missions/P487_REPAIR_RESUME_t_277fd2a6_s8/code/base_binrepair_e2e.py"
)


def _load_file(path: Path, name: str) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_symbol(binding: str) -> Any:
    """Load MODULE:SYMBOL or /absolute/file.py:SYMBOL without path guessing."""
    if not binding or ":" not in binding:
        raise RuntimeError("JOINT_V7_EXPERT_BASE must be MODULE:SYMBOL or FILE.py:SYMBOL")
    location, symbol = binding.rsplit(":", 1)
    if not location or not symbol:
        raise RuntimeError("JOINT_V7_EXPERT_BASE has an empty location or symbol")
    module = (
        _load_file(Path(location), "banana_smasher_joint_v7_expert_base")
        if location.endswith(".py") or Path(location).is_absolute()
        else importlib.import_module(location)
    )
    value = getattr(module, symbol, None)
    if not inspect.isclass(value):
        raise RuntimeError(f"JOINT_V7_EXPERT_BASE is not a class: {binding}")
    return value


def _exact_sources(plane_sources: Any, device: Any) -> dict[int, Any]:
    if not isinstance(plane_sources, Mapping):
        raise TypeError("plane_sources must be a mapping keyed by layers 0..42")
    try:
        keys = {int(key) for key in plane_sources}
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PlaneSource keys must be integer layers") from exc
    if keys != set(range(LAYERS)) or len(plane_sources) != LAYERS:
        raise RuntimeError("PlaneSource mapping must be exactly layers 0..42")
    sources = {layer: plane_sources[layer] for layer in range(LAYERS)}
    if len({id(source) for source in sources.values()}) != LAYERS:
        raise RuntimeError("all 43 supplied PlaneSource objects must be distinct")
    for layer, source in sources.items():
        if int(getattr(source, "layer", -1)) != layer:
            raise RuntimeError(f"PlaneSource layer identity drift at L{layer:03d}")
        master = getattr(source, "master", None)
        wire_lut = getattr(source, "wire_lut", None)
        member_path = getattr(source, "member_path", None)
        if master is None or not callable(wire_lut) or not callable(member_path):
            raise RuntimeError(f"L{layer:03d} is not the supplied compact-wire PlaneSource ABI")
        if tuple(master.shape) != (1024,) or str(master.dtype) != "torch.float32":
            raise RuntimeError(f"L{layer:03d} PlaneSource master must be FP32[1024]")
        if master.device != device and not (
            master.device.type == device.type == "cuda"
            and device.index is None
            and master.device.index == 0
        ):
            raise RuntimeError(
                f"L{layer:03d} PlaneSource device drift: {master.device} != {device}"
            )
    return sources


def _constructor_keyword(base_class: type) -> str | None:
    try:
        parameters = inspect.signature(base_class.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("V7 expert constructor must expose an inspectable PlaneSource seam") from exc
    for name in ("plane_source", "source"):
        if name in parameters:
            return name
    return None


def _bind_existing(instance: Any, source: Any, layer: int) -> None:
    for name in (
        "bind_plane_source",
        "install_plane_source",
        "bind_qtip_v7_source",
        "install_qtip_v7_source",
    ):
        method = getattr(instance, name, None)
        if not callable(method):
            continue
        result = method(source)
        if result not in (None, True):
            if not isinstance(result, Mapping) or int(result.get("layer", layer)) != layer:
                raise RuntimeError(f"L{layer:03d} PlaneSource binder rejected exact coverage")
        return
    raise RuntimeError(
        f"L{layer:03d} V7 expert lacks a constructor or explicit PlaneSource binder"
    )


def _make_experts(base_class: type, sources: dict[int, Any]) -> type:
    keyword = _constructor_keyword(base_class)

    class JointV7PlaneExperts(base_class):
        def __init__(self, layer: int, pilot: bool = True) -> None:
            layer = int(layer)
            if not pilot or layer not in sources:
                raise RuntimeError(f"joint V7 requires an admitted source for layer {layer}")
            source = sources[layer]
            if keyword is None:
                super().__init__(layer, pilot)
                _bind_existing(self, source, layer)
            else:
                super().__init__(layer, pilot, **{keyword: source})
            # Keep a direct identity reference without registering/copying the source.
            self.__dict__["_joint_v7_plane_source"] = source
            self._joint_v7_layer = layer

    JointV7PlaneExperts.__name__ = "JointV7PlaneExperts"
    JointV7PlaneExperts.__qualname__ = "JointV7PlaneExperts"
    return JointV7PlaneExperts


def _validate_bank(base: Any, admission: Mapping[str, Any], ordered: list[int], batch: int) -> None:
    admitted = list(
        map(
            int,
            admission["train_objective"]["full_model_train_bank"][
                "ordered_train_windows"
            ],
        )
    )
    base_bank = list(map(int, getattr(base, "TRAIN_WINS", ())))
    if batch != CANONICAL_BATCH:
        raise RuntimeError(f"historical joint batch must be {CANONICAL_BATCH}")
    if len(ordered) != 64 or ordered != admitted or ordered != base_bank:
        raise RuntimeError("exact ordered whole-model TRAIN bank drift")
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("ordered whole-model TRAIN bank contains duplicates")


def build_joint_v7_runtime(plane_sources, device, admission, ordered_train_windows, batch_size):
    """Instantiate Student/corpus/ActCache and expose the authentic joint loss."""
    if not isinstance(admission, Mapping):
        raise TypeError("admission must be the sealed mapping")
    if admission.get("framework") != "banana-smasher":
        raise RuntimeError("admission framework drift")
    sources = _exact_sources(plane_sources, device)
    ordered = list(map(int, ordered_train_windows))
    batch = int(batch_size)

    base_path = Path(os.environ.get("COMBO_BINREPAIR_BASE", str(DEFAULT_BASE)))
    base = _load_file(base_path, "banana_smasher_joint_v7_whole_model_base")
    for name in ("T", "ActCache", "batch_loss", "TRAIN_WINS"):
        if not hasattr(base, name):
            raise RuntimeError(f"whole-model base lacks required symbol {name}")
    _validate_bank(base, admission, ordered, batch)

    binding = os.environ.get("JOINT_V7_EXPERT_BASE")
    if binding is None:
        raise RuntimeError("JOINT_V7_EXPERT_BASE is required from the proven launch")
    expert_base = _load_symbol(binding)
    base.T.TrainableExperts = _make_experts(expert_base, sources)
    base.T.PILOT = tuple(range(LAYERS))

    student = base.T.Student()
    if not hasattr(student, "model") or not hasattr(student, "experts"):
        raise RuntimeError("whole-model Student ABI drift")
    if isinstance(student.experts, Mapping):
        if {int(layer) for layer in student.experts} != set(range(LAYERS)):
            raise RuntimeError("whole-model Student expert mapping must be exactly layers 0..42")
        expert_modules = [student.experts[layer] for layer in range(LAYERS)]
    else:
        expert_modules = list(student.experts)
    if len(expert_modules) != LAYERS:
        raise RuntimeError("whole-model Student must instantiate all 43 expert layers")
    for layer, module in enumerate(expert_modules):
        if getattr(module, "_joint_v7_plane_source", None) is not sources[layer]:
            raise RuntimeError(f"Student did not retain exact PlaneSource L{layer:03d}")

    # Dense repair surfaces are exposed later by the accepted runner. Freeze the
    # inherited model first, while keeping the 43 external FP32 LUT masters live.
    for parameter in student.model.parameters():
        parameter.requires_grad_(False)
    for source in sources.values():
        source.master.requires_grad_(True)

    corpus = base.T.load_corpus()
    activation_cache = base.ActCache(student)
    bank_positions = {win: index for index, win in enumerate(ordered)}

    def objective(wins, requires_grad=True):
        selected = list(map(int, wins))
        if not selected or len(selected) > batch:
            raise RuntimeError("joint TRAIN objective received an invalid batch size")
        try:
            positions = [bank_positions[win] for win in selected]
        except KeyError as exc:
            raise RuntimeError("joint TRAIN objective received a non-TRAIN window") from exc
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise RuntimeError("joint TRAIN objective window order drift")
        return base.batch_loss(
            student, corpus, activation_cache, selected, bool(requires_grad)
        )

    return {
        "B": base,
        "student": student,
        "corpus": corpus,
        "activation_cache": activation_cache,
        "plane_sources": sources,
        "objective": objective,
        "batch_loss": objective,
    }


__all__ = ["build_joint_v7_runtime"]
