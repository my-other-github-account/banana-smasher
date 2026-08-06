from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib
import json
from pathlib import Path
import time
from typing import Any

from ..production_update import run_full_depth_update
from ..token_sizing import MemoryBudget
from ..update_checkpoint import atomic_json

_REQUEST_SCHEMA = "banana-smasher-physical-repair-request-v1"
_INIT_MAX_SECONDS = 180.0
_REQUIRED_AOT_STATUS = "PASS_AUTHENTICATED_AOT"
_REQUIRED_DECODE_STATUS = "PASS_DECODED_ONCE"
_REQUIRED_LAYOUT_STATUS = "PASS_PERSISTENT_LAYOUTS"
_REQUIRED_STAGE_STATUS = "PASS_STAGED_LARGEST_FIRST"
_REQUIRED_CHECKPOINT_STATUS = "PASS_DEPTH_CHECKPOINTING"


def _load_object(specification: str) -> Any:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("runtime_factory must use the form 'module:callable'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"physical repair runtime factory is missing: {specification}"
        ) from exc


def _callable(value: Any, name: str) -> Callable[..., Any]:
    if not callable(value):
        raise RuntimeError(f"physical repair runtime lacks callable {name}")
    return value


def _require_status(value: Any, expected: str, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") != expected:
        raise RuntimeError(
            f"physical repair {name} did not pass: "
            f"{None if not isinstance(value, dict) else value.get('status')!r}"
        )
    return dict(value)


def _tensor_identity(values: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            id(value),
            int(value._version),
            tuple(int(item) for item in value.shape),
            str(value.dtype),
            str(value.device),
            bool(value.requires_grad),
        )
        for value in values
    )


def _validate_packed_indices(values: Any) -> list[Any]:
    import torch

    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError("decode_packed_indices must return at least one tensor")
    selected = list(values)
    invalid = [
        index
        for index, value in enumerate(selected)
        if not isinstance(value, torch.Tensor)
        or value.dtype
        not in {
            torch.uint8,
            torch.uint16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or value.requires_grad
        or value.grad is not None
    ]
    if invalid:
        raise RuntimeError(
            "packed indices must be resident integer tensors with no gradient surface: "
            f"invalid={invalid}"
        )
    return selected


def _parameters(modules: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result


def _validate_codebook_surface(bundle: dict[str, Any]) -> list[Any]:
    layers = list(bundle.get("layers", []))
    codebooks = list(bundle.get("codebooks", []))
    if not layers or not codebooks:
        raise RuntimeError("physical repair runtime requires layers and codebooks")
    trainable = _parameters(layers)
    if {id(value) for value in trainable} != {id(value) for value in codebooks}:
        raise RuntimeError(
            "physical repair trainable surface must contain codebooks only"
        )
    invalid = [
        index
        for index, value in enumerate(codebooks)
        if not value.requires_grad or value.grad is not None
    ]
    if invalid:
        raise RuntimeError(f"physical repair codebook surface is invalid: {invalid}")
    return codebooks


def _memory_budget(context: dict[str, Any]) -> MemoryBudget:
    supplied = context.get("memory_budget")
    if isinstance(supplied, MemoryBudget):
        return supplied
    sizing = context["memory_sizing"]
    return MemoryBudget(
        available_bytes=int(sizing["available_bytes"]),
        resident_frozen_bytes=int(sizing["resident_frozen_bytes"]),
        trainable_bytes=int(sizing["trainable_bytes"]),
        optimizer_bytes=int(sizing["optimizer_bytes"]),
        staging_bytes=int(sizing["staging_bytes"]),
        calibrated_activation_bytes_per_token=int(
            sizing["calibrated_activation_bytes_per_token"]
        ),
        os_floor_bytes=int(sizing["os_floor_bytes"]),
    )


class PhysicalRepairBackend:
    """One-initialization physical repair backend for the public update API.

    Model-specific tensor construction stays behind a runtime factory, while this
    backend owns the production ordering: authenticate the AOT, decode packed
    indices once, build persistent layer layouts, pre-stage the largest input
    slabs first, then run one checkpointed-depth cycle through the portable
    exactly-one-step update engine.
    """

    def __init__(self, request: dict[str, Any], context: dict[str, Any]) -> None:
        if request.get("schema") != _REQUEST_SCHEMA:
            raise ValueError(
                f"physical repair request schema must be {_REQUEST_SCHEMA!r}"
            )
        factory_spec = request.get("runtime_factory")
        if not isinstance(factory_spec, str):
            raise ValueError("physical repair request requires runtime_factory")
        self.request = dict(request)
        self.context = dict(context)
        self.factory_spec = factory_spec
        self._worker: dict[str, Any] | None = None

    def initialize(self) -> dict[str, Any]:
        if self._worker is not None:
            raise RuntimeError("physical repair backend cannot initialize twice")
        started = time.perf_counter()
        factory = _load_object(self.factory_spec)
        runtime = _callable(factory, self.factory_spec)(self.request, self.context)

        aot = _require_status(
            _callable(getattr(runtime, "authenticate_aot", None), "authenticate_aot")(),
            _REQUIRED_AOT_STATUS,
            "AOT authentication",
        )
        decode = _require_status(
            _callable(
                getattr(runtime, "decode_packed_indices", None),
                "decode_packed_indices",
            )(),
            _REQUIRED_DECODE_STATUS,
            "packed-index decode",
        )
        packed_indices = _validate_packed_indices(decode.get("tensors"))
        layouts = _require_status(
            _callable(
                getattr(runtime, "build_persistent_layer_layouts", None),
                "build_persistent_layer_layouts",
            )(packed_indices),
            _REQUIRED_LAYOUT_STATUS,
            "persistent layer layout",
        )
        if layouts.get("persistent") is not True:
            raise RuntimeError("physical repair layer layouts are not persistent")
        staged = _require_status(
            _callable(getattr(runtime, "stage_inputs", None), "stage_inputs")(
                largest_first=True
            ),
            _REQUIRED_STAGE_STATUS,
            "input staging",
        )
        order = staged.get("stage_order_nbytes")
        if (
            not isinstance(order, list)
            or not order
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in order)
            or order != sorted(order, reverse=True)
        ):
            raise RuntimeError("physical repair inputs were not staged largest-first")
        bundle = _callable(
            getattr(runtime, "update_bundle", None), "update_bundle"
        )()
        if not isinstance(bundle, dict):
            raise RuntimeError("physical repair update_bundle must return a dict")
        codebooks = _validate_codebook_surface(bundle)

        elapsed = time.perf_counter() - started
        if elapsed >= _INIT_MAX_SECONDS:
            raise RuntimeError(
                f"physical repair cold initialization exceeded {_INIT_MAX_SECONDS:.0f}s: "
                f"{elapsed:.6f}s"
            )
        init_receipt = {
            "schema": "banana-smasher-physical-repair-init-v1",
            "status": "PASS_INITIALIZED",
            "init_seconds": elapsed,
            "init_max_seconds": _INIT_MAX_SECONDS,
            "decoded_once": True,
            "persistent_layouts": True,
            "largest_first_staging": True,
            "aot": {key: value for key, value in aot.items() if key != "module"},
            "decode": {key: value for key, value in decode.items() if key != "tensors"},
            "layouts": layouts,
            "stage_order_nbytes": order,
        }
        self._worker = {
            "runtime": runtime,
            "bundle": bundle,
            "staged": staged,
            "packed_indices": packed_indices,
            "packed_identity": _tensor_identity(packed_indices),
            "codebooks": codebooks,
            "codebook_before": [value.detach().cpu().clone() for value in codebooks],
            "init_receipt": init_receipt,
            "cycles": 0,
        }
        return self._worker

    def cycle(
        self, worker_state: dict[str, Any], request: dict[str, Any]
    ) -> dict[str, Any]:
        import torch

        if worker_state is not self._worker or self._worker is None:
            raise RuntimeError("physical repair cycle received foreign worker state")
        if request != self.request:
            raise RuntimeError("physical repair request changed after initialization")
        if int(worker_state["cycles"]) != 0:
            raise RuntimeError("physical repair backend instance already ran a cycle")

        runtime = worker_state["runtime"]
        checkpointing = _require_status(
            _callable(
                getattr(runtime, "configure_depth_checkpointing", None),
                "configure_depth_checkpointing",
            )(required=True),
            _REQUIRED_CHECKPOINT_STATUS,
            "depth checkpointing",
        )
        bundle = worker_state["bundle"]
        staged = worker_state["staged"]
        context = self.context
        result = run_full_depth_update(
            layers=bundle["layers"],
            frozen_modules=bundle["frozen_modules"],
            input_ids=staged["input_ids"],
            teacher_targets=staged["teacher_targets"],
            teacher_mask=staged["teacher_mask"],
            positions=staged["positions"],
            requested_tokens=int(context["physical_tokens"]),
            segments=int(context["segments"]),
            batch_size=int(context["batch_size"]),
            memory_budget=_memory_budget(context),
            encode=bundle["encode"],
            loss_sum=bundle["loss_sum"],
            output=context["output"],
            identity=context["identity"],
            peak_memory_bytes=bundle["peak_memory_bytes"],
            optimizer_factory=bundle["optimizer_factory"],
            backend_sentinels=bundle["backend_sentinels"],
            receipt=context.get("receipt"),
            resume=bool(context.get("resume", True)),
            restart=bool(context.get("restart", False)),
            synchronize=bundle.get("synchronize"),
            semantic_claim=str(
                bundle.get(
                    "semantic_claim", "causal-segmented-no-equivalence-claim"
                )
            ),
            semantic_parity_tested=bool(bundle.get("semantic_parity_tested", False)),
        )
        if _tensor_identity(worker_state["packed_indices"]) != worker_state[
            "packed_identity"
        ]:
            raise RuntimeError("physical repair packed-index surface mutated")
        if any(value.grad is not None for value in worker_state["packed_indices"]):
            raise RuntimeError("physical repair packed indices received gradients")
        changed = [
            not torch.equal(before, after.detach().cpu())
            for before, after in zip(
                worker_state["codebook_before"], worker_state["codebooks"]
            )
        ]
        if not changed or not all(changed):
            raise RuntimeError(
                "physical repair optimizer did not update every trainable codebook"
            )
        worker_state["cycles"] = 1
        physical = {
            "schema": "banana-smasher-physical-repair-cycle-v1",
            "status": "PASS_PHYSICAL_REPAIR",
            "init_reused": True,
            "decoded_once": True,
            "persistent_layouts": True,
            "depth_checkpointing": checkpointing,
            "packed_indices_frozen": True,
            "codebooks_changed": True,
            "codebook_tensors": len(changed),
            "fallback_used": False,
        }
        result = {**result, "physical_repair": physical}
        sidecar = Path(context["output"]).resolve().with_name(
            f"{Path(context['output']).name}.physical-repair.json"
        )
        atomic_json(sidecar, {"init": worker_state["init_receipt"], "cycle": physical})
        return result


def run_physical_repair(
    *,
    request: Path,
    output: Path,
    receipt: Path | None,
    identity: dict[str, Any],
    requested_tokens: int,
    physical_tokens: int,
    segments: int,
    batch_size: int,
    memory_sizing: dict[str, Any],
    resume: bool,
    restart: bool,
) -> dict[str, Any]:
    """Entry-point runner for ``smash update --backend physical-repair``."""
    document = json.loads(Path(request).read_text())
    if not isinstance(document, dict):
        raise ValueError("physical repair request must be a JSON object")
    context = {
        "output": Path(output).resolve(),
        "receipt": None if receipt is None else Path(receipt).resolve(),
        "identity": dict(identity),
        "requested_tokens": int(requested_tokens),
        "physical_tokens": int(physical_tokens),
        "segments": int(segments),
        "batch_size": int(batch_size),
        "memory_sizing": dict(memory_sizing),
        "resume": bool(resume),
        "restart": bool(restart),
    }
    backend = PhysicalRepairBackend(document, context)
    worker = backend.initialize()
    return backend.cycle(worker, document)
