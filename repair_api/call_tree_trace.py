"""Durable full-call-tree tracing for exact W28 control/product rails."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils._python_dispatch import TorchDispatchMode


_SEMANTIC_PYTHON_FILES = {
    "fast_v7_expert_base.py": "provider",
    "static_w28_fast_v7_expert_base.py": "provider",
    "fast_k2_grouped.py": "product",
    "builder_B2_PUBLISHED_PRE.py": "builder",
    "official_local_planesource.py": "planesource",
    "resident_full64_accept.py": "resident",
    "sealed_pre_forward.py": "control",
}


def _semantic_python_boundary(filename: str, function: str) -> str | None:
    family = _SEMANTIC_PYTHON_FILES.get(Path(filename).name)
    return f"{family}.{function}" if family is not None else None


def _semantic_operator_boundary(callsite: dict[str, Any], operator: str) -> str | None:
    # Retry8 intentionally records no generic torch-dispatch operations. Named
    # source boundaries are captured by line tracing below.
    return None


_SEMANTIC_LINE_BOUNDARIES: dict[tuple[str, str, int], tuple[str, tuple[str, ...]]] = {
    ("static_w28_fast_v7_expert_base.py", "_project", 146): (
        "grouped_mm_dispatch_input", ("x",),
    ),
    ("static_w28_fast_v7_expert_base.py", "_project", 151): (
        "grouped_mm_dispatch_output", ("value",),
    ),
    ("static_w28_fast_v7_expert_base.py", "_project", 184): (
        "grouped_mm_dispatch_input", ("x",),
    ),
    ("static_w28_fast_v7_expert_base.py", "_project", 189): (
        "grouped_mm_dispatch_output", ("value",),
    ),
    ("fast_v7_expert_base.py", "_project", 195): (
        "grouped_mm_dispatch_input", ("x",),
    ),
    ("fast_v7_expert_base.py", "_project", 196): (
        "grouped_mm_dispatch_output", ("value",),
    ),
    ("fast_v7_expert_base.py", "forward", 297): ("w2_output", ("routed_output",)),
    ("fast_v7_expert_base.py", "forward", 302): (
        "route_weight_multiply_inputs", ("routed_output", "route_weight"),
    ),
    ("fast_v7_expert_base.py", "forward", 303): (
        "route_weight_multiply_output", ("routed_output",),
    ),
    ("fast_v7_expert_base.py", "forward", 309): (
        "weighted_per_slot_assembly_input", ("routed_output",),
    ),
    ("fast_v7_expert_base.py", "forward", 312): (
        "per_slot_assembly_output", ("final",),
    ),
    ("fast_v7_expert_base.py", "forward", 314): ("moe_output", ("final",)),
    ("moe.py", "grouped_mm_experts_forward", 463): ("w2_output", ("proj_out",)),
    ("moe.py", "grouped_mm_experts_forward", 465): (
        "route_weight_multiply_inputs", ("proj_out", "sample_weights_g"),
    ),
    ("moe.py", "grouped_mm_experts_forward", 467): (
        "route_weight_multiply_output", ("weighted_out",),
    ),
    ("moe.py", "grouped_mm_experts_forward", 470): (
        "weighted_per_slot_assembly_input", ("weighted_out",),
    ),
    ("moe.py", "grouped_mm_experts_forward", 480): (
        "per_slot_assembly_output", ("final_hidden_states",),
    ),
    ("moe.py", "grouped_mm_experts_forward", 481): (
        "moe_output", ("final_hidden_states",),
    ),
    ("modeling_deepseek_v4.py", "forward", 1151): (
        "residual_combine_inputs", ("mlp_output", "post", "comb", "hidden_states"),
    ),
}


def _callable_source_identity(value: Any) -> dict[str, Any]:
    code = getattr(value, "__code__", None)
    filename = inspect.getsourcefile(value) or (code.co_filename if code is not None else None)
    path = Path(filename).resolve() if filename else None
    raw = path.read_bytes() if path is not None and path.is_file() else b""
    return {
        "callable": f"{getattr(value, '__module__', '<unknown>')}."
        f"{getattr(value, '__qualname__', getattr(value, '__name__', '<unknown>'))}",
        "source_file": str(path) if path is not None else "<unknown>",
        "source_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "firstlineno": int(code.co_firstlineno) if code is not None else -1,
    }


def _provider_dispatch_identity(model: Any) -> dict[str, Any]:
    """Bind resident expert instances to their actual grouped dispatcher."""
    experts = getattr(model, "experts", None)
    expert_items: Any = getattr(experts, "items", None)
    if not callable(expert_items):
        return {"status": "UNBOUND", "reason": "MODEL_EXPERT_MAPPING_MISSING"}
    rows: dict[str, dict[str, Any]] = {}
    layers: list[int] = []
    for layer, expert in sorted(expert_items(), key=lambda item: int(item[0])):
        layers.append(int(layer))
        forward = type(expert).forward
        forward_identity = _callable_source_identity(forward)
        dispatcher = None
        dispatch_owner = None
        dispatch_owner_method = None
        for method_name in ("_project", "forward"):
            for owner in type(expert).__mro__:
                candidate = owner.__dict__.get(method_name)
                candidate_globals = getattr(candidate, "__globals__", {})
                candidate_names = getattr(getattr(candidate, "__code__", None), "co_names", ())
                if (
                    "grouped_packed_projection" in candidate_names
                    and callable(candidate_globals.get("grouped_packed_projection"))
                ):
                    dispatcher = candidate_globals["grouped_packed_projection"]
                    dispatch_owner = owner
                    dispatch_owner_method = method_name
                    break
            if dispatcher is not None:
                break
        dispatch_identity = (
            _callable_source_identity(dispatcher) if callable(dispatcher)
            else {"callable": "<missing>", "source_file": "<unknown>",
                  "source_sha256": None, "firstlineno": -1}
        )
        row = {
            "expert_class": f"{type(expert).__module__}.{type(expert).__qualname__}",
            "expert_forward": forward_identity["callable"],
            "expert_source_file": forward_identity["source_file"],
            "expert_source_sha256": forward_identity["source_sha256"],
            "expert_forward_firstlineno": forward_identity["firstlineno"],
            "dispatch_owner_class": (
                f"{dispatch_owner.__module__}.{dispatch_owner.__qualname__}"
                if dispatch_owner is not None else "<missing>"
            ),
            "dispatch_owner_method": dispatch_owner_method or "<missing>",
            "dispatch_callable": dispatch_identity["callable"],
            "dispatch_source_file": dispatch_identity["source_file"],
            "dispatch_source_sha256": dispatch_identity["source_sha256"],
            "dispatch_firstlineno": dispatch_identity["firstlineno"],
        }
        rows.setdefault(json.dumps(row, sort_keys=True, separators=(",", ":")), row)
    return {
        "status": "BOUND" if rows else "UNBOUND",
        "layer_count": len(layers),
        "layers": layers,
        "implementations": list(rows.values()),
    }


def _provider_project_code_bindings(model: Any) -> dict[Any, dict[str, Any]]:
    """Resolve exact installed _project code objects that call grouped dispatch."""
    experts = getattr(model, "experts", None)
    expert_items: Any = getattr(experts, "items", None)
    if not callable(expert_items):
        return {}
    bindings: dict[Any, dict[str, Any]] = {}
    items: Any = expert_items()
    for _, expert in items:
        for owner in type(expert).__mro__:
            candidate = owner.__dict__.get("_project")
            code = getattr(candidate, "__code__", None)
            names = getattr(code, "co_names", ())
            globals_value = getattr(candidate, "__globals__", {})
            if (
                code is not None
                and "grouped_packed_projection" in names
                and callable(globals_value.get("grouped_packed_projection"))
            ):
                identity = _callable_source_identity(candidate)
                bindings[code] = {
                    "owner_class": f"{owner.__module__}.{owner.__qualname__}",
                    "source_file": identity["source_file"],
                    "source_sha256": identity["source_sha256"],
                    "firstlineno": identity["firstlineno"],
                }
                break
    return bindings


def _provider_rebound_project_code_bindings(model: Any) -> dict[Any, dict[str, Any]]:
    """Bind wrapper _project code and the exact super method it delegates to."""
    experts = getattr(model, "experts", None)
    expert_items: Any = getattr(experts, "items", None)
    if not callable(expert_items):
        return {}
    bindings: dict[Any, dict[str, Any]] = {}
    items: Any = expert_items()
    for _, expert in items:
        mro = type(expert).__mro__
        for index, owner in enumerate(mro):
            candidate = owner.__dict__.get("_project")
            code = getattr(candidate, "__code__", None)
            if code is None or "super" not in getattr(code, "co_names", ()):
                continue
            super_owner = None
            super_candidate = None
            for inherited_owner in mro[index + 1:]:
                inherited = inherited_owner.__dict__.get("_project")
                if callable(inherited):
                    super_owner = inherited_owner
                    super_candidate = inherited
                    break
            if super_owner is None or super_candidate is None:
                continue
            identity = _callable_source_identity(candidate)
            super_identity = _callable_source_identity(super_candidate)
            bindings[code] = {
                "owner_class": f"{owner.__module__}.{owner.__qualname__}",
                "source_file": identity["source_file"],
                "source_sha256": identity["source_sha256"],
                "firstlineno": identity["firstlineno"],
                "super_owner_class": (
                    f"{super_owner.__module__}.{super_owner.__qualname__}"
                ),
                "super_callable": super_identity["callable"],
                "super_source_file": super_identity["source_file"],
                "super_source_sha256": super_identity["source_sha256"],
                "super_firstlineno": super_identity["firstlineno"],
            }
            break
    return bindings


def _semantic_line_boundary(
    filename: str, function: str, line: int,
) -> tuple[str, tuple[str, ...]] | None:
    return _SEMANTIC_LINE_BOUNDARIES.get((Path(filename).name, function, int(line)))


def _iter_tensors(value: Any, *, depth: int = 0) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if depth >= 2:
        return
    if isinstance(value, (tuple, list)):
        for item in value[:32]:
            yield from _iter_tensors(item, depth=depth + 1)
    elif isinstance(value, dict):
        for item in list(value.values())[:32]:
            yield from _iter_tensors(item, depth=depth + 1)


def _is_path_tensor(tensor: torch.Tensor) -> bool:
    shape = tuple(int(value) for value in tensor.shape)
    return bool(shape) and (
        2048 in shape
        or 4096 in shape
        or 8192 in shape
        or 129280 in shape
        or (tensor.ndim == 1 and shape[0] == 1024)
    )


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= 512:
        sample = flat
    else:
        sample = torch.cat((flat[:256], flat[-256:]))
    raw = sample.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "sample_numel": int(sample.numel()),
        "sample_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _relevant_tensors(value: Any) -> list[dict[str, Any]]:
    return [_tensor_metadata(tensor) for tensor in _iter_tensors(value) if _is_path_tensor(tensor)]


def _external_callsite() -> dict[str, Any]:
    frame = sys._getframe(2)
    while frame is not None:
        filename = str(Path(frame.f_code.co_filename).resolve())
        if "torch/utils/_python_dispatch.py" not in filename and filename != str(Path(__file__).resolve()):
            return {
                "file": filename,
                "function": frame.f_code.co_name,
                "line": int(frame.f_lineno),
                "firstlineno": int(frame.f_code.co_firstlineno),
            }
        frame = frame.f_back
    return {"file": "<unknown>", "function": "<unknown>", "line": -1, "firstlineno": -1}


def _module_root(model: Any) -> Any:
    """Resolve the torch module owned by the resident ShardStudent wrapper."""
    if callable(getattr(model, "named_modules", None)):
        return model
    owned = getattr(model, "model", None)
    if callable(getattr(owned, "named_modules", None)):
        return owned
    raise TypeError(f"W28_CALL_TREE_MODULE_ROOT_RED:{type(model).__module__}.{type(model).__qualname__}")


class _HiddenStateDispatchMode(TorchDispatchMode):
    def __init__(self, owner: "FullCallTreeTrace") -> None:
        super().__init__()
        self.owner = owner

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        inputs = _relevant_tensors((args, kwargs))
        result = func(*args, **kwargs)
        outputs = _relevant_tensors(result)
        if inputs or outputs:
            callsite = _external_callsite()
            boundary = _semantic_operator_boundary(callsite, str(func))
            if boundary is None:
                return result
            event = {
                "kind": "torch_dispatch",
                "operator": str(func),
                "callsite": callsite,
                "inputs": inputs,
                "outputs": outputs,
            }
            event["semantic_boundary"] = boundary
            self.owner._event(event)
        return result


class FullCallTreeTrace:
    """Capture Python calls, module forwards, and torch dispatch on the W28 tensor path."""

    def __init__(
        self,
        model: Any,
        path: str | Path,
        *,
        rail: str,
        basis_sha256: str,
        canonical_code_commit: str,
        max_events: int = 1_000_000,
    ) -> None:
        self.model = model
        self.module_root = _module_root(model)
        self.path = Path(path)
        self.rail = rail
        self.basis_sha256 = basis_sha256
        self.canonical_code_commit = canonical_code_commit
        self.max_events = int(max_events)
        self._count = 0
        self._stream: Any = None
        self._hooks: list[Any] = []
        self._dispatch = _HiddenStateDispatchMode(self)
        self._provider_project_codes = _provider_project_code_bindings(model)
        self._provider_rebound_project_codes = _provider_rebound_project_code_bindings(model)
        self._started = False
        self._stopped = False
        self._start_unix = 0.0

    def _event(self, value: dict[str, Any], *, reserved: bool = False) -> None:
        if self._stopped:
            return
        if self._count >= self.max_events and not reserved:
            raise RuntimeError(f"W28_CALL_TREE_EVENT_CAP_RED:{self.max_events}")
        value = {"ordinal": self._count, "rail": self.rail, **value}
        self._stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        self._count += 1
        if self._count % 256 == 0:
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def _trace_frame(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "call":
            rebound_binding = self._provider_rebound_project_codes.get(frame.f_code)
            if rebound_binding is not None:
                call_args = frame.f_locals.get("args")
                projection = (
                    call_args[0]
                    if isinstance(call_args, tuple) and call_args
                    else frame.f_locals.get("kwargs", {}).get("projection")
                )
                tensor = (
                    call_args[1]
                    if isinstance(call_args, tuple) and len(call_args) > 1
                    else frame.f_locals.get("kwargs", {}).get("x")
                )
                if isinstance(tensor, torch.Tensor):
                    self._event({
                        "kind": "semantic_boundary",
                        "semantic_boundary": "rebound_project_input",
                        "source_family": Path(frame.f_code.co_filename).name,
                        "projection": projection,
                        "code_binding": rebound_binding,
                        "tensors": [{"name": "x", **_tensor_metadata(tensor)}],
                    })
                return self._trace_frame
            project_binding = self._provider_project_codes.get(frame.f_code)
            if project_binding is not None:
                tensor = frame.f_locals.get("x")
                if isinstance(tensor, torch.Tensor):
                    self._event({
                        "kind": "semantic_boundary",
                        "semantic_boundary": "grouped_mm_dispatch_input",
                        "source_family": Path(frame.f_code.co_filename).name,
                        "projection": frame.f_locals.get("projection"),
                        "code_binding": project_binding,
                        "tensors": [{"name": "x", **_tensor_metadata(tensor)}],
                    })
                return self._trace_frame
            filename = str(Path(frame.f_code.co_filename).resolve())
            if filename == str(Path(__file__).resolve()):
                return None
            basename = Path(filename).name
            if basename in {key[0] for key in _SEMANTIC_LINE_BOUNDARIES}:
                return self._trace_frame
            return None
        if event == "line":
            filename = str(Path(frame.f_code.co_filename).resolve())
            spec = _semantic_line_boundary(filename, frame.f_code.co_name, frame.f_lineno)
            if spec is not None:
                boundary, names = spec
                tensors = [
                    {"name": name, **_tensor_metadata(frame.f_locals[name])}
                    for name in names
                    if isinstance(frame.f_locals.get(name), torch.Tensor)
                ]
                if tensors:
                    value = {
                        "kind": "semantic_boundary",
                        "semantic_boundary": boundary,
                        "source_family": Path(filename).name,
                        "tensors": tensors,
                    }
                    if boundary.startswith("grouped_mm_dispatch_"):
                        value["projection"] = frame.f_locals.get("projection")
                    self._event(value)
        project_binding = self._provider_project_codes.get(frame.f_code)
        rebound_binding = self._provider_rebound_project_codes.get(frame.f_code)
        if event == "return" and rebound_binding is not None and isinstance(arg, torch.Tensor):
            call_args = frame.f_locals.get("args")
            projection = (
                call_args[0]
                if isinstance(call_args, tuple) and call_args
                else frame.f_locals.get("kwargs", {}).get("projection")
            )
            self._event({
                "kind": "semantic_boundary",
                "semantic_boundary": "rebound_project_output",
                "source_family": Path(frame.f_code.co_filename).name,
                "projection": projection,
                "code_binding": rebound_binding,
                "tensors": [{"name": "value", **_tensor_metadata(arg)}],
            })
        if event == "return" and project_binding is not None and isinstance(arg, torch.Tensor):
            self._event({
                "kind": "semantic_boundary",
                "semantic_boundary": "grouped_mm_dispatch_output",
                "source_family": Path(frame.f_code.co_filename).name,
                "projection": frame.f_locals.get("projection"),
                "code_binding": project_binding,
                "tensors": [{"name": "value", **_tensor_metadata(arg)}],
            })
        if event == "return" and (
            Path(frame.f_code.co_filename).name == "modeling_deepseek_v4.py"
            and frame.f_code.co_name == "forward"
            and frame.f_code.co_firstlineno == 1129
            and isinstance(arg, torch.Tensor)
        ):
            self._event({
                "kind": "semantic_boundary",
                "semantic_boundary": "residual_output",
                "source_family": "modeling_deepseek_v4.py",
                "tensors": [{"name": "output", **_tensor_metadata(arg)}],
            })
        return self._trace_frame

    def _module_pre(self, name: str, module: Any, args: Any) -> None:
        tensors = _relevant_tensors(args)
        if tensors:
            code = getattr(type(module).forward, "__code__", None)
            self._event({
                "kind": "module_forward_enter",
                "module": name,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "file": str(Path(code.co_filename).resolve()) if code else "<native>",
                "firstlineno": int(code.co_firstlineno) if code else -1,
                "inputs": tensors,
            })

    def _module_post(self, name: str, module: Any, args: Any, output: Any) -> None:
        tensors = _relevant_tensors(output)
        if tensors:
            code = getattr(type(module).forward, "__code__", None)
            self._event({
                "kind": "module_forward_exit",
                "module": name,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "file": str(Path(code.co_filename).resolve()) if code else "<native>",
                "firstlineno": int(code.co_firstlineno) if code else -1,
                "outputs": tensors,
            })

    def start(self) -> "FullCallTreeTrace":
        if self._started:
            raise RuntimeError("W28_CALL_TREE_ALREADY_STARTED")
        if self.path.exists() or self.path.with_suffix(self.path.suffix + ".terminal.json").exists():
            raise RuntimeError(f"W28_CALL_TREE_PATH_EXISTS:{self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._start_unix = time.time()
        self._event({
            "kind": "header",
            "schema": "banana-smasher-w28-full-call-tree-jsonl-v1",
            "basis_sha256": self.basis_sha256,
            "canonical_code_commit": self.canonical_code_commit,
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "created_unix": self._start_unix,
        })
        self._event({
            "kind": "provider_dispatch_identity",
            **_provider_dispatch_identity(self.model),
        })
        sys.settrace(self._trace_frame)
        threading.settrace(self._trace_frame)
        self._started = True
        return self

    def stop(self, *, status: str = "PASS") -> dict[str, Any]:
        if not self._started or self._stopped:
            raise RuntimeError("W28_CALL_TREE_NOT_LIVE")
        sys.settrace(None)
        threading.settrace(None)
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        self._event(
            {"kind": "footer", "status": status, "event_count": self._count + 1},
            reserved=True,
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stopped = True
        raw = self.path.read_bytes()
        terminal = {
            "schema": "banana-smasher-w28-full-call-tree-terminal-v1",
            "status": status,
            "rail": self.rail,
            "basis_sha256": self.basis_sha256,
            "canonical_code_commit": self.canonical_code_commit,
            "event_stream": str(self.path),
            "event_stream_sha256": hashlib.sha256(raw).hexdigest(),
            "event_count": self._count,
            "pid": os.getpid(),
            "started_unix": self._start_unix,
            "completed_unix": time.time(),
        }
        terminal_path = self.path.with_suffix(self.path.suffix + ".terminal.json")
        terminal_raw = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode()
        with terminal_path.open("xb") as stream:
            stream.write(terminal_raw)
            stream.flush()
            os.fsync(stream.fileno())
        terminal["terminal_path"] = str(terminal_path)
        terminal["terminal_sha256"] = hashlib.sha256(terminal_raw).hexdigest()
        return terminal

    def __enter__(self) -> "FullCallTreeTrace":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._started and not self._stopped:
            self.stop(status="PASS" if exc_type is None else "ERROR")
