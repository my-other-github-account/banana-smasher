from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_sidecars import (
    CandidateSidecarWriter,
    load_teacher_support_manifest,
    load_teacher_window,
    score_anchor_sidecars,
)
from .fixed_d4 import _is_sha256, _sha256_file

_SCHEMA = "banana-smasher-fixed-d4-offline-layerwise-receipt-v1"
_PROGRESS_SCHEMA = "banana-smasher-fixed-d4-offline-layerwise-progress-v1"
_STATE_SCHEMA = "banana-smasher-fixed-d4-offline-layerwise-state-v1"
_FORBIDDEN_ADAPTER_SOURCE = (b"vllm", b"EngineCore", b"cpu_offload")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _process_read_bytes() -> int:
    proc_io = Path("/proc/self/io")
    if proc_io.is_file():
        for line in proc_io.read_text().splitlines():
            if line.startswith("rchar:"):
                return int(line.split(":", 1)[1].strip())
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_inblock) * 512


def _path_is_remote(path: Path) -> bool:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False
    candidate = path.resolve()
    matches: list[tuple[int, str, str]] = []
    for line in mountinfo.read_text().splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        trailing = after.split()
        if not separator or len(fields) < 5 or len(trailing) < 2:
            continue
        mountpoint = Path(
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\")
        )
        try:
            candidate.relative_to(mountpoint)
        except ValueError:
            continue
        matches.append((len(str(mountpoint)), trailing[0], trailing[1]))
    if not matches:
        raise ValueError(f"cannot identify filesystem for local-only input {candidate}")
    _, filesystem, source = max(matches)
    remote_filesystems = {
        "9p",
        "cifs",
        "fuse.rclone",
        "fuse.sshfs",
        "nfs",
        "nfs4",
        "smbfs",
        "sshfs",
    }
    return filesystem in remote_filesystems or source.startswith("//")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical(value))


def _atomic_npy(path: Path, value: object) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject or not array.size:
        raise ValueError("offline-layerwise activation must be one non-empty numeric array")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _read_activation(path: Path, expected_sha: str) -> np.ndarray:
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise ValueError(f"offline-layerwise activation checkpoint identity mismatch: {path}")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject or not value.size:
        raise ValueError(f"offline-layerwise activation checkpoint is invalid: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        rows.append(row)
    return rows


def _window_key(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _load_config(config: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], bytes, Path]:
    if isinstance(config, Mapping):
        parsed = dict(config)
        payload = _canonical(parsed)
        return parsed, payload, Path.cwd()
    path = Path(config).expanduser().resolve()
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid offline-layerwise producer config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("offline-layerwise producer config must be an object")
    return parsed, payload, path.parent


def _bound_path(value: object, *, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be non-empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _assert_no_resident_engine_modules() -> None:
    loaded = sorted(
        name for name in sys.modules if name == "vllm" or name.startswith("vllm.")
    )
    if loaded:
        raise ValueError(
            "offline-layerwise process imported forbidden resident engine modules: "
            + ", ".join(loaded[:8])
        )


def _load_runtime(binding: Mapping[str, Any], *, root: Path) -> type[Any]:
    path_binding = {"path", "sha256", "class", "api_version"}
    module_binding = {"module", "sha256", "class", "api_version"}
    if set(binding) not in (path_binding, module_binding):
        raise ValueError(
            "offline-layerwise runtime_adapter requires either path or module, plus "
            "sha256, class, and api_version"
        )
    module_path = binding.get("module")
    if module_path is not None:
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("offline-layerwise runtime_adapter module must be non-empty")
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            raise ValueError(
                f"cannot locate offline-layerwise runtime_adapter module {module_path}"
            )
        path = Path(spec.origin).resolve()
    else:
        path = _bound_path(binding.get("path"), root=root, label="runtime_adapter")
    expected_sha = binding.get("sha256")
    class_name = binding.get("class")
    api_version = binding.get("api_version")
    if not _is_sha256(expected_sha):
        raise ValueError("offline-layerwise runtime_adapter SHA-256 is invalid")
    if not isinstance(class_name, str) or not class_name or api_version != 1:
        raise ValueError("offline-layerwise runtime_adapter class/API version is invalid")
    payload = path.read_bytes()
    actual_sha = _sha(payload)
    if actual_sha != expected_sha:
        raise ValueError(
            f"offline-layerwise runtime_adapter SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    forbidden = [token.decode() for token in _FORBIDDEN_ADAPTER_SOURCE if token in payload]
    if forbidden:
        raise ValueError(
            "offline-layerwise runtime_adapter contains forbidden resident-engine source: "
            + ", ".join(forbidden)
        )
    if module_path is not None:
        module = importlib.import_module(module_path)
    else:
        module_name = f"_banana_smasher_offline_{actual_sha[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import offline-layerwise runtime_adapter {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    _assert_no_resident_engine_modules()
    runtime = getattr(module, class_name, None)
    required = {
        "initial_stage",
        "layer_stage",
        "terminal_stage",
        "export_activation",
        "import_activation",
        "synchronize",
        "resident_bytes",
        "peak_resident_bytes",
        "bytes_read",
    }
    if (
        not isinstance(runtime, type)
        or getattr(runtime, "API_VERSION", None) != api_version
        or any(not callable(getattr(runtime, name, None)) for name in required)
    ):
        raise ValueError(
            f"offline-layerwise runtime_adapter {class_name} does not implement API v1"
        )
    return runtime


def _validate_limits(value: object) -> dict[str, int | float | str]:
    fields = {
        "input_scope",
        "expected_read_bytes",
        "max_read_bytes",
        "first_output_deadline_seconds",
        "max_elapsed_seconds",
        "max_resident_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(
            "offline-layerwise physical_limits require input_scope, expected_read_bytes, "
            "max_read_bytes, first_output_deadline_seconds, max_elapsed_seconds, and "
            "max_resident_bytes"
        )
    limits = dict(value)
    expected = limits["expected_read_bytes"]
    maximum = limits["max_read_bytes"]
    resident = limits["max_resident_bytes"]
    if limits["input_scope"] != "local-only":
        raise ValueError("offline-layerwise input_scope must be local-only")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (expected, maximum, resident)
    ) or expected > maximum:
        raise ValueError("offline-layerwise byte limits are invalid")
    first = limits["first_output_deadline_seconds"]
    elapsed = limits["max_elapsed_seconds"]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not np.isfinite(float(item))
        or item <= 0
        for item in (first, elapsed)
    ) or first > elapsed or elapsed > 3600:
        raise ValueError("offline-layerwise time limits are invalid or exceed one hour")
    return limits


def _append_row(path: Path, row: Mapping[str, Any]) -> bytes:
    payload = _canonical(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def produce_fixed_d4_layerwise_logits(
    model_root: str | Path,
    producer_config: str | Path | Mapping[str, Any],
    bank_path: str | Path,
    output_path: str | Path,
    *,
    basis_sha256: str,
    window_id_field: str = "window_id",
    verified_pack_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a fixed-D4 candidate with one transformer layer resident at a time.

    The hash-bound runtime adapter owns architecture-specific embedding, block, D4
    dispatch, norm, and head operations. This engine owns execution order, durable
    activation/candidate checkpoints, resume identity, and physical admission.
    It never imports or delegates to a resident model engine.
    """

    _assert_no_resident_engine_modules()
    if not _is_sha256(basis_sha256):
        raise ValueError("offline-layerwise basis_sha256 must be a lowercase SHA-256")
    if not isinstance(window_id_field, str) or not window_id_field:
        raise ValueError("offline-layerwise window_id_field must be non-empty")
    model_root = Path(model_root).expanduser().resolve()
    bank_path = Path(bank_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    from .fixed_d4 import verify_fixed_d4_model

    verified_manifest = verify_fixed_d4_model(
        model_root,
        basis_sha256=basis_sha256,
        verified_pack_receipt=verified_pack_receipt,
    )
    config, config_payload, config_root = _load_config(producer_config)
    if (
        config.get("schema") != "banana-smasher-candidate-producer-v1"
        or config.get("producer") != "fixed-d4-offline-layerwise"
        or set(config) != {"schema", "producer", "parameters"}
    ):
        raise ValueError("offline-layerwise producer requires candidate-producer-v1")
    parameters = config.get("parameters")
    required_parameters = {
        "input_field",
        "positions",
        "layers",
        "teacher_support",
        "execution_mode",
        "runtime_adapter",
        "physical_limits",
    }
    if not isinstance(parameters, Mapping) or set(parameters) != required_parameters:
        raise ValueError(
            "offline-layerwise parameters require input_field, positions, layers, "
            "teacher_support, execution_mode, runtime_adapter, and physical_limits"
        )
    if parameters.get("execution_mode") != "offline-layerwise":
        raise ValueError("offline-layerwise execution_mode must be offline-layerwise")
    input_field = parameters.get("input_field")
    positions = parameters.get("positions")
    layers = parameters.get("layers")
    if not isinstance(input_field, str) or not input_field:
        raise ValueError("offline-layerwise input_field must be non-empty")
    if isinstance(positions, bool) or not isinstance(positions, int) or positions < 1:
        raise ValueError("offline-layerwise positions must be positive")
    if (
        not isinstance(layers, list)
        or not layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("offline-layerwise layers must be sorted unique non-negative integers")
    manifest_layers = list(verified_manifest.get("layers", []))
    if not manifest_layers or manifest_layers[: len(layers)] != layers:
        raise ValueError(
            "offline-layerwise configured layers must be a contiguous prefix of the verified pack"
        )
    configured_layer_count = len(layers)
    manifest_layer_count = len(manifest_layers)
    limits = _validate_limits(parameters.get("physical_limits"))
    for label, path in (("model", model_root), ("bank", bank_path)):
        if _path_is_remote(path):
            raise ValueError(f"offline-layerwise local-only policy rejects remote {label} {path}")

    bank_rows = _read_jsonl(bank_path, label="bank")
    if len(bank_rows) != 64:
        raise ValueError(
            f"offline-layerwise materialization requires exactly 64 windows, got {len(bank_rows)}"
        )
    bank_sha = _sha256_file(bank_path)
    support_binding = parameters.get("teacher_support")
    legacy_fields = {"path", "sha256", "field"}
    sidecar_fields = {"manifest", "sha256"}
    if not isinstance(support_binding, Mapping) or set(support_binding) not in (
        legacy_fields,
        sidecar_fields,
    ):
        raise ValueError(
            "offline-layerwise teacher_support requires either path/sha256/field "
            "or manifest/sha256"
        )
    support_is_sidecar = set(support_binding) == sidecar_fields
    support_path = _bound_path(
        support_binding.get("manifest" if support_is_sidecar else "path"),
        root=config_root,
        label="teacher_support",
    )
    if _path_is_remote(support_path):
        raise ValueError(f"offline-layerwise local-only policy rejects remote support {support_path}")
    support_sha = support_binding.get("sha256")
    if not _is_sha256(support_sha) or _sha256_file(support_path) != support_sha:
        raise ValueError("offline-layerwise teacher_support SHA-256 mismatch")

    support_by_window: dict[str, list[list[int]] | None] = {}
    support_width: int | None = None
    support_manifest: dict[str, Any] | None = None
    if support_is_sidecar:
        support_manifest = load_teacher_support_manifest(
            support_path, expected_bank_sha256=bank_sha
        )
        support_width = int(support_manifest["support_width"])
        if [
            _window_key(value) for value in support_manifest["window_ids"]
        ] != [
            _window_key(row.get(window_id_field)) for row in bank_rows
        ]:
            raise ValueError(
                "offline-layerwise teacher sidecar ordered windows differ from the bank"
            )
        for window_id, entry in zip(
            support_manifest["window_ids"], support_manifest["windows"], strict=True
        ):
            shapes = entry["tensors"]
            if (
                shapes["idx"]["shape"][1:] != [support_width]
                or shapes["logprob"]["shape"] != shapes["idx"]["shape"]
                or shapes["idx"]["shape"][0] < positions
            ):
                raise ValueError(
                    "offline-layerwise teacher sidecar position/support shape mismatch"
                )
            support_by_window[_window_key(window_id)] = None
    else:
        support_field = support_binding.get("field")
        if not isinstance(support_field, str) or not support_field:
            raise ValueError("offline-layerwise teacher_support field is invalid")
        support_rows = _read_jsonl(support_path, label="teacher support")
        for row in support_rows:
            if "window_id" not in row:
                raise ValueError("offline-layerwise teacher support row lacks window_id")
            matrix = row.get(support_field)
            key = _window_key(row["window_id"])
            widths = {
                len(support_row)
                for support_row in matrix
                if isinstance(support_row, list)
            } if isinstance(matrix, list) else set()
            if (
                key in support_by_window
                or not isinstance(matrix, list)
                or len(matrix) != positions
                or len(widths) != 1
                or next(iter(widths), 0) < 1
                or any(
                    not isinstance(support_row, list)
                    or len(set(support_row)) != len(support_row)
                    or any(
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or token < 0
                        for token in support_row
                    )
                    for support_row in matrix
                )
            ):
                raise ValueError("offline-layerwise teacher support matrix is invalid")
            row_width = next(iter(widths))
            if support_width is None:
                support_width = row_width
            elif row_width != support_width:
                raise ValueError(
                    "offline-layerwise teacher support row widths are inconsistent"
                )
            support_by_window[key] = matrix
    assert support_width is not None
    seen: set[str] = set()
    for row in bank_rows:
        tokens = row.get(input_field)
        if window_id_field not in row:
            raise ValueError("offline-layerwise bank row lacks window identity")
        key = _window_key(row[window_id_field])
        if (
            key in seen
            or key not in support_by_window
            or not isinstance(tokens, list)
            or len(tokens) < positions
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
        ):
            raise ValueError("offline-layerwise bank row is invalid or unbound to support")
        seen.add(key)
    if not bank_rows or set(support_by_window) != seen:
        raise ValueError("offline-layerwise bank/support coverage mismatch")

    runtime_binding = parameters.get("runtime_adapter")
    if not isinstance(runtime_binding, Mapping):
        raise ValueError("offline-layerwise runtime_adapter must be an object")
    runtime_class = _load_runtime(runtime_binding, root=config_root)
    runtime = runtime_class(model_root=model_root, parameters=dict(parameters))
    _assert_no_resident_engine_modules()

    run_root = output_path.with_name(output_path.name + ".layerwise")
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "STATE.json"
    progress_path = run_root / "PROGRESS.json"
    binding = _sha(
        _canonical(
            {
                "basis_sha256": basis_sha256,
                "model_root": str(model_root),
                "producer_config_sha256": _sha(config_payload),
                "bank_sha256": bank_sha,
                "teacher_support_sha256": support_sha,
                "runtime_adapter_sha256": runtime_binding["sha256"],
                "layers": layers,
                "positions": positions,
                "support_width": support_width,
            }
        )
    )
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("schema") != _STATE_SCHEMA or state.get("binding_sha256") != binding:
            raise ValueError("offline-layerwise resume state binding mismatch")
    else:
        state = {
            "schema": _STATE_SCHEMA,
            "binding_sha256": binding,
            "checkpoints": {},
            "completed_layers": [],
            "output_rows": [],
        }
        _atomic_json(state_path, state)

    started = time.monotonic()
    process_read_start = _process_read_bytes()
    peak_resident = int(runtime.peak_resident_bytes())
    window_layer_forwards = 0

    def metrics() -> tuple[int, int, float]:
        nonlocal peak_resident
        runtime_read = int(runtime.bytes_read())
        process_read = max(0, _process_read_bytes() - process_read_start)
        read_bytes = max(runtime_read, process_read)
        resident = int(runtime.resident_bytes())
        peak_resident = max(peak_resident, resident, int(runtime.peak_resident_bytes()))
        elapsed = time.monotonic() - started
        if read_bytes > int(limits["max_read_bytes"]):
            raise ValueError(
                f"offline-layerwise read budget exceeded: observed={read_bytes}, max={limits['max_read_bytes']}"
            )
        if resident > int(limits["max_resident_bytes"]) or peak_resident > int(limits["max_resident_bytes"]):
            raise ValueError(
                "offline-layerwise resident memory budget exceeded: "
                f"observed={max(resident, peak_resident)}, max={limits['max_resident_bytes']}"
            )
        if elapsed > float(limits["max_elapsed_seconds"]):
            raise ValueError("offline-layerwise maximum elapsed time exceeded")
        return read_bytes, resident, elapsed

    def progress(*, stage: str, layer: int | None, window: object | None) -> None:
        read_bytes, resident, elapsed = metrics()
        _atomic_json(
            progress_path,
            {
                "schema": _PROGRESS_SCHEMA,
                "status": "PASS" if stage == "complete" else "RUNNING",
                "binding_sha256": binding,
                "stage": stage,
                "layer": layer,
                "window": window,
                "layers_completed": len(state["completed_layers"]),
                "configured_layer_count": configured_layer_count,
                "manifest_layer_count": manifest_layer_count,
                "layer_windows_completed": len(state["checkpoints"].get(f"layer_{layer}", {})) if layer is not None else 0,
                "window_layer_forwards": window_layer_forwards,
                "bytes_read": read_bytes,
                "resident_bytes": resident,
                "resident_peak_bytes": peak_resident,
                "output_rows": len(state["output_rows"]),
                "elapsed_seconds": elapsed,
                "resident_engine": False,
            },
        )

    completed_layers = state.get("completed_layers")
    checkpoints = state.get("checkpoints")
    if not isinstance(completed_layers, list) or not isinstance(checkpoints, dict):
        raise ValueError("offline-layerwise resume state structure is invalid")
    if completed_layers != layers[: len(completed_layers)]:
        raise ValueError("offline-layerwise completed layer prefix is invalid")

    initial_stage = "initial"
    if completed_layers:
        source_stage = f"layer_{completed_layers[-1]}"
        source_receipts = checkpoints.get(source_stage)
        if not isinstance(source_receipts, dict) or len(source_receipts) != len(bank_rows):
            raise ValueError("offline-layerwise resumed layer checkpoint coverage mismatch")
    else:
        source_stage = initial_stage
        initial_dir = run_root / initial_stage
        initial_receipts = checkpoints.setdefault(initial_stage, {})
        with runtime.initial_stage() as embed:
            for slot, row in enumerate(bank_rows):
                slot_key = str(slot)
                if slot_key in initial_receipts:
                    continue
                activation = embed(row[input_field][:positions], window_id=row[window_id_field])
                checkpoint = initial_dir / f"window_{slot:03d}.npy"
                initial_receipts[slot_key] = _atomic_npy(
                    checkpoint, runtime.export_activation(activation)
                )
                runtime.synchronize()
                _atomic_json(state_path, state)
                progress(stage="initial", layer=None, window=row[window_id_field])
                del activation
        runtime.synchronize()
        if int(runtime.resident_bytes()) != 0:
            raise ValueError("offline-layerwise initial stage retained accelerator storage")

    for layer in layers[len(completed_layers) :]:
        target_stage = f"layer_{layer}"
        target_dir = run_root / target_stage
        target_receipts = state["checkpoints"].setdefault(target_stage, {})
        source_receipts = state["checkpoints"].get(source_stage)
        if not isinstance(source_receipts, dict) or len(source_receipts) != len(bank_rows):
            raise ValueError("offline-layerwise source activation checkpoint coverage mismatch")
        with runtime.layer_stage(layer) as forward:
            metrics()
            for slot, row in enumerate(bank_rows):
                slot_key = str(slot)
                if slot_key in target_receipts:
                    continue
                source_path = run_root / source_stage / f"window_{slot:03d}.npy"
                source_array = _read_activation(source_path, source_receipts[slot_key])
                activation = runtime.import_activation(source_array)
                output = forward(activation, window_id=row[window_id_field])
                checkpoint = target_dir / f"window_{slot:03d}.npy"
                target_receipts[slot_key] = _atomic_npy(
                    checkpoint, runtime.export_activation(output)
                )
                runtime.synchronize()
                window_layer_forwards += 1
                _atomic_json(state_path, state)
                progress(stage="layer", layer=layer, window=row[window_id_field])
                del source_array, activation, output
        runtime.synchronize()
        gc.collect()
        if int(runtime.resident_bytes()) != 0:
            raise ValueError(f"offline-layerwise layer {layer} retained accelerator storage after unload")
        if len(target_receipts) != len(bank_rows):
            raise ValueError(f"offline-layerwise layer {layer} checkpoint coverage mismatch")
        if layer not in state["completed_layers"]:
            state["completed_layers"].append(layer)
        previous = run_root / source_stage
        source_stage = target_stage
        _atomic_json(state_path, state)
        if previous != target_dir:
            shutil.rmtree(previous)
            state["checkpoints"].pop(previous.name, None)
            _atomic_json(state_path, state)
        progress(stage="layer-complete", layer=layer, window=None)

    resumed_output_keys: set[str] = set()
    candidate_writer: CandidateSidecarWriter | None = None
    if support_is_sidecar:
        model_id = verified_manifest.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                "offline-layerwise sidecar output requires model_id in the verified pack"
            )
        pack_manifest_path = model_root / "BANANA_PACK_MANIFEST.json"
        if not pack_manifest_path.is_file():
            raise ValueError(
                "offline-layerwise sidecar output requires BANANA_PACK_MANIFEST.json"
            )
        candidate_writer = CandidateSidecarWriter(
            output_path,
            teacher_manifest_path=support_path,
            window_ids=[row[window_id_field] for row in bank_rows],
            basis_sha256=basis_sha256,
            bank_sha256=bank_sha,
            model_id=model_id,
            pack_sha256=_sha256_file(pack_manifest_path),
        )
        resumed_output_keys = {
            _window_key(window_id)
            for window_id in candidate_writer.completed_window_ids
        }
    else:
        if output_path.is_file() and output_path.stat().st_size:
            for row in _read_jsonl(output_path, label="offline-layerwise candidate"):
                if set(row) != {
                    "window_id",
                    "logits",
                    "support_token_ids",
                    "top1_token_ids",
                }:
                    raise ValueError(
                        "offline-layerwise resumed candidate fields are invalid"
                    )
                key = _window_key(row["window_id"])
                if (
                    key in resumed_output_keys
                    or row["support_token_ids"] != support_by_window.get(key)
                ):
                    raise ValueError(
                        "offline-layerwise resumed candidate binding is invalid"
                    )
                resumed_output_keys.add(key)
        elif not output_path.exists():
            _atomic_bytes(output_path, b"")
    resumed_output_rows = len(resumed_output_keys)
    state["output_rows"] = sorted(resumed_output_keys)
    _atomic_json(state_path, state)

    final_receipts = state["checkpoints"].get(source_stage)
    if not isinstance(final_receipts, dict) or len(final_receipts) != len(bank_rows):
        raise ValueError("offline-layerwise terminal source checkpoint coverage mismatch")
    terminal_needed = resumed_output_rows < len(bank_rows)
    terminal_context = runtime.terminal_stage() if terminal_needed else nullcontext(None)
    with terminal_context as score:
        if terminal_needed:
            metrics()
        for slot, row in enumerate(bank_rows):
            window_id = row[window_id_field]
            key = _window_key(window_id)
            if key in resumed_output_keys:
                continue
            final_array = _read_activation(
                run_root / source_stage / f"window_{slot:03d}.npy",
                final_receipts[str(slot)],
            )
            activation = runtime.import_activation(final_array)
            if support_is_sidecar:
                assert support_manifest is not None
                support, _ = load_teacher_window(
                    support_path, window_id, manifest=support_manifest
                )
                support = support[:positions]
            else:
                support = support_by_window[key]
                assert support is not None
            scored = score(
                activation,
                support,
                window_id=window_id,
            )
            if not isinstance(scored, Mapping) or set(scored) != {
                "q_lp_at_ref",
                "q_argmax",
            }:
                raise ValueError("offline-layerwise terminal scorer returned invalid fields")
            q_lp = np.asarray(scored.get("q_lp_at_ref"))
            q_argmax = np.asarray(scored.get("q_argmax"))
            if (
                q_lp.shape != (positions, support_width)
                or q_lp.dtype != np.float16
                or not np.isfinite(q_lp).all()
                or q_argmax.shape != (positions,)
                or q_argmax.dtype != np.int32
                or np.any(q_argmax < 0)
            ):
                raise ValueError("offline-layerwise terminal scorer returned invalid position output")
            if candidate_writer is not None:
                import torch

                candidate_writer.write_window(
                    window_id,
                    q_lp_at_ref=torch.from_numpy(np.ascontiguousarray(q_lp)),
                    q_argmax=torch.from_numpy(np.ascontiguousarray(q_argmax)),
                )
            else:
                produced = {
                    "window_id": window_id,
                    "logits": q_lp.astype(np.float32).tolist(),
                    "support_token_ids": support,
                    "top1_token_ids": q_argmax.tolist(),
                }
                _append_row(output_path, produced)
            resumed_output_keys.add(key)
            state["output_rows"] = sorted(resumed_output_keys)
            runtime.synchronize()
            _atomic_json(state_path, state)
            progress(stage="terminal", layer=None, window=window_id)
            _, _, elapsed = metrics()
            if (
                resumed_output_rows == 0
                and len(resumed_output_keys) == 1
                and elapsed > float(limits["first_output_deadline_seconds"])
            ):
                raise ValueError("offline-layerwise first-output deadline exceeded")
            del final_array, activation, support, q_lp, q_argmax
    if terminal_needed:
        runtime.synchronize()
    gc.collect()
    if int(runtime.resident_bytes()) != 0:
        raise ValueError("offline-layerwise terminal stage retained accelerator storage")
    _assert_no_resident_engine_modules()
    if len(resumed_output_keys) != len(bank_rows):
        raise ValueError("offline-layerwise terminal candidate coverage mismatch")
    progress(stage="complete", layer=None, window=None)
    read_bytes, _, elapsed = metrics()
    return {
        "schema": _SCHEMA,
        "status": "PASS",
        "execution_mode": "offline-layerwise",
        "resident_engine": False,
        "basis_sha256": basis_sha256,
        "binding_sha256": binding,
        "producer_config_sha256": _sha(config_payload),
        "bank_sha256": bank_sha,
        "teacher_support_sha256": support_sha,
        "runtime_adapter_sha256": runtime_binding["sha256"],
        "layers_completed": len(layers),
        "configured_layer_count": configured_layer_count,
        "manifest_layer_count": manifest_layer_count,
        "window_layer_forwards": window_layer_forwards,
        "positions_per_window": positions,
        "support_width": support_width,
        "output_format": (
            "torch-sidecars-with-json-manifest" if support_is_sidecar else "jsonl"
        ),
        "output_rows": len(bank_rows),
        "resumed_output_rows": resumed_output_rows,
        "bytes_read": read_bytes,
        "peak_resident_bytes": peak_resident,
        "elapsed_seconds": elapsed,
        "physical_limits": limits,
        "commit_granularity": "one-fsynced-layer-window-activation-and-one-fsynced-output-window",
        "top1_source": "runtime-adapter-terminal-full-vocabulary-argmax",
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "state_path": str(state_path),
        "progress_path": str(progress_path),
    }


def rescore_fixed_d4_layerwise_terminal(
    model_root: str | Path,
    source_producer_config: str | Path | Mapping[str, Any],
    bank_path: str | Path,
    completed_state_path: str | Path,
    teacher_manifest_path: str | Path,
    output_path: str | Path,
    *,
    basis_sha256: str,
    window_id_field: str = "window_id",
    verified_pack_receipt: Mapping[str, Any] | None = None,
    terminal_runtime_adapter: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rescore preserved final activations without replaying a transformer layer."""

    _assert_no_resident_engine_modules()
    if not _is_sha256(basis_sha256):
        raise ValueError("terminal rescore basis_sha256 must be a lowercase SHA-256")
    model_root = Path(model_root).expanduser().resolve()
    bank_path = Path(bank_path).expanduser().resolve()
    state_path = Path(completed_state_path).expanduser().resolve()
    teacher_path = Path(teacher_manifest_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    from .fixed_d4 import verify_fixed_d4_model

    verified_manifest = verify_fixed_d4_model(
        model_root,
        basis_sha256=basis_sha256,
        verified_pack_receipt=verified_pack_receipt,
    )
    config, config_payload, config_root = _load_config(source_producer_config)
    parameters = config.get("parameters")
    required_parameters = {
        "input_field",
        "positions",
        "layers",
        "teacher_support",
        "execution_mode",
        "runtime_adapter",
        "physical_limits",
    }
    if (
        config.get("schema") != "banana-smasher-candidate-producer-v1"
        or config.get("producer") != "fixed-d4-offline-layerwise"
        or set(config) != {"schema", "producer", "parameters"}
        or not isinstance(parameters, Mapping)
        or set(parameters) != required_parameters
    ):
        raise ValueError("terminal rescore requires the original offline-layerwise config")
    positions = parameters["positions"]
    layers = parameters["layers"]
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or positions < 1
        or not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
    ):
        raise ValueError("terminal rescore source positions/layers are invalid")
    manifest_layers = list(verified_manifest.get("layers", []))
    if manifest_layers[: len(layers)] != layers:
        raise ValueError("terminal rescore layers differ from the verified pack")

    bank_rows = _read_jsonl(bank_path, label="bank")
    if len(bank_rows) != 64:
        raise ValueError("terminal rescore requires exactly 64 bank windows")
    window_ids = [row.get(window_id_field) for row in bank_rows]
    if len({_window_key(value) for value in window_ids}) != len(window_ids):
        raise ValueError("terminal rescore bank window identities are invalid")
    bank_sha = _sha256_file(bank_path)

    source_support = parameters["teacher_support"]
    if not isinstance(source_support, Mapping):
        raise ValueError("terminal rescore source teacher support is invalid")
    if set(source_support) == {"manifest", "sha256"}:
        source_support_path = _bound_path(
            source_support["manifest"], root=config_root, label="source teacher support"
        )
        source_teacher = load_teacher_support_manifest(
            source_support_path, expected_bank_sha256=bank_sha
        )
        source_support_width = int(source_teacher["support_width"])
    elif set(source_support) == {"path", "sha256", "field"}:
        source_support_path = _bound_path(
            source_support["path"], root=config_root, label="source teacher support"
        )
        source_field = source_support["field"]
        source_rows = _read_jsonl(source_support_path, label="source teacher support")
        widths = {
            len(row.get(source_field, [])[0])
            for row in source_rows
            if isinstance(row.get(source_field), list) and row[source_field]
        }
        if len(widths) != 1:
            raise ValueError("terminal rescore source support width is inconsistent")
        source_support_width = widths.pop()
    else:
        raise ValueError("terminal rescore source teacher support contract is invalid")
    source_support_sha = source_support.get("sha256")
    if (
        not _is_sha256(source_support_sha)
        or _sha256_file(source_support_path) != source_support_sha
    ):
        raise ValueError("terminal rescore source teacher support identity mismatch")

    runtime_binding = parameters["runtime_adapter"]
    if not isinstance(runtime_binding, Mapping):
        raise ValueError("terminal rescore runtime adapter is invalid")
    active_runtime_binding = (
        terminal_runtime_adapter
        if terminal_runtime_adapter is not None
        else runtime_binding
    )
    if not isinstance(active_runtime_binding, Mapping):
        raise ValueError("terminal rescore active runtime adapter is invalid")
    source_binding_fields = {
        "basis_sha256": basis_sha256,
        "model_root": str(model_root),
        "producer_config_sha256": _sha(config_payload),
        "bank_sha256": bank_sha,
        "teacher_support_sha256": source_support_sha,
        "runtime_adapter_sha256": runtime_binding["sha256"],
        "layers": layers,
        "positions": positions,
    }
    accepted_bindings = {
        "legacy-producer-v1": _sha(_canonical(source_binding_fields)),
        "support-width-v2": _sha(
            _canonical({**source_binding_fields, "support_width": source_support_width})
        ),
    }
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid completed offline-layerwise state {state_path}: {exc}") from exc
    source_binding_schema = next(
        (
            schema
            for schema, digest in accepted_bindings.items()
            if state.get("binding_sha256") == digest
        ),
        None,
    )
    if (
        state.get("schema") != _STATE_SCHEMA
        or source_binding_schema is None
        or state.get("completed_layers") != layers
    ):
        raise ValueError("terminal rescore completed state identity mismatch")
    binding = accepted_bindings[source_binding_schema]
    final_stage = f"layer_{layers[-1]}"
    final_receipts = state.get("checkpoints", {}).get(final_stage)
    if not isinstance(final_receipts, dict) or len(final_receipts) != len(bank_rows):
        raise ValueError("terminal rescore final activation coverage is incomplete")

    teacher = load_teacher_support_manifest(
        teacher_path, expected_bank_sha256=bank_sha
    )
    if teacher["window_ids"] != window_ids:
        raise ValueError("terminal rescore teacher windows differ from the bank")
    model_id = verified_manifest.get("model_id")
    pack_manifest_path = model_root / "BANANA_PACK_MANIFEST.json"
    if not isinstance(model_id, str) or not model_id or not pack_manifest_path.is_file():
        raise ValueError("terminal rescore requires bound model and pack identities")
    pack_sha256 = _sha256_file(pack_manifest_path)
    writer = CandidateSidecarWriter(
        output_path,
        teacher_manifest_path=teacher_path,
        window_ids=window_ids,
        basis_sha256=basis_sha256,
        bank_sha256=bank_sha,
        model_id=model_id,
        pack_sha256=pack_sha256,
    )
    resumed = len(writer.completed_window_ids)
    runtime_class = _load_runtime(active_runtime_binding, root=config_root)
    runtime = runtime_class(model_root=model_root, parameters=dict(parameters))
    terminal_needed = resumed < len(bank_rows)
    terminal_context = runtime.terminal_stage() if terminal_needed else nullcontext(None)
    with terminal_context as score:
        if terminal_needed:
            assert score is not None
        for slot, window_id in enumerate(window_ids):
            if window_id in writer.completed_window_ids:
                continue
            final_array = _read_activation(
                state_path.parent / final_stage / f"window_{slot:03d}.npy",
                final_receipts[str(slot)],
            )
            activation = runtime.import_activation(final_array)
            support, _ = load_teacher_window(teacher_path, window_id, manifest=teacher)
            support = support[:positions]
            scored = score(activation, support, window_id=window_id)
            if not isinstance(scored, Mapping) or set(scored) != {
                "q_lp_at_ref",
                "q_argmax",
            }:
                raise ValueError("terminal rescore adapter returned invalid fields")
            q_lp = np.asarray(scored["q_lp_at_ref"])
            q_argmax = np.asarray(scored["q_argmax"])
            expected_shape = (support.shape[0], teacher["support_width"])
            if (
                q_lp.shape != expected_shape
                or q_lp.dtype != np.float16
                or q_argmax.shape != (support.shape[0],)
                or q_argmax.dtype != np.int32
                or not np.isfinite(q_lp).all()
                or np.any(q_argmax < 0)
            ):
                raise ValueError("terminal rescore adapter returned invalid tensors")
            import torch

            writer.write_window(
                window_id,
                q_lp_at_ref=torch.from_numpy(np.ascontiguousarray(q_lp)),
                q_argmax=torch.from_numpy(np.ascontiguousarray(q_argmax)),
            )
            runtime.synchronize()
            del final_array, activation, support, q_lp, q_argmax
    if terminal_needed:
        runtime.synchronize()
    gc.collect()
    if int(runtime.resident_bytes()) != 0:
        raise ValueError("terminal rescore retained accelerator storage")
    _assert_no_resident_engine_modules()
    manifest_sha = _sha256_file(output_path)
    teacher_descriptor = {
        "path": str(teacher_path),
        "bytes": teacher_path.stat().st_size,
        "sha256": _sha256_file(teacher_path),
    }
    candidate_descriptor = {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": manifest_sha,
    }
    score_path = output_path.with_suffix(".score.json")
    score = score_anchor_sidecars(teacher_path, output_path)
    _atomic_json(score_path, score)
    score_descriptor = {
        "path": str(score_path),
        "bytes": score_path.stat().st_size,
        "sha256": _sha256_file(score_path),
    }
    authentic = (
        teacher["support_width"] == 8192
        and layers == manifest_layers
        and score["windows"] == 64
        and score["positions"] == 64 * 1024
        and all(row["positions"] == 1024 for row in score["per_window"])
    )
    return {
        "schema": "banana-smasher-fixed-d4-terminal-rescore-receipt-v1",
        "status": "PASS",
        "execution_mode": "offline-layerwise-terminal-only",
        "terminal_only": True,
        "resident_engine": False,
        "window_layer_forwards": 0,
        "transformer_layer_forwards": 0,
        "source_state_path": str(state_path),
        "source_binding_sha256": binding,
        "source_binding_schema": source_binding_schema,
        "basis_sha256": basis_sha256,
        "bank_sha256": bank_sha,
        "teacher_sha256": teacher["identities"]["teacher_sha256"],
        "model_id": model_id,
        "pack_sha256": pack_sha256,
        "source_runtime_adapter_sha256": runtime_binding["sha256"],
        "runtime_adapter_sha256": active_runtime_binding["sha256"],
        "support_width": teacher["support_width"],
        "position_cutoff": 1024,
        "kld_semantics": "support-renormalized",
        "top1_semantics": "full-vocabulary-argmax",
        "output_rows": len(writer.completed_window_ids),
        "resumed_output_rows": resumed,
        "windows": score["windows"],
        "positions": score["positions"],
        "kld_sum": score["kld_sum"],
        "mean_kld": score["mean_kld"],
        "top1_matches": score["top1_matches"],
        "top1_agreement": score["top1_agreement"],
        "output_format": "torch-sidecars-with-json-manifest",
        "output": str(output_path),
        "output_sha256": manifest_sha,
        "teacher_manifest_sha256": teacher_descriptor["sha256"],
        "teacher_support_sidecar_manifest": teacher_descriptor,
        "candidate_output_sidecar_manifest": candidate_descriptor,
        "score": score_descriptor,
        "quality_rail": {
            "support_width": teacher["support_width"],
            "position_cutoff": 1024,
            "kld_semantics": "support-renormalized",
            "top1_semantics": "full-vocabulary-argmax",
            "teacher_support_sidecar_manifest": teacher_descriptor,
            "candidate_output_sidecar_manifest": candidate_descriptor,
            "score": score_descriptor,
        },
        "classification": (
            "authentic-top8192-anchor" if authentic else "backend-smoke-only"
        ),
    }
