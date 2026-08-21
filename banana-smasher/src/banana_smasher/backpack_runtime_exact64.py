from __future__ import annotations


import hashlib
import json
import os
import shutil
import tempfile
import time
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anchor_sidecars import (
    CandidateSidecarWriter,
    load_teacher_support_manifest,
    load_teacher_window,
    _score_anchor_sidecars,
)
from .hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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
        raise ValueError("exact64 activation must be one non-empty numeric array")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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
        raise ValueError(f"exact64 activation checkpoint identity mismatch: {path}")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject or not value.size:
        raise ValueError(f"exact64 activation checkpoint is invalid: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _validate_whole_model_accounting(document: Mapping[str, Any]) -> Mapping[str, Any]:
    accounting = document.get("whole_model_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("exact64 requires standardized whole-model accounting")

    integer_fields = (
        "expert_physical_wire_bytes",
        "dense_nonrouted_bytes",
        "repair_bytes",
        "metadata_bytes",
        "fixed_nonexpert_bytes",
        "whole_shipping_bytes",
        "shipping_bytes_cap",
        "shipping_slack_bytes",
        "logical_base_parameters",
        "whole_model_bpw_numerator_bits",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = accounting.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"exact64 whole-model accounting field is invalid: {field}"
            )
        values[field] = value
    if values["shipping_bytes_cap"] == 0 or values["logical_base_parameters"] == 0:
        raise ValueError(
            "exact64 whole-model target and logical denominator must be positive"
        )

    fixed = (
        values["dense_nonrouted_bytes"]
        + values["repair_bytes"]
        + values["metadata_bytes"]
    )
    if values["fixed_nonexpert_bytes"] != fixed:
        raise ValueError("exact64 fixed_nonexpert_bytes equation mismatch")
    whole = values["expert_physical_wire_bytes"] + fixed
    if values["whole_shipping_bytes"] != whole:
        raise ValueError("exact64 whole_shipping_bytes equation mismatch")
    if whole > values["shipping_bytes_cap"]:
        raise ValueError("exact64 whole-model shipping target exceeded")
    if values["shipping_slack_bytes"] != values["shipping_bytes_cap"] - whole:
        raise ValueError("exact64 shipping_slack_bytes equation mismatch")
    numerator = whole * 8
    if values["whole_model_bpw_numerator_bits"] != numerator:
        raise ValueError("exact64 whole-model BPW numerator mismatch")
    ratio = f"{numerator}/{values['logical_base_parameters']}"
    if accounting.get("whole_model_bpw_exact_ratio") != ratio:
        raise ValueError("exact64 whole-model BPW exact ratio mismatch")
    with localcontext() as context:
        context.prec = 80
        decimal_value = format(
            Decimal(numerator) / Decimal(values["logical_base_parameters"]),
            "f",
        )
    if accounting.get("whole_model_bpw_decimal") != decimal_value:
        raise ValueError("exact64 whole-model BPW decimal mismatch")
    return accounting


def _revision_bind_teacher_manifest(
    source_path: Path,
    output_root: Path,
    *,
    bank_sha256: str,
    basis_sha256: str,
    model_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Load a teacher manifest that already proves the active model revision."""

    source_document = json.loads(source_path.read_text())
    if source_document.get("schema") != "banana-smasher-anchor-teacher-sidecars-v2":
        raise ValueError("exact64 requires a revision-bound teacher manifest")
    source = load_teacher_support_manifest(
        source_path,
        expected_bank_sha256=bank_sha256,
        expected_basis_sha256=basis_sha256,
        expected_model_id=model_id,
        require_revision_binding=True,
    )
    return source_path, source


def _run_backpack_exact64(
    *,
    model_root: str | Path,
    bank_path: str | Path,
    teacher_manifest_path: str | Path,
    virtual_manifest_path: str | Path,
    materialization_index_path: str | Path,
    qtip2_root_map_path: str | Path | None = None,
    qtip3_root_map_path: str | Path | None = None,
    qtip2_v7_root_map_path: str | Path | None = None,
    qtip2_v7_shared_lut_path: str | Path | None = None,
    qtip2_v7_member_roster_path: str | Path | None = None,
    output_root: str | Path,
    basis_sha256: str,
    expected_windows: int = 64,
    slice_id: str | None = None,
    class_by_window: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a virtual mixed Backpack assignment on one exact bound window rail.

    The public default remains exact64. The measured SPSA caller explicitly
    selects eight revision-bound TRAIN windows and supplies their class map.
    """

    model_root = Path(model_root).resolve()
    bank_path = Path(bank_path).resolve()
    teacher_manifest_path = Path(teacher_manifest_path).resolve()
    virtual_manifest_path = Path(virtual_manifest_path).resolve()
    materialization_index_path = Path(materialization_index_path).resolve()
    qtip2_root_map_path = (
        None if qtip2_root_map_path is None else Path(qtip2_root_map_path).resolve()
    )
    qtip3_root_map_path = (
        None if qtip3_root_map_path is None else Path(qtip3_root_map_path).resolve()
    )
    v7_values = (
        qtip2_v7_root_map_path,
        qtip2_v7_shared_lut_path,
        qtip2_v7_member_roster_path,
    )
    if any(value is not None for value in v7_values) != all(
        value is not None for value in v7_values
    ):
        raise ValueError("exact64 QTIP2 V7 bindings must be supplied together")
    qtip2_v7_root_map_path = (
        None
        if qtip2_v7_root_map_path is None
        else Path(qtip2_v7_root_map_path).resolve()
    )
    qtip2_v7_shared_lut_path = (
        None
        if qtip2_v7_shared_lut_path is None
        else Path(qtip2_v7_shared_lut_path).resolve()
    )
    qtip2_v7_member_roster_path = (
        None
        if qtip2_v7_member_roster_path is None
        else Path(qtip2_v7_member_roster_path).resolve()
    )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipts = output_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)

    index_path = model_root / "model.safetensors.index.json"
    if _sha256_file(index_path) != basis_sha256:
        raise ValueError("exact64 model basis mismatch")
    bank_sha256 = _sha256_file(bank_path)
    bank_rows = _read_jsonl(bank_path)
    if expected_windows not in {8, 64} or len(bank_rows) != expected_windows:
        raise ValueError(
            f"Backpack rail requires exactly {expected_windows} bank rows, "
            f"got {len(bank_rows)}"
        )
    model_id = "deepseek-ai/DeepSeek-V4-Flash-0731"
    teacher_manifest_path, teacher = _revision_bind_teacher_manifest(
        teacher_manifest_path,
        output_root,
        bank_sha256=bank_sha256,
        basis_sha256=basis_sha256,
        model_id=model_id,
    )
    if teacher["support_width"] != 8192:
        raise ValueError("exact64 requires Top-8192 teacher support")
    window_ids = [row.get("window_id") for row in bank_rows]
    if window_ids != teacher["window_ids"]:
        raise ValueError("exact64 bank/teacher window order mismatch")
    if expected_windows == 8:
        if not isinstance(slice_id, str) or not slice_id:
            raise ValueError("measured TRAIN8 requires a slice_id")
        if (
            not isinstance(class_by_window, Mapping)
            or set(class_by_window) != {str(value) for value in window_ids}
            or set(class_by_window.values())
            != {
                "agentic",
                "chat",
                "code",
                "multilingual",
                "prose",
                "reasoning",
            }
        ):
            raise ValueError(
                "measured TRAIN8 requires complete six-class window metadata"
            )
    for row in bank_rows:
        tokens = row.get("token_ids")
        if (
            not isinstance(tokens, list)
            or len(tokens) < 1024
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
        ):
            raise ValueError("exact64 bank token row is invalid")

    virtual_manifest = json.loads(virtual_manifest_path.read_text())
    if (
        virtual_manifest.get("status") != "PASS_LOGICAL_FULL_WIRE"
        or virtual_manifest.get("basis_sha256") != basis_sha256
        or virtual_manifest.get("storage", {}).get("tensor_payload_copy_bytes") != 0
    ):
        raise ValueError("exact64 virtual manifest identity mismatch")
    _validate_whole_model_accounting(virtual_manifest)
    virtual_files = []
    for path in (
        virtual_manifest_path.parent / "ASSIGNMENT.json",
        virtual_manifest_path,
        materialization_index_path,
    ):
        virtual_files.append(
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    virtual_files.sort(key=lambda row: row["file"])
    pack_sha256 = hashlib.sha256(
        (
            json.dumps(virtual_files, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()
    parameters = {
        "positions": 1024,
        "backpack_runtime": {
            "basis_sha256": basis_sha256,
            "virtual_manifest": str(virtual_manifest_path),
            "materialization_index": str(materialization_index_path),
        },
    }
    binding_inputs = parameters["backpack_runtime"]
    root_map_hashes = {}
    for source_key, root_map_path in (
        ("qtip2", qtip2_root_map_path),
        ("qtip3", qtip3_root_map_path),
    ):
        if root_map_path is not None:
            binding_inputs[f"{source_key}_root_map"] = str(root_map_path)
            root_map_hashes[f"{source_key}_root_map_sha256"] = _sha256_file(
                root_map_path
            )
    if qtip2_v7_root_map_path is not None:
        binding_inputs.update(
            {
                "qtip2_v7_root_map": str(qtip2_v7_root_map_path),
                "qtip2_v7_shared_lut": str(qtip2_v7_shared_lut_path),
                "qtip2_v7_member_roster": str(qtip2_v7_member_roster_path),
            }
        )
    v7_hashes = (
        {}
        if qtip2_v7_root_map_path is None
        else {
            "qtip2_v7_root_map_sha256": _sha256_file(qtip2_v7_root_map_path),
            "qtip2_v7_shared_lut_sha256": _sha256_file(qtip2_v7_shared_lut_path),
            "qtip2_v7_member_roster_sha256": _sha256_file(qtip2_v7_member_roster_path),
        }
    )
    binding = hashlib.sha256(
        json.dumps(
            {
                "basis_sha256": basis_sha256,
                "bank_sha256": bank_sha256,
                "teacher_manifest_sha256": _sha256_file(teacher_manifest_path),
                "pack_sha256": pack_sha256,
                "materialization_index_sha256": _sha256_file(
                    materialization_index_path
                ),
                **root_map_hashes,
                **v7_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    run_root = output_root / "layerwise"
    state_path = run_root / "STATE.json"
    progress_path = receipts / "PROGRESS.json"
    run_root.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("binding_sha256") != binding:
            raise ValueError("exact64 resume binding mismatch")
    else:
        state = {
            "schema": "banana-smasher-backpack-exact64-state-v1",
            "binding_sha256": binding,
            "checkpoints": {},
            "completed_layers": [],
        }
        _atomic_json(state_path, state)

    runtime = DeepseekV4BackpackRuntime(model_root=model_root, parameters=parameters)
    started = time.monotonic()
    forwards = 0

    def progress(
        stage: str, *, layer: int | None = None, slot: int | None = None
    ) -> None:
        _atomic_json(
            progress_path,
            {
                "schema": "banana-smasher-backpack-exact64-progress-v1",
                "status": "PASS" if stage == "complete" else "RUNNING",
                "binding_sha256": binding,
                "stage": stage,
                "layer": layer,
                "window_slot": slot,
                "completed_layers": list(state["completed_layers"]),
                "window_layer_forwards": forwards,
                "candidate_windows": len(writer.completed_window_ids)
                if "writer" in locals()
                else 0,
                "elapsed_seconds": time.monotonic() - started,
                "resident_bytes": runtime.resident_bytes(),
                "peak_resident_bytes": runtime.peak_resident_bytes(),
                "bytes_read": runtime.bytes_read(),
            },
        )

    checkpoints = state["checkpoints"]
    completed_layers = state["completed_layers"]
    if completed_layers != list(range(len(completed_layers))):
        raise ValueError("exact64 completed layer prefix mismatch")
    if completed_layers:
        source_stage = f"layer_{completed_layers[-1]}"
    else:
        source_stage = "initial"
        initial_receipts = checkpoints.setdefault(source_stage, {})
        initial_dir = run_root / source_stage
        with runtime.initial_stage() as embed:
            for slot, row in enumerate(bank_rows):
                slot_key = str(slot)
                if slot_key in initial_receipts:
                    continue
                activation = embed(row["token_ids"][:1024], window_id=row["window_id"])
                initial_receipts[slot_key] = _atomic_npy(
                    initial_dir / f"window_{slot:03d}.npy",
                    runtime.export_activation(activation),
                )
                runtime.synchronize()
                _atomic_json(state_path, state)
                progress("initial", slot=slot)
                del activation

    for layer in range(len(completed_layers), 43):
        target_stage = f"layer_{layer}"
        target_dir = run_root / target_stage
        target_receipts = checkpoints.setdefault(target_stage, {})
        source_receipts = checkpoints.get(source_stage)
        if (
            not isinstance(source_receipts, Mapping)
            or len(source_receipts) != expected_windows
        ):
            raise ValueError("exact64 source checkpoint coverage mismatch")
        with runtime.layer_stage(layer) as forward:
            for slot, row in enumerate(bank_rows):
                slot_key = str(slot)
                if slot_key in target_receipts:
                    continue
                source = _read_activation(
                    run_root / source_stage / f"window_{slot:03d}.npy",
                    source_receipts[slot_key],
                )
                activation = runtime.import_activation(source)
                output = forward(activation, window_id=row["window_id"])
                target_receipts[slot_key] = _atomic_npy(
                    target_dir / f"window_{slot:03d}.npy",
                    runtime.export_activation(output),
                )
                runtime.synchronize()
                forwards += 1
                _atomic_json(state_path, state)
                progress("layer", layer=layer, slot=slot)
                del source, activation, output
        if len(target_receipts) != expected_windows:
            raise ValueError(f"exact64 layer {layer} checkpoint coverage mismatch")
        state["completed_layers"].append(layer)
        previous = run_root / source_stage
        source_stage = target_stage
        _atomic_json(state_path, state)
        if previous != target_dir:
            shutil.rmtree(previous)
            checkpoints.pop(previous.name, None)
            _atomic_json(state_path, state)
        progress("layer-complete", layer=layer)

    candidate_manifest = output_root / "candidate_top8192.json"
    writer = CandidateSidecarWriter(
        candidate_manifest,
        teacher_manifest_path=teacher_manifest_path,
        window_ids=window_ids,
        basis_sha256=basis_sha256,
        bank_sha256=bank_sha256,
        model_id=model_id,
        pack_sha256=pack_sha256,
    )
    final_receipts = checkpoints.get(source_stage)
    if (
        not isinstance(final_receipts, Mapping)
        or len(final_receipts) != expected_windows
    ):
        raise ValueError("exact64 terminal checkpoint coverage mismatch")
    with runtime.terminal_stage() as score:
        for slot, row in enumerate(bank_rows):
            if row["window_id"] in writer.completed_window_ids:
                continue
            source = _read_activation(
                run_root / source_stage / f"window_{slot:03d}.npy",
                final_receipts[str(slot)],
            )
            activation = runtime.import_activation(source)
            support, _ = load_teacher_window(
                teacher_manifest_path,
                row["window_id"],
                manifest=teacher,
            )
            scored = score(
                activation,
                support[:1024],
                window_id=row["window_id"],
            )
            writer.write_window(
                row["window_id"],
                q_lp_at_ref=scored["q_lp_at_ref"],
                q_argmax=scored["q_argmax"],
            )
            progress("terminal", slot=slot)
            del source, activation, support, scored

    score_result = _score_anchor_sidecars(teacher_manifest_path, candidate_manifest)
    class_kld: dict[str, float] | None = None
    class_top1: dict[str, dict[str, float | int]] | None = None
    if class_by_window is not None:
        class_kld_sum: dict[str, float] = {}
        class_positions: dict[str, int] = {}
        class_matches: dict[str, int] = {}
        for row in score_result["per_window"]:
            name = class_by_window[str(row["window_id"])]
            class_kld_sum[name] = class_kld_sum.get(name, 0.0) + float(row["kld_sum"])
            class_positions[name] = class_positions.get(name, 0) + int(row["positions"])
            class_matches[name] = class_matches.get(name, 0) + int(row["top1_matches"])
        class_kld = {
            name: class_kld_sum[name] / class_positions[name]
            for name in sorted(class_positions)
        }
        class_top1 = {
            name: {
                "matches": class_matches[name],
                "positions": class_positions[name],
                "agreement": class_matches[name] / class_positions[name],
            }
            for name in sorted(class_positions)
        }
    top1_receipt = {
        "schema": "banana-smasher-backpack-exact64-top1-v1",
        "status": "PASS",
        "binding_sha256": binding,
        "windows": score_result["windows"],
        "positions": score_result["positions"],
        "top1_matches": score_result["top1_matches"],
        "top1_agreement": score_result["top1_agreement"],
        "top1_semantics": score_result["top1_semantics"],
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
    }
    _atomic_json(receipts / "TOP1.json", top1_receipt)
    _atomic_json(receipts / "SCORE.json", score_result)
    result = {
        "schema": (
            "banana-smasher-backpack-exact64-terminal-v1"
            if expected_windows == 64
            else "banana-smasher-backpack-train8-terminal-v1"
        ),
        "status": "PASS",
        "binding_sha256": binding,
        "basis_sha256": basis_sha256,
        "bank_sha256": bank_sha256,
        "pack_sha256": pack_sha256,
        "assignment_sha256": virtual_manifest["assignment_map_sha256"],
        "slice_id": slice_id,
        "window_ids": window_ids,
        "windows": score_result["windows"],
        "positions": score_result["positions"],
        "support_width": score_result["support_width"],
        "top1_matches": score_result["top1_matches"],
        "top1_agreement": score_result["top1_agreement"],
        "mean_kld": score_result["mean_kld"],
        "kld_sum": score_result["kld_sum"],
        **(
            {"class_kld": class_kld, "class_top1": class_top1}
            if class_kld is not None
            else {}
        ),
        "holdout_used": False,
        "repair_applied": False,
        "full_window_forwards": expected_windows,
        "transformer_layer_forwards": expected_windows * 43,
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        "top1_receipt_sha256": _sha256_file(receipts / "TOP1.json"),
        "score_receipt_sha256": _sha256_file(receipts / "SCORE.json"),
        "runtime_adapter": "banana_smasher.hf_deepseek_v4_backpack_adapter.DeepseekV4BackpackRuntime",
        "runtime_adapter_sha256": _sha256_file(
            Path(__file__).with_name("hf_deepseek_v4_backpack_adapter.py")
        ),
        "elapsed_seconds": time.monotonic() - started,
        "bytes_read": runtime.bytes_read(),
        "peak_resident_bytes": runtime.peak_resident_bytes(),
    }
    _atomic_json(receipts / "TERMINAL.json", result)
    progress("complete")
    return result


def _run_backpack_train8(**kwargs: Any) -> dict[str, Any]:
    """Run one balanced eight-window TRAIN SPSA measurement."""

    return _run_backpack_exact64(expected_windows=8, **kwargs)
