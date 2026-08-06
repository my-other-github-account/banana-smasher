from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anchor_sidecars import (
    CandidateSidecarWriter,
    load_teacher_support_manifest,
    load_teacher_window,
    score_anchor_sidecars,
)
from .hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime
from .locality import require_local_backpack_inputs
from .offline_layerwise import _atomic_json, _atomic_npy, _read_activation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _expert_chunk_width(layer: int) -> int:
    """Use measured expert slices as checkpoint pressure grows across later layers."""

    if layer == 13:
        return 8
    return 4 if layer >= 14 else 32


def _revision_bind_teacher_manifest(
    source_path: Path,
    output_root: Path,
    *,
    basis_sha256: str,
    model_id: str,
) -> Path:
    """Promote a validated historical manifest without changing teacher tensors."""

    source = load_teacher_support_manifest(source_path)
    if source["schema"] == "banana-smasher-anchor-teacher-sidecars-v2":
        return source_path
    target = output_root / "teacher_support" / "teacher_top8192.v2.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    sidecars = target.parent / "teacher_top8192.v2.sidecars"
    sidecars.mkdir(parents=True, exist_ok=True)
    for slot, entry in enumerate(source["windows"]):
        source_sidecar = (source_path.parent / entry["path"]).resolve()
        relative = Path("teacher_top8192.v2.sidecars") / f"t8192_{slot:03d}.pt"
        target_sidecar = target.parent / relative
        if not target_sidecar.exists():
            try:
                os.link(source_sidecar, target_sidecar)
            except OSError:
                shutil.copyfile(source_sidecar, target_sidecar)
        if (
            target_sidecar.stat().st_size != entry["bytes"]
            or _sha256_file(target_sidecar) != entry["sha256"]
        ):
            raise ValueError("revision-bound teacher sidecar identity mismatch")
        entries.append({**entry, "path": relative.as_posix()})
    promoted = {
        "schema": "banana-smasher-anchor-teacher-sidecars-v2",
        "support_width": source["support_width"],
        "window_ids": source["window_ids"],
        "identities": {
            "bank_sha256": source["identities"]["bank_sha256"],
            "teacher_sha256": source["identities"]["teacher_sha256"],
            "basis_sha256": basis_sha256,
            "model_id": model_id,
        },
        "windows": entries,
    }
    if target.exists():
        existing = json.loads(target.read_text())
        if existing != promoted:
            raise ValueError("revision-bound teacher manifest resume mismatch")
    else:
        _atomic_json(target, promoted)
    load_teacher_support_manifest(
        target,
        expected_basis_sha256=basis_sha256,
        expected_model_id=model_id,
        require_revision_binding=True,
    )
    return target


def run_backpack_exact64(
    *,
    model_root: str | Path,
    bank_path: str | Path,
    teacher_manifest_path: str | Path,
    virtual_manifest_path: str | Path,
    materialization_index_path: str | Path,
    qtip2_root_map_path: str | Path,
    qtip3_root_map_path: str | Path,
    output_root: str | Path,
    basis_sha256: str,
) -> dict[str, Any]:
    """Run a virtual mixed Backpack assignment on the exact 64-window rail."""

    require_local_backpack_inputs(
        model_root=model_root,
        bank_path=bank_path,
        teacher_manifest_path=teacher_manifest_path,
        virtual_manifest_path=virtual_manifest_path,
        materialization_index_path=materialization_index_path,
        qtip2_root_map_path=qtip2_root_map_path,
        qtip3_root_map_path=qtip3_root_map_path,
        output_root=output_root,
    )

    model_root = Path(model_root).resolve()
    bank_path = Path(bank_path).resolve()
    teacher_manifest_path = Path(teacher_manifest_path).resolve()
    virtual_manifest_path = Path(virtual_manifest_path).resolve()
    materialization_index_path = Path(materialization_index_path).resolve()
    qtip2_root_map_path = Path(qtip2_root_map_path).resolve()
    qtip3_root_map_path = Path(qtip3_root_map_path).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipts = output_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)

    index_path = model_root / "model.safetensors.index.json"
    if _sha256_file(index_path) != basis_sha256:
        raise ValueError("exact64 model basis mismatch")
    bank_sha256 = _sha256_file(bank_path)
    bank_rows = _read_jsonl(bank_path)
    if len(bank_rows) != 64:
        raise ValueError(f"exact64 requires 64 bank rows, got {len(bank_rows)}")
    model_id = "deepseek-ai/DeepSeek-V4-Flash-0731"
    teacher_manifest_path = _revision_bind_teacher_manifest(
        teacher_manifest_path,
        output_root,
        basis_sha256=basis_sha256,
        model_id=model_id,
    )
    teacher = load_teacher_support_manifest(
        teacher_manifest_path,
        expected_bank_sha256=bank_sha256,
        expected_basis_sha256=basis_sha256,
        expected_model_id=model_id,
        require_revision_binding=True,
    )
    if teacher["support_width"] != 8192:
        raise ValueError("exact64 requires Top-8192 teacher support")
    window_ids = [row.get("window_id") for row in bank_rows]
    if window_ids != teacher["window_ids"]:
        raise ValueError("exact64 bank/teacher window order mismatch")
    for row in bank_rows:
        tokens = row.get("token_ids")
        if (
            not isinstance(tokens, list)
            or len(tokens) < 1024
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
        ):
            raise ValueError("exact64 bank token row is invalid")

    virtual_manifest = json.loads(virtual_manifest_path.read_text())
    if (
        virtual_manifest.get("status") != "PASS_LOGICAL_FULL_WIRE"
        or virtual_manifest.get("basis_sha256") != basis_sha256
        or virtual_manifest.get("storage", {}).get("tensor_payload_copy_bytes") != 0
    ):
        raise ValueError("exact64 virtual manifest identity mismatch")
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
        (json.dumps(virtual_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    parameters = {
        "positions": 1024,
        "backpack_runtime": {
            "basis_sha256": basis_sha256,
            "virtual_manifest": str(virtual_manifest_path),
            "materialization_index": str(materialization_index_path),
            "qtip2_root_map": str(qtip2_root_map_path),
            "qtip3_root_map": str(qtip3_root_map_path),
        },
    }
    binding = hashlib.sha256(
        json.dumps(
            {
                "basis_sha256": basis_sha256,
                "bank_sha256": bank_sha256,
                "teacher_manifest_sha256": _sha256_file(teacher_manifest_path),
                "pack_sha256": pack_sha256,
                "materialization_index_sha256": _sha256_file(materialization_index_path),
                "qtip2_root_map_sha256": _sha256_file(qtip2_root_map_path),
                "qtip3_root_map_sha256": _sha256_file(qtip3_root_map_path),
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

    def progress(stage: str, *, layer: int | None = None, slot: int | None = None) -> None:
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
                "candidate_windows": len(writer.completed_window_ids) if "writer" in locals() else 0,
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
        if not isinstance(source_receipts, Mapping) or len(source_receipts) != 64:
            raise ValueError("exact64 source checkpoint coverage mismatch")
        # Split attention from the 12 GiB mixed-expert materialization before
        # the full-layer path can violate the mandatory 4 GiB CUDA guard.
        if layer >= 10:
            attention_stage = f"layer_{layer}_attention"
            attention_dir = run_root / attention_stage
            attention_receipts = checkpoints.setdefault(attention_stage, {})
            with runtime.attention_stage(layer) as forward_attention:
                for slot, row in enumerate(bank_rows):
                    slot_key = str(slot)
                    if slot_key in attention_receipts:
                        continue
                    source = _read_activation(
                        run_root / source_stage / f"window_{slot:03d}.npy",
                        source_receipts[slot_key],
                    )
                    activation = runtime.import_activation(source)
                    output = forward_attention(
                        activation, window_id=row["window_id"]
                    )
                    attention_receipts[slot_key] = _atomic_npy(
                        attention_dir / f"window_{slot:03d}.npy",
                        runtime.export_activation(output),
                    )
                    runtime.synchronize()
                    _atomic_json(state_path, state)
                    progress("layer-attention", layer=layer, slot=slot)
                    del source, activation, output
            if len(attention_receipts) != 64:
                raise ValueError(f"exact64 layer {layer} attention coverage mismatch")
            # Once attention is sealed, this layer no longer reads its source
            # activation. Drop the superseded task-owned stage before expert
            # materialization so tmpfs pages do not consume the CUDA guard.
            source_dir = run_root / source_stage
            if source_dir != target_dir and source_dir.exists():
                shutil.rmtree(source_dir)
            chunk_prefix = f"layer_{layer}_mlp_experts_"
            resumable_chunks: list[tuple[int, str]] = []
            for stage_name, stage_receipts in checkpoints.items():
                if not stage_name.startswith(chunk_prefix) or len(stage_receipts) != 64:
                    continue
                _, stop_text = stage_name.removeprefix(chunk_prefix).split("_", 1)
                resumable_chunks.append((int(stop_text), stage_name))
            if resumable_chunks:
                next_expert, previous_chunk_stage = max(resumable_chunks)
            else:
                next_expert, previous_chunk_stage = 0, None
            chunk_width = _expert_chunk_width(layer)
            expert_chunks = tuple(
                (start, min(start + chunk_width, 256))
                for start in range(next_expert, 256, chunk_width)
            )
            for chunk_start, chunk_stop in expert_chunks:
                final_chunk = chunk_stop == 256
                chunk_stage = (
                    target_stage
                    if final_chunk
                    else f"layer_{layer}_mlp_experts_{chunk_start}_{chunk_stop}"
                )
                chunk_dir = run_root / chunk_stage
                chunk_receipts = checkpoints.setdefault(chunk_stage, {})
                with runtime.mlp_chunk_stage(
                    layer, chunk_start, chunk_stop
                ) as forward_mlp_chunk:
                    for slot, row in enumerate(bank_rows):
                        slot_key = str(slot)
                        if slot_key in chunk_receipts:
                            continue
                        source = _read_activation(
                            attention_dir / f"window_{slot:03d}.npy",
                            attention_receipts[slot_key],
                        )
                        activation = runtime.import_activation(source)
                        accumulated = None
                        previous_source = None
                        if previous_chunk_stage is not None:
                            previous_receipts = checkpoints[previous_chunk_stage]
                            previous_source = _read_activation(
                                run_root
                                / previous_chunk_stage
                                / f"window_{slot:03d}.npy",
                                previous_receipts[slot_key],
                            )
                            accumulated = runtime.import_activation(previous_source)
                        output = forward_mlp_chunk(
                            activation,
                            window_id=row["window_id"],
                            accumulated=accumulated,
                            include_shared_residual=chunk_start == 0,
                        )
                        chunk_receipts[slot_key] = _atomic_npy(
                            chunk_dir / f"window_{slot:03d}.npy",
                            runtime.export_activation(output),
                        )
                        runtime.synchronize()
                        if final_chunk:
                            forwards += 1
                        _atomic_json(state_path, state)
                        progress(
                            f"layer-mlp-experts-{chunk_start}-{chunk_stop}",
                            layer=layer,
                            slot=slot,
                        )
                        del source, activation, accumulated, output
                        if previous_source is not None:
                            del previous_source
                if len(chunk_receipts) != 64:
                    raise ValueError(
                        f"exact64 layer {layer} MLP expert chunk "
                        f"{chunk_start}:{chunk_stop} coverage mismatch"
                    )
                if previous_chunk_stage is not None:
                    previous_chunk_dir = run_root / previous_chunk_stage
                    if previous_chunk_dir != target_dir:
                        shutil.rmtree(previous_chunk_dir)
                        checkpoints.pop(previous_chunk_stage, None)
                        _atomic_json(state_path, state)
                previous_chunk_stage = chunk_stage
            if len(target_receipts) != 64:
                raise ValueError(f"exact64 layer {layer} MLP coverage mismatch")
            state["completed_layers"].append(layer)
            previous = run_root / source_stage
            source_stage = target_stage
            _atomic_json(state_path, state)
            if previous != target_dir:
                shutil.rmtree(previous, ignore_errors=True)
                checkpoints.pop(previous.name, None)
            shutil.rmtree(attention_dir)
            checkpoints.pop(attention_stage, None)
            _atomic_json(state_path, state)
            progress("layer-complete", layer=layer)
            continue
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
        if len(target_receipts) != 64:
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
    if not isinstance(final_receipts, Mapping) or len(final_receipts) != 64:
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

    score_result = score_anchor_sidecars(teacher_manifest_path, candidate_manifest)
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
        "schema": "banana-smasher-backpack-exact64-terminal-v1",
        "status": "PASS",
        "binding_sha256": binding,
        "basis_sha256": basis_sha256,
        "bank_sha256": bank_sha256,
        "pack_sha256": pack_sha256,
        "windows": score_result["windows"],
        "positions": score_result["positions"],
        "support_width": score_result["support_width"],
        "top1_matches": score_result["top1_matches"],
        "top1_agreement": score_result["top1_agreement"],
        "mean_kld": score_result["mean_kld"],
        "kld_sum": score_result["kld_sum"],
        "holdout_used": False,
        "repair_applied": False,
        "full_window_forwards": 64,
        "transformer_layer_forwards": 64 * 43,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run virtual Backpack exact64 scoring")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--virtual-manifest", required=True)
    parser.add_argument("--materialization-index", required=True)
    parser.add_argument("--qtip2-root-map", required=True)
    parser.add_argument("--qtip3-root-map", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--basis-sha256", required=True)
    args = parser.parse_args(argv)
    result = run_backpack_exact64(
        model_root=args.model_root,
        bank_path=args.bank,
        teacher_manifest_path=args.teacher_manifest,
        virtual_manifest_path=args.virtual_manifest,
        materialization_index_path=args.materialization_index,
        qtip2_root_map_path=args.qtip2_root_map,
        qtip3_root_map_path=args.qtip3_root_map,
        output_root=args.output_root,
        basis_sha256=args.basis_sha256,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
