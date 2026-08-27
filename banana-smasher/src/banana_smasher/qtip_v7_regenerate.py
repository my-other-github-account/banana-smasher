"""Manifest-driven QTIP2 V7 physical-member regeneration.

This module is orchestration around the public QTIP V7 producer primitives.  It
contains no alternate quantizer and refuses any source/model or Hessian identity
that is not hash-bound by the request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import qtip_k2
from .fixed_d4 import _SafetensorsShard, _source_tensor
from .q2_codec import k2_lut_fp16
from .qtip_v7_batch import (
    buffered_ldlq_cross_unit,
    finalize_batch_unit,
    prepare_v7_unit,
)

BASIS_SCHEMA = "banana-smasher-qtip2-v7-regenerate-request-v1"
MEMBER_SCHEMA = "banana-smasher-qtip2-v7-regenerated-member-v1"
LAYER_SCHEMA = "banana-smasher-qtip2-v7-regenerated-layer-terminal-v1"
TERMINAL_SCHEMA = "banana-smasher-qtip2-v7-regeneration-terminal-v1"
PROJECTIONS = ("w1", "w2", "w3")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: Any) -> bytes:
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_file() and path.stat().st_size == len(payload) and _sha256_file(path) == digest:
            return digest
        raise RuntimeError(f"immutable output collision: {path}")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
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
    return digest


def _atomic_json(path: Path, value: object) -> str:
    return _atomic_bytes(path, _canonical(value))


def _require_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _decode_nvfp4(packed: np.ndarray, scales: np.ndarray, *, device: str) -> Any:
    import torch
    packed_shape = tuple(int(value) for value in packed.shape)
    scale_shape = tuple(int(value) for value in scales.shape)
    if len(packed_shape) != 2 or len(scale_shape) != 2:
        raise ValueError("native NVFP4 source components must be matrices")
    if packed_shape[0] != scale_shape[0] or packed_shape[1] != scale_shape[1] * 16:
        raise ValueError("native NVFP4 source component geometry mismatch")
    blocks = torch.from_numpy(np.asarray(packed, dtype=np.uint8).copy()).to(device).reshape(scale_shape[0], scale_shape[1], 16)
    scale_tensor = torch.from_numpy(np.asarray(scales, dtype=np.uint8).copy()).to(device).reshape(scale_shape)
    lut = torch.tensor(E2M1, dtype=torch.float32, device=device)
    values = torch.stack((lut[blocks.int() & 15], lut[blocks.int() >> 4]), dim=-1).reshape(scale_shape[0], scale_shape[1], 32)
    values = values * torch.exp2(scale_tensor.float() - 127.0).unsqueeze(-1)
    return values.reshape(scale_shape[0], -1).to(torch.bfloat16).contiguous()


def _load_source(model_root: Path, weight_map: Mapping[str, object], shards: dict[Path, _SafetensorsShard], row: Mapping[str, Any]) -> Any:
    layer, expert, projection = int(row["layer"]), int(row["expert"]), str(row["projection"])
    prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
    packed = _source_tensor(model_root, weight_map, shards, prefix + ".weight", dtype="I8")
    scales = _source_tensor(model_root, weight_map, shards, prefix + ".scale", dtype="F8_E8M0")
    source = _decode_nvfp4(packed, scales, device="cuda")
    return source


def _load_hessian(row: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    spec = row.get("raw_hessian")
    if not isinstance(spec, Mapping):
        raise ValueError("member row lacks raw_hessian binding")
    path = Path(str(spec.get("path", ""))).expanduser().resolve()
    expected_sha = str(spec.get("sha256", ""))
    if not path.is_file() or len(expected_sha) != 64 or _sha256_file(path) != expected_sha:
        raise ValueError(f"raw Hessian file identity mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"raw Hessian geometry mismatch: {path}")
    data_sha = hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
    if data_sha != spec.get("data_sha256"):
        raise ValueError(f"raw Hessian data identity mismatch: {path}")
    return torch.from_numpy(np.asarray(array).copy()).to("cuda"), dict(spec)


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("schema") != BASIS_SCHEMA:
        raise ValueError(f"request schema must be {BASIS_SCHEMA}")
    basis = str(request.get("basis_sha256", ""))
    commit = str(request.get("canonical_commit_sha", ""))
    model_root = Path(str(request.get("model_root", ""))).expanduser().resolve()
    index = model_root / "model.safetensors.index.json"
    if len(basis) != 64 or not index.is_file() or _sha256_file(index) != basis:
        raise ValueError("source model index does not match intended basis")
    if len(commit) != 40:
        raise ValueError("canonical_commit_sha must be a full git SHA")
    rows = request.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("request members must be a non-empty array")
    identities = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("member rows must be objects")
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        if not 0 <= identity[1] < 256 or identity[2] not in PROJECTIONS:
            raise ValueError(f"invalid member identity: {identity}")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate member identities are forbidden")
    expected = request.get("expected_identities")
    if expected is not None:
        declared = {(int(x["layer"]), int(x["expert"]), str(x["projection"])) for x in expected}
        if set(identities) != declared:
            raise ValueError("request member identities differ from expected_identities")
    return {**dict(request), "basis_sha256": basis, "canonical_commit_sha": commit, "model_root": model_root, "members": [dict(row) for row in rows]}


def _ordered_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["member"]).encode() + b"\0")
        digest.update(str(row["sha256"]).encode() + b"\0")
        digest.update(str(int(row["bytes"])).encode() + b"\n")
    return digest.hexdigest()


def regenerate_qtip2_v7_physical(request: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate an exact, hash-bound set of QTIP2 V7 BF16 physical members."""
    import torch
    source_request = _require_object(Path(request).expanduser().resolve()) if not isinstance(request, Mapping) else dict(request)
    bound = _validate_request(source_request)
    output_root = Path(str(bound["output_root"])).expanduser().resolve()
    receipts = output_root / "receipts"
    progress_path = receipts / "PROGRESS.json"
    index = json.loads((bound["model_root"] / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("source model index lacks weight_map")
    if not torch.cuda.is_available():
        raise RuntimeError("QTIP2 V7 regeneration requires CUDA")
    parent_lut = torch.from_numpy(k2_lut_fp16().copy()).to("cuda")
    qtip_k2._cuda_extension()
    shards: dict[Path, _SafetensorsShard] = {}
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in bound["members"]:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)
    all_member_rows: list[dict[str, Any]] = []
    layer_terminals = []
    total_counters = {"qfn_calls": 0, "extension_calls": 0, "cuda_tiles": 0, "fallback_calls": 0}
    for layer in sorted(rows_by_layer):
        layer_rows = sorted(rows_by_layer[layer], key=lambda row: (int(row["expert"]), PROJECTIONS.index(str(row["projection"]))))
        experts = sorted({int(row["expert"]) for row in layer_rows})
        expected = {(expert, projection) for expert in experts for projection in PROJECTIONS}
        observed = {(int(row["expert"]), str(row["projection"])) for row in layer_rows}
        if observed != expected:
            raise ValueError(f"layer {layer} is not projection-complete for its expert roster")
        layer_member_rows: list[dict[str, Any]] = []
        for batch_start in range(0, len(experts), 10):
            batch_experts = set(experts[batch_start:batch_start + 10])
            batch_rows = [row for row in layer_rows if int(row["expert"]) in batch_experts]
            units = []
            for row in batch_rows:
                raw_h, h_spec = _load_hessian(row)
                source = _load_source(bound["model_root"], weight_map, shards, row)
                units.append({"layer": layer, "expert": int(row["expert"]), "projection": str(row["projection"]), "source": source, "raw_h": raw_h, "raw_h_count": int(h_spec["count"]), "input_identity": {"raw_hessian_data_sha256": h_spec["data_sha256"], "raw_hessian_sha256": h_spec["sha256"]}})
            hessian_cache: dict[tuple[str, int, int], tuple[Any, str, Any, str]] = {}
            prepared = []
            for unit in units:
                item = prepare_v7_unit(qtip_k2, unit, parent_lut, hessian_cache)
                key = (unit["input_identity"]["raw_hessian_data_sha256"], int(unit["raw_h_count"]), int(unit["source"].shape[1]))
                item["proxy_hessian"] = hessian_cache[key][0].detach().cpu()
                prepared.append(item)
            for projection in PROJECTIONS:
                group = [item for item in prepared if item["projection"] == projection]
                quantized, states, counters = buffered_ldlq_cross_unit(qtip_k2, group, parent_lut)
                if int(counters.get("fallback_calls", -1)) != 0 or int(counters.get("cuda_tiles", 0)) <= 0:
                    raise RuntimeError(f"QTIP2 V7 public CUDA path refused layer={layer} projection={projection}: {counters}")
                for key in total_counters:
                    total_counters[key] += int(counters.get(key, 0))
                for item, quantized_inner, state in zip(group, quantized, states, strict=True):
                    result = finalize_batch_unit(qtip_k2, item, quantized_inner, state)
                    physical = result["physical_bfloat16"]
                    payload = _tensor_bytes(physical)
                    member = str(result["member"])
                    expert = int(item["expert"])
                    target = output_root / "products" / f"L{layer:03d}" / f"E{expert:03d}" / f"{projection}.physical.bf16.bin"
                    payload_sha = _atomic_bytes(target, payload)
                    member_receipt = {"schema": MEMBER_SCHEMA, "status": "PASS", "task_id": bound["task_id"], "board_run_id": int(bound["board_run_id"]), "basis_sha256": bound["basis_sha256"], "canonical_commit_sha": bound["canonical_commit_sha"], "member": member, "layer": layer, "expert": expert, "projection": projection, "bytes": len(payload), "sha256": payload_sha, "path": str(target), "source_only": True, "fallback_calls": 0, "cuda_positive": True, "boundaries": result["boundaries"], "raw_hessian_sha256": item["input_identity"]["raw_hessian_sha256"], "created_unix": time.time()}
                    member_receipt_path = receipts / "members" / f"L{layer:03d}_E{expert:03d}_{projection}.json"
                    member_receipt_sha = _atomic_json(member_receipt_path, member_receipt)
                    row = {"member": member, "bytes": len(payload), "sha256": payload_sha, "path": str(target), "receipt": str(member_receipt_path), "receipt_sha256": member_receipt_sha}
                    layer_member_rows.append(row)
                    all_member_rows.append(row)
            _atomic_json(progress_path, {"schema": "banana-smasher-qtip2-v7-regen-progress-v1", "status": "RUNNING", "task_id": bound["task_id"], "board_run_id": int(bound["board_run_id"]), "basis_sha256": bound["basis_sha256"], "members_complete": len(all_member_rows), "members_expected": len(bound["members"]), "layer": layer, "batch_experts": sorted(batch_experts), "fallback_calls": 0, "updated_unix": time.time()})
            del units, prepared, hessian_cache
            torch.cuda.empty_cache()
        layer_member_rows.sort(key=lambda row: row["member"])
        if len(layer_member_rows) != len(layer_rows) or len({row["member"] for row in layer_member_rows}) != len(layer_rows):
            raise RuntimeError(f"layer {layer} physical closure failed")
        terminal = {"schema": LAYER_SCHEMA, "status": "PASS", "task_id": bound["task_id"], "board_run_id": int(bound["board_run_id"]), "basis_sha256": bound["basis_sha256"], "canonical_commit_sha": bound["canonical_commit_sha"], "layer": layer, "members": layer_member_rows, "members_complete": len(layer_member_rows), "members_expected": len(layer_rows), "bytes": sum(int(row["bytes"]) for row in layer_member_rows), "gaps": 0, "duplicates": 0, "fallback_calls": 0, "ordered_sha256": _ordered_sha(layer_member_rows), "created_unix": time.time()}
        path = receipts / f"L{layer:03d}_TERMINAL.json"
        layer_terminals.append({"layer": layer, "path": str(path), "sha256": _atomic_json(path, terminal), "members": len(layer_member_rows), "bytes": terminal["bytes"], "ordered_sha256": terminal["ordered_sha256"]})
    all_member_rows.sort(key=lambda row: row["member"])
    if len(all_member_rows) != len(bound["members"]) or len({row["member"] for row in all_member_rows}) != len(bound["members"]):
        raise RuntimeError("aggregate physical closure failed")
    terminal = {"schema": TERMINAL_SCHEMA, "status": "PASS", "task_id": bound["task_id"], "board_run_id": int(bound["board_run_id"]), "basis_sha256": bound["basis_sha256"], "canonical_commit_sha": bound["canonical_commit_sha"], "qsfp_locator": bound.get("qsfp_locator", "192.168.200.6"), "root": str(output_root), "members": all_member_rows, "members_complete": len(all_member_rows), "members_expected": len(bound["members"]), "bytes": sum(int(row["bytes"]) for row in all_member_rows), "gaps": 0, "duplicates": 0, "fallback_calls": 0, "ordered_sha256": _ordered_sha(all_member_rows), "layer_terminals": layer_terminals, "cuda": total_counters, "references": bound.get("references", []), "created_unix": time.time()}
    terminal_path = receipts / "AGGREGATE_TERMINAL.json"
    terminal_sha = _atomic_json(terminal_path, terminal)
    _atomic_json(progress_path, {**terminal, "aggregate_terminal_path": str(terminal_path), "aggregate_terminal_sha256": terminal_sha})
    return {**terminal, "terminal": str(terminal_path), "terminal_sha256": terminal_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate hash-bound QTIP2 V7 physical members")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    result = regenerate_qtip2_v7_physical(args.request)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"], "terminal_sha256": result["terminal_sha256"], "members": result["members_complete"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
