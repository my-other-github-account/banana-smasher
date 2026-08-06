from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from banana_smasher.qtip_periodic_signal import (
    AVG_MEMBER_BASELINE_RECEIPT_SHA256,
    FF0731_MODEL_INDEX_SHA256,
    PERIODIC_SIGNAL_CANDIDATES,
    TEACHER_TOP8192_MANIFEST_SHA256,
    TRAIN64_BANK_MANIFEST_SHA256,
    TRAIN8_POSITION_CUTOFF,
    TRAIN8_ROW_IDS,
    TRAIN8_SUPPORT_WIDTH,
)

MANIFEST_SCHEMA = "banana-smasher-qtip25-periodic-two-cell-train8-run-v1"
PROGRESS_SCHEMA = "banana-smasher-qtip25-periodic-two-cell-train8-progress-v1"
ARM_CELL_SOURCES = {
    "qtip_k2": ("qtip_k2", "source"),
    "qtip_k3": ("source", "qtip_k3"),
    "qtip25_avg_member": ("qtip_k2", "qtip_k3"),
    "qtip25_periodic_23": ("periodic", "periodic"),
}
E2M1_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: str | Path, value: object, *, overwrite: bool = True) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination, follow_symlinks=False)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _require_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def arm_cell_sources() -> dict[str, tuple[str, str]]:
    """Return the exact two-cell overlay semantics used by the train8 runner."""
    return dict(ARM_CELL_SOURCES)


def matched_measurements(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, int]]:
    """Derive honest single-cell controls and matched-pair measurements."""
    if len(cells) != 2:
        raise ValueError("two-cell train8 runner requires exactly two cells")
    by_control = {str(cell["control"]): cell for cell in cells}
    if set(by_control) != {"qtip_k2", "qtip_k3"}:
        raise ValueError("two-cell train8 runner requires one K2 and one K3 control")
    k2 = by_control["qtip_k2"]
    k3 = by_control["qtip_k3"]
    k2_error = float(k2["direct_error"]["control_sse"])
    k3_error = float(k3["direct_error"]["control_sse"])
    periodic_error = sum(float(cell["direct_error"]["periodic_sse"]) for cell in cells)
    errors = {
        "qtip_k2": k2_error,
        "qtip_k3": k3_error,
        "qtip25_avg_member": k2_error + k3_error,
        "qtip25_periodic_23": periodic_error,
    }
    k2_bits = int(k2["accounting"]["weights"]) * 2
    k3_bits = int(k3["accounting"]["weights"]) * 3
    bits = {
        "qtip_k2": k2_bits,
        "qtip_k3": k3_bits,
        "qtip25_avg_member": k2_bits + k3_bits,
        "qtip25_periodic_23": sum(
            int(cell["accounting"]["code_bits"]) for cell in cells
        ),
    }
    if bits["qtip25_avg_member"] != bits["qtip25_periodic_23"]:
        raise ValueError("AVG-MEMBER and PERIODIC matched-pair code bits differ")
    if not all(math.isfinite(value) and value >= 0.0 for value in errors.values()):
        raise ValueError("two-cell direct errors must be finite and nonnegative")
    return errors, bits


def _hashed_path(value: Mapping[str, Any], name: str, *, verify: bool) -> Path:
    path = Path(str(value["path"]))
    expected = _require_sha(f"{name}.sha256", value["sha256"])
    if verify:
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{name} hash mismatch: {observed} != {expected}")
    return path


def _validate_cell_payload(cell: Mapping[str, Any]) -> None:
    import torch

    control = str(cell["control"])
    expected_k = {"qtip_k2": 2, "qtip_k3": 3}[control]
    unit = torch.load(
        cell["control_unit"]["path"], map_location="cpu", mmap=True, weights_only=True
    )
    required = {"shape", "geometry", "trellis", "SU", "SV", "Wscale", "tlut"}
    if not isinstance(unit, Mapping) or not required.issubset(unit):
        raise ValueError(f"{control} unit is missing required payload fields")
    if tuple(int(value) for value in unit["shape"]) != (4096, 2048):
        raise ValueError(f"{control} unit has wrong down-projection shape")
    geometry = unit["geometry"]
    expected_geometry = {"L": 16, "K": expected_k, "V": 2, "tlut_bits": 9}
    if {key: int(geometry.get(key, -1)) for key in expected_geometry} != expected_geometry:
        raise ValueError(f"{control} unit has wrong QTIP geometry")
    weights = int(cell["accounting"]["weights"])
    if weights != 4096 * 2048:
        raise ValueError(f"{control} accounting has wrong weight count")
    trellis = unit["trellis"]
    if trellis.numel() * trellis.element_size() * 8 != weights * expected_k:
        raise ValueError(f"{control} trellis bytes do not close nominal code bits")
    if tuple(unit["SU"].shape) != (2048,) or tuple(unit["SV"].shape) != (4096,):
        raise ValueError(f"{control} transform vectors have wrong geometry")
    if unit["Wscale"].numel() != 1 or tuple(unit["tlut"].shape) != (512, 2):
        raise ValueError(f"{control} scale/TLUT geometry mismatch")
    packed = np.load(cell["periodic_codes"]["path"], mmap_mode="r", allow_pickle=False)
    if packed.dtype != np.uint8 or tuple(packed.shape) != (32768, 80):
        raise ValueError(f"{control} periodic packed payload has wrong layout")
    if packed.nbytes * 8 != int(cell["accounting"]["code_bits"]):
        raise ValueError(f"{control} periodic packed bytes do not close accounting")


def _validate_import_source(module: Any, descriptor: Mapping[str, Any], name: str) -> None:
    loaded = Path(str(module.__file__)).resolve()
    declared = Path(str(descriptor["path"])).resolve()
    if loaded != declared or sha256_file(loaded) != descriptor["sha256"]:
        raise RuntimeError(f"executed {name} source differs from manifest binding")


def validate_manifest(manifest: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    """Validate the current-FF0731 two-cell manifest before CUDA is imported."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected train8 manifest schema: {manifest.get('schema')!r}")
    if manifest.get("task_id") != "t_7002ac79":
        raise ValueError("train8 manifest belongs to the wrong task")
    if manifest.get("basis_sha256") != FF0731_MODEL_INDEX_SHA256:
        raise ValueError("train8 manifest is not bound to current FF0731")
    if tuple(str(value) for value in manifest.get("row_ids", ())) != TRAIN8_ROW_IDS:
        raise ValueError("train8 manifest has the wrong frozen row IDs")
    if manifest.get("attention_implementation") != "eager":
        raise ValueError("two-cell decision runner requires eager attention")
    if int(manifest.get("position_cutoff", -1)) != TRAIN8_POSITION_CUTOFF:
        raise ValueError("train8 manifest has the wrong position cutoff")
    if int(manifest.get("support_width", -1)) != TRAIN8_SUPPORT_WIDTH:
        raise ValueError("train8 manifest has the wrong support width")

    claim = manifest.get("claim")
    shard_claim = manifest.get("shards")
    if not isinstance(claim, Mapping) or not isinstance(shard_claim, Mapping):
        raise ValueError("train8 manifest must bind HOST_CLAIM and SHARDS")
    claim_path = _hashed_path(claim, "claim", verify=verify_files)
    shards_path = _hashed_path(shard_claim, "shards", verify=verify_files)
    if verify_files:
        claim_value = json.loads(claim_path.read_text())
        if claim_value.get("task_id") != "t_7002ac79" or claim_value.get("state") != "CLAIMED":
            raise ValueError("HOST_CLAIM owner/state mismatch")
        shards_value = json.loads(shards_path.read_text())
        if shards_value.get("intended_basis") != FF0731_MODEL_INDEX_SHA256:
            raise ValueError("SHARDS intended basis mismatch")

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("train8 manifest is missing model identity")
    config_path = _hashed_path(model["config"], "model.config", verify=verify_files)
    index_path = _hashed_path(model["index"], "model.index", verify=verify_files)
    if model["index"]["sha256"] != FF0731_MODEL_INDEX_SHA256:
        raise ValueError("model index hash differs from current FF0731")
    if verify_files:
        config = json.loads(config_path.read_text())
        expected_topology = {
            "num_hidden_layers": 43,
            "hidden_size": 4096,
            "vocab_size": 129280,
            "n_routed_experts": 256,
        }
        if {key: config.get(key) for key in expected_topology} != expected_topology:
            raise ValueError("current FF0731 topology mismatch")
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        required_keys = {"embed.weight", "head.weight", "norm.weight"}
        if not required_keys.issubset(weight_map):
            raise ValueError("current FF0731 index is missing static model tensors")
        layer_ids = {
            int(key.split(".")[1])
            for key in weight_map
            if key.startswith("layers.")
        }
        if layer_ids != set(range(43)):
            raise ValueError("current FF0731 index does not close all 43 layers")
    source_root_text = str(model.get("source_root", ""))
    source_root = Path(source_root_text)
    if not source_root_text or not source_root.is_absolute():
        raise ValueError("model.source_root must be a nonempty absolute path")
    model_shards = model.get("shards")
    if not isinstance(model_shards, Mapping):
        raise ValueError("model.shards must bind every referenced FF0731 shard")
    referenced_shards = set(weight_map.values()) if verify_files else set(model_shards)
    if set(model_shards) != referenced_shards:
        raise ValueError("model.shards does not exactly close the index weight map")
    for shard, descriptor in model_shards.items():
        if Path(str(shard)).name != shard or not str(shard).endswith(".safetensors"):
            raise ValueError(f"invalid model shard identity: {shard!r}")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"missing model shard descriptor: {shard}")
        _require_sha(f"model.shards[{shard}].sha256", descriptor.get("sha256"))
        if not isinstance(descriptor.get("bytes"), int) or isinstance(descriptor.get("bytes"), bool) or int(descriptor["bytes"]) <= 0:
            raise ValueError(f"model.shards[{shard}].bytes must be a positive integer")

    corpus = manifest.get("corpus")
    bank = manifest.get("bank")
    teacher = manifest.get("teacher")
    if not all(isinstance(value, Mapping) for value in (corpus, bank, teacher)):
        raise ValueError("train8 manifest must bind corpus, bank, and teacher")
    corpus_path = _hashed_path(corpus, "corpus", verify=verify_files)
    bank_path = _hashed_path(bank, "bank", verify=verify_files)
    teacher_manifest_path = _hashed_path(teacher["manifest"], "teacher.manifest", verify=verify_files)
    if bank["sha256"] != TRAIN64_BANK_MANIFEST_SHA256:
        raise ValueError("train8 manifest bank is not the frozen train64 bank")
    if teacher["manifest"]["sha256"] != TEACHER_TOP8192_MANIFEST_SHA256:
        raise ValueError("train8 manifest teacher identity mismatch")
    teacher_rows = teacher.get("rows")
    if not isinstance(teacher_rows, Sequence) or tuple(
        str(row.get("row_id")) for row in teacher_rows
    ) != TRAIN8_ROW_IDS:
        raise ValueError("train8 manifest teacher rows do not match the frozen order")
    for row in teacher_rows:
        _hashed_path(row, f"teacher.rows[{row['row_id']}]", verify=verify_files)
    corpus_rows = corpus.get("rows")
    if not isinstance(corpus_rows, Sequence) or tuple(str(row.get("row_id")) for row in corpus_rows) != TRAIN8_ROW_IDS:
        raise ValueError("corpus row bindings do not match the frozen train8 order")
    for row in corpus_rows:
        _require_sha(f"corpus.rows[{row.get('row_id')}].sha256", row.get("sha256"))
    if verify_files:
        corpus_value = json.loads(corpus_path.read_text())
        bank_value = json.loads(bank_path.read_text())
        teacher_manifest_value = json.loads(teacher_manifest_path.read_text())
        bank_rows = tuple(str(row.get("id")) for row in bank_value.get("windows", ())[: len(TRAIN8_ROW_IDS)])
        if bank_rows != TRAIN8_ROW_IDS:
            raise ValueError("frozen bank does not begin with the train8 cohort")
        teacher_members = {
            str(row.get("window_id")): row for row in teacher_manifest_value.get("members", ())
        }
        for row_binding, teacher_row in zip(corpus_rows, teacher_rows, strict=True):
            row_id = str(row_binding["row_id"])
            corpus_row = corpus_value[int(row_id)]
            row_payload = {
                "token_ids": [int(value) for value in corpus_row["token_ids"]],
                "real_len": int(corpus_row["real_len"]),
            }
            observed_row_sha = hashlib.sha256(_canonical_json(row_payload)).hexdigest()
            if observed_row_sha != row_binding["sha256"]:
                raise ValueError(f"corpus row {row_id} payload hash mismatch")
            member = teacher_members.get(row_id)
            if member is None or member.get("sha256") != teacher_row["sha256"] or int(member.get("bytes", -1)) != int(teacher_row["bytes"]):
                raise ValueError(f"teacher row {row_id} is not bound by the teacher terminal")

    cells = manifest.get("cells")
    if not isinstance(cells, Sequence) or len(cells) != 2:
        raise ValueError("train8 manifest must contain exactly two cells")
    identities = []
    for cell in cells:
        identity = cell.get("identity", {})
        identities.append(
            (int(identity.get("layer", -1)), int(identity.get("expert", -1)), str(identity.get("projection", "")))
        )
        _hashed_path(cell["control_unit"], f"cell[{cell.get('control')}].control_unit", verify=verify_files)
        _hashed_path(cell["periodic_codes"], f"cell[{cell.get('control')}].periodic_codes", verify=verify_files)
        _require_sha("cell.source_weight_sha256", cell["source_weight_sha256"])
    if identities != [(0, 0, "down"), (0, 1, "down")]:
        raise ValueError("train8 manifest cells must be L000 E000/E001 down in order")
    if [str(cell.get("control")) for cell in cells] != ["qtip_k2", "qtip_k3"]:
        raise ValueError("train8 cells must bind E000 to K2 and E001 to K3")
    if verify_files:
        for cell in cells:
            _validate_cell_payload(cell)
    errors, bits = matched_measurements(cells)

    runtime = manifest.get("runtime")
    output = manifest.get("output")
    if not isinstance(runtime, Mapping) or not isinstance(output, Mapping):
        raise ValueError("train8 manifest is missing runtime/output configuration")
    for key in ("banana_smasher_source", "public_site", "qtip_root", "shard_stage_dir"):
        path = Path(str(runtime[key]))
        if verify_files and not path.exists():
            raise FileNotFoundError(f"runtime path does not exist: {key}={path}")
    executed_sources = runtime.get("executed_sources")
    if not isinstance(executed_sources, Mapping) or set(executed_sources) != {
        "runner", "periodic_signal", "qtip_runner", "periodic_plugin"
    }:
        raise ValueError("runtime.executed_sources must bind the complete executed source set")
    for name, descriptor in executed_sources.items():
        _hashed_path(descriptor, f"runtime.executed_sources[{name}]", verify=verify_files)
    if verify_files and executed_sources["runner"]["sha256"] != sha256_file(__file__):
        raise ValueError("manifest runner source hash differs from executed runner")
    if int(runtime.get("microbatch", 0)) < 1:
        raise ValueError("microbatch must be positive")
    if int(runtime.get("readout_chunk_positions", 0)) < 1:
        raise ValueError("readout_chunk_positions must be positive")
    if not str(output.get("root", "")):
        raise ValueError("output root is required")

    return {
        "claim_path": claim_path,
        "shards_path": shards_path,
        "config_path": config_path,
        "index_path": index_path,
        "model_shards": dict(model_shards),
        "direct_error": errors,
        "nominal_code_bits": bits,
    }


def load_manifest(path: str | Path, *, verify_files: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.pop("content_sha256", None)
    observed = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if expected != observed:
        raise ValueError(f"manifest content hash mismatch: {observed} != {expected}")
    manifest["content_sha256"] = expected
    return manifest, validate_manifest(manifest, verify_files=verify_files)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    value = dict(manifest)
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(_canonical_json(value)).hexdigest()
    return atomic_json(path, value, overwrite=False)


def _startticks(pid: int) -> int:
    return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


def _require_authority(manifest: Mapping[str, Any]) -> None:
    for key in ("claim", "shards"):
        expected = str(manifest[key]["sha256"])
        observed = sha256_file(manifest[key]["path"])
        if observed != expected:
            raise RuntimeError(f"{key} drift: {observed} != {expected}")
    claim = json.loads(Path(manifest["claim"]["path"]).read_text())
    if claim.get("task_id") != "t_7002ac79" or claim.get("state") != "CLAIMED":
        raise RuntimeError("HOST_CLAIM owner/state mismatch")


def run_stage_server(manifest_path: str | Path) -> int:
    """Copy one requested FF0731 shard at a time into the task-owned bind root."""
    manifest, _ = load_manifest(manifest_path, verify_files=True)
    source_root = Path(str(manifest["model"]["source_root"]))
    stage = Path(str(manifest["runtime"]["shard_stage_dir"]))
    request_path = stage / "REQUEST.json"
    ready_path = stage / "READY.json"
    release_path = stage / "RELEASE.json"
    terminal_path = stage / "STAGER_TERMINAL.json"
    stage.mkdir(parents=True, exist_ok=True)
    served: list[dict[str, Any]] = []
    while True:
        _require_authority(manifest)
        if terminal_path.exists():
            return 0
        if not request_path.exists():
            time.sleep(0.2)
            continue
        request = json.loads(request_path.read_text())
        if request.get("basis_sha256") != FF0731_MODEL_INDEX_SHA256:
            raise RuntimeError("shard request basis mismatch")
        shard = str(request["shard"])
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise RuntimeError(f"invalid shard request: {shard!r}")
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            if ready.get("request_sha256") == sha256_file(request_path):
                time.sleep(0.2)
                continue
            raise RuntimeError("new shard request arrived before prior release")
        source = source_root / shard
        if not source.is_file():
            raise FileNotFoundError(f"missing authoritative FF0731 shard: {source}")
        destination = stage / shard
        descriptor, temporary_name = tempfile.mkstemp(
            dir=stage, prefix=f".{shard}.", suffix=".copy"
        )
        source_digest = hashlib.sha256()
        try:
            with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
                for block in iter(lambda: src.read(16 << 20), b""):
                    source_digest.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(temporary_name, destination)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        destination_sha = sha256_file(destination)
        expected_sha = str(manifest["model"]["shards"][shard]["sha256"])
        expected_bytes = int(manifest["model"]["shards"][shard]["bytes"])
        if destination_sha != source_digest.hexdigest() or destination_sha != expected_sha or destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"staged shard copy mismatch: {shard}")
        ready = {
            "schema": "banana-smasher-qtip25-periodic-shard-ready-v1",
            "status": "READY",
            "basis_sha256": FF0731_MODEL_INDEX_SHA256,
            "task_id": "t_7002ac79",
            "shard": shard,
            "bytes": destination.stat().st_size,
            "sha256": destination_sha,
            "request_sha256": sha256_file(request_path),
            "source_path": str(source),
            "destination_path": str(destination),
            "ready_unix": time.time(),
        }
        atomic_json(ready_path, ready)
        while True:
            _require_authority(manifest)
            if release_path.exists():
                release = json.loads(release_path.read_text())
                if release.get("ready_sha256") == sha256_file(ready_path):
                    break
            time.sleep(0.2)
        served.append(ready)
        destination.unlink()
        request_path.unlink()
        ready_path.unlink()
        release_path.unlink()
        directory = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _request_shard(manifest: Mapping[str, Any], shard: str) -> tuple[Path, str]:
    stage = Path(str(manifest["runtime"]["shard_stage_dir"]))
    request_path = stage / "REQUEST.json"
    ready_path = stage / "READY.json"
    if any(path.exists() for path in (request_path, ready_path, stage / "RELEASE.json")):
        raise RuntimeError("shard exchange directory is not at a clean boundary")
    request = {
        "schema": "banana-smasher-qtip25-periodic-shard-request-v1",
        "task_id": "t_7002ac79",
        "basis_sha256": FF0731_MODEL_INDEX_SHA256,
        "shard": shard,
        "requested_unix": time.time(),
    }
    request_sha = atomic_json(request_path, request)
    while not ready_path.exists():
        _require_authority(manifest)
        time.sleep(0.2)
    ready = json.loads(ready_path.read_text())
    if (
        ready.get("status") != "READY"
        or ready.get("shard") != shard
        or ready.get("request_sha256") != request_sha
        or ready.get("basis_sha256") != FF0731_MODEL_INDEX_SHA256
        or ready.get("sha256") != manifest["model"]["shards"][shard]["sha256"]
        or int(ready.get("bytes", -1)) != int(manifest["model"]["shards"][shard]["bytes"])
    ):
        raise RuntimeError(f"invalid staged shard receipt: {ready}")
    destination = Path(str(ready["destination_path"]))
    if sha256_file(destination) != ready["sha256"]:
        raise RuntimeError(f"staged shard hash drift: {shard}")
    return destination, sha256_file(ready_path)


def _release_shard(manifest: Mapping[str, Any], ready_sha: str) -> None:
    stage = Path(str(manifest["runtime"]["shard_stage_dir"]))
    atomic_json(
        stage / "RELEASE.json",
        {"schema": "banana-smasher-qtip25-periodic-shard-release-v1", "ready_sha256": ready_sha},
    )
    while any((stage / name).exists() for name in ("REQUEST.json", "READY.json", "RELEASE.json")):
        time.sleep(0.1)


def _fwht(torch: Any, value: Any) -> Any:
    count = value.shape[-1]
    if count <= 0 or count & (count - 1):
        raise ValueError("FWHT axis must be a positive power of two")
    result = value.contiguous()
    width = 1
    while width < count:
        grouped = result.reshape(*result.shape[:-1], count // (2 * width), 2, width)
        left, right = grouped[..., 0, :], grouped[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            *result.shape[:-1], count
        )
        width *= 2
    return result / math.sqrt(count)


def _expanded_lut(torch: Any, tlut: Any, device: Any) -> Any:
    table = tlut.float().to(device)
    state = torch.arange(1 << 16, device=device, dtype=torch.int64)
    quadratic = (state + 1) * state
    sign = 1 - ((quadratic >> 15) & 1) * 2
    index = (quadratic >> 6) & 511
    result = table[index].clone()
    result[:, 0] *= sign
    return result


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _deq_fp8(torch: Any, weight: Any, scale: Any, device: Any) -> Any:
    weight = weight.to(device)
    scale = torch.exp2(scale.to(device).view(torch.uint8).float() - 127.0)
    rows, columns = weight.shape
    scale = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :columns]
    return (weight.float() * scale).to(torch.bfloat16)


def _deq_fp4(torch: Any, packed: Any, scales: Any, device: Any) -> Any:
    packed = packed.to(device).view(torch.uint8)
    scale = torch.exp2(scales.to(device).view(torch.uint8).float() - 127.0)
    levels = torch.as_tensor(E2M1_VALUES, device=device)
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return (levels[values.long()] * scale.repeat_interleave(32, -1)).to(torch.bfloat16)


def _build_layer_state(torch: Any, layer: int, weight_map: Mapping[str, str], handle: Any, device: Any) -> dict[str, Any]:
    prefix = f"layers.{layer}."
    keys = [key for key in weight_map if key.startswith(prefix)]
    state: dict[str, Any] = {}
    consumed: set[str] = set()

    def tensor(name: str) -> Any:
        key = prefix + name
        consumed.add(key)
        return handle.get_tensor(key)

    def has(name: str) -> bool:
        return prefix + name in weight_map

    def fp8(name: str) -> Any:
        return _deq_fp8(torch, tensor(name + ".weight"), tensor(name + ".scale"), device)

    def bf(name: str) -> Any:
        return tensor(name).to(device).to(torch.bfloat16)

    def f32(name: str) -> Any:
        return tensor(name).to(device).float()

    state["self_attn.q_a_proj.weight"] = fp8("attn.wq_a")
    state["self_attn.q_b_proj.weight"] = fp8("attn.wq_b")
    state["self_attn.kv_proj.weight"] = fp8("attn.wkv")
    state["self_attn.o_a_proj.weight"] = fp8("attn.wo_a")
    state["self_attn.o_b_proj.weight"] = fp8("attn.wo_b")
    state["self_attn.sinks"] = f32("attn.attn_sink")
    state["self_attn.q_a_norm.weight"] = bf("attn.q_norm.weight")
    state["self_attn.kv_norm.weight"] = bf("attn.kv_norm.weight")
    state["input_layernorm.weight"] = bf("attn_norm.weight")
    state["post_attention_layernorm.weight"] = bf("ffn_norm.weight")
    if has("attn.compressor.wkv.weight"):
        state["self_attn.compressor.position_bias"] = f32("attn.compressor.ape")
        state["self_attn.compressor.kv_norm.weight"] = bf("attn.compressor.norm.weight")
        state["self_attn.compressor.kv_proj.weight"] = bf("attn.compressor.wkv.weight")
        state["self_attn.compressor.gate_proj.weight"] = bf("attn.compressor.wgate.weight")
    if has("attn.indexer.wq_b.weight"):
        indexer = "self_attn.compressor.indexer."
        state[indexer + "position_bias"] = f32("attn.indexer.compressor.ape")
        state[indexer + "kv_norm.weight"] = bf("attn.indexer.compressor.norm.weight")
        state[indexer + "kv_proj.weight"] = bf("attn.indexer.compressor.wkv.weight")
        state[indexer + "gate_proj.weight"] = bf("attn.indexer.compressor.wgate.weight")
        state[indexer + "q_b_proj.weight"] = fp8("attn.indexer.wq_b")
        state[indexer + "scorer.weights_proj.weight"] = bf("attn.indexer.weights_proj.weight")
    state["mlp.gate.weight"] = bf("ffn.gate.weight")
    if has("ffn.gate.tid2eid"):
        state["mlp.gate.tid2eid"] = tensor("ffn.gate.tid2eid").to(device)
    if has("ffn.gate.bias"):
        state["mlp.gate.e_score_correction_bias"] = f32("ffn.gate.bias")
    state["attn_hc.fn"] = f32("hc_attn_fn")
    state["attn_hc.base"] = f32("hc_attn_base")
    state["attn_hc.scale"] = f32("hc_attn_scale")
    state["ffn_hc.fn"] = f32("hc_ffn_fn")
    state["ffn_hc.base"] = f32("hc_ffn_base")
    state["ffn_hc.scale"] = f32("hc_ffn_scale")
    state["mlp.shared_experts.gate_proj.weight"] = fp8("ffn.shared_experts.w1")
    state["mlp.shared_experts.up_proj.weight"] = fp8("ffn.shared_experts.w3")
    state["mlp.shared_experts.down_proj.weight"] = fp8("ffn.shared_experts.w2")

    gate_up = torch.empty((256, 4096, 4096), dtype=torch.bfloat16, device=device)
    down = torch.empty((256, 4096, 2048), dtype=torch.bfloat16, device=device)
    for first in range(0, 256, 8):
        experts = range(first, min(first + 8, 256))
        for weight_name, destination, rows in (
            ("w1", gate_up, slice(0, 2048)),
            ("w3", gate_up, slice(2048, 4096)),
            ("w2", down, slice(0, 4096)),
        ):
            packed = torch.stack(
                [tensor(f"ffn.experts.{expert}.{weight_name}.weight") for expert in experts]
            )
            scales = torch.stack(
                [tensor(f"ffn.experts.{expert}.{weight_name}.scale") for expert in experts]
            )
            destination[first:first + len(experts), rows] = _deq_fp4(
                torch, packed, scales, device
            )
            del packed, scales
    state["mlp.experts.gate_up_proj"] = gate_up
    state["mlp.experts.down_proj"] = down
    missed = set(keys) - consumed
    if missed:
        raise RuntimeError(f"layer {layer} has unconsumed checkpoint keys: {sorted(missed)[:8]}")
    return state


def _materialize_layer(torch: Any, model: Any, layer: int, state: Mapping[str, Any], config: Any, device: Any) -> Any:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    module = model.model.layers[layer]
    missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"layer {layer} unexpected state keys: {unexpected[:8]}")
    for name, child in list(module.named_modules()):
        if isinstance(child, DeepseekV4RotaryEmbedding):
            parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            setattr(parent, name.rsplit(".", 1)[-1], DeepseekV4RotaryEmbedding(config).to(device))
    unresolved = [name for name, value in module.named_parameters() if value.is_meta]
    unresolved += [name for name, value in module.named_buffers() if value.is_meta]
    if unresolved:
        raise RuntimeError(f"layer {layer} still has meta values: {unresolved[:8]}")
    return module


def _dematerialize_layer(torch: Any, model: Any, layer: int) -> None:
    module = model.model.layers[layer]
    for child in module.modules():
        for name, parameter in list(child._parameters.items()):
            if parameter is not None:
                child._parameters[name] = torch.nn.Parameter(
                    torch.empty(parameter.shape, device="meta", dtype=parameter.dtype),
                    requires_grad=False,
                )
        for name, buffer in list(child._buffers.items()):
            if buffer is not None:
                child._buffers[name] = torch.empty(buffer.shape, device="meta", dtype=buffer.dtype)
    torch.cuda.empty_cache()


def _decode_two_cell_overlays(torch: Any, manifest: Mapping[str, Any], device: Any) -> dict[str, list[Any]]:
    sys.path.insert(0, str(manifest["runtime"]["banana_smasher_source"]))
    sys.path.insert(1, str(manifest["runtime"]["public_site"]))
    from banana_smasher import qtip_periodic_signal, qtip_runner
    from banana_smasher_plugin import periodic_qtip

    sources = manifest["runtime"]["executed_sources"]
    _validate_import_source(qtip_periodic_signal, sources["periodic_signal"], "periodic_signal")
    _validate_import_source(qtip_runner, sources["qtip_runner"], "qtip_runner")
    _validate_import_source(periodic_qtip, sources["periodic_plugin"], "periodic_plugin")
    dequantize_periodic_blocks = periodic_qtip.dequantize_periodic_blocks

    qtip_runner.QTIP = Path(str(manifest["runtime"]["qtip_root"]))
    _, _, _, kernel_decode = qtip_runner.load_official_qtip()
    controls: list[Any] = []
    periodic: list[Any] = []
    for cell in manifest["cells"]:
        unit = torch.load(cell["control_unit"]["path"], map_location="cpu", mmap=True, weights_only=True)
        geometry = unit["geometry"]
        rows, columns = [int(value) for value in unit["shape"]]
        lut = _expanded_lut(torch, unit["tlut"], device)
        raw = kernel_decode.decode_compressed(
            int(geometry["L"]), int(geometry["tlut_bits"]), int(geometry["K"]),
            int(geometry["V"]) - 1, rows, columns,
            unit["trellis"].to(device).reshape(-1), lut,
        )
        control = raw * unit["Wscale"].to(device)
        control = _fwht(torch, control.T).T * unit["SV"].float().to(device)[:, None]
        control = _fwht(torch, control) * unit["SU"].float().to(device)
        controls.append(control.to(torch.bfloat16))

        packed = np.load(cell["periodic_codes"]["path"], mmap_mode="r", allow_pickle=False)
        chunks = []
        for start in range(0, len(packed), 512):
            source = torch.from_numpy(np.asarray(packed[start:start + 512]).copy()).to(device)
            chunks.append(dequantize_periodic_blocks(source, lut).cpu())
        decoded = torch.cat(chunks).reshape(rows // 16, columns // 16, 16, 16)
        decoded = decoded.transpose(1, 2).reshape(rows, columns).to(device)
        decoded = decoded * unit["Wscale"].to(device)
        decoded = _fwht(torch, decoded.T).T * unit["SV"].float().to(device)[:, None]
        decoded = _fwht(torch, decoded) * unit["SU"].float().to(device)
        periodic.append(decoded.to(torch.bfloat16))
    return {"controls": controls, "periodic": periodic}


def _atomic_torch_save(torch: Any, path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as stream:
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
    return sha256_file(path)


def run_forward(manifest_path: str | Path) -> int:
    manifest, derived = load_manifest(manifest_path, verify_files=True)
    if sha256_file(__file__) != manifest["runtime"]["executed_sources"]["runner"]["sha256"]:
        raise RuntimeError("executed runner source drifted after preflight")
    _require_authority(manifest)
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exact train8 forward")
    free, _ = torch.cuda.mem_get_info()
    if free < int(manifest["runtime"].get("minimum_free_cuda_bytes", 8 << 30)):
        raise RuntimeError(f"insufficient CUDA memory before train8 forward: {free}")
    device = torch.device("cuda")
    output_root = Path(str(manifest["output"]["root"]))
    progress_path = output_root / "PROGRESS.json"
    output_root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    progress = {
        "schema": PROGRESS_SCHEMA,
        "status": "RUNNING",
        "task_id": "t_7002ac79",
        "pid": pid,
        "startticks": _startticks(pid),
        "basis_sha256": FF0731_MODEL_INDEX_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "started_unix": time.time(),
        "phase": "preflight_complete",
        "layer": None,
        "arm": None,
        "rows_complete": 0,
    }
    atomic_json(progress_path, progress)

    config = AutoConfig.from_pretrained(derived["config_path"].parent)
    weight_map = json.loads(derived["index_path"].read_text())["weight_map"]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    embed_path, embed_ready = _request_shard(manifest, weight_map["embed.weight"])
    with safe_open(embed_path, framework="pt", device="cpu") as handle:
        model.model.embed_tokens.weight = torch.nn.Parameter(
            handle.get_tensor("embed.weight").to(device).to(torch.bfloat16),
            requires_grad=False,
        )
    _release_shard(manifest, embed_ready)
    model.model.rotary_emb = DeepseekV4RotaryEmbedding(config).to(device)

    corpus = json.loads(Path(manifest["corpus"]["path"]).read_text())
    row_ids = [int(value) for value in TRAIN8_ROW_IDS]
    sequence_length = int(manifest["runtime"].get("sequence_length", 2048))
    ids = torch.full((len(row_ids), sequence_length), 1, dtype=torch.long)
    real_lengths = []
    for index, row_id in enumerate(row_ids):
        row = corpus[row_id]
        tokens = row["token_ids"]
        if len(tokens) > sequence_length:
            raise ValueError(f"row {row_id} exceeds sequence length {sequence_length}")
        ids[index, :len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
        real_lengths.append(int(row["real_len"]))
    ids = ids.to(device)
    positions = torch.arange(sequence_length, device=device).unsqueeze(0)
    microbatch = int(manifest["runtime"]["microbatch"])
    slices = [slice(start, min(start + microbatch, len(row_ids))) for start in range(0, len(row_ids), microbatch)]
    with torch.no_grad():
        embeds = model.model.embed_tokens(ids)
        position_embeddings = {
            "main": model.model.rotary_emb(embeds[:1], position_ids=positions, layer_type="main"),
            "compress": model.model.rotary_emb(embeds[:1], position_ids=positions, layer_type="compress"),
        }
        masks = [
            create_sliding_window_causal_mask(
                config=config, inputs_embeds=embeds[batch], attention_mask=None,
                past_key_values=DynamicCache(config=config), position_ids=positions,
            )
            for batch in slices
        ]
        hidden = {
            arm: [
                embeds[batch].unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
                for batch in slices
            ]
            for arm in PERIODIC_SIGNAL_CANDIDATES
        }
        caches = {
            arm: [DynamicCache(config=config) for _ in slices]
            for arm in PERIODIC_SIGNAL_CANDIDATES
        }
        del embeds
        overlays = _decode_two_cell_overlays(torch, manifest, device)

        for layer in range(config.num_hidden_layers):
            _require_authority(manifest)
            shard = next(iter({value for key, value in weight_map.items() if key.startswith(f"layers.{layer}.")}))
            shard_path, ready_sha = _request_shard(manifest, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                state = _build_layer_state(torch, layer, weight_map, handle, device)
            _release_shard(manifest, ready_sha)
            module = _materialize_layer(torch, model, layer, state, config, device)
            del state
            down = module.mlp.experts.down_proj
            if layer == 0:
                source_cells = [down[0].clone(), down[1].clone()]
                for index, cell in enumerate(manifest["cells"]):
                    source_f32 = source_cells[index].float()
                    if _tensor_sha256(source_f32) != cell["source_weight_sha256"]:
                        raise RuntimeError(f"L000 source component mismatch for cell {index}")
            for arm in PERIODIC_SIGNAL_CANDIDATES:
                if layer == 0:
                    selected = []
                    for index, source_name in enumerate(ARM_CELL_SOURCES[arm]):
                        if source_name == "source":
                            selected.append(source_cells[index])
                        elif source_name == "periodic":
                            selected.append(overlays["periodic"][index])
                        else:
                            selected.append(overlays["controls"][index])
                    down[0].copy_(selected[0])
                    down[1].copy_(selected[1])
                for batch_index, batch in enumerate(slices):
                    hidden[arm][batch_index] = module(
                        hidden[arm][batch_index],
                        position_embeddings=position_embeddings,
                        position_ids=positions,
                        attention_mask=masks[batch_index],
                        input_ids=ids[batch],
                        past_key_values=caches[arm][batch_index],
                    )
                progress.update({"phase": "forward", "layer": layer, "arm": arm, "updated_unix": time.time()})
                atomic_json(progress_path, progress)
            _dematerialize_layer(torch, model, layer)

        readout_path, readout_ready = _request_shard(manifest, weight_map["head.weight"])
        with safe_open(readout_path, framework="pt", device="cpu") as handle:
            model.lm_head.weight = torch.nn.Parameter(
                handle.get_tensor("head.weight").to(device).to(torch.bfloat16), requires_grad=False
            )
            model.model.norm.weight = torch.nn.Parameter(
                handle.get_tensor("norm.weight").to(device).to(torch.bfloat16), requires_grad=False
            )
            model.model.hc_head.hc_fn = torch.nn.Parameter(
                handle.get_tensor("hc_head_fn").to(device).float(), requires_grad=False
            )
            model.model.hc_head.hc_base = torch.nn.Parameter(
                handle.get_tensor("hc_head_base").to(device).float(), requires_grad=False
            )
            model.model.hc_head.hc_scale = torch.nn.Parameter(
                handle.get_tensor("hc_head_scale").to(device).float(), requires_grad=False
            )
        _release_shard(manifest, readout_ready)

        row_members = {str(row["row_id"]): row for row in manifest["teacher"]["rows"]}
        artifacts: dict[str, list[dict[str, Any]]] = {arm: [] for arm in PERIODIC_SIGNAL_CANDIDATES}
        readout_chunk = int(manifest["runtime"]["readout_chunk_positions"])
        for arm in PERIODIC_SIGNAL_CANDIDATES:
            for batch_index, batch in enumerate(slices):
                final = model.model.norm(model.model.hc_head(hidden[arm][batch_index]))
                for local_index in range(final.shape[0]):
                    row_index = batch.start + local_index
                    row_id = str(row_ids[row_index])
                    teacher = torch.load(row_members[row_id]["path"], map_location="cpu", weights_only=True)
                    support_ids = teacher["idx"][:TRAIN8_POSITION_CUTOFF].to(torch.long)
                    if tuple(support_ids.shape) != (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH):
                        raise RuntimeError(f"teacher support shape mismatch for row {row_id}")
                    candidate_logits = torch.empty(
                        (TRAIN8_POSITION_CUTOFF, TRAIN8_SUPPORT_WIDTH), dtype=torch.float16
                    )
                    candidate_argmax = torch.empty(TRAIN8_POSITION_CUTOFF, dtype=torch.int32)
                    source_hidden = final[local_index, :TRAIN8_POSITION_CUTOFF].to(torch.bfloat16)
                    for start in range(0, TRAIN8_POSITION_CUTOFF, readout_chunk):
                        stop = min(start + readout_chunk, TRAIN8_POSITION_CUTOFF)
                        full_logits = model.lm_head(source_hidden[start:stop]).float()
                        support = support_ids[start:stop].to(device)
                        candidate_logits[start:stop] = full_logits.gather(1, support).to(torch.float16).cpu()
                        candidate_argmax[start:stop] = full_logits.argmax(-1).to(torch.int32).cpu()
                        del full_logits, support
                    destination = output_root / arm / f"q8192_win{row_id}.pt"
                    file_sha = _atomic_torch_save(
                        torch,
                        destination,
                        {"candidate_logits": candidate_logits, "candidate_argmax": candidate_argmax},
                    )
                    artifacts[arm].append(
                        {"row_id": row_id, "path": str(destination), "bytes": destination.stat().st_size, "sha256": file_sha}
                    )
                    progress.update({
                        "phase": "readout", "arm": arm, "layer": 42,
                        "rows_complete": sum(len(rows) for rows in artifacts.values()),
                        "updated_unix": time.time(),
                    })
                    atomic_json(progress_path, progress)
                del final

    candidate_artifacts: dict[str, str] = {}
    for arm, rows in artifacts.items():
        artifact_manifest = {
            "schema": "banana-smasher-qtip25-periodic-train8-arm-v1",
            "basis_sha256": FF0731_MODEL_INDEX_SHA256,
            "arm": arm,
            "cell_sources": list(ARM_CELL_SOURCES[arm]),
            "rows": rows,
        }
        artifact_path = output_root / arm / "ARTIFACT.json"
        candidate_artifacts[arm] = atomic_json(artifact_path, artifact_manifest, overwrite=False)

    from banana_smasher.qtip_periodic_signal import (
        _measurement_values_sha256,
        _update_array_hash,
        score_periodic_train8_signal,
        write_periodic_train8_signal_receipt,
    )

    payload_hashes = {
        "teacher_support": hashlib.sha256(),
        **{arm: hashlib.sha256() for arm in PERIODIC_SIGNAL_CANDIDATES},
    }
    score_rows = []
    for row_id in TRAIN8_ROW_IDS:
        teacher = torch.load(row_members[row_id]["path"], map_location="cpu", weights_only=True)
        support_ids = teacher["idx"][:TRAIN8_POSITION_CUTOFF].numpy()
        teacher_logits = teacher["logprob"][:TRAIN8_POSITION_CUTOFF].numpy()
        candidate_logits = {}
        candidate_argmax = {}
        _update_array_hash(payload_hashes["teacher_support"], "support_ids", row_id, support_ids)
        _update_array_hash(payload_hashes["teacher_support"], "teacher_logits", row_id, teacher_logits)
        for arm in PERIODIC_SIGNAL_CANDIDATES:
            value = torch.load(output_root / arm / f"q8192_win{row_id}.pt", map_location="cpu", weights_only=True)
            candidate_logits[arm] = value["candidate_logits"].numpy()
            candidate_argmax[arm] = value["candidate_argmax"].numpy()
            _update_array_hash(payload_hashes[arm], "candidate_logits", row_id, candidate_logits[arm])
            _update_array_hash(payload_hashes[arm], "candidate_argmax", row_id, candidate_argmax[arm])
        score_rows.append({
            "row_id": row_id,
            "support_ids": support_ids,
            "teacher_logits": teacher_logits,
            "candidate_logits": candidate_logits,
            "candidate_argmax": candidate_argmax,
        })
    expected_payloads = {name: digest.hexdigest() for name, digest in payload_hashes.items()}
    provenance = {
        "avg_member_receipt_sha256": AVG_MEMBER_BASELINE_RECEIPT_SHA256,
        "bank_manifest_sha256": TRAIN64_BANK_MANIFEST_SHA256,
        "teacher_manifest_sha256": TEACHER_TOP8192_MANIFEST_SHA256,
        "candidate_artifact_sha256": candidate_artifacts,
        "expected_scored_payload_sha256": expected_payloads,
        "measurement_values_sha256": _measurement_values_sha256(
            candidate_artifacts=candidate_artifacts,
            direct_error=derived["direct_error"],
            nominal_code_bits=derived["nominal_code_bits"],
        ),
    }
    receipt = score_periodic_train8_signal(
        rows=score_rows,
        expected_row_ids=TRAIN8_ROW_IDS,
        intended_basis_sha256=FF0731_MODEL_INDEX_SHA256,
        observed_basis_sha256=manifest["basis_sha256"],
        direct_error=derived["direct_error"],
        nominal_code_bits=derived["nominal_code_bits"],
        provenance=provenance,
        chunk_positions=16,
    )
    receipt["runner"] = {
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": sha256_file(manifest_path),
        "arm_cell_sources": {key: list(value) for key, value in ARM_CELL_SOURCES.items()},
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
    }
    receipt_path = output_root / "TRAIN8_SIGNAL.json"
    receipt_sha = write_periodic_train8_signal_receipt(receipt_path, receipt)
    progress.update({
        "status": "PASS", "phase": "complete", "arm": None,
        "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha,
        "completed_unix": time.time(),
    })
    atomic_json(progress_path, progress)
    stage_terminal = Path(str(manifest["runtime"]["shard_stage_dir"])) / "STAGER_TERMINAL.json"
    atomic_json(stage_terminal, {"status": "PASS", "receipt_sha256": receipt_sha})
    print(json.dumps({"status": "PASS", "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Current-FF0731 QTIP2.5-PERIODIC two-cell train8 runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("preflight", "stage-server", "run"), default="preflight")
    arguments = parser.parse_args(argv)
    if arguments.mode == "preflight":
        manifest, derived = load_manifest(arguments.manifest, verify_files=True)
        print(json.dumps({
            "status": "PASS", "schema": manifest["schema"],
            "basis_sha256": manifest["basis_sha256"],
            "direct_error": derived["direct_error"],
            "nominal_code_bits": derived["nominal_code_bits"],
            "arm_cell_sources": {key: list(value) for key, value in ARM_CELL_SOURCES.items()},
        }, sort_keys=True))
        return 0
    if arguments.mode == "stage-server":
        return run_stage_server(arguments.manifest)
    return run_forward(arguments.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
