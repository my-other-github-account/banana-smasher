import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(os.environ["QTIP3_ROOT"])
from banana_smasher.qtip25_native_v4_api import build_qtip_native_cell, build_qtip_native_cells
from banana_smasher.qtip3_api_producer import (
    CellSpec,
    Qtip3ApiConfig,
    Qtip3ApiPlan,
    admit_host_and_shard,
    load_cell_roster,
    release_bounded_host,
    release_host,
    release_smoke_host,
    release_unstarted_admission,
    run_cells,
    run_cells_batched,
)

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
TASK_ID = os.environ["QTIP3_TASK_ID"]
BOARD_RUN_ID = int(os.environ["QTIP3_BOARD_RUN_ID"])
HOST = os.environ["QTIP3_HOST"]
TIER_CONFIG = Qtip3ApiConfig.for_tier(os.environ.get("QTIP3_TIER", "qtip3_v7"))
if "QTIP3_BPW" in os.environ and float(os.environ["QTIP3_BPW"]) != TIER_CONFIG.bpw:
    raise RuntimeError("QTIP3_TIER and QTIP3_BPW select inconsistent V7 geometry")
ALLOC = os.environ["QTIP3_ALLOCATION"]
DRIVER_SHA = os.environ["QTIP3_DRIVER_SHA"]
EXPECTED_CLAIM = os.environ["QTIP3_EXPECTED_CLAIM"]
CLAIM_PATH = Path(os.environ.get("QTIP3_CLAIM_PATH", "/home/dnola/HOST_CLAIM.json"))
INDEX = Path(
    os.environ.get(
        "QTIP3_MODEL_INDEX",
        "/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json",
    )
)
MODEL = INDEX.parent
LAYERS = tuple(int(value) for value in os.environ["QTIP3_LAYERS"].split(",") if value)
if not LAYERS or len(set(LAYERS)) != len(LAYERS) or any(layer < 0 or layer > 42 for layer in LAYERS):
    raise RuntimeError(f"invalid routed L000-L042 layer scope: {LAYERS}")
# The proven producer module has an immutable campaign-specific default.  This
# task owns a disjoint routed continuation scope, so bind its module-level
# validator to the exact card-assigned subset before constructing plans or CellSpecs.
from banana_smasher import qtip3_api_producer as producer_module
producer_module.LAYERS = LAYERS
producer_module.EXPECTED_CELLS = len(LAYERS) * 512
CELL_ROSTER_PATH = os.environ.get("QTIP3_CELL_ROSTER_PATH")
CELL_ROSTER_EXPECTED_COUNT = int(os.environ.get("QTIP3_CELL_ROSTER_EXPECTED_COUNT", "5992"))
if CELL_ROSTER_EXPECTED_COUNT <= 0:
    raise RuntimeError("QTIP3_CELL_ROSTER_EXPECTED_COUNT must be positive")
CELL_ROSTER = (
    load_cell_roster(
        CELL_ROSTER_PATH,
        intended_basis_sha256=BASIS,
        expected_count=CELL_ROSTER_EXPECTED_COUNT,
    )
    if CELL_ROSTER_PATH else ()
)
if CELL_ROSTER and not {row[0] for row in CELL_ROSTER}.issubset(set(LAYERS)):
    raise RuntimeError("cell roster contains a layer outside QTIP3_LAYERS")
PROJECTIONS = ("fused13", "down")
EXPERTS = tuple(range(256))
REC = ROOT / "receipts"
WORK = ROOT / "working_full_api"
SOURCE = WORK / "source.npy"
OUTPUT = Path(os.environ.get("QTIP3_OUTPUT_ROOT", str(ROOT / "outputs/full_api")))
SMOKE_COUNT = int(os.environ.get("QTIP3_SMOKE_COUNT", "0"))
MAX_NEW_BATCHES = int(os.environ.get("QTIP3_MAX_NEW_BATCHES", "0")) or None


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_probe():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return tuple(x.strip() for x in out.splitlines() if x.strip())


def plan(expected_claim):
    return Qtip3ApiPlan(
        task_id=TASK_ID,
        board_run_id=BOARD_RUN_ID,
        host=HOST,
        allocation=ALLOC,
        intended_basis_sha256=BASIS,
        driver_goals_path=Path(os.environ.get("QTIP3_AUTHORITY_PATH", str(ROOT / "CARD_AUTHORITY_t_1186bc8a.md"))),
        driver_goals_sha256=DRIVER_SHA,
        claim_path=CLAIM_PATH,
        shards_path=ROOT / "SHARDS.json",
        mission_root=ROOT,
        model_index_path=INDEX,
        tlut_path=ROOT / "inputs/qtip_tlut.npy",
        expected_claim_sha256=expected_claim,
        layers=LAYERS,
        cell_roster=CELL_ROSTER,
    )


LUT = np.asarray(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=np.float32,
)


def decode_mxfp4(packed, scale_bytes):
    packed_bytes = np.ascontiguousarray(packed).view(np.uint8)
    indices = np.stack((packed_bytes & 0x0F, packed_bytes >> 4), axis=-1).reshape(
        packed_bytes.shape[0], -1
    )
    values = LUT[indices]
    scale_bytes = np.ascontiguousarray(scale_bytes)
    if scale_bytes.dtype != np.uint8:
        raise RuntimeError(
            f"E8M0 scale bridge must yield uint8 storage, got {scale_bytes.dtype}"
        )
    scales = np.exp2(scale_bytes.astype(np.float32) - np.float32(127.0))
    scales = np.repeat(scales, 32, axis=1)[:, : values.shape[1]]
    if scales.shape != values.shape:
        raise RuntimeError(
            f"MXFP4/E8M0 geometry mismatch: {values.shape} != {scales.shape}"
        )
    return np.ascontiguousarray(values * scales, dtype=np.float32)


weight_map = json.loads(INDEX.read_text())["weight_map"]
numpy_handles = {}
torch_handles = {}


def packed_tensor(key):
    shard = MODEL / weight_map[key]
    handle = numpy_handles.get(str(shard))
    if handle is None:
        handle = safe_open(str(shard), framework="numpy")
        numpy_handles[str(shard)] = handle
    return handle.get_tensor(key)


def scale_storage_bytes(key):
    shard = MODEL / weight_map[key]
    handle = torch_handles.get(str(shard))
    if handle is None:
        handle = safe_open(str(shard), framework="pt", device="cpu")
        torch_handles[str(shard)] = handle
    return handle.get_tensor(key).contiguous().view(torch.uint8).cpu().numpy()


def materialize(cell):
    prefix = f"layers.{cell.layer}.ffn.experts.{cell.expert}"
    if cell.projection == "fused13":
        values = np.concatenate(
            (
                decode_mxfp4(
                    packed_tensor(prefix + ".w1.weight"),
                    scale_storage_bytes(prefix + ".w1.scale"),
                ),
                decode_mxfp4(
                    packed_tensor(prefix + ".w3.weight"),
                    scale_storage_bytes(prefix + ".w3.scale"),
                ),
            ),
            axis=0,
        )
    else:
        values = decode_mxfp4(
            packed_tensor(prefix + ".w2.weight"),
            scale_storage_bytes(prefix + ".w2.scale"),
        )
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = SOURCE.with_name(".source.tmp.npy")
    np.save(tmp, values, allow_pickle=False)
    os.replace(tmp, SOURCE)
    del values
    gc.collect()


def cleanup(cell):
    # Retain only the compact codes and immutable receipts needed by the sealed
    # single-host scorer. Large decoded/source intermediates would overflow the
    # task-private tmpfs across the full closure and are reproducible from the
    # retained codes + closure-bound control + TLUT + selected scale.
    keep = {"PUBLIC_CELL_RECEIPT.json", "CELL_RECEIPT.json", "codes.npy"}
    for path in cell.output.iterdir():
        if path.name not in keep and path.is_file():
            path.unlink()


CONTROL_ROOT = Path(os.environ["QTIP3_CONTROL_ROOT"])
CONTROL_MAP = {int(layer): Path(prefix) for layer, prefix in json.loads(os.environ["QTIP3_CONTROL_MAP"]).items()}
if set(CONTROL_MAP) != set(LAYERS):
    raise RuntimeError(f"control map mismatch: layers={LAYERS} map={sorted(CONTROL_MAP)}")


def control_for(layer, expert, projection):
    return CONTROL_ROOT / CONTROL_MAP[layer] / f"L{layer:03d}/E{expert:03d}_{projection}/QTIP_UNIT.pt"


def all_cells():
    cells = []
    scope = CELL_ROSTER or tuple(
        (layer, expert, projection)
        for layer in LAYERS for expert in EXPERTS for projection in PROJECTIONS
    )
    for layer, expert, projection in scope:
                cells.append(
                    CellSpec(
                        layer=layer,
                        expert=expert,
                        projection=projection,
                        source=SOURCE,
                        control=control_for(layer, expert, projection),
                        output=OUTPUT / f"L{layer:03d}_E{expert:03d}_{projection}",
                    )
                )
    return cells


def smoke(new_plan, cells):
    prepared = []
    batch_source_root = Path(os.environ.get("QTIP3_SMOKE_SOURCE_ROOT", str(WORK / "batch_sources")))
    accelerated_root = Path(os.environ.get("QTIP3_SMOKE_ACCEL_ROOT", str(ROOT / "outputs/smoke_accelerated")))
    common = dict(
        bpw=TIER_CONFIG.bpw,
        codec_version="v6",
        backend="cuda",
        intended_basis_sha256=BASIS,
        observed_basis_sha256=BASIS,
        scale_factors=(1.0,),
        ldlq_scale_semantics="rms_ratio",
        feedback_mode="off",
        trellis_objective="sse",
        decode_repeats=1,
        reserve_bytes=256 << 20,
    )
    for cell in cells[:SMOKE_COUNT]:
        started = time.perf_counter()
        materialize(cell)
        materialize_seconds = time.perf_counter() - started
        batch_source = batch_source_root / f"{cell.key.replace('/', '_')}.npy"
        batch_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE, batch_source)
        SOURCE.unlink()
        baseline = ROOT / "outputs/smoke_baseline" / cell.key.replace("/", "_")
        accelerated = accelerated_root / cell.key.replace("/", "_")
        shutil.rmtree(baseline, ignore_errors=True)
        shutil.rmtree(accelerated, ignore_errors=True)
        started = time.perf_counter()
        old = build_qtip_native_cell(
            batch_source,
            cell.control,
            new_plan.tlut_path,
            baseline,
            solve_batch=2048,
            decode_batch=2048,
            cyclic_fixed_point_fast_path=False,
            **common,
        )
        baseline_wall = time.perf_counter() - started
        baseline_artifacts = {
            name: {
                "baseline_sha256": sha256_file(Path(old["artifacts"][name]["path"])),
                "baseline_bytes": Path(old["artifacts"][name]["path"]).read_bytes(),
            }
            for name in ("codes", "decoded", "SU", "SV", "Wscale")
        }
        old_receipt_sha256 = old["receipt_sha256"]
        shutil.rmtree(baseline)
        prepared.append((cell, batch_source, accelerated, baseline_artifacts, old_receipt_sha256, materialize_seconds, baseline_wall))

    started = time.perf_counter()
    accelerated_results = build_qtip_native_cells(
        [
            {"source": source, "control": cell.control, "output": accelerated}
            for cell, source, accelerated, _artifacts, _old_receipt, _materialize, _baseline_wall in prepared
        ],
        new_plan.tlut_path,
        solve_batch=65536,
        decode_batch=65536,
        cyclic_fixed_point_fast_path=True,
        **common,
    )
    batch_wall = time.perf_counter() - started
    accelerated_share = batch_wall / len(prepared)
    rows = []
    accelerated_seconds = sum(item[5] for item in prepared) + batch_wall
    baseline_seconds = sum(item[5] + item[6] for item in prepared)
    for (cell, batch_source, accelerated, baseline_artifacts, old_receipt_sha256, materialize_seconds, baseline_wall), new in zip(prepared, accelerated_results, strict=True):
        accelerated_wall = accelerated_share
        comparisons = {}
        for name in ("codes", "decoded", "SU", "SV", "Wscale"):
            new_path = Path(new["artifacts"][name]["path"])
            comparisons[name] = {
                "baseline_sha256": baseline_artifacts[name]["baseline_sha256"],
                "accelerated_sha256": sha256_file(new_path),
                "bitwise": baseline_artifacts[name]["baseline_bytes"] == new_path.read_bytes(),
            }
        if not all(value["bitwise"] for value in comparisons.values()):
            raise RuntimeError(f"SMOKE_PARITY_REFUSED {cell.key} {comparisons}")
        row = {
            "cell": cell.key,
            "materialize_seconds": materialize_seconds,
            "baseline_api_seconds": baseline_wall,
            "accelerated_api_seconds": accelerated_wall,
            "baseline_total_seconds": materialize_seconds + baseline_wall,
            "accelerated_total_seconds": materialize_seconds + accelerated_wall,
            "speedup": (materialize_seconds + baseline_wall)
            / (materialize_seconds + accelerated_wall),
            "comparisons": comparisons,
            "baseline_receipt_sha256": old_receipt_sha256,
            "accelerated_receipt_sha256": new["receipt_sha256"],
        }
        atomic(REC / "smoke" / f"{cell.key.replace('/', '_')}.json", row)
        rows.append(row)
        shutil.rmtree(accelerated)
        batch_source.unlink(missing_ok=True)
    cells_per_minute = len(rows) * 60.0 / accelerated_seconds
    terminal = {
        "schema": "banana-smasher-qtip3-v7-public-api-smoke-v1",
        "status": "PASS" if len(rows) >= 20 and cells_per_minute >= 20.0 else "FAIL",
        "task_id": TASK_ID,
        "board_run_id": BOARD_RUN_ID,
        "host": HOST,
        "basis_sha256": BASIS,
        "cells": len(rows),
        "layer": 2,
        "parity": "bitwise",
        "cells_per_minute": cells_per_minute,
        "accelerated_total_seconds": accelerated_seconds,
        "baseline_total_seconds": baseline_seconds,
        "speedup": baseline_seconds / accelerated_seconds,
        "solve_batch": 65536,
        "baseline_solve_batch": 2048,
        "cross_cell_batch_cells": len(rows),
        "cross_cell_batch_wall_seconds": batch_wall,
        "rows": rows,
    }
    terminal_path = REC / "SMOKE_TERMINAL.json"
    terminal["receipt_sha256"] = atomic(terminal_path, terminal)
    if terminal["status"] != "PASS":
        raise RuntimeError(f"SMOKE_THROUGHPUT_REFUSED {cells_per_minute}")
    release = release_smoke_host(new_plan, terminal_path)
    atomic(REC / "FULL_API_SMOKE_RELEASE.json", release)
    print(json.dumps(terminal, sort_keys=True), flush=True)


new_plan = plan(EXPECTED_CLAIM)
admission = admit_host_and_shard(new_plan, gpu_probe=gpu_probe, config=TIER_CONFIG)
atomic(
    REC / "FULL_API_START.json",
    {
        "schema": "banana-smasher-qtip3-v7-full-api-start-v1",
        "status": "RUNNING",
        "task_id": new_plan.task_id,
        "cells": SMOKE_COUNT or new_plan.expected_cells,
        "admission": admission,
        "reserve_bytes": 256 << 20,
        "pid": os.getpid(),
    },
)
cells = all_cells()
if SMOKE_COUNT:
    smoke(new_plan, cells)
else:
    config = TIER_CONFIG
    terminal = run_cells_batched(
        new_plan, config, cells, batch_api=build_qtip_native_cells, batch_size=40,
        prepare_cell=materialize, cleanup_cell=cleanup,
        batch_source_root=Path(
            os.environ.get(
                "QTIP3_BATCH_SOURCE_ROOT",
                str(ROOT / "working_full_api" / "batch_sources"),
            )
        ),
        max_new_batches=MAX_NEW_BATCHES,
    )
    atomic(REC / "FULL_API_CONTROLLER_TERMINAL.json", terminal)
    release = (
        release_bounded_host(new_plan, REC / "PRODUCER_TERMINAL.json")
        if MAX_NEW_BATCHES is not None
        else release_host(new_plan, REC / "PRODUCER_TERMINAL.json")
    )
    atomic(REC / "FULL_API_RELEASE.json", release)
    print(json.dumps(terminal, sort_keys=True), flush=True)
