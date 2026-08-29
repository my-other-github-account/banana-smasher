#!/usr/bin/env python3
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import torch

from repair_api import ResidentRepairAPI
from repair_api.modern_green_resident import ModernGreenResidentEngine
from repair_api.sealed_pre_forward import bind_sealed_pre_resident_config

TASK = os.environ.get("BANANA_SMASHER_TASK_ID", "t_d4dac464")
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
W28_KLD = 0.1364830042977786
W28_TOP1 = 880
W28_ADOPTION_TASK = "t_8b1b3a3f"
SEALED_SINGLETON_L042_SHA256 = "2dba6948c490a95477a6dd5d310bc8a4993b5a01a0f2bd062dcea78d359a99ec"
ADOPTED_TASK_ID = W28_ADOPTION_TASK
ADOPTED_PROVIDER_WRAPPER_SHA256 = "ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
ADOPTED_PROVIDER_EXPERT_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"
CURRENT_PROVIDER_WRAPPER_SHA256 = ADOPTED_PROVIDER_WRAPPER_SHA256
CURRENT_PROVIDER_EXPERT_SHA256 = ADOPTED_PROVIDER_EXPERT_SHA256
CUDA_MEMORY_FRACTION = 0.45


def _apply_cuda_memory_fraction(cuda: Any) -> float:
    """Fence this process's CUDA allocator before NCCL or model allocation."""
    cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
    observed = float(cuda.get_per_process_memory_fraction())
    if not math.isclose(observed, CUDA_MEMORY_FRACTION, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"CUDA_MEMORY_FRACTION_READBACK_MISMATCH:{observed}")
    return observed


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def aggregate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(int(row["positions"]) for row in rows)
    kld_sum = math.fsum(float(row["kld_sum_binary64"]) for row in rows)
    top1 = sum(int(row["top1"]) for row in rows)
    return {"positions": positions, "kld_sum": kld_sum, "kld_mean": kld_sum / positions,
            "top1": top1, "top1_rate": top1 / positions}


def _tensor_tap(value: Any) -> dict[str, Any]:
    tensor = value.detach().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(value.stride()),
    }


def _pre_gemm_tensor_witness(value: Any, *, role: str) -> dict[str, Any]:
    """Bind an operand's bytes and physical view without changing its lifetime."""
    tensor = value.detach()
    contiguous = tensor.contiguous()
    raw = contiguous.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "stride": [int(step) for step in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "is_contiguous": bool(tensor.is_contiguous()),
        "byte_serialization": "detach.contiguous.uint8.cpu",
    }


def _compare_pre_gemm_witnesses(
    control: dict[str, Any],
    variant: dict[str, Any],
    *,
    control_invocation: dict[str, Any],
    variant_invocation: dict[str, Any],
) -> dict[str, Any]:
    """Name the first unequal pre-GEMM operand or invocation boundary."""
    ordered = ("hidden", "gate_weight")
    boundaries: dict[str, Any] = {}
    for name in ordered:
        control_value = control.get(name)
        variant_value = variant.get(name)
        exact = isinstance(control_value, dict) and control_value == variant_value
        boundaries[name] = {
            "exact": exact,
            "control": control_value,
            "variant": variant_value,
        }
    geometry_keys = (
        "operator", "weight_transpose_semantics", "m", "n", "k",
        "input_dtype", "weight_dtype", "output_dtype", "accumulation_contract",
    )
    control_geometry = {key: control_invocation.get(key) for key in geometry_keys}
    variant_geometry = {key: variant_invocation.get(key) for key in geometry_keys}
    boundaries["invocation_geometry"] = {
        "exact": control_geometry == variant_geometry,
        "control": control_invocation,
        "variant": variant_invocation,
        "compared_geometry": list(geometry_keys),
    }
    order = (*ordered, "invocation_geometry")
    first = next((name for name in order if not boundaries[name]["exact"]), None)
    return {
        "status": (
            "PRE_GEMM_INPUT_WEIGHT_GEOMETRY_PARITY"
            if first is None
            else "PRE_GEMM_DIVERGENCE_LOCALIZED"
        ),
        "boundary_order": list(order),
        "boundaries": boundaries,
        "first_unequal_boundary": first,
    }


def _capture_product_layer_taps(engine: Any, hidden: Any, ids: Any) -> tuple[dict[str, Any], Any]:
    """Capture the existing product assembly without changing its layer walk."""
    taps: dict[str, Any] = {}
    layers = engine.student.model.model.layers
    originals: dict[int, Any] = {}
    for index in range(43):
        layer = layers[index]
        originals[index] = layer.forward

        def capture(*args: Any, _index: int = index, _forward: Any = layer.forward, **kwargs: Any) -> Any:
            value = _forward(*args, **kwargs)
            taps[f"L{_index:03d}"] = _tensor_tap(value)
            return value

        layer.forward = capture
    try:
        hidden = engine._run_layers(hidden, ids, False)
        torch.cuda.synchronize()
    finally:
        for index, forward in originals.items():
            layers[index].forward = forward
    return taps, hidden


def _capture_reference_layer_taps(engine: Any, hidden: Any, ids: Any) -> dict[str, Any]:
    """Use the sealed shared-cache handoff from official_k2_resident_score.py:2729-2748."""
    from transformers.cache_utils import DynamicCache

    taps: dict[str, Any] = {}
    template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
    active_cache = DynamicCache(config=engine.student.config)
    pos, pe, mask = engine._positional(ids, template, active_cache)
    for index in range(43):
        layer = engine.student.model.model.layers[index]
        hidden = layer(
            hidden,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=mask,
            input_ids=ids,
            past_key_values=active_cache,
        )
        taps[f"L{index:03d}"] = _tensor_tap(hidden)
        entry = active_cache.layers[index]
        if bool(getattr(entry, "is_initialized", False)):
            entry.keys = entry.keys.new_empty((0,))
            entry.values = entry.values.new_empty((0,))
            entry.is_initialized = False
    torch.cuda.synchronize()
    return taps


def _whole_chain_bisect(engine: Any, *, window: int, root: Path, rank: int, pin: str) -> dict[str, Any]:
    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    ids = prepared["ids"][window]
    embeddings = engine.student.model.model.embed_tokens(ids)
    hidden = embeddings.unsqueeze(2).expand(
        -1, -1, engine.student.config.hc_mult, -1
    ).contiguous()
    engine.student.model.eval()
    with torch.no_grad():
        product_taps, _product_hidden = _capture_product_layer_taps(
            engine, hidden.clone(), ids
        )
        reference_taps = _capture_reference_layer_taps(engine, hidden.clone(), ids)
    if tuple(product_taps) != tuple(f"L{index:03d}" for index in range(43)):
        raise RuntimeError("PRODUCT_TAP_COVERAGE_RED")
    if tuple(reference_taps) != tuple(product_taps):
        raise RuntimeError("REFERENCE_TAP_COVERAGE_RED")
    first_divergent = next(
        (name for name in product_taps if product_taps[name]["sha256"] != reference_taps[name]["sha256"]),
        None,
    )
    local = {
        "rank": rank,
        "product": product_taps,
        "diagnostic": reference_taps,
        "first_divergent_layer": first_divergent,
    }
    gathered: list[Any] = [None, None]
    torch.distributed.all_gather_object(gathered, local)
    receipt = {
        "schema": "banana-smasher-whole-chain-product-diagnostic-bisect-v1",
        "status": "EXACT_43_LAYER_PARITY" if all(row["first_divergent_layer"] is None for row in gathered) else "WHOLE_CHAIN_DIVERGENCE_LOCALIZED",
        "task_id": TASK,
        "canonical_code_commit": pin,
        "basis_sha256": BASIS,
        "checkpoint_sha256": CHECKPOINT,
        "window": window,
        "one_variable": "product assembly cache handoff: isolated per-layer DynamicCache vs sealed shared-and-cleared DynamicCache",
        "reference_source": "repair_api/official_k2_resident_score.py:2729-2748",
        "product_source": "repair_api/modern_green_resident.py:1264-1403",
        "ranks": gathered,
        "first_divergent_layer": next((row["first_divergent_layer"] for row in gathered if row["first_divergent_layer"] is not None), None),
        "created_unix": time.time(),
    }
    path = root / "attempt37-whole-chain-bisect" / f"WHOLE_CHAIN_BISECT.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt


def _law4_public_product_taps(api: Any, engine: Any, *, window: int,
                              root: Path, rank: int, pin: str) -> dict[str, Any]:
    """Tap the product only while public ResidentRepairAPI.validate runs."""
    model = engine.student.model
    local: dict[str, Any] = {}
    retained_payloads: dict[str, Any] = {}
    originals: list[tuple[Any, Any]] = []
    captured_logits: list[Any] = []

    def wrap(module: Any, name: str, transform: Any = None) -> None:
        forward = module.forward
        originals.append((module, forward))
        def capture(*args: Any, **kwargs: Any) -> Any:
            value = forward(*args, **kwargs)
            tapped = transform(value) if transform is not None else value
            local[name] = _tensor_tap(tapped)
            if name in ("L001_attention_input", "L001_attention_return"):
                retained_payloads[name] = tapped.detach().to("cpu").contiguous()
            if name == "logits":
                captured_logits[:] = [tapped.detach()]
            return value
        module.forward = capture

    if rank == 0:
        wrap(model.model.embed_tokens, "embeddings")
        if os.environ.get("LAW4_L001_ATTENTION_INPUT_PAYLOAD_ONLY", "0") == "1":
            wrap(model.model.layers[1].input_layernorm, "L001_attention_input")
        if os.environ.get("LAW4_L001_ATTENTION_PAYLOAD_ONLY", "0") == "1":
            wrap(model.model.layers[1].self_attn, "L001_attention_return", _first_tensor)
    for index in range(engine.first, engine.last + 1):
        wrap(model.model.layers[index], f"L{index:03d}")
    if rank == 1:
        wrap(model.model.hc_head, "hc_head")
        wrap(model.model.norm, "norm")
        wrap(model.lm_head, "logits", lambda value: value[:1024])
    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    if rank == 0:
        local = {"ids": _tensor_tap(prepared["ids"][window]), **local}
    try:
        measurement = api.validate(engine, (window,), engine.config["validation_teacher_root"])
    finally:
        for module, forward in originals:
            module.forward = forward
    attention_input_only = os.environ.get("LAW4_L001_ATTENTION_INPUT_PAYLOAD_ONLY", "0") == "1"
    attention_return_only = os.environ.get("LAW4_L001_ATTENTION_PAYLOAD_ONLY", "0") == "1"
    if attention_input_only and attention_return_only:
        raise RuntimeError("LAW4_L001_ATTENTION_DIAGNOSTIC_AMBIGUOUS")
    if rank == 0 and (attention_input_only or attention_return_only):
        payload_name = "L001_attention_input" if attention_input_only else "L001_attention_return"
        payload = retained_payloads.get(payload_name)
        if payload is None:
            raise RuntimeError(f"LAW4_{payload_name.upper()}_PAYLOAD_MISSING")
        payload_file = "RESIDENT_L001_ATTENTION_INPUT.pt" if attention_input_only else "RESIDENT_L001_ATTENTION_RETURN.pt"
        payload_path = root / "receipts" / payload_file
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = payload_path.with_name(f".{payload_path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, payload_path)
        directory_fd = os.open(payload_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if rank == 1:
        if len(captured_logits) != 1:
            raise RuntimeError("LAW4_PRODUCT_LOGITS_CAPTURE_RED")
        from repair_api.balanced64 import POSITIONS_PER_WINDOW, SUPPORT, _load_torch
        teacher = _load_torch(Path(engine.config["validation_teacher_root"]) / f"t8192_win{window}.pt")
        indices = teacher["idx"][:POSITIONS_PER_WINDOW, :SUPPORT].to(
            dtype=torch.int64, device=captured_logits[0].device)
        logprob = torch.log_softmax(captured_logits[0].float(), dim=-1)
        local["q_lp_at_ref"] = _tensor_tap(logprob.gather(1, indices).to(torch.float16))
        local["q_argmax"] = _tensor_tap(logprob.argmax(-1).to(torch.int32))
    gathered: list[Any] = [None, None]
    torch.distributed.all_gather_object(gathered, local)
    merged: dict[str, Any] = {}
    for row in gathered:
        overlap = set(merged) & set(row)
        if overlap:
            raise RuntimeError(f"LAW4_PRODUCT_TAP_OVERLAP:{sorted(overlap)}")
        merged.update(row)
    layer_taps: list[str] = []
    for index in range(43):
        if index == 1 and os.environ.get("LAW4_L001_ATTENTION_INPUT_PAYLOAD_ONLY", "0") == "1":
            layer_taps.append("L001_attention_input")
        if index == 1 and os.environ.get("LAW4_L001_ATTENTION_PAYLOAD_ONLY", "0") == "1":
            layer_taps.append("L001_attention_return")
        layer_taps.append(f"L{index:03d}")
    required = ("ids", "embeddings", *layer_taps,
                "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax")
    if tuple(merged) != required:
        raise RuntimeError("LAW4_PRODUCT_TAP_ORDER_RED")
    receipt = {
        "schema": "banana-smasher-law4-public-product-tap-v1", "status": "PASS",
        "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
        "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT, "window": window,
        "public_api": {"method": "ResidentRepairAPI.validate", "version": "v1"},
        "product_source": "repair_api/modern_green_resident.py:1264-1403",
        "measurement": measurement, "taps": merged, "created_unix": time.time(),
    }
    path = root / "receipts" / f"LAW4_PUBLIC_PRODUCT_TAPS.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    return receipt


def _readout_binding_ab(engine: Any, *, window: int, root: Path, rank: int, pin: str) -> dict[str, Any]:
    """Compare product and sealed diagnostic readout math on one exact L042 tensor."""
    import numpy as np
    from repair_api.balanced64 import POSITIONS_PER_WINDOW, SUPPORT, _load_torch

    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    ids = prepared["ids"][window]
    embeddings = engine.student.model.model.embed_tokens(ids)
    hidden = embeddings.unsqueeze(2).expand(
        -1, -1, engine.student.config.hc_mult, -1
    ).contiguous()
    engine.student.model.eval()
    with torch.no_grad():
        layer_taps, hidden = _capture_product_layer_taps(engine, hidden, ids)
    local: dict[str, Any] = {"rank": rank, "L042": layer_taps["L042"]}
    if rank == 1:
        teacher_path = Path(engine.config["validation_teacher_root"]) / f"t8192_win{window}.pt"
        teacher = _load_torch(teacher_path)
        teacher_idx = teacher["idx"][:POSITIONS_PER_WINDOW, :SUPPORT].to(
            dtype=torch.int64, device="cpu"
        ).contiguous()
        teacher_logprob = teacher["logprob"][:POSITIONS_PER_WINDOW, :SUPPORT].to(
            dtype=torch.float16, device="cpu"
        ).contiguous()
        idx_device = teacher_idx.to(device=engine.student.device)

        def readout() -> tuple[dict[str, Any], dict[str, Any]]:
            hc = engine.student.model.model.hc_head(hidden)
            final = engine.student.model.model.norm(hc)
            logits = engine.student.model.lm_head(
                final[0, :POSITIONS_PER_WINDOW].to(torch.bfloat16)
            ).float()
            logprob = torch.log_softmax(logits, dim=-1)
            q_lp = logprob.gather(1, idx_device).to(torch.float16)
            q_argmax = logprob.argmax(-1).to(torch.int64)
            q_np = q_lp.cpu().numpy().astype(np.float64, copy=False)
            ref_np = teacher_logprob.numpy().astype(np.float64, copy=False)
            ref_max = np.max(ref_np, axis=1, keepdims=True)
            cand_max = np.max(q_np, axis=1, keepdims=True)
            ref_norm = ref_np - (ref_max + np.log(np.exp(ref_np - ref_max).sum(axis=1, keepdims=True)))
            cand_norm = q_np - (cand_max + np.log(np.exp(q_np - cand_max).sum(axis=1, keepdims=True)))
            terms = np.sum(np.exp(ref_norm) * (ref_norm - cand_norm), axis=1, dtype=np.float64)
            stages = {
                "L042": layer_taps["L042"], "hc_head": _tensor_tap(hc),
                "norm": _tensor_tap(final), "logits": _tensor_tap(logits),
                "q_lp_at_ref": _tensor_tap(q_lp), "q_argmax": _tensor_tap(q_argmax),
            }
            metrics = {
                "kld_mean": math.fsum(float(value) for value in terms.tolist()) / POSITIONS_PER_WINDOW,
                "top1": int(np.count_nonzero(q_argmax.cpu().numpy() == teacher_idx[:, 0].numpy())),
            }
            return stages, metrics

        with torch.no_grad():
            product_stages, product_metrics = readout()
            diagnostic_stages, diagnostic_metrics = readout()
        local.update({
            "product": {"stages": product_stages, "metrics": product_metrics},
            "diagnostic": {"stages": diagnostic_stages, "metrics": diagnostic_metrics},
            "teacher": {
                "path": str(teacher_path), "sha256": sha(teacher_path),
                "idx": _tensor_tap(teacher_idx), "logprob": _tensor_tap(teacher_logprob),
            },
        })
    gathered: list[Any] = [None, None]
    torch.distributed.all_gather_object(gathered, local)
    readout_rank = gathered[1]
    product = readout_rank["product"]
    diagnostic = readout_rank["diagnostic"]
    stage_order = ("L042", "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax")
    first_divergent = next(
        (name for name in stage_order if product["stages"][name]["sha256"] != diagnostic["stages"][name]["sha256"]),
        None,
    )
    receipt = {
        "schema": "banana-smasher-readout-binding-ab-v1",
        "status": "READOUT_DIVERGENCE_LOCALIZED" if first_divergent else "EXACT_READOUT_PARITY",
        "task_id": TASK, "canonical_code_commit": pin, "basis_sha256": BASIS,
        "checkpoint_sha256": CHECKPOINT, "window": window,
        "sealed_hidden_gate": {
            "rank0_sha256": gathered[0]["L042"]["sha256"],
            "rank1_sha256": gathered[1]["L042"]["sha256"],
            "attempt37b_receipts": ["5cc1909b86095d5434bcbc1b764381776a5a8c2459e7dec29528fd91b0ba8855", "1992ab3c66af23c54a7f475d86ebe3d9e5dce1373b0134bfd361d6994ba68840"],
        },
        "product_source": "repair_api/modern_green_resident.py:2130-2178 (rank1 readout)",
        "diagnostic_source": "repair_api/official_k2_resident_score.py:2782-2831 (sealed parity readout)",
        "one_variable": "post-L042 readout/teacher handoff only",
        "product_rank1": product, "diagnostic_rank1": diagnostic,
        "teacher_reference": readout_rank["teacher"],
        "first_divergent_stage": first_divergent, "created_unix": time.time(),
    }
    path = root / "attempt39-readout-binding" / f"READOUT_BINDING_AB.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt


def _accumulate_token_major_stable_routes(
    weighted_slot_major: Any,
    top_k_index: Any,
    *,
    hidden_dtype: Any,
) -> Any:
    """Match accepted 942c route ordering and BF16 rounding lifetime."""
    route_shape = tuple(int(value) for value in top_k_index.shape)
    expected_shape = (route_shape[1], route_shape[0])
    if tuple(weighted_slot_major.shape[:2]) != expected_shape:
        raise RuntimeError(
            "R20_WEIGHTED_ROUTE_GEOMETRY_RED:"
            f"{tuple(weighted_slot_major.shape)}:{expected_shape}"
        )
    token_major = weighted_slot_major.transpose(0, 1)
    expert_order = torch.argsort(top_k_index, dim=1, stable=True)
    ordered_output = torch.gather(
        token_major,
        1,
        expert_order.unsqueeze(-1).expand_as(token_major),
    )
    final = torch.zeros(
        (route_shape[0], weighted_slot_major.shape[-1]),
        dtype=hidden_dtype,
        device=weighted_slot_major.device,
    )
    for route_slot in range(route_shape[1]):
        final = (final + ordered_output[:, route_slot]).to(hidden_dtype)
    return final


def _authentic_scoring_readout_boundary(
    api: Any, engine: Any, *, window: int, root: Path, rank: int, pin: str,
) -> dict[str, Any]:
    """Tap the real validation L042/readout path against its sealed singleton gate."""
    stages: dict[str, Any] = {}

    def capture(name: str, value: Any) -> None:
        if name in stages:
            raise RuntimeError(f"AUTHENTIC_READOUT_DUPLICATE_STAGE:{name}")
        stages[name] = _tensor_tap(value)

    if rank == 1:
        engine.authentic_scoring_readout_boundary_tap = capture
    try:
        measurement = _score_admission_windows(
            api, engine, (window,), Path(engine.config["validation_teacher_root"])
        )
        torch.cuda.synchronize()
    finally:
        if rank == 1:
            del engine.authentic_scoring_readout_boundary_tap
    local = {"rank": rank, "stages": stages}
    gathered: list[Any] = [None, None]
    engine.dist.all_gather_object(gathered, local)
    readout = gathered[1]["stages"]
    stage_order = ("L042", "hc_head", "norm", "logits", "q_lp", "q_argmax")
    if tuple(readout) != stage_order:
        raise RuntimeError(f"AUTHENTIC_READOUT_COVERAGE_RED:{tuple(readout)}")
    l042_exact = readout["L042"]["sha256"] == SEALED_SINGLETON_L042_SHA256
    metric_exact = (
        measurement.get("windows") == [window]
        and measurement.get("kld_mean") == W28_KLD
        and measurement.get("top1") == W28_TOP1
    )
    first_divergent = None if l042_exact and metric_exact else (
        "L042" if not l042_exact else "post_L042_readout_or_teacher_binding"
    )
    receipt = {
        "schema": "banana-smasher-authentic-scoring-readout-boundary-v1",
        "status": "EXACT_AUTHENTIC_READOUT_PARITY" if first_divergent is None else "AUTHENTIC_READOUT_DIVERGENCE_LOCALIZED",
        "task_id": TASK, "canonical_code_commit": pin,
        "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
        "window": window,
        "product_source": "repair_api/modern_green_resident.py:_validate_preloaded authentic _run_layers/readout path",
        "sealed_control": {
            "attempt106ay_rank_receipt_sha256": "83b49688f0639b60f0ba0af0c351d4d8dfa008dbd9ee0bea3a9d94148453b6cb",
            "L042_sha256": SEALED_SINGLETON_L042_SHA256,
            "kld_mean": W28_KLD, "top1": W28_TOP1,
        },
        "product_stages": readout,
        "measurement": measurement,
        "first_divergent_boundary": first_divergent,
        "created_unix": time.time(),
    }
    path = root / "receipts" / f"AUTHENTIC_SCORING_READOUT_BOUNDARY.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt
def _first_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _routed_return_assembly_witness(
    hidden_states: Any,
    routed_output: Any,
    top_k_index: Any,
    top_k_weights: Any,
) -> dict[str, Any]:
    """Witness the immutable provider's final route assembly without changing it."""
    token_index = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .unsqueeze(1).expand_as(top_k_index).reshape(-1)
    )
    expert_index = top_k_index.reshape(-1).to(torch.int64)
    weighted = (
        routed_output * top_k_weights.reshape(-1, 1).float()
    ).to(hidden_states.dtype)
    weighted_slot_major = weighted.reshape(
        hidden_states.shape[0], top_k_index.shape[1], weighted.shape[-1]
    ).transpose(0, 1).reshape(-1, weighted.shape[-1])
    final = torch.zeros_like(hidden_states)
    destination_zero = final.detach().clone()
    for expert in torch.unique(expert_index, sorted=True):
        mask = expert_index == expert
        final.index_add_(0, token_index[mask], weighted[mask])

    def metadata(value: Any) -> dict[str, Any]:
        witness = _tensor_tap(value)
        witness.update({
            "stride": [int(item) for item in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "contiguous": bool(value.is_contiguous()),
            "data_ptr": int(value.data_ptr()),
        })
        if int(value.numel()) <= 64:
            witness["values"] = value.detach().cpu().reshape(-1).tolist()
        return witness

    zero = metadata(destination_zero)
    zero["nonzero"] = int(torch.count_nonzero(destination_zero).item())
    return {
        "assembly_contract": "token-major-flat_then_ascending-expert-index-add",
        "source_routed_output": metadata(routed_output),
        "route_weights": metadata(top_k_weights),
        "token_index": metadata(token_index),
        "expert_index": metadata(expert_index),
        "destination_zero": zero,
        "weighted_routes": metadata(weighted),
        "weighted_routes_slot_major": metadata(weighted_slot_major),
        "final_return": metadata(final),
        "source_file_line": "repair_api/modern_green_resident.py:520-543",
        "operations": [
            "reshape top_k_weights token-major",
            "multiply routed_output by route weights in float32",
            "cast weighted routes to hidden dtype",
            "flatten token/expert indices token-major",
            "ascending-expert index_add_ into zero destination",
        ],
    }


def _compare_tensor_boundary(control: Any, variant: Any) -> dict[str, Any]:
    control_tap = _tensor_tap(control)
    variant_tap = _tensor_tap(variant)
    first_difference = None
    if control_tap["shape"] != variant_tap["shape"] or control_tap["dtype"] != variant_tap["dtype"]:
        first_difference = {
            "kind": "tensor_identity",
            "control_shape": control_tap["shape"],
            "variant_shape": variant_tap["shape"],
            "control_dtype": control_tap["dtype"],
            "variant_dtype": variant_tap["dtype"],
        }
    elif control_tap["sha256"] != variant_tap["sha256"]:
        unequal = control.reshape(-1) != variant.reshape(-1)
        index = int(torch.nonzero(unequal, as_tuple=False)[0].item())
        remaining = index
        coordinates = []
        for extent in reversed(control.shape):
            coordinates.append(remaining % int(extent))
            remaining //= int(extent)
        coordinates.reverse()
        first_difference = {
            "kind": "value",
            "flat_index": index,
            "index": coordinates,
            "control_value": control.reshape(-1)[index].item(),
            "variant_value": variant.reshape(-1)[index].item(),
        }
    return {
        "exact": first_difference is None,
        "control": control_tap,
        "variant": variant_tap,
        "first_difference": first_difference,
    }


class _AuthenticRouteCaptureMode:
    """Retain aliases at the immutable provider's weighted-route seam."""

    def __new__(cls) -> Any:
        from torch.utils._python_dispatch import TorchDispatchMode

        class CaptureMode(TorchDispatchMode):
            def __init__(self) -> None:
                super().__init__()
                self.top_k_index = None
                self.top_k_weights = None
                self.w2_output = None
                self.expert_indices = None
                self.token_indices = None
                self.route_weights = None
                self.weighted_buffer = None
                self.ordered_weighted_buffer = None
                self._raw_weighted = None
                self._route_count = None

            @staticmethod
            def _same_storage(left: Any, right: Any) -> bool:
                return (
                    isinstance(left, torch.Tensor)
                    and isinstance(right, torch.Tensor)
                    and left.device == right.device
                    and int(left.data_ptr()) == int(right.data_ptr())
                )

            def bind_route_inputs(self, top_k_index: Any, top_k_weights: Any) -> None:
                self.top_k_index = top_k_index
                self.top_k_weights = top_k_weights
                self._route_count = int(top_k_index.numel())

            def capture_w2(self, value: Any, assignments: Any) -> None:
                self.w2_output = value
                self.expert_indices = assignments

            def capture_route(
                self, w2_output: Any, expert_indices: Any, route_weights: Any,
                weighted_buffer: Any, ordered_weighted_buffer: Any,
            ) -> None:
                self.w2_output = w2_output
                self.expert_indices = expert_indices
                self.route_weights = route_weights
                self.weighted_buffer = weighted_buffer
                self.ordered_weighted_buffer = ordered_weighted_buffer

            def capture_ordered(self, value: Any) -> None:
                if (
                    self.w2_output is not None
                    and self.ordered_weighted_buffer is None
                    and isinstance(value, torch.Tensor)
                    and value.ndim == 3
                ):
                    self.ordered_weighted_buffer = value

            def __torch_dispatch__(
                self, func: Any, types: Any, args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                del types
                result = func(*args, **(kwargs or {}))
                if self._route_count is None:
                    return result
                if (
                    self.token_indices is None
                    and isinstance(result, torch.Tensor)
                    and result.dtype == torch.int64
                    and result.ndim == 1
                    and int(result.numel()) == self._route_count
                    and not self._same_storage(result, self.top_k_index)
                ):
                    self.token_indices = result
                if (
                    func is torch.ops.aten.mul.Tensor
                    and self.w2_output is not None
                    and len(args) >= 2
                    and self._same_storage(args[0], self.w2_output)
                ):
                    self.route_weights = args[1]
                    self._raw_weighted = result
                    if result.dtype == self.w2_output.dtype:
                        self.weighted_buffer = result
                elif (
                    func is torch.ops.aten._to_copy.default
                    and self._raw_weighted is not None
                    and args
                    and self._same_storage(args[0], self._raw_weighted)
                    and isinstance(result, torch.Tensor)
                    and result.dtype == self.w2_output.dtype
                ):
                    self.weighted_buffer = result
                elif (
                    self.weighted_buffer is not None
                    and self.w2_output is not None
                    and self.ordered_weighted_buffer is None
                    and isinstance(result, torch.Tensor)
                    and result.ndim == 3
                    and result.dtype == self.w2_output.dtype
                    and not self._same_storage(result, self.weighted_buffer)
                ):
                    self.ordered_weighted_buffer = result
                return result

        return CaptureMode()


def _replay_a30_route_schedules(
    hidden_states: Any, top_k_index: Any, weighted_buffer: Any,
) -> dict[str, Any]:
    """Replay the two accepted BF16 accumulation schedules from captured bytes."""
    route_shape = tuple(int(value) for value in top_k_index.shape)
    routed = weighted_buffer.reshape(
        route_shape[0], route_shape[1], hidden_states.shape[-1]
    )
    expert_order = torch.argsort(top_k_index, dim=1, stable=True)
    ordered = torch.gather(
        routed, 1, expert_order.unsqueeze(-1).expand_as(routed)
    )
    token_major = torch.zeros_like(hidden_states)
    for route_slot in range(route_shape[1]):
        token_major = (token_major + ordered[:, route_slot]).to(hidden_states.dtype)

    token_index = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .unsqueeze(1).expand_as(top_k_index).reshape(-1)
    )
    expert_index = top_k_index.reshape(-1).to(torch.int64)
    expert_major = torch.zeros_like(hidden_states)
    for expert in torch.unique(expert_index, sorted=True):
        mask = expert_index == expert
        expert_major.index_add_(
            0, token_index[mask], weighted_buffer[mask].to(expert_major.dtype)
        )
    return {
        "token_major": token_major,
        "expert_major": expert_major,
        "ordered_weighted": ordered,
    }


def _replay_source_return_assemblies(
    hidden_states: Any,
    top_k_index: Any,
    weighted_slot_major_flat: Any,
    control_return: Any,
    provider_return: Any,
) -> dict[str, Any]:
    """Compare accepted torch.where assembly with provider flat-mask assembly."""
    import torch.nn.functional as F

    tokens, slots = (int(top_k_index.shape[0]), int(top_k_index.shape[1]))
    slot_major = weighted_slot_major_flat.reshape(slots, tokens, -1)
    num_experts = max(256, int(top_k_index.max().item()) + 1)

    def accepted_where_index_add() -> Any:
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_row in hit:
            expert = expert_row[0]
            top_k_pos, token_index = torch.where(mask[expert])
            final.index_add_(
                0, token_index, slot_major[top_k_pos, token_index].to(final.dtype)
            )
        return final

    accepted_a = accepted_where_index_add()
    accepted_b = accepted_where_index_add()
    if not torch.equal(accepted_a, accepted_b):
        raise RuntimeError("RUN6519_ACCEPTED_ASSEMBLY_SELF_COMPARE_RED")

    token_index = (
        torch.arange(tokens, device=top_k_index.device)
        .unsqueeze(1).expand_as(top_k_index).reshape(-1)
    )
    expert_index = top_k_index.reshape(-1).to(torch.int64)
    token_major_weighted = slot_major.transpose(0, 1).reshape(-1, slot_major.shape[-1])
    provider_flat = torch.zeros_like(hidden_states)
    for expert in torch.unique(expert_index, sorted=True):
        selected = expert_index == expert
        provider_flat.index_add_(
            0, token_index[selected], token_major_weighted[selected].to(provider_flat.dtype)
        )

    accepted_exact = torch.equal(accepted_a, control_return)
    provider_exact = torch.equal(provider_flat, provider_return)
    return {
        "status": "RETURN_ASSEMBLY_PRIMITIVE_LOCALIZED" if accepted_exact and provider_exact else "RETURN_ASSEMBLY_PRIMITIVE_INCONCLUSIVE",
        "installed_provider_source": "repair_api/modern_green_resident.py:538-573",
        "accepted_builder_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:456-473; accepted DeepseekV4Experts forward torch.where/index_add primitive",
        "instrument_control_self_compare_exact": True,
        "accepted_where_index_add": _tensor_tap(accepted_a),
        "provider_flat_mask_index_add": _tensor_tap(provider_flat),
        "authentic_control_return": _tensor_tap(control_return),
        "authentic_provider_return": _tensor_tap(provider_return),
        "accepted_matches_authentic_control": accepted_exact,
        "provider_flat_matches_authentic_provider": provider_exact,
        "first_assembly_operation_divergence": (
            "provider_flattened_token_major_boolean_mask_index_add_invocation"
            if accepted_exact and provider_exact and not torch.equal(accepted_a, provider_flat)
            else None
        ),
    }


def _adjudicate_temporal_interleaving(
    authentic_control_return: Any,
    source_interleaved_return: Any,
    post_materialized_return: Any,
) -> dict[str, Any]:
    """Adjudicate source-temporal projection/add against delayed assembly."""
    source_exact = torch.equal(source_interleaved_return, authentic_control_return)
    delayed_exact = torch.equal(post_materialized_return, authentic_control_return)
    return {
        "status": (
            "TEMPORAL_INTERLEAVING_LOCALIZED"
            if source_exact and not delayed_exact
            else "TEMPORAL_INTERLEAVING_INCONCLUSIVE"
        ),
        "installed_provider_source": "repair_api/modern_green_resident.py:538-573",
        "accepted_builder_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:456-473",
        "one_variable": (
            "per-expert projection -> weight -> immediate index_add versus "
            "post-materialized all-route assembly"
        ),
        "instrument_control_self_compare_exact": source_exact,
        "source_interleaved_matches_authentic_control": source_exact,
        "post_materialized_matches_authentic_control": delayed_exact,
        "authentic_control_return": _tensor_tap(authentic_control_return),
        "source_interleaved_return": _tensor_tap(source_interleaved_return),
        "post_materialized_return": _tensor_tap(post_materialized_return),
        "first_temporal_operation_divergence": (
            "deferred_all_route_materialization_before_index_add"
            if source_exact and not delayed_exact else None
        ),
    }


def _adjudicate_source_workspace_lifetime(
    authentic_control_return: Any,
    retained_alias_return: Any,
    fresh_workspace_return: Any,
) -> dict[str, Any]:
    """Compare source projection with retained aliases versus fresh workspaces."""
    retained_exact = torch.equal(retained_alias_return, authentic_control_return)
    fresh_exact = torch.equal(fresh_workspace_return, authentic_control_return)
    return {
        "status": (
            "SOURCE_PROJECTION_WORKSPACE_LIFETIME_LOCALIZED"
            if fresh_exact and not retained_exact
            else "SOURCE_PROJECTION_WORKSPACE_LIFETIME_INCONCLUSIVE"
        ),
        "installed_provider_source": "repair_api/modern_green_resident.py:538-573",
        "accepted_builder_source": (
            "repair_api/assets/builder_B2_PUBLISHED_PRE.py:456-473; "
            "accepted DeepseekV4Experts source projection"
        ),
        "one_variable": (
            "retained source projection operand aliases versus byte-identical "
            "fresh contiguous operand workspaces immediately before F.linear"
        ),
        "instrument_control_self_compare_exact": fresh_exact,
        "retained_alias_matches_authentic_control": retained_exact,
        "fresh_workspace_matches_authentic_control": fresh_exact,
        "authentic_control_return": _tensor_tap(authentic_control_return),
        "retained_alias_return": _tensor_tap(retained_alias_return),
        "fresh_workspace_return": _tensor_tap(fresh_workspace_return),
        "first_workspace_operation_divergence": (
            "source_projection_operand_workspace_before_F.linear"
            if fresh_exact and not retained_exact else None
        ),
    }


def _run_one_layer_with_attention(engine: Any, layer: Any, hidden: Any, ids: Any) -> tuple[Any, Any]:
    from transformers.cache_utils import DynamicCache

    attention: dict[str, Any] = {}

    def capture(_module: Any, _args: Any, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            raise RuntimeError("ONE_LAYER_ATTENTION_TENSOR_ABSENT")
        attention["tensor"] = tensor.detach().clone()

    handle = layer.self_attn.register_forward_hook(capture)
    try:
        cache = DynamicCache(config=engine.student.config)
        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        pos, pe, mask = engine._positional(ids, template, cache)
        output = layer(
            hidden,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=mask,
            input_ids=ids,
            past_key_values=cache,
        )
    finally:
        handle.remove()
    if "tensor" not in attention:
        raise RuntimeError("ONE_LAYER_ATTENTION_HOOK_NOT_CALLED")
    return output, attention["tensor"]


def _run_one_layer_with_expert_trace(
    engine: Any, layer: Any, hidden: Any, ids: Any, *, resident: bool
) -> tuple[Any, dict[str, Any]]:
    """Run the authentic expert arithmetic while retaining named route seams."""
    import types
    import torch.nn.functional as F

    experts = layer.mlp.experts
    had_instance_forward = "forward" in experts.__dict__
    prior_instance_forward = experts.__dict__.get("forward")
    original_forward = experts.forward
    replay_inputs: dict[str, Any] = {}
    authentic_capture: dict[str, Any] = {}
    pre_gemm_capture: dict[str, Any] = {}
    assembly_capture: dict[str, Any] = {}
    captured: dict[str, list[Any]] = {
        name: [] for name in (
            "route_key", "gate", "up", "activated", "w2_down",
            "weighted_routed_output", "full_weighted_routed_output",
            "full_route_expert", "per_slot_accumulation", "authentic_routed_return",
        )
    }

    def keep(name: str, value: Any) -> None:
        # Retain an alias only.  Attempt106 step 2 proved that synchronizing and
        # copying every expert boundary to CPU inside the expert loop changes
        # the later BF16 GEMM algorithm/workspace and therefore the output.  All
        # copies and accumulation replay happen after the authentic arithmetic.
        captured[name].append(value.detach())

    def replay_control(
        module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any,
        *, fresh_workspace: bool = False, record_trace: bool = True,
    ) -> Any:
        def source_linear(value: Any, weight: Any) -> Any:
            if fresh_workspace:
                value = value.clone(memory_format=torch.contiguous_format)
                weight = weight.clone(memory_format=torch.contiguous_format)
            return F.linear(value, weight)

        final = torch.zeros_like(hidden_states)
        full_weighted = torch.empty(
            (top_k_index.numel(), hidden_states.shape[-1]),
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        full_route_expert = top_k_index.transpose(0, 1).reshape(-1).to(torch.int64)
        with torch.no_grad():
            mask = F.one_hot(
                top_k_index, num_classes=module.num_experts
            ).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            trace_expert = torch.tensor(204, device=hit.device, dtype=hit.dtype)
            if not torch.any(hit[:, 0] == trace_expert):
                raise RuntimeError("EXPERT_TRACE_TARGET_204_INACTIVE")
        trace_tokens = None
        for expert_index_row in hit:
            expert_index = expert_index_row[0]
            if expert_index == module.num_experts:
                continue
            top_k_pos, token_index = torch.where(mask[expert_index])
            routed = hidden_states[token_index]
            if expert_index != trace_expert:
                # Byte-for-byte source shape of DeepseekV4Experts.forward: do
                # not extend the lifetime of gate_up/down temporaries for any
                # expert that precedes the single traced final expert.
                current = module._apply_gate(source_linear(
                    routed, module.gate_up_proj[expert_index]
                ))
                current = source_linear(
                    current, module.down_proj[expert_index]
                ) * top_k_weights[token_index, top_k_pos, None]
                flat_route = top_k_pos * hidden_states.shape[0] + token_index
                full_weighted[flat_route] = current.to(full_weighted.dtype)
                final.index_add_(0, token_index, current.to(final.dtype))
                continue
            gate_up = source_linear(routed, module.gate_up_proj[expert_index])
            split = int(module.intermediate_dim)
            gate_weight = module.gate_up_proj[expert_index][:split]
            pre_gemm_capture["control"] = {
                "hidden": _pre_gemm_tensor_witness(
                    routed, role="expert_204_active_hidden_rows"
                ),
                "gate_weight": _pre_gemm_tensor_witness(
                    gate_weight, role="expert_204_decoded_gate_weight"
                ),
                "invocation": {
                    "operator": "torch.nn.functional.linear",
                    "input_expression": "routed",
                    "weight_expression": "module.gate_up_proj[expert_index]",
                    "weight_transpose_semantics": "input @ weight.T",
                    "m": int(routed.shape[0]),
                    "n": int(module.gate_up_proj[expert_index].shape[0]),
                    "k": int(routed.shape[1]),
                    "input_dtype": str(routed.dtype),
                    "weight_dtype": str(module.gate_up_proj.dtype),
                    "output_dtype": str(gate_up.dtype),
                    "accumulation_contract": "PyTorch CUDA BF16 F.linear default",
                },
            }
            trace_tokens = token_index
            activated = module._apply_gate(gate_up)
            down = source_linear(activated, module.down_proj[expert_index])
            weighted = down * top_k_weights[token_index, top_k_pos, None]
            flat_route = top_k_pos * hidden_states.shape[0] + token_index
            full_weighted[flat_route] = weighted.to(full_weighted.dtype)
            final.index_add_(0, token_index, weighted.to(final.dtype))
            # Retain only the first active expert. Its boundaries are produced
            # before any diagnostic alias can perturb a later BF16 GEMM.
            gate, up = gate_up.chunk(2, dim=-1)
            if record_trace:
                keep("route_key", torch.stack((top_k_pos, token_index), dim=1))
                keep("gate", gate)
                keep("up", up)
                keep("activated", activated)
                keep("w2_down", down)
                keep("weighted_routed_output", weighted)
        if record_trace:
            keep("full_weighted_routed_output", full_weighted)
            keep("full_route_expert", full_route_expert)
        if trace_tokens is None:
            raise RuntimeError("EXPERT_TRACE_TARGET_204_INACTIVE")
        if record_trace:
            keep("per_slot_accumulation", final[trace_tokens])
        return final

    def replay_resident(
        module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any
    ) -> Any:
        token_index = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            .unsqueeze(0).expand(top_k_index.shape[1], -1).reshape(-1)
        )
        slot_index = (
            torch.arange(top_k_index.shape[1], device=hidden_states.device)
            .unsqueeze(1).expand(-1, hidden_states.shape[0]).reshape(-1)
        )
        expert_index = top_k_index.transpose(0, 1).reshape(-1).to(torch.int64)
        route_weight = top_k_weights.transpose(0, 1).reshape(-1, 1)
        routed_hidden = hidden_states[token_index].contiguous()
        lut_master = module.plane_source.wire_lut().reshape(-1).contiguous()
        gate = module._project(
            "w1", routed_hidden, expert_index, module.packed_w1,
            lut_master, module.su_w1, module.sv_w1,
        )
        up = module._project(
            "w3", routed_hidden, expert_index, module.packed_w3,
            lut_master, module.su_w3, module.sv_w3,
        )
        if os.environ.get("FAST_K2_SEALED_NO_SWIGLU_CLAMP", "0") != "1":
            gate = gate.clamp(max=module.limit)
            up = up.clamp(min=-module.limit, max=module.limit)
        activated = module.act(gate) * up
        down = module._project(
            "w2", activated, expert_index, module.packed_w2,
            lut_master, module.su_w2, module.sv_w2,
        )
        weighted = down * route_weight.to(dtype=down.dtype)
        keep("full_weighted_routed_output", weighted)
        keep("full_route_expert", expert_index)
        active_experts = torch.unique(expert_index, sorted=True)
        trace_expert = torch.tensor(204, device=active_experts.device, dtype=active_experts.dtype)
        if not torch.any(active_experts == trace_expert):
            raise RuntimeError("EXPERT_TRACE_TARGET_204_INACTIVE")
        trace_mask = expert_index == trace_expert
        trace_tokens = token_index[trace_mask]
        diagnostic_decoder = _load_r20_grouped_decoder()
        expert_number = int(trace_expert.item())
        gate_weight = diagnostic_decoder.sealed_bf16_full_weight(
            module.packed_w1[expert_number],
            lut_master,
            module.su_w1[expert_number],
            module.sv_w1[expert_number],
        ).transpose(0, 1).contiguous()
        pre_gemm_capture["variant"] = {
            "hidden": _pre_gemm_tensor_witness(
                routed_hidden[trace_mask], role="expert_204_active_hidden_rows"
            ),
            "gate_weight": _pre_gemm_tensor_witness(
                gate_weight, role="expert_204_decoded_gate_weight"
            ),
            "invocation": {
                "operator": "torch.nn.functional.linear",
                "input_expression": "expert_x",
                "weight_expression": "torch.cat((gate_weight, up_weight), dim=0)",
                "weight_transpose_semantics": "input @ weight.T",
                "m": int(routed_hidden[trace_mask].shape[0]),
                "n": int(module.sv_w1.shape[1] + module.sv_w3.shape[1]),
                "k": int(routed_hidden.shape[1]),
                "input_dtype": str(routed_hidden.dtype),
                "weight_dtype": str(gate_weight.dtype),
                "output_dtype": str(gate.dtype),
                "accumulation_contract": "PyTorch CUDA BF16 F.linear default",
            },
        }
        final = _accumulate_token_major_stable_routes(
            weighted.reshape(
                top_k_index.shape[1], hidden_states.shape[0], weighted.shape[-1]
            ),
            top_k_index,
            hidden_dtype=hidden_states.dtype,
        )
        keep("route_key", torch.stack((slot_index[trace_mask], trace_tokens), dim=1))
        keep("gate", gate[trace_mask])
        keep("up", up[trace_mask])
        keep("activated", activated[trace_mask])
        keep("w2_down", down[trace_mask])
        keep("weighted_routed_output", weighted[trace_mask])
        keep("per_slot_accumulation", final[trace_tokens])
        module.cpu_relay_bytes += 0
        module.reconstruction_calls += 0
        return final

    def transparent_forward(
        module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any
    ) -> Any:
        # Record aliases only, then invoke the exact bound product method. The
        # prior wrappers reimplemented the expert loop and retained projection
        # temporaries before shared-expert/residual arithmetic, changing the
        # BF16 workspace selection. Boundary reconstruction happens only after
        # the complete authentic layer output has been produced and sealed.
        replay_inputs.update({
            "hidden_states": hidden_states,
            "top_k_index": top_k_index,
            "top_k_weights": top_k_weights,
        })
        if not resident:
            authentic = original_forward(hidden_states, top_k_index, top_k_weights)
            keep("authentic_routed_return", authentic)
            return authentic
        route_capture = _AuthenticRouteCaptureMode()
        route_capture.bind_route_inputs(top_k_index, top_k_weights)
        module._a30_route_capture = route_capture
        original_torch_gather = torch.gather
        from repair_api import modern_green_resident as resident_runtime

        original_accumulate = resident_runtime._sealed_builder_accumulate_routes

        def capture_authentic_accumulate(
            assembly_hidden: Any,
            routed_output: Any,
            assembly_top_k_index: Any,
            assembly_top_k_weights: Any,
            *,
            route_observer: Any = None,
        ) -> Any:
            result = original_accumulate(
                assembly_hidden, routed_output,
                assembly_top_k_index, assembly_top_k_weights,
                route_observer=route_observer,
            )
            assembly_capture["variant"] = {
                "hidden_states": assembly_hidden.detach(),
                "routed_output": routed_output.detach(),
                "top_k_index": assembly_top_k_index.detach(),
                "top_k_weights": assembly_top_k_weights.detach(),
                "final_return": result.detach(),
            }
            return result

        def capture_authentic_gather(*args: Any, **kwargs: Any) -> Any:
            result = original_torch_gather(*args, **kwargs)
            route_capture.capture_ordered(result)
            return result

        try:
            with (
                patch.object(torch, "gather", capture_authentic_gather),
                patch.object(
                    resident_runtime,
                    "_sealed_builder_accumulate_routes",
                    capture_authentic_accumulate,
                ),
                route_capture,
            ):
                authentic = original_forward(hidden_states, top_k_index, top_k_weights)
        finally:
            module.__dict__.pop("_a30_route_capture", None)
        required = {
            "authentic_w2_output": route_capture.w2_output,
            "authentic_route_weights": route_capture.route_weights,
            "authentic_token_indices": route_capture.token_indices,
            "authentic_expert_indices": route_capture.expert_indices,
            "authentic_weighted_buffer": route_capture.weighted_buffer,
            "authentic_ordered_weighted_buffer": route_capture.ordered_weighted_buffer,
        }
        if any(value is None for value in required.values()):
            missing = sorted(name for name, value in required.items() if value is None)
            raise RuntimeError("A30_AUTHENTIC_ROUTE_CAPTURE_MISSING:" + ",".join(missing))
        authentic_capture.update(required)
        keep("authentic_routed_return", authentic)
        return authentic

    experts.forward = types.MethodType(transparent_forward, experts)
    try:
        output, _attention = _run_one_layer_with_attention(engine, layer, hidden, ids)
    finally:
        if had_instance_forward:
            experts.forward = prior_instance_forward
        else:
            experts.__dict__.pop("forward", None)
    if tuple(replay_inputs) != ("hidden_states", "top_k_index", "top_k_weights"):
        raise RuntimeError("EXPERT_TRACE_TRANSPARENT_WRAPPER_NOT_CALLED")
    replay = replay_resident if resident else replay_control
    fresh_workspace_result = None
    replay_result = replay(
        experts,
        replay_inputs["hidden_states"],
        replay_inputs["top_k_index"],
        replay_inputs["top_k_weights"],
    )
    if not resident:
        fresh_workspace_result = replay_control(
            experts,
            replay_inputs["hidden_states"],
            replay_inputs["top_k_index"],
            replay_inputs["top_k_weights"],
            fresh_workspace=True,
            record_trace=False,
        )
    if any(not values for values in captured.values()):
        raise RuntimeError("EXPERT_TRACE_COVERAGE_RED")
    route_key = torch.cat(captured.pop("route_key"), dim=0)
    full_trace = {
        name: torch.cat(captured.pop(name), dim=0).cpu()
        for name in (
            "full_weighted_routed_output", "full_route_expert", "authentic_routed_return"
        )
    }
    if not resident:
        # replay_control is the accepted source loop itself: each expert's
        # projections, weighting, and index_add are temporally interleaved.
        if fresh_workspace_result is None:
            raise RuntimeError("RUN6521_FRESH_WORKSPACE_REPLAY_MISSING")
        full_trace["temporal_interleaved_return"] = replay_result.cpu()
        full_trace["fresh_workspace_return"] = fresh_workspace_result.cpu()
    if resident:
        schedules = _replay_a30_route_schedules(
            replay_inputs["hidden_states"], replay_inputs["top_k_index"],
            authentic_capture["authentic_weighted_buffer"],
        )
        authentic_capture.update({
            "a30_token_major_return": schedules["token_major"],
            "a30_expert_major_return": schedules["expert_major"],
            "a30_replayed_ordered_weighted": schedules["ordered_weighted"],
        })
        full_trace.update({name: value.cpu() for name, value in authentic_capture.items()})
    order = torch.argsort(route_key[:, 0] * hidden.reshape(-1, hidden.shape[-1]).shape[0] + route_key[:, 1])
    trace: dict[str, Any] = {"route_key": route_key[order]}
    trace.update({
        name: torch.cat(values, dim=0)[order].cpu()
        for name, values in captured.items()
    })
    trace.update(full_trace)
    if resident:
        if "variant" not in assembly_capture:
            raise RuntimeError("A30O_AUTHENTIC_RETURN_ASSEMBLY_CAPTURE_MISSING")
        assembly = assembly_capture["variant"]
        trace["routed_return_assembly"] = _routed_return_assembly_witness(
            assembly["hidden_states"], assembly["routed_output"],
            assembly["top_k_index"], assembly["top_k_weights"],
        )
        trace["routed_return_assembly"]["captured_final_return"] = _tensor_tap(
            assembly["final_return"]
        )
    selected = "variant" if resident else "control"
    if selected not in pre_gemm_capture:
        raise RuntimeError(f"PRE_GEMM_CAPTURE_MISSING:{selected}")
    trace["pre_gemm"] = pre_gemm_capture[selected]
    return output, trace


def _load_r20_grouped_decoder() -> Any:
    """Load the hash-bound decoder that materializes the sealed BF16 weight."""
    import importlib.util

    assets = Path(__file__).resolve().parent / "assets"
    path = assets / "fast_k2_grouped.py"
    expected = "5ff7e60b1b7d21abee2dbdc3202a1cf2c3787c3bd4744af34f1a9b6ace5ff361"
    if sha(path) != expected:
        raise RuntimeError("R20_DIAGNOSTIC_DECODER_HASH_MISMATCH")
    spec = importlib.util.spec_from_file_location("r20_tensor_ab_diagnostic_decoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("R20_DIAGNOSTIC_DECODER_IMPORT_REFUSED")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    return decoder


def _sealed_full_weight_projection(
    x: Any, assignments: Any, packed: Any, lut_master: Any, su: Any, sv: Any,
    *, full_weight_builder: Any,
) -> Any:
    """Match the sealed full-weight BF16 F.linear boundary expert by expert."""
    experts = int(packed.shape[0])
    order = torch.argsort(assignments, stable=True)
    inverse_order = torch.argsort(order)
    sorted_assignments = assignments[order]
    sorted_x = x[order].to(torch.bfloat16).contiguous()
    counts = torch.bincount(sorted_assignments, minlength=experts).to(torch.int64)
    offsets = torch.cat((
        torch.zeros(1, device=x.device, dtype=torch.int64), counts.cumsum(0)
    ))
    sorted_output = torch.empty(
        (x.shape[0], sv.shape[1]), device=x.device, dtype=torch.bfloat16
    )
    for expert_index in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        start = int(offsets[expert_index].item())
        stop = start + int(counts[expert_index].item())
        full_weight = full_weight_builder(
            packed[expert_index], lut_master, su[expert_index], sv[expert_index]
        )
        sorted_output[start:stop] = torch.matmul(sorted_x[start:stop], full_weight)
    # Match the canonical resident adapter in modern_green_resident.py:548-556:
    # the sealed dense builder exposes BF16 projection tensors to clamp/SwiGLU.
    return sorted_output[inverse_order]


def _concat_official_gate_up_weight(gate_weight: Any, up_weight: Any) -> Any:
    """Concatenate in official [2M,K] storage, then expose its [K,2M] view."""
    return torch.cat((gate_weight.T, up_weight.T), dim=0).contiguous().T


def _sealed_full_weight_gate_up_projection(
    x: Any,
    assignments: Any,
    packed_w1: Any,
    lut_master: Any,
    su_w1: Any,
    sv_w1: Any,
    packed_w3: Any,
    su_w3: Any,
    sv_w3: Any,
    *,
    full_weight_builder: Any,
) -> tuple[Any, Any]:
    """Match the sealed builder's one concatenated BF16 gate_up GEMM."""
    experts = int(packed_w1.shape[0])
    order = torch.argsort(assignments, stable=True)
    inverse_order = torch.argsort(order)
    sorted_assignments = assignments[order]
    sorted_x = x[order].to(torch.bfloat16).contiguous()
    counts = torch.bincount(sorted_assignments, minlength=experts).to(torch.int64)
    offsets = torch.cat((
        torch.zeros(1, device=x.device, dtype=torch.int64), counts.cumsum(0)
    ))
    gate_width = int(sv_w1.shape[1])
    gate_up_width = gate_width + int(sv_w3.shape[1])
    sorted_gate_up = torch.empty(
        (x.shape[0], gate_up_width), device=x.device, dtype=torch.bfloat16
    )
    for expert_index in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        start = int(offsets[expert_index].item())
        stop = start + int(counts[expert_index].item())
        gate_weight = full_weight_builder(
            packed_w1[expert_index], lut_master, su_w1[expert_index], sv_w1[expert_index]
        )
        up_weight = full_weight_builder(
            packed_w3[expert_index], lut_master, su_w3[expert_index], sv_w3[expert_index]
        )
        gate_up_weight = _concat_official_gate_up_weight(gate_weight, up_weight)
        sorted_gate_up[start:stop] = torch.matmul(
            sorted_x[start:stop], gate_up_weight
        )
    gate_up = sorted_gate_up[inverse_order]
    return gate_up[:, :gate_width], gate_up[:, gate_width:]


def _install_r20_full_weight_projection(engine: Any) -> dict[str, Any]:
    """Match A16's concatenated gate_up GEMM without changing provider bytes."""
    decoder = _load_r20_grouped_decoder()
    bound = []
    for layer_index, expert in sorted(engine.student.experts.items()):
        original_project = expert._project
        pending_up: dict[str, Any] = {}

        def exact_project(
            projection: str, x: Any, assignments: Any, packed: Any,
            lut_master: Any, su: Any, sv: Any,
            route_rows_per_sample: int | None = None,
            route_metadata: Any = None,
            *, _builder: Any = decoder.sealed_bf16_full_weight,
            _expert: Any = expert,
            _pending_up: dict[str, Any] = pending_up,
        ) -> Any:
            del route_rows_per_sample, route_metadata
            if projection == "w1":
                if _pending_up:
                    raise RuntimeError("R20_GATE_UP_CACHE_NOT_CONSUMED")
                gate, up = _sealed_full_weight_gate_up_projection(
                    x,
                    assignments,
                    _expert.packed_w1,
                    lut_master,
                    _expert.su_w1,
                    _expert.sv_w1,
                    _expert.packed_w3,
                    _expert.su_w3,
                    _expert.sv_w3,
                    full_weight_builder=_builder,
                )
                _pending_up.update(
                    value=up,
                    x_data_ptr=int(x.data_ptr()),
                    assignments_data_ptr=int(assignments.data_ptr()),
                )
                return gate
            if projection == "w3":
                if (
                    _pending_up.get("x_data_ptr") != int(x.data_ptr())
                    or _pending_up.get("assignments_data_ptr")
                    != int(assignments.data_ptr())
                ):
                    raise RuntimeError("R20_GATE_UP_CACHE_IDENTITY_MISMATCH")
                up = _pending_up.pop("value")
                _pending_up.pop("x_data_ptr")
                _pending_up.pop("assignments_data_ptr")
                return up
            value = _sealed_full_weight_projection(
                x, assignments, packed, lut_master, su, sv,
                full_weight_builder=_builder,
            )
            # This instance-level projection binding is installed after the
            # construction-time class adapter. Capture the active w2 result at
            # this final dispatch seam so the immutable provider's forward can
            # be accumulated with the sealed builder's BF16 index_add schedule.
            if projection == "w2" and getattr(_expert, "_sealed_capture_w2", False):
                _expert._sealed_routed_output = value
                route_capture = getattr(_expert, "_a30_route_capture", None)
                if route_capture is not None:
                    route_capture.capture_w2(value, assignments)
            return value

        expert._r20_original_project = original_project
        expert._project = exact_project
        bound.append(int(layer_index))
    return {
        "status": "BOUND_SEALED_FULL_WEIGHT_BF16_GEMM",
        "layers": bound,
        "provider_expert_sha256": ADOPTED_PROVIDER_EXPERT_SHA256,
        "decoder_sha256": "86d560f494646fd9bcdc0a2297e8fdea1afaf3362c24c33aad9111f19b512005",
    }


def _decode_r20_expert_w1(expert: Any) -> Any:
    """Decode the R20 wire with a hash-bound diagnostic already in that git pin."""
    diagnostic_decoder = _load_r20_grouped_decoder()
    return diagnostic_decoder.sealed_bf16_full_weight(
        expert.packed_w1[0], expert.plane_source.wire_lut().reshape(-1).contiguous(),
        expert.su_w1[0], expert.sv_w1[0],
    ).transpose(0, 1).contiguous()


def _call_with_grouped_mm_operation_probe(
    experts: Any, forward: Any, hidden_states: Any, top_k_index: Any,
    top_k_weights: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """Compare each authentic grouped-mm GEMM with source-shaped F.linear."""
    import transformers.integrations.moe as moe

    implementation = str(getattr(experts.config, "_experts_implementation", ""))
    if implementation != "grouped_mm":
        raise RuntimeError(f"GROUPED_MM_OPERATION_IMPLEMENTATION_RED:{implementation}")
    implementation_globals = moe.grouped_mm_experts_forward.__globals__
    original_grouped_linear = implementation_globals["_grouped_linear"]
    operations: list[dict[str, Any]] = []

    def observed_grouped_linear(
        inputs: Any, weights: Any, offsets: Any, *, bias: Any = None,
        is_transposed: bool = False,
    ) -> Any:
        grouped = original_grouped_linear(
            inputs, weights, offsets, bias=bias, is_transposed=is_transposed
        )
        if is_transposed:
            raise RuntimeError("GROUPED_MM_OPERATION_TRANSPOSED_UNSUPPORTED")
        source = torch.empty_like(grouped)
        start = 0
        for expert_index, stop_value in enumerate(offsets.detach().cpu().tolist()):
            stop = int(stop_value)
            if stop > start:
                expert_bias = None if bias is None else bias[expert_index]
                source[start:stop] = torch.nn.functional.linear(
                    inputs[start:stop], weights[expert_index], expert_bias
                )
            start = stop
        if start != int(inputs.shape[0]):
            raise RuntimeError(
                f"GROUPED_MM_OPERATION_SENTINEL_ROWS_UNSUPPORTED:{start}:{inputs.shape[0]}"
            )
        operations.append({
            "call_index": len(operations),
            "source": "transformers/integrations/moe.py:380-481 grouped_mm_experts_forward",
            "accepted_source": "transformers modeling_deepseek_v4.py:1006-1022 per-expert F.linear",
            "inputs": _tensor_tap(inputs),
            "weights": _tensor_tap(weights),
            "offsets": _tensor_tap(offsets),
            "grouped_mm_output": _tensor_tap(grouped),
            "source_flinear_output": _tensor_tap(source),
            "grouped_vs_source_exact": bool(torch.equal(grouped, source)),
        })
        return grouped

    implementation_globals["_grouped_linear"] = observed_grouped_linear
    try:
        output = forward(hidden_states, top_k_index, top_k_weights)
    finally:
        if implementation_globals.get("_grouped_linear") is not observed_grouped_linear:
            raise RuntimeError("GROUPED_MM_OPERATION_PROBE_ALIAS_DRIFT")
        implementation_globals["_grouped_linear"] = original_grouped_linear
    if len(operations) != 2:
        raise RuntimeError(f"GROUPED_MM_OPERATION_CALL_COUNT_RED:{len(operations)}")
    return output, operations


def _call_with_routed_reduction_probe(
    experts: Any, forward: Any, hidden_states: Any, top_k_index: Any,
    top_k_weights: Any,
) -> tuple[Any, dict[str, Any]]:
    """Tap the grouped provider's final token-major reshape+sum."""
    from torch.utils._python_dispatch import TorchDispatchMode

    implementation = str(getattr(experts.config, "_experts_implementation", ""))
    if implementation != "grouped_mm":
        raise RuntimeError(f"ROUTED_REDUCTION_IMPLEMENTATION_RED:{implementation}")
    captures: list[dict[str, Any]] = []

    class ReductionMode(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None,
        ) -> Any:
            del types
            call_kwargs = {} if kwargs is None else kwargs
            result = func(*args, **call_kwargs)
            if func is torch.ops.aten.sum.dim_IntList:
                source = args[0]
                dimensions = tuple(args[1])
                if source.ndim == 3 and dimensions == (1,):
                    captures.append({
                        "boundary": (
                            "transformers.integrations.moe.grouped_mm_experts_forward "
                            "weighted_out.view(num_tokens,num_top_k,hidden_dim).sum(dim=1)"
                        ),
                        "weighted_out_token_major": _tensor_tap(source),
                        "grouped_reshape_sum": _tensor_tap(result),
                    })
            return result

    with ReductionMode():
        output = forward(hidden_states, top_k_index, top_k_weights)
    if len(captures) != 1:
        raise RuntimeError(f"ROUTED_REDUCTION_CAPTURE_COUNT_RED:{len(captures)}")
    return output, captures[0]


def _run_one_layer_with_authentic_projection_control(
    engine: Any, layer: Any, hidden: Any, ids: Any,
) -> tuple[Any, dict[str, Any]]:
    """Invoke the exact bound source expert path twice on identical operands."""
    import inspect
    import types

    experts = layer.mlp.experts
    had_instance_forward = "forward" in experts.__dict__
    prior_instance_forward = experts.__dict__.get("forward")
    original_forward = experts.forward
    source_forward = inspect.unwrap(original_forward)
    captured: dict[str, Any] = {}

    def transparent_forward(
        module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any,
    ) -> Any:
        if os.environ.get("RUN6910_ROUTED_REDUCTION_AB_ONLY", "0") == "1":
            implementation = str(
                getattr(getattr(module, "config", None), "_experts_implementation", "")
            )
            if implementation == "grouped_mm":
                authentic, routed_reduction = _call_with_routed_reduction_probe(
                    module, original_forward, hidden_states, top_k_index, top_k_weights
                )
            elif implementation == "eager":
                authentic = original_forward(
                    hidden_states, top_k_index, top_k_weights
                )
                routed_reduction = {
                    "boundary": (
                        "transformers use_experts_implementation selected the original "
                        "DeepseekV4Experts.forward index_add_ source body"
                    ),
                    "selected_implementation": "eager",
                    "pre_repair_control_receipt_sha256": (
                        "4b48665f3a5aee7b3c3211f2c4d21ad1fe007fbb5bc961bac4db66a811f439fa"
                    ),
                    "pre_repair_grouped_output_sha256": (
                        "9fe707b06ed5f465b22383e1adbe34445f25267b9ea6e4676bdcb4defaeb1a9d"
                    ),
                    "pre_repair_source_output_sha256": (
                        "0c5e981a0189cc1b01e5b11cbd4579335499e80e238400c4cde2483cd14783ec"
                    ),
                    "current_source_index_add": _tensor_tap(authentic),
                }
            else:
                raise RuntimeError(
                    f"ROUTED_REDUCTION_REPAIR_IMPLEMENTATION_RED:{implementation}"
                )
            grouped_mm_operations = []
        elif os.environ.get("RUN6873_GROUPED_MM_OPERATION_COMPARATOR_ONLY", "0") == "1":
            authentic, grouped_mm_operations = _call_with_grouped_mm_operation_probe(
                module, original_forward, hidden_states, top_k_index, top_k_weights
            )
            routed_reduction = {}
        else:
            authentic = original_forward(hidden_states, top_k_index, top_k_weights)
            grouped_mm_operations = []
            routed_reduction = {}
        immediate_duplicate = original_forward(hidden_states, top_k_index, top_k_weights)
        if inspect.ismethod(source_forward):
            undecorated_source = source_forward(
                hidden_states, top_k_index, top_k_weights
            )
        else:
            undecorated_source = source_forward(
                module, hidden_states, top_k_index, top_k_weights
            )
        captured.update(
            hidden_states=hidden_states.detach(),
            top_k_index=top_k_index.detach(),
            top_k_weights=top_k_weights.detach(),
            authentic=authentic.detach(),
            replay=immediate_duplicate.detach(),
            undecorated_source=undecorated_source.detach(),
            grouped_mm_operations=grouped_mm_operations,
            routed_reduction=routed_reduction,
        )
        return authentic

    experts.forward = types.MethodType(transparent_forward, experts)
    try:
        output, _attention = _run_one_layer_with_attention(engine, layer, hidden, ids)
        post_layer_return = original_forward(
            captured["hidden_states"], captured["top_k_index"], captured["top_k_weights"]
        )
    finally:
        if had_instance_forward:
            experts.forward = prior_instance_forward
        else:
            experts.__dict__.pop("forward", None)
    required = (
        "hidden_states", "top_k_index", "top_k_weights", "authentic", "replay",
        "undecorated_source", "grouped_mm_operations", "routed_reduction",
    )
    if tuple(captured) != required:
        raise RuntimeError("AUTHENTIC_SOURCE_PROJECTION_CONTROL_MISSING")
    exact = torch.equal(captured["authentic"], captured["replay"])
    context_exact = torch.equal(captured["authentic"], post_layer_return)
    source_dispatch_exact = torch.equal(
        captured["authentic"], captured["undecorated_source"]
    )
    return output, {
        "status": (
            "AUTHENTIC_SOURCE_PROJECTION_CONTROL_EXACT"
            if exact else "AUTHENTIC_SOURCE_PROJECTION_CONTROL_RED"
        ),
        "instrument_control_self_compare_exact": exact,
        "source_expert_invocation_count": 2,
        "one_variable": "none: immediate duplicate of the exact bound authentic source expert invocation",
        "accepted_source": "transformers modeling_deepseek_v4.py:1006-1022 DeepseekV4Experts.forward",
        "materialization_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:456-473",
        "provider_dispatch_binding": getattr(engine, "experts_dispatch_binding", None),
        "hidden_states": _tensor_tap(captured["hidden_states"]),
        "top_k_index": _tensor_tap(captured["top_k_index"]),
        "top_k_weights": _tensor_tap(captured["top_k_weights"]),
        "authentic_projection_path_return": _tensor_tap(captured["authentic"]),
        "immediate_duplicate_projection_path_return": _tensor_tap(captured["replay"]),
        "post_layer_source_projection_path_return": _tensor_tap(post_layer_return),
        "undecorated_source_body_return": _tensor_tap(captured["undecorated_source"]),
        "source_implementation_dispatch": {
            "status": (
                "SOURCE_IMPLEMENTATION_DISPATCH_PARITY"
                if source_dispatch_exact else "SOURCE_IMPLEMENTATION_DISPATCH_LOCALIZED"
            ),
            "one_variable": (
                "@use_experts_implementation decorated dispatch versus its exact "
                "undecorated DeepseekV4Experts.forward source body"
            ),
            "decorated_vs_undecorated_exact": source_dispatch_exact,
            "configured_implementation": str(
                getattr(getattr(experts, "config", None), "_experts_implementation", None)
            ),
            "control": "two decorated dispatch invocations self-compare exactly",
            "kill_gate": (
                "authorize localization only when decorated duplicate is exact and "
                "undecorated source body differs"
            ),
            "repair_authorized": False,
        },
        "grouped_mm_operation_comparator": {
            "status": (
                "GROUPED_MM_OPERATION_LOCALIZED"
                if captured["grouped_mm_operations"]
                and any(
                    not operation["grouped_vs_source_exact"]
                    for operation in captured["grouped_mm_operations"]
                )
                else "GROUPED_MM_OPERATION_PARITY"
            ),
            "one_variable": (
                "authentic grouped_mm GEMM versus per-expert source F.linear "
                "on identical operands"
            ),
            "control": "decorated grouped_mm duplicate return is byte-exact",
            "operations": captured["grouped_mm_operations"],
            "repair_authorized": False,
        },
        "post_second_gemm_routed_reduction": {
            "status": (
                "ROUTED_REDUCTION_REPAIR_EXACT"
                if captured["routed_reduction"].get("selected_implementation") == "eager"
                and source_dispatch_exact
                else "ROUTED_REDUCTION_REPAIR_MISMATCH"
                if captured["routed_reduction"].get("selected_implementation") == "eager"
                else "ROUTED_REDUCTION_PARITY"
                if captured["routed_reduction"] and source_dispatch_exact
                else "ROUTED_REDUCTION_LOCALIZED"
                if captured["routed_reduction"]
                else "ROUTED_REDUCTION_NOT_RUN"
            ),
            "one_variable": (
                "grouped token-major reshape+sum versus undecorated source index_add_ "
                "after sealed grouped_mm operation parity"
            ),
            "control": "instrumented decorated grouped_mm return self-compares byte-exactly",
            "selected_dispatch": captured["routed_reduction"],
            "undecorated_source_index_add": _tensor_tap(captured["undecorated_source"]),
            "grouped_vs_undecorated_exact": source_dispatch_exact,
            "max_abs_delta": float(
                (captured["authentic"].float() - captured["undecorated_source"].float())
                .abs().max().item()
            ),
            "repair_authorized": bool(
                captured["routed_reduction"]
                and captured["routed_reduction"].get("selected_implementation") != "eager"
                and not source_dispatch_exact
            ),
        },
        "source_caller_context": {
            "status": (
                "SOURCE_CALLER_CONTEXT_PARITY"
                if context_exact else "SOURCE_CALLER_CONTEXT_LOCALIZED"
            ),
            "one_variable": (
                "exact bound DeepseekV4Experts.forward inside authentic layer MLP caller "
                "versus immediately after complete layer return"
            ),
            "inside_vs_post_layer_exact": context_exact,
            "repair_authorized": False,
        },
    }


def _sealed_authentic_source_projection_control(
    engine: Any, *, window: int, root: Path, rank: int, pin: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run only the repaired known-equal source-projection instrument."""
    from repair_api.sealed_pre_forward import _prepare_exact_modules

    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    ids = prepared["ids"][window]
    local: dict[str, Any] = {"rank": rank}
    control_binding = None
    known_control_hash = "11cc07869ffcf71c39699e5631fa352cdb3aba52a003b04b659ceb5cfa4c0662"
    with torch.no_grad():
        if rank == 0:
            embeddings = engine.student.model.model.embed_tokens(ids)
            hidden = embeddings.unsqueeze(2).expand(
                -1, -1, engine.student.config.hc_mult, -1
            ).contiguous()
            import gc
            for layer_index in range(1, 21):
                engine.student.model.model.layers[layer_index] = torch.nn.Identity()
                engine.student.experts.pop(layer_index, None)
                engine.student.sources.pop(layer_index, None)
            gc.collect()
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if int(free_bytes) < (18 << 30):
                raise RuntimeError(f"EXPERT_TRACE_GPU_HEADROOM_RED:{free_bytes}:{18 << 30}")
            torch.cuda.set_per_process_memory_fraction(0.75)
            builder, planesource_module, control_binding = _prepare_exact_modules(
                task=TASK, rank=rank, root=root, config=engine.config,
                checkpoint=checkpoint_path,
            )
            planes = planesource_module.PlaneSource(str(root / "SEALED_PRE_CONTRACT.json"))
            control_sd = builder.build_layer_sd(
                0, engine.student.wm, engine.student.get_tensor, "planes", planes
            )
            from transformers import AutoModelForCausalLM
            with torch.device("meta"):
                control_model = AutoModelForCausalLM.from_config(
                    engine.student.config, attn_implementation="eager"
                )
            control_model.eval()
            control_layer = builder.materialize_layer(
                control_model, 0, control_sd, engine.student.config
            )
            unmodified, _attention = _run_one_layer_with_attention(
                engine, control_layer, hidden.clone(), ids
            )
            unmodified_tap = _tensor_tap(unmodified)
            if unmodified_tap["sha256"] != known_control_hash:
                raise RuntimeError(
                    "EXPERT_TRACE_UNMODIFIED_CONTROL_RED:" + unmodified_tap["sha256"]
                )
            instrumented, projection_control = _run_one_layer_with_authentic_projection_control(
                engine, control_layer, hidden.clone(), ids
            )
            transparent = torch.equal(unmodified, instrumented)
            if not transparent:
                raise RuntimeError(
                    "AUTHENTIC_SOURCE_PROJECTION_INSTRUMENT_TRANSPARENCY_RED:"
                    f"{unmodified_tap['sha256']}:{_tensor_tap(instrumented)['sha256']}"
                )
            if not projection_control["instrument_control_self_compare_exact"]:
                raise RuntimeError("AUTHENTIC_SOURCE_PROJECTION_CONTROL_RED")
            local.update(
                unmodified_control=unmodified_tap,
                instrumented_control=_tensor_tap(instrumented),
                instrument_transparent=True,
                source_projection_control=projection_control,
            )
            del control_model, control_layer, control_sd
            torch.cuda.empty_cache()
    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    engine.dist.all_gather_object(gathered, local)
    control = gathered[0].get("source_projection_control")
    if not isinstance(control, dict):
        raise RuntimeError("AUTHENTIC_SOURCE_PROJECTION_CONTROL_RECEIPT_MISSING")
    dispatch_probe = os.environ.get(
        "RUN6524_SOURCE_IMPLEMENTATION_DISPATCH_ONLY", "0"
    ) == "1"
    operation_probe = os.environ.get(
        "RUN6873_GROUPED_MM_OPERATION_COMPARATOR_ONLY", "0"
    ) == "1"
    reduction_probe = os.environ.get("RUN6910_ROUTED_REDUCTION_AB_ONLY", "0") == "1"
    receipt = {
        "schema": (
            "banana-smasher-post-second-gemm-routed-reduction-v1"
            if reduction_probe
            else "banana-smasher-grouped-mm-operation-comparator-v1"
            if operation_probe
            else "banana-smasher-source-implementation-dispatch-v1"
            if dispatch_probe
            else "banana-smasher-authentic-source-projection-control-v1"
        ),
        "status": control["status"],
        "task_id": TASK, "canonical_code_commit": pin,
        "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
        "window": window, "layer": 0,
        "control_source_binding": control_binding if rank == 0 else None,
        "unmodified_control": gathered[0]["unmodified_control"],
        "instrumented_control": gathered[0]["instrumented_control"],
        "instrument_transparent": gathered[0]["instrument_transparent"],
        "source_projection_control": control,
        "repair_authorized": False,
        "successor_step": "read authentic source and launch one new source-backed comparator",
        "created_unix": time.time(),
    }
    receipt_stem = (
        "POST_SECOND_GEMM_ROUTED_REDUCTION"
        if reduction_probe else "GROUPED_MM_OPERATION_COMPARATOR"
        if operation_probe else "SOURCE_IMPLEMENTATION_DISPATCH"
        if dispatch_probe else "AUTHENTIC_SOURCE_PROJECTION_CONTROL"
    )
    path = root / "receipts" / f"{receipt_stem}.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt


def _sealed_runtime_expert_trace_ab(
    engine: Any, *, window: int, root: Path, rank: int, pin: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Localize the first resident divergence inside one authentic expert layer."""
    from repair_api.sealed_pre_forward import _prepare_exact_modules

    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    ids = prepared["ids"][window]
    local: dict[str, Any] = {"rank": rank, "boundaries": {}}
    control_binding = None
    known_control_hash = "11cc07869ffcf71c39699e5631fa352cdb3aba52a003b04b659ceb5cfa4c0662"
    with torch.no_grad():
        if rank == 0:
            embeddings = engine.student.model.model.embed_tokens(ids)
            hidden = embeddings.unsqueeze(2).expand(
                -1, -1, engine.student.config.hc_mult, -1
            ).contiguous()
            variant_layer = engine.student.model.model.layers[0]
            import gc
            # Keep the authentic resident layer-0 object for step 3, but retire
            # every other local layer before materializing the dense control.
            # Attempt105 ran the resident wrapper before proving its control;
            # that ordering made a wrapper RED scientifically ambiguous.
            for layer_index in range(1, 21):
                engine.student.model.model.layers[layer_index] = torch.nn.Identity()
                engine.student.experts.pop(layer_index, None)
                engine.student.sources.pop(layer_index, None)
            gc.collect()
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if int(free_bytes) < (18 << 30):
                raise RuntimeError(f"EXPERT_TRACE_GPU_HEADROOM_RED:{free_bytes}:{18 << 30}")
            torch.cuda.set_per_process_memory_fraction(0.75)
            builder, planesource_module, control_binding = _prepare_exact_modules(
                task=TASK, rank=rank, root=root, config=engine.config,
                checkpoint=checkpoint_path,
            )
            planes = planesource_module.PlaneSource(str(root / "SEALED_PRE_CONTRACT.json"))
            control_sd = builder.build_layer_sd(
                0, engine.student.wm, engine.student.get_tensor, "planes", planes
            )
            from transformers import AutoModelForCausalLM
            with torch.device("meta"):
                control_model = AutoModelForCausalLM.from_config(
                    engine.student.config, attn_implementation="eager"
                )
            control_model.eval()
            control_layer = builder.materialize_layer(
                control_model, 0, control_sd, engine.student.config
            )
            # Step 1: the unmodified sealed control is the truth gate.  No trace
            # wrapper has been attached or executed before this point.
            control_hidden, _control_attention = _run_one_layer_with_attention(
                engine, control_layer, hidden.clone(), ids
            )
            control_tap = _tensor_tap(control_hidden)
            control_flat66 = float(control_hidden.reshape(-1)[66].item())
            if control_tap["sha256"] != known_control_hash or control_flat66 != 0.00048828125:
                raise RuntimeError(
                    "EXPERT_TRACE_UNMODIFIED_CONTROL_RED:"
                    f"{control_tap['sha256']}:{control_flat66}"
                )
            progress_path = root / "ATTEMPT106_PROGRESS.json"
            atomic(progress_path, {
                "schema": "banana-smasher-attempt106-progress-v1",
                "task_id": TASK, "canonical_code_commit": pin,
                "completed_steps": [1],
                "step1_unmodified_control": {
                    "status": "EXACT", "post_expert": control_tap,
                    "flat66": control_flat66,
                },
                "updated_unix": time.time(),
            })

            # Step 2: attach the trace wrapper only after the original forward is
            # sealed, then require byte identity before admitting any tap.
            traced_control_hidden, control_trace = _run_one_layer_with_expert_trace(
                engine, control_layer, hidden.clone(), ids, resident=False
            )
            traced_control_tap = _tensor_tap(traced_control_hidden)
            if traced_control_tap != control_tap or not torch.equal(
                traced_control_hidden, control_hidden
            ):
                raise RuntimeError(
                    "EXPERT_TRACE_WRAPPER_TRANSPARENCY_RED:"
                    f"{control_tap['sha256']}:{traced_control_tap['sha256']}"
                )
            atomic(progress_path, {
                "schema": "banana-smasher-attempt106-progress-v1",
                "task_id": TASK, "canonical_code_commit": pin,
                "completed_steps": [1, 2],
                "step1_unmodified_control": {
                    "status": "EXACT", "post_expert": control_tap,
                    "flat66": control_flat66,
                },
                "step2_traced_control": {
                    "status": "BYTE_IDENTICAL", "post_expert": traced_control_tap,
                },
                "updated_unix": time.time(),
            })

            del control_model, control_sd
            torch.cuda.empty_cache()
            # Step 3 starts only after both control gates.  It uses the retained
            # canonical resident layer and the now-admitted normalized taps.
            variant_hidden, variant_trace = _run_one_layer_with_expert_trace(
                engine, variant_layer, hidden.clone(), ids, resident=True
            )
            control_a27 = _adjudicate_a27_active_rows({
                "expert": 204,
                **{
                    name: _tensor_tap(control_trace[name])
                    for name in ("route_key", "gate", "up", "activated", "w2_down")
                },
            })
            if control_a27["first_unequal_boundary"] is not None:
                raise RuntimeError(
                    "PRE_GEMM_A27_CONTROL_REPRODUCTION_RED:"
                    + str(control_a27["first_unequal_boundary"])
                )
            local["pre_gemm_comparison"] = _compare_pre_gemm_witnesses(
                control_trace["pre_gemm"],
                variant_trace["pre_gemm"],
                control_invocation=control_trace["pre_gemm"]["invocation"],
                variant_invocation=variant_trace["pre_gemm"]["invocation"],
            )
            local["pre_gemm_comparison"]["a27_control_reproduction"] = control_a27
            boundary_order = (
                "route_key", "gate", "up", "activated", "w2_down",
                "weighted_routed_output", "full_weighted_routed_output",
                "per_slot_accumulation", "authentic_routed_return",
                "post_expert_hidden",
            )
            local["boundaries"] = {
                name: _compare_tensor_boundary(control_trace[name], variant_trace[name])
                for name in boundary_order[:-1]
            }
            local["boundaries"]["post_expert_hidden"] = _compare_tensor_boundary(
                control_hidden, variant_hidden
            )
            for required_exact in (
                "route_key", "full_weighted_routed_output", "per_slot_accumulation"
            ):
                if not local["boundaries"][required_exact]["exact"]:
                    raise RuntimeError(
                        "A30O_ASSEMBLY_KILL_GATE_RED:" + required_exact
                    )
            routed_return_assembly = variant_trace["routed_return_assembly"]
            replay_weighted = _tensor_tap(
                variant_trace["full_weighted_routed_output"]
            )
            control_return = _tensor_tap(
                control_trace["authentic_routed_return"]
            )
            variant_return = _tensor_tap(
                variant_trace["authentic_routed_return"]
            )
            slot_major_exact = (
                routed_return_assembly["weighted_routes_slot_major"]["sha256"]
                == replay_weighted["sha256"]
            )
            captured_final_exact = (
                routed_return_assembly["final_return"]["sha256"]
                == routed_return_assembly["captured_final_return"]["sha256"]
                == variant_return["sha256"]
            )
            control_final_exact = (
                routed_return_assembly["final_return"]["sha256"]
                == control_return["sha256"]
            )
            first_unequal_assembly_operation = (
                "routed_output_route_weight_token_major_alignment"
                if not slot_major_exact else
                "captured_helper_return_or_alias_lifetime"
                if not captured_final_exact else
                "ascending_expert_index_add_order"
                if not control_final_exact else None
            )
            local["routed_return_assembly"] = {
                **routed_return_assembly,
                "control_return": control_return,
                "variant_return": variant_return,
                "replay_full_weighted_slot_major": replay_weighted,
                "comparisons": {
                    "weighted_slot_major_vs_exact_replay": slot_major_exact,
                    "reconstructed_vs_captured_variant_return": captured_final_exact,
                    "reconstructed_vs_control_return": control_final_exact,
                },
                "first_unequal_assembly_operation": first_unequal_assembly_operation,
            }
            if tuple(local["boundaries"]) != boundary_order:
                raise RuntimeError("EXPERT_TRACE_BOUNDARY_ORDER_RED")
            local_first_divergent = next(
                (name for name in boundary_order if not local["boundaries"][name]["exact"]),
                None,
            )
            full_difference = local["boundaries"]["full_weighted_routed_output"].get(
                "first_difference"
            )
            local["first_weighted_divergent_expert"] = (
                int(variant_trace["full_route_expert"][full_difference["index"][0]].item())
                if full_difference and full_difference.get("kind") == "value" else None
            )
            authentic_weighted = variant_trace["authentic_weighted_buffer"]
            route_tokens = int(variant_trace["authentic_routed_return"].shape[0])
            route_slots = int(authentic_weighted.shape[0]) // route_tokens
            authentic_slot_major = (
                authentic_weighted.reshape(
                    route_tokens, route_slots, authentic_weighted.shape[-1]
                ).transpose(0, 1).reshape(-1, authentic_weighted.shape[-1])
            )
            a30_comparisons = {
                "weighted_rows_vs_normalized_slot_major": _compare_tensor_boundary(
                    variant_trace["full_weighted_routed_output"], authentic_slot_major
                ),
                "captured_vs_replayed_ordered_weighted": _compare_tensor_boundary(
                    variant_trace["authentic_ordered_weighted_buffer"],
                    variant_trace["a30_replayed_ordered_weighted"],
                ),
                "token_major_vs_control_return": _compare_tensor_boundary(
                    control_trace["authentic_routed_return"],
                    variant_trace["a30_token_major_return"],
                ),
                "expert_major_vs_control_return": _compare_tensor_boundary(
                    control_trace["authentic_routed_return"],
                    variant_trace["a30_expert_major_return"],
                ),
                "token_major_vs_variant_return": _compare_tensor_boundary(
                    variant_trace["authentic_routed_return"],
                    variant_trace["a30_token_major_return"],
                ),
                "expert_major_vs_variant_return": _compare_tensor_boundary(
                    variant_trace["authentic_routed_return"],
                    variant_trace["a30_expert_major_return"],
                ),
            }
            route_tokens = int(variant_trace["authentic_routed_return"].shape[0])
            source_top_k_index = variant_trace["authentic_expert_indices"].reshape(
                route_tokens, -1
            )
            source_return_assembly = _replay_source_return_assemblies(
                control_trace["authentic_routed_return"].new_zeros(
                    control_trace["authentic_routed_return"].shape
                ),
                source_top_k_index,
                variant_trace["full_weighted_routed_output"],
                control_trace["authentic_routed_return"],
                variant_trace["authentic_routed_return"],
            )
            local["source_return_assembly"] = source_return_assembly
            local["temporal_interleaving"] = _adjudicate_temporal_interleaving(
                control_trace["authentic_routed_return"],
                control_trace["temporal_interleaved_return"],
                variant_trace["a30_expert_major_return"],
            )
            local["source_workspace_lifetime"] = _adjudicate_source_workspace_lifetime(
                control_trace["authentic_routed_return"],
                control_trace["temporal_interleaved_return"],
                control_trace["fresh_workspace_return"],
            )
            local["a30_authentic_route_capture"] = {
                "w2_output": _tensor_tap(variant_trace["authentic_w2_output"]),
                "route_weights": _tensor_tap(variant_trace["authentic_route_weights"]),
                "token_indices": _tensor_tap(variant_trace["authentic_token_indices"]),
                "expert_indices": _tensor_tap(variant_trace["authentic_expert_indices"]),
                "weighted_buffer": _tensor_tap(authentic_weighted),
                "ordered_weighted_buffer": _tensor_tap(
                    variant_trace["authentic_ordered_weighted_buffer"]
                ),
                "comparisons": a30_comparisons,
            }
            mechanism = {
                "dtype": "torch.bfloat16 projection outputs with float32 route weights",
                "layout": "trace-normalized slot-major/token-major route keys",
                "accumulation": (
                    "sealed dense per-expert F.linear plus ascending-expert BF16 index_add "
                    "versus resident packed grouped projection plus ascending-expert BF16 index_add"
                ),
                "first_divergent_boundary": local_first_divergent,
            }
            atomic(progress_path, {
                "schema": "banana-smasher-attempt106-progress-v1",
                "task_id": TASK, "canonical_code_commit": pin,
                "completed_steps": [1, 2, 3],
                "step1_unmodified_control": {
                    "status": "EXACT", "post_expert": control_tap,
                    "flat66": control_flat66,
                },
                "step2_traced_control": {
                    "status": "BYTE_IDENTICAL", "post_expert": traced_control_tap,
                },
                "step3_resident_comparison": {
                    "status": "EXACT" if local_first_divergent is None else "DIVERGENT",
                    "first_divergent_boundary": local_first_divergent,
                    "mechanism": mechanism,
                    "routed_return_assembly": routed_return_assembly,
                },
                "updated_unix": time.time(),
            })
            local["mechanism"] = mechanism
            del control_layer, control_trace, variant_trace
            torch.cuda.empty_cache()
    gathered: list[Any] = [None, None]
    engine.dist.all_gather_object(gathered, local)
    boundaries = gathered[0]["boundaries"]
    boundary_order = (
        "route_key", "gate", "up", "activated", "w2_down",
        "weighted_routed_output", "full_weighted_routed_output",
        "per_slot_accumulation", "authentic_routed_return",
        "post_expert_hidden",
    )
    first_divergent = next(
        (name for name in boundary_order if not boundaries[name]["exact"]), None
    )
    pre_gemm = gathered[0].get("pre_gemm_comparison")
    if not isinstance(pre_gemm, dict):
        raise RuntimeError("PRE_GEMM_COMPARISON_MISSING")
    routed_return_assembly = gathered[0].get("routed_return_assembly")
    if not isinstance(routed_return_assembly, dict):
        raise RuntimeError("A30O_ROUTED_RETURN_ASSEMBLY_WITNESS_MISSING")
    source_return_assembly = gathered[0].get("source_return_assembly")
    if not isinstance(source_return_assembly, dict):
        raise RuntimeError("RUN6519_SOURCE_RETURN_ASSEMBLY_MISSING")
    temporal_interleaving = gathered[0].get("temporal_interleaving")
    if not isinstance(temporal_interleaving, dict):
        raise RuntimeError("RUN6520_TEMPORAL_INTERLEAVING_MISSING")
    source_workspace_lifetime = gathered[0].get("source_workspace_lifetime")
    if not isinstance(source_workspace_lifetime, dict):
        raise RuntimeError("RUN6521_SOURCE_WORKSPACE_LIFETIME_MISSING")
    first_assembly_operation = source_return_assembly.get(
        "first_assembly_operation_divergence"
    ) or routed_return_assembly.get("first_unequal_assembly_operation")
    candidate = {
        "status": (
            "A30O_AUTHENTIC_RETURN_ASSEMBLY_PARITY"
            if first_assembly_operation is None
            else "A30O_AUTHENTIC_RETURN_ASSEMBLY_DIVERGENCE_LOCALIZED"
        ),
        "first_unequal_assembly_operation": first_assembly_operation,
        "pre_gemm_prior_disposition": "MULTIPLE_GEMM_EXECUTION_MECHANISMS_PLAUSIBLE_STOP",
        "repair_authorized": False,
    }
    receipt = {
        "schema": "banana-smasher-sealed-runtime-expert-trace-ab-v1",
        "status": "EXACT_EXPERT_TRACE_PARITY" if first_divergent is None else "EXPERT_TRACE_DIVERGENCE_LOCALIZED",
        "task_id": TASK, "canonical_code_commit": pin,
        "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
        "window": window, "layer": 0,
        "control_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:593-643 + repair_api/assets/official_local_planesource.py:592-624",
        "variant_source": f"repair_api/modern_green_resident.py resident reconstruction at {pin}",
        "control_source_binding": control_binding if rank == 0 else None,
        "boundary_order": list(boundary_order), "boundaries": boundaries,
        "first_divergent_boundary": first_divergent,
        "first_weighted_divergent_expert": gathered[0].get("first_weighted_divergent_expert"),
        "a30_authentic_route_capture": gathered[0].get("a30_authentic_route_capture"),
        "pre_gemm_comparison": pre_gemm,
        "routed_return_assembly": routed_return_assembly,
        "source_return_assembly": source_return_assembly,
        "temporal_interleaving": temporal_interleaving,
        "source_workspace_lifetime": source_workspace_lifetime,
        "assembly_control_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:456-473 + accepted DeepseekV4Experts torch.where/index_add forward",
        "assembly_variant_source": "repair_api/modern_green_resident.py:520-543 (_sealed_builder_accumulate_routes)",
        "source_backed_repair_candidate": candidate,
        "mechanism": gathered[0].get("mechanism"),
        "created_unix": time.time(),
    }
    path = root / "receipts" / f"SEALED_RUNTIME_EXPERT_TRACE_AB.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt


def _run_separate_gate_up_geometry(
    engine: Any, layer: Any, hidden: Any, ids: Any
) -> tuple[Any, Any]:
    """Run the sealed dense layer with only gate/up GEMM geometry split."""
    import types
    import torch.nn.functional as F

    experts = layer.mlp.experts
    had_instance_forward = "forward" in experts.__dict__
    prior_instance_forward = experts.__dict__.get("forward")

    def separate_forward(
        module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any
    ) -> Any:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(
                top_k_index, num_classes=module.num_experts
            ).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_index_row in hit:
            expert_index = expert_index_row[0]
            if expert_index == module.num_experts:
                continue
            top_k_pos, token_index = torch.where(mask[expert_index])
            routed = hidden_states[token_index]
            gate_up_weight = module.gate_up_proj[expert_index]
            split = int(module.intermediate_dim)
            # Sole variable: two [2048,4096] BF16 GEMMs replace the sealed
            # concatenated [4096,4096] BF16 gate_up GEMM. Weights, routing,
            # clamp/SwiGLU, down projection, accumulation, and layout are fixed.
            gate = F.linear(routed, gate_up_weight[:split])
            up = F.linear(routed, gate_up_weight[split:])
            current = module._apply_gate(torch.cat((gate, up), dim=-1))
            current = F.linear(current, module.down_proj[expert_index])
            current = current * top_k_weights[token_index, top_k_pos, None]
            final.index_add_(0, token_index, current.to(final.dtype))
        return final

    experts.forward = types.MethodType(separate_forward, experts)
    try:
        return _run_one_layer_with_attention(engine, layer, hidden, ids)
    finally:
        if had_instance_forward:
            experts.forward = prior_instance_forward
        else:
            experts.__dict__.pop("forward", None)


_A27_ACTIVE_ROW_CONTROL = {
    "route_key": "77f7c2dea33f31f549f1170d41b90c6b620e8db99e70807aab5367d50b3ae1ae",
    "gate": "e6a418dbb208c4e83fcb25aa55f4246ff1387aa5abdfa2c15835d10764221e6c",
    "up": "c2350b61f0d24b0b2b2b812df20e5693ebfc699b4afab74f19120c7dba220463",
    "activated": "377fedca101670cf7d35a51579aec7a383d473e98ad21e4bc692c3a1b7b7a150",
    "w2_down": "07e386fb4a72f7ef31f9603e93c26f3bf758e03aec886720875c3516f9447e31",
}


def _adjudicate_a27_active_rows(witness: Any) -> dict[str, Any]:
    """Compare one product capture only to immutable accepted A27 truth."""
    if not isinstance(witness, dict) or witness.get("expert") != 204:
        raise RuntimeError("A27_ALIGNED_ACTIVE_ROW_EXPERT_MISMATCH")
    boundaries: dict[str, Any] = {}
    for name, control_sha in _A27_ACTIVE_ROW_CONTROL.items():
        observed = witness.get(name)
        variant_sha = observed.get("sha256") if isinstance(observed, dict) else None
        if not isinstance(variant_sha, str) or len(variant_sha) != 64:
            raise RuntimeError(f"A27_ALIGNED_ACTIVE_ROW_WITNESS_MISSING:{name}")
        boundaries[name] = {
            "control_sha256": control_sha,
            "variant_sha256": variant_sha,
            "exact": variant_sha == control_sha,
        }
    order = tuple(_A27_ACTIVE_ROW_CONTROL)
    first_unequal = next(
        (name for name in order if not boundaries[name]["exact"]), None
    )
    return {
        "status": (
            "A27_ALIGNED_ACTIVE_ROW_PARITY"
            if first_unequal is None
            else "A27_ALIGNED_ACTIVE_ROW_DIVERGENCE_LOCALIZED"
        ),
        "control": {
            "task_id": TASK,
            "canonical_code_commit": "b7f1250576d9551f58477bcbd2fce3fc620fd417",
            "rank0_receipt_sha256": "8af1f8d3766215177ae9766fc8d7ca027427f51aae4a0313f6ab7de63117ecbb",
            "rank1_receipt_sha256": "329b37dca1240ae7e605799fa515497cfd2f10eb45474bf049375e9c7ea36b9d",
            "rank_agreement": True,
            "source": "accepted A27_TRACE active expert 204 rows",
        },
        "boundary_order": list(order),
        "boundaries": boundaries,
        "first_unequal_boundary": first_unequal,
    }


def _sealed_runtime_tensor_ab(
    engine: Any,
    *,
    window: int,
    root: Path,
    rank: int,
    pin: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Compare one exact sealed-builder layer with the resident reconstruction."""
    from repair_api.sealed_pre_forward import _prepare_exact_modules

    prepared = engine.preload_validation((window,), engine.config["validation_teacher_root"])
    ids = prepared["ids"][window]
    shape = (
        1, ids.shape[1], int(engine.student.config.hc_mult),
        int(engine.student.config.hidden_size),
    )
    local: dict[str, Any] = {"rank": rank, "boundaries": {}}
    control_binding = None
    control_known_hash = "11cc07869ffcf71c39699e5631fa352cdb3aba52a003b04b659ceb5cfa4c0662"
    variant_known_hash = "6c4dc981b8ece4f5e1fd81573a289a48c2ae742efd20546f8d83f594c4e12f1d"
    with torch.no_grad():
        if rank == 0:
            embeddings = engine.student.model.model.embed_tokens(ids)
            hidden = embeddings.unsqueeze(2).expand(
                -1, -1, engine.student.config.hc_mult, -1
            ).contiguous()
            variant_layer = engine.student.model.model.layers[0]
            # Retire the resident layer stack before materializing the sealed
            # dense control, while retaining the exact canonical layer-0
            # provider object that the repaired public runtime actually calls.
            import gc
            for layer_index in range(1, 21):
                engine.student.model.model.layers[layer_index] = torch.nn.Identity()
                engine.student.experts.pop(layer_index, None)
                engine.student.sources.pop(layer_index, None)
            gc.collect()
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            required_free = 18 << 30
            if int(free_bytes) < required_free:
                raise RuntimeError(f"SEALED_RUNTIME_TENSOR_AB_GPU_HEADROOM_RED:{free_bytes}:{required_free}")
            torch.cuda.set_per_process_memory_fraction(0.75)
            builder, planesource_module, control_binding = _prepare_exact_modules(
                task=TASK, rank=rank, root=root, config=engine.config,
                checkpoint=checkpoint_path,
            )
            planes = planesource_module.PlaneSource(str(root / "SEALED_PRE_CONTRACT.json"))
            control_sd = builder.build_layer_sd(0, engine.student.wm, engine.student.get_tensor, "planes", planes)
            from transformers import AutoModelForCausalLM

            with torch.device("meta"):
                control_model = AutoModelForCausalLM.from_config(
                    engine.student.config, attn_implementation="eager"
                )
            control_model.eval()
            control_layer = builder.materialize_layer(
                control_model, 0, control_sd, engine.student.config
            )
            control_hidden, control_attention = _run_one_layer_with_attention(
                engine, control_layer, hidden.clone(), ids
            )
            control_tap = _tensor_tap(control_hidden)
            control_first = float(control_hidden.reshape(-1)[66].item())
            if control_tap["sha256"] != control_known_hash or control_first != 0.00048828125:
                raise RuntimeError(
                    f"SEALED_CONTROL_REPRODUCTION_RED:{control_tap['sha256']}:{control_first}"
                )
            variant_hidden, variant_attention = _run_one_layer_with_attention(
                engine, variant_layer, hidden.clone(), ids
            )
            torch.cuda.synchronize()
            local["projection_runtime_witness"] = (
                engine.sealed_gate_up_runtime_witness(require_activation=True)
            )
            local["boundaries"] = {
                "input_tensor": _compare_tensor_boundary(hidden, hidden.clone()),
                "attention_tensor": _compare_tensor_boundary(control_attention, variant_attention),
                "post_expert_hidden": _compare_tensor_boundary(control_hidden, variant_hidden),
            }
            engine.dist.send(variant_hidden.contiguous(), dst=1)
            engine.dist.send(control_hidden.contiguous(), dst=1)
            del control_layer, control_model, control_sd
            torch.cuda.empty_cache()
        else:
            variant_hidden = torch.empty(shape, dtype=torch.bfloat16, device=engine.student.device)
            control_hidden = torch.empty_like(variant_hidden)
            engine.dist.recv(variant_hidden, src=0)
            engine.dist.recv(control_hidden, src=0)

            def readout(current: Any) -> Any:
                hc = engine.student.model.model.hc_head(current)
                final = engine.student.model.model.norm(hc)
                return engine.student.model.lm_head(final[0, :1024].to(torch.bfloat16)).float()

            variant_logits = readout(variant_hidden).cpu()
            torch.cuda.empty_cache()
            control_logits = readout(control_hidden).cpu()
            torch.cuda.synchronize()
            local["boundaries"] = {
                "lm_head_logits": _compare_tensor_boundary(control_logits, variant_logits)
            }
    gathered: list[Any] = [None, None]
    engine.dist.all_gather_object(gathered, local)
    boundaries = {**gathered[0]["boundaries"], **gathered[1]["boundaries"]}
    boundary_order = ("input_tensor", "attention_tensor", "post_expert_hidden", "lm_head_logits")
    if tuple(boundaries) != boundary_order:
        raise RuntimeError(f"SEALED_RUNTIME_TENSOR_AB_COVERAGE_RED:{tuple(boundaries)}")
    first_divergent = next(
        (name for name in boundary_order if not boundaries[name]["exact"]), None
    )
    projection_witness = gathered[0].get("projection_runtime_witness")
    aligned_comparison = None
    if isinstance(projection_witness, dict):
        layer0 = projection_witness.get("layers", {}).get("0")
        if isinstance(layer0, dict) and "aligned_active_rows" in layer0:
            aligned_comparison = _adjudicate_a27_active_rows(
                layer0["aligned_active_rows"]
            )
    receipt = {
        "schema": "banana-smasher-sealed-runtime-tensor-ab-v1",
        "status": "EXACT_RUNTIME_TENSOR_PARITY" if first_divergent is None else "RUNTIME_TENSOR_DIVERGENCE_LOCALIZED",
        "task_id": TASK, "canonical_code_commit": pin,
        "variant_base_commit": pin,
        "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
        "window": window, "layer": 0, "expert": 0,
        "control_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:593-643 + repair_api/assets/official_local_planesource.py:592-624",
        "variant_source": f"accepted 942c resident layer through repair_api/modern_green_resident.py at {pin}",
        "sole_variable": "sealed_builder_layer_vs_canonical_repaired_resident_layer",
        "mechanism": {
            "accumulation": "BF16 GEMM reduction geometry changes with output width 4096 versus 2048+2048",
            "dtype": "torch.bfloat16",
            "layout": "identical contiguous routed hidden and identical row slices of gate_up_proj",
            "control_known_hash": control_known_hash if rank == 0 else None,
            "variant_known_hash": variant_known_hash if rank == 0 else None,
            "variant_matches_resident_known_hash": boundaries["post_expert_hidden"]["variant"]["sha256"] == variant_known_hash,
        },
        "control_source_binding": control_binding if rank == 0 else None,
        "projection_runtime_witness": projection_witness,
        "aligned_active_row_comparison": aligned_comparison,
        "boundaries": boundaries,
        "first_divergent_boundary": first_divergent,
        "created_unix": time.time(),
    }
    path = root / "receipts" / f"SEALED_RUNTIME_TENSOR_AB.rank{rank}.json"
    receipt["receipt_sha256"] = atomic(path, receipt)
    print(json.dumps({"receipt_path": str(path), **receipt}, sort_keys=True), flush=True)
    return receipt


def _adopt_w28_admission(
    path: Path,
    expected_sha256: str,
    *,
    rank: int,
    expected_task_id: str = W28_ADOPTION_TASK,
) -> dict[str, Any]:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"ADOPT_W28_SHA_MISMATCH:{observed}:{expected_sha256}")
    row = json.loads(raw)
    measurement = row.get("measurement", {})
    if (
        row.get("schema") != "banana-smasher-resident-w28-admission-v1"
        or row.get("status") != "PASS"
        or row.get("task_id") != expected_task_id
        or int(row.get("rank", -1)) != rank
        or row.get("basis_sha256") != BASIS
        or row.get("checkpoint_sha256") != CHECKPOINT
        or measurement.get("windows") != [28]
        or measurement.get("kld_mean") != W28_KLD
        or measurement.get("top1") != W28_TOP1
    ):
        raise RuntimeError("ADOPT_W28_IDENTITY_OR_METRIC_MISMATCH")
    return {**row, "receipt_sha256": observed, "admission_adopted": True}


def _resolve_config_path(root: Path, *, task: str, rank: int) -> Path:
    explicit = os.environ.get("BANANA_SMASHER_CONFIG_PATH")
    if not explicit:
        return root / f"CONFIG.{task}.rank{rank}.json"
    physical_root = root.resolve()
    selected = Path(explicit).expanduser().resolve()
    try:
        selected.relative_to(physical_root)
    except ValueError as error:
        raise RuntimeError("CONFIG_PATH_OUTSIDE_PHYSICAL_ROOT") from error
    if not selected.is_file():
        raise RuntimeError("EXPLICIT_CONFIG_PATH_MISSING")
    if not selected.name.startswith(f"CONFIG.{task}.") or not selected.name.endswith(f".rank{rank}.json"):
        raise RuntimeError("EXPLICIT_CONFIG_PATH_IDENTITY_MISMATCH")
    return selected


def _provider_binding_compatible(previous: Any, observed: Any) -> bool:
    """Accept only the declared parent-to-Attempt24 provider transition."""
    if previous == observed:
        return True
    if not isinstance(previous, dict) or not isinstance(observed, dict):
        return False
    provider_keys = {"provider_wrapper_sha256", "provider_expert_sha256"}
    previous_rest = {key: value for key, value in previous.items() if key not in provider_keys}
    observed_rest = {key: value for key, value in observed.items() if key not in provider_keys}
    return (
        previous_rest == observed_rest
        and previous.get("provider_wrapper_sha256") == ADOPTED_PROVIDER_WRAPPER_SHA256
        and previous.get("provider_expert_sha256") == ADOPTED_PROVIDER_EXPERT_SHA256
        and observed.get("provider_wrapper_sha256") == CURRENT_PROVIDER_WRAPPER_SHA256
        and observed.get("provider_expert_sha256") == CURRENT_PROVIDER_EXPERT_SHA256
    )


def _score_admission_windows(
    api: Any,
    engine: Any,
    windows: tuple[int, ...],
    teacher_root: Path,
) -> dict[str, Any]:
    """The sole imported zero-reload forward used by admission and production."""
    call_tree_path = os.environ.get("W28_FULL_CALL_TREE_PATH")
    if not call_tree_path:
        return api.validate(engine, windows, teacher_root)
    if tuple(windows) != (28,):
        raise RuntimeError("W28_FULL_CALL_TREE_REQUIRES_EXACT_SINGLETON_28")
    canonical_pin = os.environ.get("BANANA_SMASHER_CANONICAL_PIN")
    if not canonical_pin:
        raise RuntimeError("W28_FULL_CALL_TREE_CANONICAL_PIN_REQUIRED")
    from repair_api.call_tree_trace import FullCallTreeTrace

    with FullCallTreeTrace(
        engine.student,
        call_tree_path,
        rail="product_w28_admission",
        basis_sha256=BASIS,
        canonical_code_commit=canonical_pin,
    ) as call_tree:
        measurement = api.validate(engine, windows, teacher_root)
    print(json.dumps({
        "status": "W28_PRODUCT_FULL_CALL_TREE_SEALED",
        "terminal": str(call_tree.path.with_suffix(call_tree.path.suffix + ".terminal.json")),
    }, sort_keys=True), flush=True)
    return measurement


def _prewarm_candidate_extension() -> dict[str, Any]:
    """Load the exact grouped-K2 extension and prove its cached no-op is read-free."""
    from repair_api.official_k2_resident_score import PayloadModelReadCounter

    wrapper = sys.modules["fast_k2_grouped"]
    wrapper_file = getattr(wrapper, "__file__", None)
    if not isinstance(wrapper_file, str):
        raise RuntimeError("CANDIDATE_EXTENSION_WRAPPER_FILE_MISSING")
    extension = wrapper._cuda_extension()
    torch.cuda.synchronize()
    extension_file = getattr(extension, "__file__", None)
    if not isinstance(extension_file, str):
        raise RuntimeError("CANDIDATE_EXTENSION_FILE_MISSING")
    extension_path = Path(extension_file).resolve()
    reads = PayloadModelReadCounter((Path(wrapper_file).resolve().parent,))
    ready = reads.mark_resident_ready()
    try:
        cached_extension = wrapper._cuda_extension()
        torch.cuda.synchronize()
    finally:
        reads.active = False
    noop_reads = reads.delta(ready)
    if cached_extension is not extension or noop_reads != 0:
        raise RuntimeError(
            f"CANDIDATE_EXTENSION_PREWARM_NOOP_RED:reads={noop_reads}:paths={reads.paths}"
        )
    return {
        "status": "PASS",
        "wrapper_module_file": str(Path(wrapper_file).resolve()),
        "extension_module_file": str(extension_path),
        "extension_module_sha256": sha(extension_path),
        "prewarm_noop_source_reads": noop_reads,
        "prewarm_noop_source_read_paths": list(reads.paths),
    }


def validate_scheduled_pair_group(
    api: Any,
    engine: Any,
    windows: tuple[int, ...],
    teacher_root: Path,
    receipt_dir: Path,
    *,
    rank: int,
    canonical_code_commit: str,
    attempt: str,
    first_pair_index: int,
) -> dict[str, Any]:
    """Run one admitted collective for disjoint pairs and seal each pair row set."""
    if not windows or len(windows) % 2 or first_pair_index < 0:
        raise RuntimeError("SCHEDULED_PAIR_GROUP_GEOMETRY_MISMATCH")
    if not attempt or not attempt.replace("-", "").replace("_", "").isalnum():
        raise RuntimeError("SCHEDULED_PAIR_GROUP_ATTEMPT_MALFORMED")
    measurement = _score_admission_windows(api, engine, windows, teacher_root)
    if measurement.get("windows") != list(windows):
        raise RuntimeError("SCHEDULED_PAIR_GROUP_WINDOW_MISMATCH")
    counters = measurement.get("runtime_counters", {})
    if any(int(counters.get(name, -1)) != 0 for name in (
        "checkpoint_reloads", "fallback_calls", "reconstruction_calls",
        "timed_model_payload_reads", "timed_score_file_reads",
    )):
        raise RuntimeError("SCHEDULED_PAIR_GROUP_ZERO_RELOAD_MISMATCH")
    measured_rows = list(measurement.get("per_window", []))
    if [int(row.get("window", -1)) for row in measured_rows] != list(windows):
        raise RuntimeError("SCHEDULED_PAIR_GROUP_ROW_COVERAGE_MISMATCH")
    profiles = measurement.get("phase_profiles_by_rank")
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise RuntimeError("SCHEDULED_PAIR_GROUP_PROFILE_MISMATCH")
    group_raw = json.dumps(measurement, sort_keys=True, separators=(",", ":")).encode()
    group_sha256 = hashlib.sha256(group_raw).hexdigest()
    pair_receipts: list[dict[str, Any]] = []
    for local_index, offset in enumerate(range(0, len(windows), 2)):
        pair_index = first_pair_index + local_index
        pair_windows = windows[offset:offset + 2]
        path = receipt_dir / (
            f"FULL64_SCHEDULED_PAIR.{attempt}.rank{rank}.{pair_index:02d}.json"
        )
        if path.exists():
            raise RuntimeError(f"SCHEDULED_PAIR_RECEIPT_EXISTS:{pair_index}")
        pair_measurement = {
            "windows": list(pair_windows),
            "per_window": measured_rows[offset:offset + 2],
            "runtime_counters": counters,
            "validation_corpus_sha256": measurement.get("validation_corpus_sha256"),
            "validation_teacher_sha256_by_window": {
                str(window): measurement.get("validation_teacher_sha256_by_window", {}).get(str(window))
                for window in pair_windows
            },
            "sealed_builder_binding": measurement.get("sealed_builder_binding"),
            "scheduled_group_measurement_sha256": group_sha256,
            "scheduled_group_windows": list(windows),
            "scheduled_group_phase_profiles_by_rank": profiles,
        }
        row = {
            "schema": "banana-smasher-resident-full64-scheduled-pair-v1",
            "status": "PASS", "task_id": TASK, "rank": rank,
            "canonical_code_commit": canonical_code_commit,
            "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
            "pair_index": pair_index, "windows": list(pair_windows),
            "measurement": pair_measurement,
        }
        digest = atomic(path, row)
        pair_receipts.append({
            "pair_index": pair_index, "windows": list(pair_windows),
            "path": str(path), "sha256": digest,
        })
    return {**measurement, "scheduled_pair_receipts": pair_receipts,
            "scheduled_group_measurement_sha256": group_sha256}


def _adopt_full64_pair0(
    path: Path,
    expected_sha256: str,
    *,
    rank: int,
    expected_code_commit: str,
    expected_rows: dict[int, dict[str, Any]],
    expected_task_id: str = TASK,
) -> dict[str, Any]:
    """Authenticate the sealed pair0 and apply the card's aggregate tolerance."""
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError("ADOPT_FULL64_PAIR0_SHA_MISMATCH")
    row = json.loads(raw)
    measurement = row.get("measurement", {})
    measured_rows = list(measurement.get("per_window", []))
    if (
        row.get("schema") != "banana-smasher-resident-full64-admission-pair-v1"
        or row.get("status") != "PASS"
        or row.get("task_id") != expected_task_id
        or int(row.get("rank", -1)) != rank
        or row.get("canonical_code_commit") != expected_code_commit
        or row.get("basis_sha256") != BASIS
        or row.get("checkpoint_sha256") != CHECKPOINT
        or int(row.get("batch_index", -1)) != 0
        or row.get("windows") != [28, 56]
        or measurement.get("windows") != [28, 56]
        or [int(value.get("window", -1)) for value in measured_rows] != [28, 56]
    ):
        raise RuntimeError("ADOPT_FULL64_PAIR0_IDENTITY_MISMATCH")
    observed_aggregate = aggregate_from_rows(measured_rows)
    expected_sum = math.fsum(
        float(expected_rows[int(value["window"])]["kld_mean"]) * int(value["positions"])
        for value in measured_rows
    )
    expected_positions = sum(int(value["positions"]) for value in measured_rows)
    aggregate_delta = observed_aggregate["kld_mean"] - expected_sum / expected_positions
    directional_shift = math.fsum(
        float(value["kld_sum_binary64"]) / int(value["positions"])
        - float(expected_rows[int(value["window"])]["kld_mean"])
        for value in measured_rows
    ) / len(measured_rows)
    if abs(aggregate_delta) > 5e-4 or abs(directional_shift) > 5e-4:
        raise RuntimeError("ADOPT_FULL64_PAIR0_DIRECTIONAL_SHIFT")
    return {
        **row,
        "receipt_sha256": observed_sha256,
        "pair0_adopted": True,
        "aggregate_kld_delta": aggregate_delta,
        "directional_kld_shift": directional_shift,
    }


def validate_full64_admission_pairs(
    api: Any,
    engine: Any,
    windows: tuple[int, ...],
    teacher_root: Path,
    receipt_dir: Path,
    *,
    rank: int,
    canonical_code_commit: str,
    attempt: str,
    expected_rows: dict[int, dict[str, Any]],
    adopted_pair0_path: Path | None = None,
    adopted_pair0_sha256: str | None = None,
    adopted_pair0_code_commit: str | None = None,
    adopted_pair0_task_id: str = TASK,
) -> dict[str, Any]:
    """Iterate the exact admitted pair forward; no second batch implementation."""
    batch_size = 2
    if len(windows) % batch_size:
        raise RuntimeError("FULL64_BATCH_GEOMETRY_MISMATCH")
    all_rows: list[dict[str, Any]] = []
    profiles: list[list[dict[str, Any]]] = [[], []]
    pair_receipts: list[dict[str, Any]] = []
    corpus_sha256: str | None = None
    teacher_sha256_by_window: dict[str, str] = {}
    sealed_builder_binding: Any = None
    adopted = 0
    computed = 0
    for batch_index, offset in enumerate(range(0, len(windows), batch_size)):
        batch = windows[offset:offset + batch_size]
        path = receipt_dir / (
            f"FULL64_ADMISSION_PAIR.{attempt}.rank{rank}.{batch_index:02d}.json"
        )
        pair_adopted = batch_index == 0 and adopted_pair0_path is not None
        if pair_adopted:
            if not adopted_pair0_sha256 or not adopted_pair0_code_commit:
                raise RuntimeError("ADOPT_FULL64_PAIR0_IDENTITY_REQUIRED")
            assert adopted_pair0_path is not None
            row = _adopt_full64_pair0(
                adopted_pair0_path, adopted_pair0_sha256, rank=rank,
                expected_code_commit=adopted_pair0_code_commit,
                expected_rows=expected_rows,
                expected_task_id=adopted_pair0_task_id,
            )
            measurement = row["measurement"]
            digest = row["receipt_sha256"]
            adopted += 1
        else:
            if path.exists():
                raise RuntimeError(f"FULL64_ADMISSION_PAIR_RECEIPT_EXISTS:{batch_index}")
            measurement = _score_admission_windows(api, engine, batch, teacher_root)
            if measurement.get("windows") != list(batch):
                raise RuntimeError(f"FULL64_BATCH_WINDOW_MISMATCH:{batch_index}")
            row = {
                "schema": "banana-smasher-resident-full64-admission-pair-v1",
                "status": "PASS", "task_id": TASK, "rank": rank,
                "canonical_code_commit": canonical_code_commit,
                "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                "batch_index": batch_index, "windows": list(batch),
                "measurement": measurement,
            }
            digest = atomic(path, row)
            computed += 1
            print(json.dumps({"status": "FULL64_ADMISSION_PAIR_ACCEPTED", "rank": rank,
                              "batch_index": batch_index, "windows": list(batch),
                              "receipt": str(path), "receipt_sha256": digest}, sort_keys=True), flush=True)
        counters = measurement.get("runtime_counters", {})
        if any(int(counters.get(name, -1)) != 0 for name in (
            "checkpoint_reloads", "fallback_calls", "reconstruction_calls",
            "timed_model_payload_reads", "timed_score_file_reads",
        )):
            raise RuntimeError(f"FULL64_BATCH_ZERO_RELOAD_MISMATCH:{batch_index}")
        measured_rows = list(measurement.get("per_window", []))
        if [int(value.get("window", -1)) for value in measured_rows] != list(batch):
            raise RuntimeError(f"FULL64_BATCH_ROW_COVERAGE_MISMATCH:{batch_index}")
        for value in measured_rows:
            all_rows.append({**value, "ordinal": len(all_rows)})
        batch_profiles = measurement.get("phase_profiles_by_rank")
        if not isinstance(batch_profiles, list) or len(batch_profiles) != 2:
            raise RuntimeError(f"FULL64_BATCH_PROFILE_MISMATCH:{batch_index}")
        for profile_rank in (0, 1):
            profiles[profile_rank].extend(batch_profiles[profile_rank])
        observed_corpus = measurement.get("validation_corpus_sha256")
        if corpus_sha256 is not None and observed_corpus != corpus_sha256:
            raise RuntimeError("FULL64_BATCH_CORPUS_DRIFT")
        corpus_sha256 = observed_corpus
        teacher_sha256_by_window.update(measurement.get("validation_teacher_sha256_by_window", {}))
        observed_binding = measurement.get("sealed_builder_binding")
        if sealed_builder_binding is not None and not _provider_binding_compatible(
            sealed_builder_binding, observed_binding
        ):
            raise RuntimeError("FULL64_BATCH_PROVIDER_BINDING_DRIFT")
        sealed_builder_binding = observed_binding
        receipt_path = adopted_pair0_path if pair_adopted else path
        pair_receipts.append({"pair_index": batch_index, "path": str(receipt_path), "sha256": digest,
                              "windows": list(batch), "adopted": pair_adopted})
    aggregate = aggregate_from_rows(all_rows)
    return {
        "schema": "banana-smasher-resident-trainer-validate-batched-v1",
        "windows": list(windows), "per_window": all_rows,
        "positions": aggregate["positions"], "kld_mean": aggregate["kld_mean"],
        "top1": aggregate["top1"], "physical_fixture_batch_size": batch_size,
        "execution_mode": "resident_in_memory_bounded_batches",
        "phase_profiles_by_rank": profiles,
        "runtime_counters": {"checkpoint_reloads": 0, "fallback_calls": 0,
                             "reconstruction_calls": 0, "timed_model_payload_reads": 0,
                             "timed_score_file_reads": 0},
        "validation_corpus_sha256": corpus_sha256,
        "validation_teacher_sha256_by_window": teacher_sha256_by_window,
        "sealed_builder_binding": sealed_builder_binding,
        "admission_pair_receipts": pair_receipts,
        "admission_pair_count": len(pair_receipts),
        "adopted_pair_count": adopted,
        "computed_pair_count": computed,
        "public_api": {"method": "ResidentRepairAPI.validate",
                       "version": "resident-trainer-validate-v1"},
    }


def _admit_initial_w28_geometry(
    config: dict[str, Any], *, singleton_public_parity_tap_only: bool
) -> str:
    """Admit singleton geometry only for the returning public tap diagnostic."""
    if singleton_public_parity_tap_only:
        if (
            int(config.get("score_window_batch_size", 0)) != 1
            or int(config.get("sealed_builder_window_microbatch", 0)) != 1
        ):
            raise RuntimeError("PARITY_TAP_REQUIRES_SINGLETON_GEOMETRY")
        return "SINGLETON_PUBLIC_PARITY_TAP_ONLY"

    if int(config.get("score_window_batch_size", 0)) != 2:
        raise RuntimeError("FULL64_REQUIRES_ADMITTED_BATCH_GEOMETRY")
    accepted_w28_geometry = {
        "score_window_batch_size": 2,
        "sealed_builder_window_microbatch": 2,
    }
    if (
        any(config.get(name) != value for name, value in accepted_w28_geometry.items())
        or any(
            name in config
            for name in (
                "score_pair_stream_concurrency",
                "score_pipeline_overlap",
                "attention_query_chunk_size",
            )
        )
    ):
        raise RuntimeError("W28_ACCEPTED_INCOMING_GEOMETRY_MISMATCH")
    return "FULL64_ADMITTED_MB2_GEOMETRY"


def main() -> None:
    rank = int(os.environ["RANK"])
    attempt = os.environ.get("BANANA_SMASHER_ATTEMPT", "")
    if attempt and not attempt.replace("-", "").replace("_", "").isalnum():
        raise RuntimeError("ATTEMPT_LABEL_MALFORMED")
    attempt_suffix = f".{attempt}" if attempt else ""
    root = Path(os.environ["PHYSICAL_ROOT"])
    config_path = _resolve_config_path(root, task=TASK, rank=rank)
    config = json.loads(config_path.read_text())
    w28_only = os.environ.get("W28_ONLY", "0") == "1"
    exact102_admission = None
    if w28_only:
        admission_value = os.environ.get("EXACT102_ADMISSION_RECEIPT")
        admission_sha = os.environ.get("EXACT102_ADMISSION_SHA256")
        if not admission_value or not admission_sha:
            raise RuntimeError("W28_ONLY_REQUIRES_EXACT102_ADMISSION")
        exact102_path = Path(admission_value).expanduser().resolve()
        if sha(exact102_path) != admission_sha:
            raise RuntimeError("EXACT102_ADMISSION_SHA_MISMATCH")
        exact102_admission = json.loads(exact102_path.read_text())
        if (
            exact102_admission.get("schema")
            != "banana-smasher-exact102-public-admission-v2"
            or exact102_admission.get("status") != "PASS_ADMISSION_READY"
            or exact102_admission.get("task_id") != TASK
            or exact102_admission.get("basis_sha256") != BASIS
            or exact102_admission.get("checkpoint_sha256") != CHECKPOINT
            or int(exact102_admission.get("provenance_members", -1)) != 22016
        ):
            raise RuntimeError("EXACT102_ADMISSION_CONTRACT_MISMATCH")
        terminal_value = config.get("backpack_virtual_terminal_path")
        terminal_sha = config.get("backpack_virtual_terminal_sha256")
        manifest_sha = config.get("backpack_virtual_manifest_sha256")
        if not all(isinstance(value, str) and value for value in (
            terminal_value, terminal_sha, manifest_sha
        )):
            raise RuntimeError("EXACT102_MIXED_ARTIFACT_BINDING_MISMATCH")
        terminal_path = Path(terminal_value).expanduser().resolve()
        if not terminal_path.is_file() or sha(terminal_path) != terminal_sha:
            raise RuntimeError("EXACT102_MIXED_ARTIFACT_BINDING_MISMATCH")
        virtual_terminal = json.loads(terminal_path.read_text())
        manifest_path = Path(str(virtual_terminal.get("virtual_manifest_path", ""))).expanduser().resolve()
        accounting = virtual_terminal.get("whole_model_accounting", {})
        if (
            virtual_terminal.get("schema") != "banana-smasher-mixed-exact102-virtual-terminal-v1"
            or virtual_terminal.get("status") != "PASS"
            or virtual_terminal.get("task_id") != TASK
            or virtual_terminal.get("basis_sha256") != BASIS
            or virtual_terminal.get("virtual_manifest_sha256") != manifest_sha
            or accounting.get("whole_shipping_bytes") != 102000000000
            or not manifest_path.is_file()
            or sha(manifest_path) != manifest_sha
            or Path(str(config.get("artifact_root", ""))).expanduser().resolve()
            != manifest_path.parent
        ):
            raise RuntimeError("EXACT102_MIXED_ARTIFACT_BINDING_MISMATCH")
        virtual_manifest = json.loads(manifest_path.read_text())
        if (
            virtual_manifest.get("basis_sha256") != BASIS
            or virtual_manifest.get("source_component_counts", {}).get("qtip3") != 14773
            or virtual_manifest.get("whole_model_accounting", {}).get("whole_shipping_bytes")
            != 102000000000
        ):
            raise RuntimeError("EXACT102_MIXED_ARTIFACT_BINDING_MISMATCH")
    packed_boundary_tap_only = os.environ.get("PACKED_BOUNDARY_TAP_ONLY", "0") == "1"
    sealed_runtime_tensor_ab_only = os.environ.get("SEALED_RUNTIME_TENSOR_AB_ONLY", "0") == "1"
    aligned_active_row_capture_only = os.environ.get(
        "SEALED_ALIGNED_ACTIVE_ROW_CAPTURE_ONLY", "0"
    ) == "1"
    sealed_runtime_expert_trace_ab_only = os.environ.get("SEALED_RUNTIME_EXPERT_TRACE_AB_ONLY", "0") == "1"
    changed_input_w28_only = os.environ.get("CHANGED_INPUT_W28_ONLY", "0") == "1"
    matched_sdpa_w28_only = os.environ.get("MATCHED_SDPA_W28_ONLY", "0") == "1"
    pair_scheduling_ab_only = os.environ.get("PAIR_SCHEDULING_AB_ONLY", "0") == "1"
    singleton_public_parity_tap_only = (
        os.environ.get("LAW4_PUBLIC_PRODUCT_TAP_ONLY", "0") == "1"
        or os.environ.get("RUN6910_ROUTED_REDUCTION_AB_ONLY", "0") == "1"
    )
    if singleton_public_parity_tap_only:
        config["singleton_public_parity_tap_only"] = True
    adopted_w28 = bool(os.environ.get("ADOPT_W28_RECEIPT"))
    adopt_keep_provider = os.environ.get("ADOPT_W28_KEEP_PROVIDER", "0") == "1"
    # With authenticated W28 adoption there is no pre-production forward: bind
    # the only resident engine to the packed Attempt24 provider before its
    # modules are constructed. Mutating config after construction is inert.
    if packed_boundary_tap_only:
        config["resident_validation_expert_implementation"] = "packed_cuda_bf16_boundary"
    elif adopted_w28 and not adopt_keep_provider:
        # The authenticated prefix and terminal were produced by the sealed
        # PlaneSource decode -> inverse-transform -> BF16 physical matrix ->
        # BF16 GEMM boundary.  Construct the sole resident engine with that
        # same existing provider; the packed child is source-bound separately.
        config["resident_validation_expert_implementation"] = "sealed_bf16_full_weight"
    pin = str(config["canonical_code_commit"])
    if len(pin) != 40 or config.get("basis_sha256") != BASIS:
        raise RuntimeError("CONFIG_PIN_OR_BASIS_MISMATCH")
    index = Path(config["model_root"]) / "model.safetensors.index.json"
    if sha(index) != BASIS:
        raise RuntimeError("BASIS_GATE_MISMATCH")
    api = ResidentRepairAPI.open(Path(config["artifact_root"]))
    config = api.bind_routed_return_accumulation(
        config,
        provider_expert_sha256=CURRENT_PROVIDER_EXPERT_SHA256,
    )
    config = api.bind_combined_gate_up_projection(
        config,
        provider_expert_sha256=CURRENT_PROVIDER_EXPERT_SHA256,
        capture_witness=(sealed_runtime_tensor_ab_only or aligned_active_row_capture_only),
        active_row_expert=204 if aligned_active_row_capture_only else None,
    )
    checkpoint_path = api.artifact.checkpoint_path("PRE")
    if sha(checkpoint_path) != CHECKPOINT:
        raise RuntimeError("PUBLISHED_PRE_CHECKPOINT_MISMATCH")
    reference_path = Path(config["reference_terminal"])
    reference = json.loads(reference_path.read_text())
    if reference.get("basis_sha256") != BASIS or reference.get("checkpoint_sha256") != CHECKPOINT:
        raise RuntimeError("REFERENCE_IDENTITY_MISMATCH")
    windows = tuple(int(value) for value in reference["coverage"]["expected_windows"])
    if len(windows) != 64 or len(set(windows)) != 64 or windows[0] != 28:
        raise RuntimeError("REFERENCE_WINDOW_ROSTER_MISMATCH")
    expected_rows = {int(row["window"]): row for row in reference["per_window"]}
    if set(expected_rows) != set(windows):
        raise RuntimeError("REFERENCE_ROW_COVERAGE_MISMATCH")
    # Hash-bind the imported sealed builder/PlaneSource identities, then execute
    # their admitted single-window geometry through the existing zero-reload
    # resident API rather than the cold layer-streaming producer.
    bind_sealed_pre_resident_config(config)
    _admit_initial_w28_geometry(
        config,
        singleton_public_parity_tap_only=singleton_public_parity_tap_only,
    )
    allowed_experts = {None, "accepted_static_w28"}
    if packed_boundary_tap_only:
        allowed_experts.add("packed_cuda_bf16_boundary")
    if adopted_w28 or config.get("sealed_pre_source_binding"):
        allowed_experts.add("sealed_bf16_full_weight")
    if config.get("resident_validation_expert_implementation") not in allowed_experts:
        raise RuntimeError("FULL64_REQUIRES_ACCEPTED_PROVIDER")
    if matched_sdpa_w28_only:
        config["resident_validation_attention_implementation"] = "sdpa"
        config.pop("attention_query_chunk_size", None)
        config.pop("resident_validation_stock_hf_attention", None)
        config.pop("resident_validation_stock_hf_sdpa_math_backend", None)
    elif changed_input_w28_only:
        # Changed-input gate only: preserve the installed DeepseekV4 eager
        # compound arithmetic (BF16 QK, row-max subtraction, BF16 softmax,
        # sink-column drop, then value matmul) while bounding only the query
        # workspace. A 2048-row serial chunk is the Attempt24 window-isolation
        # seam; it does not alter provider, basis, PRE, score geometry, or rows.
        config["resident_validation_attention_implementation"] = "eager"
        config["attention_query_chunk_size"] = 2048
        config["resident_validation_official_decoder_dispatch"] = True
        config.pop("resident_validation_stock_hf_attention", None)
        config.pop("resident_validation_stock_hf_sdpa_math_backend", None)
    else:
        # This lane changes pair-axis scheduling only. Preserve the exact imported
        # eager forward/scorer math that produced the immutable Attempt57 rows.
        config["resident_validation_attention_implementation"] = "eager"
        config.pop("resident_validation_stock_hf_attention", None)
        config.pop("resident_validation_stock_hf_sdpa_math_backend", None)
    # The immutable accepted W28 producer retained the incoming score geometry
    # through engine construction and admission.  Production-only scheduling
    # is installed after the exact W28 gate below.
    # Bind the physical boundary before the sole engine is constructed.
    # A7 localized the first matched divergence to the gate GEMM itself: the
    # packed grouped reduction rounds differently from the sealed full-weight
    # BF16 F.linear even though the decoded weight bytes are exact. Keep the
    # accepted 942c provider bytes and repair only that projection seam.
    os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "1"
    os.environ.pop("FAST_K2_SEALED_COMPLETE_EXPERT_BF16", None)
    os.environ.pop("FAST_K2_SEALED_PROJECTION_BF16", None)

    os.environ["NCCL_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    os.environ["GLOO_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    if (
        sealed_runtime_tensor_ab_only
        or aligned_active_row_capture_only
        or sealed_runtime_expert_trace_ab_only
    ):
        # The accepted R20 tensor rail relied on the 88 GiB cgroup limit and
        # carried no narrower per-process CUDA cap. A8 proved that the later
        # 0.45 cap fails during unchanged nonexpert construction, before any
        # tensor boundary executes.
        cuda_memory_fraction = None
        cuda_memory_cap_status = "BYPASSED_R20_TENSOR_AB_CGROUP_ONLY"
    else:
        cuda_memory_fraction = _apply_cuda_memory_fraction(torch.cuda)
        cuda_memory_cap_status = "PASS"
    cap_path = root / "receipts" / f"CUDA_MEMORY_CAP.{TASK}.rank{rank}.json"
    cap_row = {
        "schema": "banana-smasher-cuda-memory-cap-v1",
        "status": cuda_memory_cap_status,
        "task_id": TASK,
        "rank": rank,
        "pid": os.getpid(),
        "canonical_code_commit": pin,
        "basis_sha256": BASIS,
        "cuda_memory_fraction": cuda_memory_fraction,
        "created_unix": time.time(),
    }
    cap_row["receipt_sha256"] = atomic(cap_path, cap_row)
    print(json.dumps({"cuda_memory_cap_path": str(cap_path), **cap_row}, sort_keys=True), flush=True)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=900))
    grouped_mm_operation_probe = (
        os.environ.get("RUN6873_GROUPED_MM_OPERATION_COMPARATOR_ONLY", "0") == "1"
        or os.environ.get("RUN6910_ROUTED_REDUCTION_AB_ONLY", "0") == "1"
    )
    expected_world_size = 1 if grouped_mm_operation_probe else 2
    if (
        torch.distributed.get_world_size() != expected_world_size
        or torch.distributed.get_rank() != rank
    ):
        raise RuntimeError("DIST_GEOMETRY_MISMATCH")

    process_started = time.perf_counter()
    load_started = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layer_ranges = {0: (0, 20)} if grouped_mm_operation_probe else {
        0: (0, 20), 1: (21, 42)
    }
    engine = ModernGreenResidentEngine(
        payload=payload, config=config, rank=rank, layer_ranges=layer_ranges
    )
    del payload
    projection_binding = engine.sealed_gate_up_runtime_witness(
        require_activation=False
    )
    torch.cuda.synchronize()
    resident_load_seconds = time.perf_counter() - load_started
    extension_prewarm = _prewarm_candidate_extension()

    if (
        os.environ.get("RUN6522_AUTHENTIC_SOURCE_PROJECTION_CONTROL_ONLY", "0") == "1"
        or os.environ.get("RUN6524_SOURCE_IMPLEMENTATION_DISPATCH_ONLY", "0") == "1"
        or os.environ.get("RUN6873_GROUPED_MM_OPERATION_COMPARATOR_ONLY", "0") == "1"
        or os.environ.get("RUN6910_ROUTED_REDUCTION_AB_ONLY", "0") == "1"
    ):
        _sealed_authentic_source_projection_control(
            engine, window=28, root=root, rank=rank, pin=pin,
            checkpoint_path=checkpoint_path,
        )
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if sealed_runtime_expert_trace_ab_only:
        _sealed_runtime_expert_trace_ab(
            engine, window=28, root=root, rank=rank, pin=pin,
            checkpoint_path=checkpoint_path,
        )
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if sealed_runtime_tensor_ab_only or aligned_active_row_capture_only:
        _sealed_runtime_tensor_ab(
            engine, window=28, root=root, rank=rank, pin=pin,
            checkpoint_path=checkpoint_path,
        )
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if os.environ.get("WHOLE_CHAIN_BISECT_ONLY", "0") == "1":
        _whole_chain_bisect(engine, window=28, root=root, rank=rank, pin=pin)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if os.environ.get("AUTHENTIC_SCORING_READOUT_BOUNDARY_ONLY", "0") == "1":
        _authentic_scoring_readout_boundary(
            api, engine, window=28, root=root, rank=rank, pin=pin
        )
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if singleton_public_parity_tap_only:
        _law4_public_product_taps(
            api, engine, window=28, root=root, rank=rank, pin=pin)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if os.environ.get("READOUT_BINDING_AB_ONLY", "0") == "1":
        _readout_binding_ab(engine, window=28, root=root, rank=rank, pin=pin)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # The accepted W28 gate was captured in the sealed builder's aligned mb=2
    # fixture. Production then switches only the physical validation geometry
    # to the admitted batch4 resident path; the engine and weights stay loaded.
    adopt_path_value = os.environ.get("ADOPT_W28_RECEIPT")
    if adopt_path_value:
        expected_adopt_sha = os.environ.get("ADOPT_W28_SHA256")
        if not expected_adopt_sha:
            raise RuntimeError("ADOPT_W28_SHA_REQUIRED")
        admission_path = Path(adopt_path_value).expanduser().resolve()
        admission_row = _adopt_w28_admission(
            admission_path, expected_adopt_sha, rank=rank,
            expected_task_id=os.environ.get("ADOPT_W28_TASK_ID", W28_ADOPTION_TASK),
        )
        admission_wall = float(admission_row["admission_wall_seconds"])
    else:
        config["sealed_builder_window_microbatch"] = 2
        admission_started = time.perf_counter()
        admission = _score_admission_windows(
            api, engine, (28,), Path(config["validation_teacher_root"])
        )
        torch.cuda.synchronize()
        admission_wall = time.perf_counter() - admission_started
        if matched_sdpa_w28_only:
            teacher_sha = str(config.get("matched_sdpa_teacher_sha256", ""))
            builder_sha = str(config.get("matched_sdpa_teacher_builder_sha256", ""))
            for label, digest in (("teacher", teacher_sha), ("builder", builder_sha)):
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise RuntimeError(f"MATCHED_SDPA_{label.upper()}_SHA_MALFORMED")
            if rank == 0:
                terminal_path = root / "receipts" / "MATCHED_SDPA_W28_TERMINAL.json"
                terminal = {
                    "schema": "banana-smasher-matched-sdpa-w28-v1",
                    "status": "PARALLEL_METRIC",
                    "task_id": TASK,
                    "canonical_code_commit": pin,
                    "basis_sha256": BASIS,
                    "checkpoint_sha256": CHECKPOINT,
                    "window": 28,
                    "teacher_attention_implementation": "sdpa",
                    "student_attention_implementation": "sdpa",
                    "teacher_row_sha256": teacher_sha,
                    "teacher_builder_sha256": builder_sha,
                    "measurement": admission,
                    "resident_load_seconds": resident_load_seconds,
                    "extension_prewarm": extension_prewarm,
                    "admission_wall_seconds": admission_wall,
                    "eager_pair_anchor": {
                        "kld_mean": W28_KLD,
                        "top1": W28_TOP1,
                        "publication_role": "canonical_cross_check_anchor",
                    },
                    "publication_role": "parallel_metric_requires_own_u0_rebaseline",
                }
                terminal["receipt_sha256"] = atomic(terminal_path, terminal)
                print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
            engine.close()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        if changed_input_w28_only:
            # The construction-time binding witness is necessarily zero before
            # the first forward. Refresh it at the W28 boundary and fail closed
            # unless the provider-global wrapper actually served this score.
            projection_binding = engine.sealed_gate_up_runtime_witness(
                require_activation=True
            )
            observed_kld = float(admission.get("kld_mean", float("nan")))
            kld_shift = observed_kld - W28_KLD
            terminal = {
                "schema": "banana-smasher-changed-input-w28-terminal-v1",
                "status": "PARITY" if abs(kld_shift) <= 1.0e-3 else "SDPA_RED",
                "task_id": TASK, "rank": rank,
                "canonical_code_commit": pin, "basis_sha256": BASIS,
                "checkpoint_sha256": CHECKPOINT,
                "reference": {"attention_implementation": "eager", "kld_mean": W28_KLD,
                              "top1": W28_TOP1},
                "candidate": {"attention_implementation": "sdpa", "measurement": admission},
                "projection_runtime_witness": projection_binding,
                "kld_shift": kld_shift, "wall_seconds": admission_wall,
            }
            terminal_path = root / "receipts" / f"CHANGED_INPUT_W28.{TASK}.rank{rank}.json"
            terminal["receipt_sha256"] = atomic(terminal_path, terminal)
            print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
            engine.close()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        if packed_boundary_tap_only:
            tap_terminal = {
                "schema": "banana-smasher-packed-w28-boundary-tap-terminal-v1",
                "status": "DIAGNOSTIC_ONLY", "task_id": TASK, "rank": rank,
                "canonical_code_commit": pin, "basis_sha256": BASIS,
                "checkpoint_sha256": CHECKPOINT, "measurement": admission,
                "wall_seconds": admission_wall,
            }
            atomic(root / "receipts" / f"PACKED_W28_BOUNDARY_TAP.{TASK}.rank{rank}.json", tap_terminal)
            engine.close()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        if admission.get("windows") != [28] or admission.get("kld_mean") != W28_KLD or admission.get("top1") != W28_TOP1:
            raise RuntimeError(f"W28_ADMISSION_RED:{admission.get('kld_mean')}:{admission.get('top1')}")
        admission_path = root / "receipts" / f"W28_ADMISSION.{TASK}.rank{rank}.json"
        admission_row = {"schema": "banana-smasher-resident-w28-admission-v1", "status": "PASS",
                         "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
                         "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                         "resident_load_seconds": resident_load_seconds, "admission_wall_seconds": admission_wall,
                         "measurement": admission}
        admission_row["receipt_sha256"] = atomic(admission_path, admission_row)

    if w28_only:
        assert exact102_admission is not None
        terminal_path = root / "receipts" / f"W28_ONLY_TERMINAL.{TASK}.rank{rank}.json"
        terminal = {
            "schema": "banana-smasher-exact102-imported-w28-v1",
            "status": "PASS",
            "task_id": TASK,
            "rank": rank,
            "canonical_code_commit": pin,
            "basis_sha256": BASIS,
            "checkpoint_sha256": CHECKPOINT,
            "exact102_admission_sha256": os.environ["EXACT102_ADMISSION_SHA256"],
            "exact102_virtual_artifact_sha256": exact102_admission[
                "virtual_artifact_sha256"
            ],
            "provenance_members": exact102_admission["provenance_members"],
            "resident_load_seconds": resident_load_seconds,
            "admission_wall_seconds": admission_wall,
            "measurement": admission_row["measurement"],
        }
        terminal["receipt_sha256"] = atomic(terminal_path, terminal)
        print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # W28 is now sealed against the immutable accepted producer. Production
    # retains its single-window physical context and calls the same admission
    # forward for every ordered row. There is no second full64 geometry or
    # input-builder path.
    os.environ["FAST_K2_SEALED_PROJECTION_BF16"] = "0"
    # The sealed PRE truth used one physical window per builder forward. Keep
    # the exact eager/provider arithmetic and expose all64 once to the existing
    # two-rank pipeline without introducing an mb2 CUDA batch context.
    config["score_window_batch_size"] = 1
    engine.score_pipeline_microbatch = 1
    config["score_pair_stream_concurrency"] = 1
    config["score_pipeline_overlap"] = True

    # The packed provider was selected before engine construction whenever W28
    # was adopted. Keep that single provider identity through production.
    # The same admitted inverse-transform→BF16 physical-weight boundary remains
    # active through canary and production; do not revert to the fused-float path.
    # Keep the exact admitted reference FWHT backend; a post-gate backend switch
    # would create a second forward implementation.
    if pair_scheduling_ab_only:
        config["score_window_batch_size"] = 1
        engine.score_pipeline_microbatch = 1
        # The 0.45 allocator fence intentionally turns the former 77 GiB UVM
        # runaway into a recoverable OOM. Bound only eager attention workspace
        # through the existing bitwise 64-query seam; provider/scorer/order stay fixed.
        config["attention_query_chunk_size"] = 64
        config["indexer_scorer_query_chunk_size"] = 128
        config["score_pair_stream_concurrency"] = 1
        config["score_pipeline_overlap"] = False
        config["score_pair_group_single_stream"] = True
        canary_windows = windows[2:10]
        canary_started = time.perf_counter()
        canary = validate_scheduled_pair_group(
            api, engine, canary_windows, Path(config["validation_teacher_root"]),
            root / "receipts", rank=rank, canonical_code_commit=pin,
            attempt=attempt or "pair-scheduling-ab", first_pair_index=1,
        )
        torch.cuda.synchronize()
        canary_wall = time.perf_counter() - canary_started
        canary_rows = list(canary.get("per_window", []))
        row_diffs = []
        for row in canary_rows:
            expected = expected_rows[int(row["window"])]
            observed_mean = float(row["kld_sum_binary64"]) / int(row["positions"])
            row_diffs.append({
                "window": int(row["window"]),
                "kld_delta": observed_mean - float(expected["kld_mean"]),
                "top1_delta": int(row["top1"]) - int(expected["top1"]),
            })
        exact_rows = len(row_diffs) == 8 and all(
            row["kld_delta"] == 0.0 and row["top1_delta"] == 0
            for row in row_diffs
        )
        profiles = canary.get("phase_profiles_by_rank", [[], []])
        if len(profiles) != 2 or len(profiles[0]) != 1 or len(profiles[1]) != 1:
            raise RuntimeError("PAIR_SCHEDULING_AB_PROFILE_GEOMETRY_MISMATCH")
        rank0_stage_ms = float(profiles[0][0]["forward_ms"]) + float(profiles[0][0]["p2p_ms"])
        rank1_stage_ms = float(profiles[1][0]["forward_ms"]) + float(profiles[1][0]["readout_ms"])
        full_groups = len(windows) // len(canary_windows)
        projected_full64_wall = (
            rank0_stage_ms + full_groups * max(rank0_stage_ms, rank1_stage_ms)
        ) / 1000.0
        status = (
            "PASS_PARITY_PROJECTED_SUB300"
            if exact_rows and projected_full64_wall < 300.0
            else "RATE_LOW" if exact_rows else "PARITY_RED"
        )
        receipt = {
            "schema": "banana-smasher-pair-scheduling-ab-v1", "status": status,
            "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
            "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
            "windows": list(canary_windows), "post_load_wall_seconds": canary_wall,
            "cuda_memory_fraction": cuda_memory_fraction,
            "projected_full64_wall_seconds": projected_full64_wall,
            "projection_formula": "rank0_stage + 8 * max(rank0_stage, rank1_stage)",
            "row_diffs": row_diffs, "per_window": canary_rows,
            "phase_profiles_by_rank": profiles,
            "scheduled_pair_receipts": canary.get("scheduled_pair_receipts"),
            "mechanism_counters": canary.get("runtime_counters", {}),
        }
        receipt_path = root / "receipts" / f"PAIR_SCHEDULING_AB.{TASK}{attempt_suffix}.rank{rank}.json"
        receipt["receipt_sha256"] = atomic(receipt_path, receipt)
        print(json.dumps({"receipt_path": str(receipt_path), **receipt}, sort_keys=True), flush=True)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if status != "PASS_PARITY_PROJECTED_SUB300":
            raise RuntimeError(status)
        return

    ep_canary = tuple(int(value) for value in config.get("expert_parallel_canary_windows", ()))
    if ep_canary:
        if (
            not bool(config.get("expert_parallel_all_layers", False))
            or len(ep_canary) != 2
            or 28 in ep_canary
            or not set(ep_canary).issubset(expected_rows)
        ):
            raise RuntimeError("EXPERT_PARALLEL_CANARY_GEOMETRY_MISMATCH")
        canary_started = time.perf_counter()
        canary = _score_admission_windows(
            api, engine, ep_canary, Path(config["validation_teacher_root"])
        )
        torch.cuda.synchronize()
        canary_wall = time.perf_counter() - canary_started
        canary_rows = list(canary.get("per_window", []))
        exact_rows = len(canary_rows) == 2
        for row in canary_rows:
            expected = expected_rows[int(row["window"])]
            exact_rows = exact_rows and (
                float(row["kld_sum_binary64"]) / int(row["positions"])
                == float(expected["kld_mean"])
                and int(row["top1"]) == int(expected["top1"])
            )
        projected_wall = canary_wall * 64.0 / 2.0
        canary_status = (
            "PASS_EXACT_PROJECTED_SUB300"
            if exact_rows and projected_wall < 300.0
            else "RATE_LOW_CANARY" if exact_rows else "PARITY_RED_CANARY"
        )
        canary_path = root / "receipts" / (
            f"EXPERT_PARALLEL_CANARY.{TASK}{attempt_suffix}.rank{rank}.json"
        )
        canary_receipt = {
            "schema": "banana-smasher-expert-parallel-canary-v1",
            "status": canary_status, "task_id": TASK, "rank": rank,
            "canonical_code_commit": pin, "basis_sha256": BASIS,
            "checkpoint_sha256": CHECKPOINT, "windows": list(ep_canary),
            "post_load_wall_seconds": canary_wall,
            "projected_full64_wall_seconds": projected_wall,
            "exact_reference_rows": exact_rows, "per_window": canary_rows,
            "phase_profiles_by_rank": canary.get("phase_profiles_by_rank"),
            "mechanism_counters": canary.get("runtime_counters", {}),
        }
        canary_receipt["receipt_sha256"] = atomic(canary_path, canary_receipt)
        print(json.dumps({"canary_path": str(canary_path), **canary_receipt}, sort_keys=True), flush=True)
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if canary_status != "PASS_EXACT_PROJECTED_SUB300":
            raise RuntimeError(canary_status)
        return
    full_started = time.perf_counter()
    adopted_pair0_value = os.environ.get("ADOPT_FULL64_PAIR0_RECEIPT")
    adopted_pair0_path = (
        Path(adopted_pair0_value).expanduser().resolve()
        if adopted_pair0_value else None
    )
    if adopted_pair0_path is not None:
        try:
            adopted_pair0_path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError("ADOPT_FULL64_PAIR0_OUTSIDE_PHYSICAL_ROOT") from error
    if adopted_pair0_path is not None:
        raise RuntimeError("SCHEDULED_FULL64_DOES_NOT_ADOPT_A_SERIAL_PAIR")
    full = validate_scheduled_pair_group(
        api, engine, windows, Path(config["validation_teacher_root"]),
        root / "receipts", rank=rank, canonical_code_commit=pin,
        attempt=attempt or "production", first_pair_index=0,
    )
    torch.cuda.synchronize()
    post_load_wall = time.perf_counter() - full_started
    rows = list(full.get("per_window", []))
    if len(rows) != 64 or [int(row["window"]) for row in rows] != list(windows):
        raise RuntimeError("FULL64_ROW_COVERAGE_MISMATCH")
    aggregate = aggregate_from_rows(rows)
    diffs = []
    for row in rows:
        window = int(row["window"])
        expected = expected_rows[window]
        observed_mean = float(row["kld_sum_binary64"]) / int(row["positions"])
        expected_mean = float(expected["kld_mean"])
        diffs.append({"window": window, "kld_mean": observed_mean,
                      "expected_kld_mean": expected_mean, "kld_delta": observed_mean - expected_mean,
                      "top1": int(row["top1"]), "expected_top1": int(expected["top1"]),
                      "top1_delta": int(row["top1"]) - int(expected["top1"])})
    directional_shift = math.fsum(item["kld_delta"] for item in diffs) / len(diffs)
    expected_aggregate = reference["aggregate"]
    if post_load_wall >= 300.0:
        rate_low_path = root / "receipts" / f"FULL64_RATE_LOW.{TASK}.rank{rank}.json"
        rate_low = {
            "schema": "banana-smasher-resident-full64-rate-low-v2", "status": "RATE_LOW",
            "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
            "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
            "post_load_wall_seconds": post_load_wall, "threshold_seconds": 300.0,
            "aggregate": aggregate, "per_window": rows,
            "phase_profiles_by_rank": full.get("phase_profiles_by_rank"),
            "mechanism_counters": full.get("runtime_counters", {}),
        }
        rate_low["receipt_sha256"] = atomic(rate_low_path, rate_low)
        raise RuntimeError(f"RATE_LOW:{post_load_wall}")
    if abs(aggregate["kld_mean"] - float(expected_aggregate["kld_mean"])) > 5e-4:
        raise RuntimeError(f"AGGREGATE_KLD_SHIFT:{aggregate['kld_mean']}")
    if abs(directional_shift) > 5e-4:
        raise RuntimeError(f"DIRECTIONAL_SHIFT:{directional_shift}")
    mechanism = full.get("runtime_counters", {})
    if int(mechanism.get("checkpoint_reloads", -1)) != 0 or int(mechanism.get("reconstruction_calls", -1)) != 0:
        raise RuntimeError("WEIGHT_RELOAD_OR_RECONSTRUCTION_OBSERVED")

    terminal = {"schema": "banana-smasher-resident-full64-terminal-v1", "status": "PASS",
                "task_id": TASK, "rank": rank, "pid": os.getpid(), "canonical_code_commit": pin,
                "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                "runtime_provider": "vllm/vllm-openai:v0.24.0", "teacher_root": config["validation_teacher_root"],
                "validation_teacher_sha256_by_window": full.get("validation_teacher_sha256_by_window"),
                "reference_terminal_sha256": sha(reference_path), "resident_load_seconds": resident_load_seconds,
                "admission_receipt": str(admission_path), "admission_receipt_sha256": admission_row["receipt_sha256"],
                "admission_wall_seconds": admission_wall, "post_load_wall_seconds": post_load_wall,
                "cuda_memory_fraction": cuda_memory_fraction,
                "process_wall_seconds": time.perf_counter() - process_started,
                "zero_weight_reload_proof": {"resident_engine_instances": 1, "checkpoint_loads": 1,
                                             "reconstruction_calls": 0},
                "aggregate": aggregate, "expected_aggregate": expected_aggregate,
                "directional_kld_shift": directional_shift, "per_window": rows,
                "per_window_diff": diffs, "phase_profiles_by_rank": full.get("phase_profiles_by_rank"),
                "mechanism_counters": mechanism}
    terminal_path = root / "receipts" / f"FULL64_TERMINAL.{TASK}.rank{rank}.json"
    terminal["receipt_sha256"] = atomic(terminal_path, terminal)
    print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
    engine.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
