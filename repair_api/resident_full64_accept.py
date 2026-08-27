#!/usr/bin/env python3
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from repair_api import ResidentRepairAPI
from repair_api.modern_green_resident import ModernGreenResidentEngine

TASK = os.environ.get("BANANA_SMASHER_TASK_ID", "t_d4dac464")
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
W28_KLD = 0.13712959240533734
W28_TOP1 = 877
W28_ADOPTION_TASK = "t_8b1b3a3f"
ADOPTED_TASK_ID = W28_ADOPTION_TASK
ADOPTED_PROVIDER_WRAPPER_SHA256 = "ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
ADOPTED_PROVIDER_EXPERT_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"
CURRENT_PROVIDER_WRAPPER_SHA256 = "97f7379b65782acf1b4360704375d5ad1da43f2f95544dda8ebc175ee7458e6e"
CURRENT_PROVIDER_EXPERT_SHA256 = "4b1d5d99734cf75a0d50a9d6be69467b5fea129fc9415d53c476d2f103f92269"


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


def validate_full64_batches(
    api: Any,
    engine: Any,
    windows: tuple[int, ...],
    teacher_root: Path,
    receipt_dir: Path,
    *,
    rank: int,
    canonical_code_commit: str,
    adopted_prefix_code_commit: str,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Run and durably seal bounded resident batches, adopting exact sealed prefixes."""
    if batch_size < 1 or len(windows) % batch_size:
        raise RuntimeError("FULL64_BATCH_GEOMETRY_MISMATCH")
    all_rows: list[dict[str, Any]] = []
    profiles: list[list[dict[str, Any]]] = [[], []]
    batch_receipts: list[dict[str, Any]] = []
    corpus_sha256: str | None = None
    teacher_sha256_by_window: dict[str, str] = {}
    sealed_builder_binding: Any = None
    resumed = 0
    computed = 0
    for batch_index, offset in enumerate(range(0, len(windows), batch_size)):
        batch = windows[offset:offset + batch_size]
        path = receipt_dir / f"FULL64_BATCH.rank{rank}.{batch_index:02d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            measurement = row.get("measurement", {})
            same_attempt = (
                row.get("task_id") == TASK
                and row.get("canonical_code_commit") == canonical_code_commit
            )
            adopted_prefix = (
                row.get("task_id") == ADOPTED_TASK_ID
                and row.get("canonical_code_commit") == adopted_prefix_code_commit
            )
            exact = (
                row.get("schema") == "banana-smasher-resident-full64-batch-v1"
                and row.get("status") == "PASS"
                and (same_attempt or adopted_prefix)
                and row.get("rank") == rank
                and row.get("basis_sha256") == BASIS
                and row.get("checkpoint_sha256") == CHECKPOINT
                and row.get("batch_index") == batch_index
                and row.get("windows") == list(batch)
                and measurement.get("windows") == list(batch)
            )
            if not exact:
                raise RuntimeError(f"FULL64_BATCH_RECEIPT_IDENTITY_MISMATCH:{batch_index}")
            resumed += 1
        else:
            measurement = api.validate(engine, batch, teacher_root)
            if measurement.get("windows") != list(batch):
                raise RuntimeError(f"FULL64_BATCH_WINDOW_MISMATCH:{batch_index}")
            row = {
                "schema": "banana-smasher-resident-full64-batch-v1",
                "status": "PASS", "task_id": TASK, "rank": rank,
                "canonical_code_commit": canonical_code_commit,
                "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                "batch_index": batch_index, "windows": list(batch),
                "measurement": measurement,
            }
            digest = atomic(path, row)
            computed += 1
            print(json.dumps({"status": "FULL64_BATCH_ACCEPTED", "rank": rank,
                              "batch_index": batch_index, "windows": list(batch),
                              "receipt": str(path), "receipt_sha256": digest}, sort_keys=True), flush=True)
            release = getattr(engine, "release_validation_inputs", None)
            if not callable(release) or not release(batch, teacher_root):
                raise RuntimeError(f"FULL64_BATCH_INPUT_RELEASE_FAILED:{batch_index}")
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
        batch_receipts.append({"batch_index": batch_index, "path": str(path), "sha256": sha(path),
                               "windows": list(batch)})
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
        "batch_receipts": batch_receipts,
        "resumed_batch_count": resumed, "computed_batch_count": computed,
        "public_api": {"method": "ResidentRepairAPI.validate",
                       "version": "resident-trainer-validate-v1"},
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    attempt = os.environ.get("BANANA_SMASHER_ATTEMPT", "")
    if attempt and not attempt.replace("-", "").replace("_", "").isalnum():
        raise RuntimeError("ATTEMPT_LABEL_MALFORMED")
    attempt_suffix = f".{attempt}" if attempt else ""
    root = Path(os.environ["PHYSICAL_ROOT"])
    config_path = _resolve_config_path(root, task=TASK, rank=rank)
    config = json.loads(config_path.read_text())
    packed_boundary_tap_only = os.environ.get("PACKED_BOUNDARY_TAP_ONLY", "0") == "1"
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
    if int(config.get("score_window_batch_size", 0)) not in {4, 8}:
        raise RuntimeError("FULL64_REQUIRES_ADMITTED_BATCH_GEOMETRY")
    accepted_w28_geometry = {
        # The authenticated accepted W28 producer uses batch8/mb2. Keep that
        # incoming geometry through construction; production scheduling starts
        # only after the exact W28 admission gate.
        "score_window_batch_size": 8,
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
    allowed_experts = {None, "accepted_static_w28"}
    if packed_boundary_tap_only:
        allowed_experts.add("packed_cuda_bf16_boundary")
    if adopted_w28:
        allowed_experts.add("sealed_bf16_full_weight")
    if config.get("resident_validation_expert_implementation") not in allowed_experts:
        raise RuntimeError("FULL64_REQUIRES_ACCEPTED_PROVIDER")
    # Eager remains the exact accepted admission rail.  Production uses the
    # DeepseekV4-aware SDPA adapter, which replaces only softmax(QK^T)V while
    # retaining compressor masks, sink logits, inverse value rotation, and the
    # grouped output projection in the model's own attention forward.
    config["resident_validation_attention_implementation"] = "eager"
    # The immutable accepted W28 producer retained the incoming score geometry
    # through engine construction and admission.  Production-only scheduling
    # is installed after the exact W28 gate below.
    # Bind the physical boundary before the sole engine is constructed.
    if adopted_w28 and not adopt_keep_provider:
        os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "1"
    else:
        os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "0"
    os.environ.pop("FAST_K2_SEALED_COMPLETE_EXPERT_BF16", None)
    os.environ.pop("FAST_K2_SEALED_PROJECTION_BF16", None)

    os.environ["NCCL_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    os.environ["GLOO_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=900))
    if torch.distributed.get_world_size() != 2 or torch.distributed.get_rank() != rank:
        raise RuntimeError("DIST_GEOMETRY_MISMATCH")

    process_started = time.perf_counter()
    load_started = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    engine = ModernGreenResidentEngine(payload=payload, config=config, rank=rank,
                                       layer_ranges={0: (0, 20), 1: (21, 42)})
    del payload
    torch.cuda.synchronize()
    resident_load_seconds = time.perf_counter() - load_started

    if os.environ.get("WHOLE_CHAIN_BISECT_ONLY", "0") == "1":
        _whole_chain_bisect(engine, window=28, root=root, rank=rank, pin=pin)
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
        admission = api.validate(engine, (28,), config["validation_teacher_root"])
        torch.cuda.synchronize()
        admission_wall = time.perf_counter() - admission_started
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

    # W28 is now sealed against the immutable accepted producer.  Only after
    # that exact gate may production install its scheduling and post-source
    # BF16 boundaries.
    config["score_window_batch_size"] = 2
    config["score_pair_stream_concurrency"] = 1
    config["score_pipeline_overlap"] = True
    config["attention_query_chunk_size"] = 512
    # Keep the adopted PlaneSource physical matrix boundary through scoring.
    # Switching provider families after construction is inert and was the
    # measured 11.72911233476603/top1=0 suffix failure.
    os.environ["FAST_K2_SEALED_PROJECTION_BF16"] = "0"

    # The packed provider was selected before engine construction whenever W28
    # was adopted. Keep that single provider identity through production.
    # The same admitted inverse-transform→BF16 physical-weight boundary remains
    # active through canary and production; do not revert to the fused-float path.
    # Keep the exact immutable W28 producer's reference FWHT backend through
    # production. Attempt54 measured that switching this already-loaded engine
    # to Quack changed batch0 from exact W28 parity to 10.40359289672382/top1=1.
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
        canary = api.validate(engine, ep_canary, config["validation_teacher_root"])
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
    score_receipt_dir = Path(config["score_resume_root"]).expanduser().resolve() / "receipts"
    try:
        score_receipt_dir.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError("SCORE_RECEIPT_ROOT_OUTSIDE_PHYSICAL_ROOT") from error
    full = validate_full64_batches(
        api, engine, windows, Path(config["validation_teacher_root"]), score_receipt_dir,
        rank=rank, canonical_code_commit=pin,
        adopted_prefix_code_commit="ae27abc53f3ca69f6efa9a64c1f6e1d4f0193d1e",
        batch_size=2,
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
