"""Resident controller for exact cross-unit QTIP2 build batches."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .qtip_batch import build_qtip_batch


_ACTIVE_BUILD_ACCELERATIONS = (
    "persistent-prefix-full16",
    "kernel-cache",
    "shared-capture/single-process-staging",
    "batched-block-LDL",
    "cross-unit-LDLQ",
    "FWHT",
    "bounded-batch-matrix-lifetime",
    "canonical-pack-from-states",
    "packed-byte-reconstruction",
)


def _common(label: str, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError(f"QTIP batch lacks {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"QTIP batch requires one {label}")
    return first


def main_batch(
    config_paths: Sequence[Path],
    root: Path,
    layer: int,
    *,
    kernel_cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Solve and seal same-shape current K2 configs in one resident process.

    Existing successful units are never recomputed.  Callers that mix existing
    and missing units must validate/filter them before invoking this function.
    """
    from . import solver_qtip_profile as solver_module

    if not config_paths:
        raise ValueError("QTIP cross-unit batch requires at least one config")
    paths = [Path(path).resolve() for path in config_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("QTIP cross-unit batch contains duplicate configs")
    configs = [solver_module._read_qtip_config(path) for path in paths]
    if any(int(config["layer"]) != layer for config in configs):
        raise ValueError("QTIP batch config layer differs from selected layer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    identities = [
        (
            int(config["expert"]),
            solver_module.validate_qtip_projection(config["projection"]),
        )
        for config in configs
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("QTIP cross-unit batch contains duplicate unit identities")
    for path in paths:
        existing = solver_module._validated_existing_unit(
            path, root, layer, profile_mode=False
        )
        if existing is not None:
            raise RuntimeError(
                f"QTIP cross-unit batch refuses to recompute sealed unit: {path}"
            )

    outer_started = time.perf_counter()
    epoch_started = time.time()
    basis_gates = [solver_module._verify_basis(config, root) for config in configs]
    _common("model basis", [gate["index_sha256"] for gate in basis_gates])

    runner_bindings = [
        solver_module._declared_public_qtip_runner(config) for config in configs
    ]
    runner_path, runner_sha256 = _common("public runner binding", runner_bindings)
    qtip_roots = [solver_module._config_path(config, "qtip_root") for config in configs]
    qtip_root = _common("QTIP source root", qtip_roots)
    model_roots = [solver_module._config_path(config, "model_root") for config in configs]
    model_root = _common("model root", model_roots)
    tlut_paths = [solver_module._config_path(config, "tlut_source") for config in configs]
    tlut_path = _common("TLUT source", tlut_paths)
    geometries = [config.get("geometry", {"L": 16, "K": 3, "V": 2}) for config in configs]
    geometry = _common("geometry", geometries)
    sealed_geometry = tuple(int(geometry[key]) for key in ("L", "K", "V"))
    if sealed_geometry[0] != 16 or sealed_geometry[2] != 2 or sealed_geometry[1] not in (1, 2, 3, 4):
        raise ValueError(
            f"cross-unit controller supports L16/V2 with K in 1..4, got {sealed_geometry}"
        )
    codebooks = [solver_module._resolve_config_codebook(config, geometry) for config in configs]
    codebook = _common("codebook identity", codebooks)

    runner = solver_module._load_public_qtip_runner(runner_path, runner_sha256)
    runner.QTIP = qtip_root
    bitshift, _ldlq, _math_utils, kernel_decode = runner.load_official_qtip()
    from .glm_qtip_source_adapter import bind_source_closure
    source_closure = bind_source_closure(model_root, configs, runner, {
        "bitshift": bitshift, "ldlq": _ldlq, "math_utils": _math_utils,
        "kernel_decompress": kernel_decode,
    })
    from . import qtip_viterbi as exact

    references = [
        torch.load(
            solver_module._config_path(config, "reference_unit"),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        for config in configs
    ]
    seeds = [
        solver_module._resolve_rht_seed(
            config,
            reference,
            layer=layer,
            expert=identity[0],
            projection=identity[1],
        )
        for config, reference, identity in zip(
            configs, references, identities, strict=True
        )
    ]
    rht_seeds = [seed for seed, _policy in seeds]
    pinned_tlut = solver_module._load_tlut(tlut_path)
    expected_tlut_digests = [str(reference["tlut_sha256"]) for reference in references]
    actual_tlut_digest = solver_module._tensor_sha256(pinned_tlut)
    if any(digest != actual_tlut_digest for digest in expected_tlut_digests):
        raise RuntimeError("TLUT digest differs from a sealed reference unit")

    codebook_instance = bitshift.bitshift_codebook(
        L=int(geometry["L"]),
        K=int(geometry["K"]),
        V=int(geometry["V"]),
        tlut_bits=int(codebook["tlut_bits"]),
        decode_mode=str(codebook["decode_mode"]),
        tlut=pinned_tlut.to("cuda"),
    ).to("cuda")
    kernel_prepare_started = time.perf_counter()
    from .qtip_kernel_cache import build_qtip_kernels

    kernel_bpw = configs[0].get("bpw")
    if not isinstance(kernel_bpw, str):
        kernel_bpw = "2.00"
    if any(
        (config.get("bpw") if isinstance(config.get("bpw"), str) else "2.00")
        != kernel_bpw
        for config in configs
    ):
        raise ValueError("QTIP batch requires one kernel bpw identity")
    kernel_cache = build_qtip_kernels(kernel_bpw, cache_root=kernel_cache_root)
    timers = solver_module._ExactTimers()
    solver_identity = solver_module._install_configured_viterbi(
        codebook_instance,
        exact,
        timers,
        configs[0],
        profile_mode=False,
    )
    if any(
        solver_module.backend_for_geometry(sealed_geometry)
        != config.get("backend", solver_module.backend_for_geometry(sealed_geometry))
        for config in configs
    ):
        raise ValueError("QTIP batch contains a divergent solver backend")
    kernel_prepare_seconds = time.perf_counter() - kernel_prepare_started

    hessian_bindings = [
        solver_module._bind_hessian_layer_manifest(config, layer=layer)
        for config in configs
    ]
    capture_binding = _common(
        "Hessian capture binding",
        [(str(root_path), windows, binding["sha256"]) for root_path, windows, binding in hessian_bindings],
    )
    capture_root = Path(capture_binding[0])
    fit_window_count = int(capture_binding[1])
    hessian_binding = hessian_bindings[0][2]
    captures = solver_module._load_captures(
        capture_root, layer, fit_window_count
    )
    device = torch.device("cuda")
    fit_windows_batch = []
    fit_sources = []
    source_weights = []
    source_refs = []
    pack_binding: Mapping[str, object] | None = None
    try:
        for config, identity in zip(configs, identities, strict=True):
            expert, projection = identity
            fit_windows, fit_source = solver_module._prepare_fit_windows(
                runner,
                captures,
                model_root=model_root,
                layer=layer,
                expert=expert,
                projection=projection,
                device=device,
            )
            source_weight, source_ref = solver_module._load_weight(
                model_root, layer, expert, projection
            )
            binding = solver_module._bind_public_runner_pack_contract(
                codebook_instance, config, source_weight
            )
            if binding is None:
                raise RuntimeError("current QTIP config lacks canonical pack contract")
            if pack_binding is None:
                pack_binding = binding
            elif dict(binding) != dict(pack_binding):
                raise ValueError("QTIP batch requires one canonical packed shape")
            fit_windows_batch.append(fit_windows)
            fit_sources.append(fit_source)
            source_weights.append(source_weight)
            source_refs.append(source_ref)
    finally:
        solver_module._release_capture_bank(
            capture_root, layer, fit_window_count, captures
        )
        del captures
    staging_seconds = (
        time.perf_counter() - outer_started - kernel_prepare_seconds
    )

    torch.cuda.reset_peak_memory_stats()
    build_started = time.perf_counter()
    candidates, batch_build = build_qtip_batch(
        runner,
        source_weights,
        fit_windows_batch,
        codebook_instance,
        kernel_decode,
        device,
        rht_seeds,
    )
    torch.cuda.synchronize()
    build_wall_seconds = time.perf_counter() - build_started
    if len(candidates) != len(paths):
        raise RuntimeError("QTIP batch builder returned the wrong unit count")

    artifact_rows = []
    artifact_fsync_seconds = []
    for path, config, identity, source_weight, candidate in zip(
        paths, configs, identities, source_weights, candidates, strict=True
    ):
        expert, projection = identity
        solver_module._validate_candidate_packed_shape(
            candidate, config, source_weight
        )
        solver_module._bind_candidate_geometry(candidate, config)
        reconstructed = candidate.pop("reconstructed_weight", None)
        if reconstructed is None:
            raise RuntimeError("QTIP batch omitted reconstructed_weight before wire seal")
        out = root / "solve" / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
        out.mkdir(parents=True, exist_ok=True)
        artifact_path = out / "QTIP_UNIT.pt"
        candidate["schema"] = solver_module._QTIP_UNIT_PAYLOAD_SCHEMA
        started = time.perf_counter()
        solver_module._atomic_torch(artifact_path, candidate)
        artifact_fsync_seconds.append(time.perf_counter() - started)
        artifact_rows.append(
            {
                "path": artifact_path,
                "sha256": solver_module._sha256(artifact_path),
                "candidate": candidate,
                "reconstructed": reconstructed,
            }
        )

    units = len(paths)
    mean_build_seconds = build_wall_seconds / units
    mean_staging_seconds = staging_seconds / units
    mean_kernel_prepare_seconds = kernel_prepare_seconds / units
    mean_build_phases = {
        name: float(seconds) / units
        for name, seconds in batch_build["phase_seconds"].items()
    }
    mean_decode_seconds = mean_build_phases["packed_decode_conformance"]
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    receipt_rows: list[dict[str, Any]] = []
    receipt_paths = []
    receipt_fsync_seconds = []
    batch_process = solver_module._process_receipt()
    batch_id_payload = json.dumps(
        [path.name for path in paths], separators=(",", ":")
    ).encode()
    batch_id = hashlib.sha256(batch_id_payload).hexdigest()[:16]

    for unit, (
        path,
        config,
        identity,
        basis_gate,
        source_ref,
        fit_source,
        seed_row,
        artifact_row,
    ) in enumerate(
        zip(
            paths,
            configs,
            identities,
            basis_gates,
            source_refs,
            fit_sources,
            seeds,
            artifact_rows,
            strict=True,
        )
    ):
        expert, projection = identity
        seed, seed_policy = seed_row
        phase_seconds = {
            "staging": mean_staging_seconds,
            "kernel_prepare": mean_kernel_prepare_seconds,
            "progress_receipt_fsync": 0.0,
            "solve": mean_build_seconds,
            "solve_core_excluding_packed_decode_conformance": max(
                0.0, mean_build_seconds - mean_decode_seconds
            ),
            "packed_decode_conformance": mean_decode_seconds,
            "artifact_fsync": artifact_fsync_seconds[unit],
            "receipt_fsync": 0.0,
            "remainder": 0.0,
        }
        unit_build = {
            "schema": "banana-smasher-qtip-cross-unit-build-member-v1",
            "status": "PASS",
            "rht_seed": seed,
            "quant_seconds": mean_build_phases["batched_ldlq"],
            "fit_rows": batch_build["fit_rows"][unit],
            "fit_route_mass": batch_build["fit_route_mass"][unit],
            "canonical_pack": batch_build["canonical_pack"][unit],
            "packed_decode": batch_build["packed_decode"][unit],
            "phase_seconds": mean_build_phases,
            "batch": {
                "schema": batch_build["schema"],
                "implementation": batch_build["implementation"],
                "batch_id": batch_id,
                "batch_units": units,
                "member_index": unit,
                "batch_wall_seconds": batch_build["batch_wall_seconds"],
                "mean_build_wall_seconds": batch_build["mean_build_wall_seconds"],
                "phase_seconds": batch_build["phase_seconds"],
                "solver_geometry": batch_build["solver_geometry"],
                "matrix_lifetime": batch_build["matrix_lifetime"],
                "independent_unit_state": True,
            },
        }
        receipt = {
            "schema": solver_module._QTIP_SOLVE_RECEIPT_SCHEMA,
            "status": "PASS",
            "host": os.uname().nodename,
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "fresh_no_warm_start": True,
            "public_command_config": str(path),
            "config_sha256": solver_module._sha256(path),
            "basis_gate": basis_gate,
            "epoch_started": epoch_started,
            "epoch_ended": time.time(),
            "total_wall_seconds": 0.0,
            "staging_seconds": mean_staging_seconds,
            "solve_seconds": mean_build_seconds,
            "phase_seconds": phase_seconds,
            "assignment_sha256": solver_module._tensor_sha256(
                artifact_row["candidate"]["trellis"]
            ),
            "artifact": str(artifact_row["path"]),
            "artifact_sha256": artifact_row["sha256"],
            "viterbi_launches": timers.calls,
            "viterbi_sequences": timers.sequences // units,
            "transition_decisions": (
                (timers.sequences // units)
                * (int(solver_identity["steps"]) - 1)
                * int(solver_identity["branches_per_prefix"])
            ),
            "solver": solver_identity,
            "kernel_cache": kernel_cache,
            "build": unit_build,
            "source_weight": source_ref,
            "glm_source_closure": source_closure,
            "fit_source": fit_source,
            "fit_windows": fit_window_count,
            "hessian_layer_manifest": hessian_binding,
            "rht_seed": seed,
            "rht_seed_policy": seed_policy,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "cross_unit_batch": {
                "batch_id": batch_id,
                "batch_units": units,
                "member_index": unit,
                "process": batch_process,
                "config_names": [batch_path.name for batch_path in paths],
            },
        }
        receipt = solver_module._public_receipt(receipt)
        receipt_path = (
            root
            / "solve"
            / f"L{layer:03d}"
            / f"E{expert:03d}_{projection}"
            / "QTIP_SOLVE_RECEIPT.json"
        )
        started = time.perf_counter()
        solver_module._atomic_json(receipt_path, receipt)
        receipt_fsync_seconds.append(time.perf_counter() - started)
        receipt_paths.append(receipt_path)
        receipt_rows.append(receipt)

    total_batch_wall_seconds = time.perf_counter() - outer_started
    mean_total_seconds = total_batch_wall_seconds / units
    for unit, (receipt, receipt_path) in enumerate(
        zip(receipt_rows, receipt_paths, strict=True)
    ):
        receipt["total_wall_seconds"] = mean_total_seconds
        receipt["epoch_ended"] = time.time()
        receipt["phase_seconds"]["receipt_fsync"] = receipt_fsync_seconds[unit]
        closed = sum(
            float(value)
            for name, value in receipt["phase_seconds"].items()
            if name not in {"remainder", "solve_core_excluding_packed_decode_conformance"}
        )
        receipt["phase_seconds"]["remainder"] = max(0.0, mean_total_seconds - closed)
        solver_module._atomic_json(receipt_path, receipt)

    aggregate = {
        "schema": "banana-smasher-qtip-cross-unit-controller-v1",
        "status": "PASS",
        "layer": layer,
        "batch_id": batch_id,
        "batch_units": units,
        "config_names": [path.name for path in paths],
        "basis_sha256": basis_gates[0]["index_sha256"],
        "process": batch_process,
        "total_wall_seconds": total_batch_wall_seconds,
        "mean_total_wall_seconds": mean_total_seconds,
        "staging_seconds": staging_seconds,
        "mean_staging_seconds": mean_staging_seconds,
        "kernel_prepare_seconds": kernel_prepare_seconds,
        "build_wall_seconds": build_wall_seconds,
        "mean_build_wall_seconds": mean_build_seconds,
        "build_phase_seconds": batch_build["phase_seconds"],
        "mean_build_phase_seconds": mean_build_phases,
        "matrix_lifetime": batch_build["matrix_lifetime"],
        "accelerations": {
            "schema": "banana-smasher-qtip-active-build-accelerations-v1",
            "active": list(_ACTIVE_BUILD_ACCELERATIONS),
            "historical_k3_alternating_branch_pruning": False,
        },
        "solver": solver_identity,
        "assignment_sha256": [receipt["assignment_sha256"] for receipt in receipt_rows],
        "artifact_sha256": [row["sha256"] for row in artifact_rows],
        "canonical_pack_roundtrip_exact": [
            row["canonical_pack"]["canonical_pack_roundtrip_exact"]
            for row in (receipt["build"] for receipt in receipt_rows)
        ],
        "packed_decode_fp16_bit_exact": [
            row["packed_decode"]["fp16_bit_exact"]
            for row in (receipt["build"] for receipt in receipt_rows)
        ],
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "receipt_names": [path.name for path in receipt_paths],
        "epoch_started": epoch_started,
        "epoch_ended": time.time(),
    }
    aggregate_path = (
        root
        / "solve"
        / f"L{layer:03d}"
        / "batches"
        / batch_id
        / "QTIP_CROSS_UNIT_BATCH_RECEIPT.json"
    )
    solver_module._atomic_json(
        aggregate_path, solver_module._public_receipt(aggregate)
    )
    return receipt_rows


__all__ = ["main_batch"]
