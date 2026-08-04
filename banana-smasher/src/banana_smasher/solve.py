from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class SolveInputError(ValueError):
    """The public solve-input bundle is incomplete or inconsistent."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bundle_payload(
    root: Path, raw: object, *, field: str
) -> tuple[bytes, dict[str, object]]:
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
        raise SolveInputError(f"{field} must bind path, bytes, and sha256")
    relative = raw["path"]
    expected_bytes = raw["bytes"]
    expected_sha256 = raw["sha256"]
    if not isinstance(relative, str) or not relative:
        raise SolveInputError(f"{field}.path must be a non-empty relative path")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise SolveInputError(f"{field}.bytes must be a non-negative integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise SolveInputError(f"{field}.sha256 must be lowercase SHA-256")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SolveInputError(f"{field}.path escapes the solve-input root")
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise SolveInputError(f"{field}.path must not contain a symlink")
    if not candidate.is_file():
        raise SolveInputError(f"{field}.path is missing: {relative}")
    payload = candidate.read_bytes()
    if len(payload) != expected_bytes:
        raise SolveInputError(f"{field} byte count mismatch")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SolveInputError(f"{field} sha256 mismatch")
    return payload, {
        "path": relative_path.as_posix(),
        "bytes": len(payload),
        "sha256": actual_sha256,
    }


def _load_matrix(payload: bytes, *, field: str) -> np.ndarray:
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise SolveInputError(f"{field} must be one rank-2 NPY array")
    if value.shape[1] != 4:
        raise SolveInputError(f"{field} must use the exact D=4 geometry")
    if value.shape[0] == 0:
        raise SolveInputError(f"{field} must contain at least one row")
    if value.dtype != np.dtype("float32"):
        raise SolveInputError(f"{field} must have dtype float32")
    if not np.isfinite(value).all():
        raise SolveInputError(f"{field} contains NaN or infinity")
    return np.ascontiguousarray(value)


def _load_manifest(root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    manifest_path = root / "solve.json"
    if not manifest_path.is_file():
        raise SolveInputError(f"solve-input manifest is missing: {manifest_path}")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise SolveInputError("solve.json must contain one JSON object")
    if manifest.get("schema") != "banana-smasher-solve-input-v1":
        raise SolveInputError(f"unsupported solve-input schema: {manifest.get('schema')!r}")
    layer = manifest.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise SolveInputError("solve-input layer must be a non-negative integer")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SolveInputError("solve-input cells must be a non-empty array")
    return manifest, {
        "path": "solve.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run_solve(
    source_root: str | Path,
    output: str | Path,
    *,
    device: str = "cuda",
    reference_search: bool = False,
    verbose_receipts: bool = False,
) -> dict[str, Any]:
    """Run exact full-codebook search for every declared solve cell.

    The ordinary path requires CUDA plus Triton and never silently falls back.
    The exhaustive implementation is an explicit hidden developer/CI mode.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "solve requires torch; install torch on a supported host"
        ) from exc

    from . import exact_codebook

    source_root = Path(source_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise SolveInputError(f"solve-input root is missing: {source_root}")
    if output.exists():
        raise FileExistsError(output)

    manifest, manifest_evidence = _load_manifest(source_root)
    backend = "reference-search" if reference_search else "exact-gemm"
    if not reference_search:
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("exact-gemm requires a CUDA device; no reference fallback is performed")
        if exact_codebook.triton is None:
            raise RuntimeError(
                "exact-gemm requires Triton; install torch and Triton on a supported CUDA host"
            )
    candidate_count: int | None = None
    seen_cells: set[str] = set()
    input_evidence: list[dict[str, object]] = []
    prepared_cells: list[dict[str, Any]] = []

    for index, raw_cell in enumerate(manifest["cells"]):
        if not isinstance(raw_cell, dict):
            raise SolveInputError(f"cells[{index}] must be an object")
        cell = raw_cell.get("cell")
        if not isinstance(cell, str) or not cell or cell in seen_cells:
            raise SolveInputError(f"cells[{index}].cell must be a unique non-empty string")
        seen_cells.add(cell)
        vectors_payload, vectors_evidence = _bundle_payload(
            source_root, raw_cell.get("vectors"), field=f"cells[{index}].vectors"
        )
        codebook_payload, codebook_evidence = _bundle_payload(
            source_root, raw_cell.get("codebook"), field=f"cells[{index}].codebook"
        )
        input_evidence.extend(
            (
                {"cell": cell, "field": "vectors", **vectors_evidence},
                {"cell": cell, "field": "codebook", **codebook_evidence},
            )
        )
        vectors_array = _load_matrix(vectors_payload, field=f"cells[{index}].vectors")
        codebook_array = _load_matrix(codebook_payload, field=f"cells[{index}].codebook")
        if codebook_array.shape[0] < 2:
            raise SolveInputError(f"cells[{index}].codebook needs at least two candidates")
        if candidate_count is None:
            candidate_count = int(codebook_array.shape[0])
        elif candidate_count != int(codebook_array.shape[0]):
            raise SolveInputError("all solve cells must use one common candidate count")
        if not reference_search and candidate_count % 64:
            raise SolveInputError("exact-gemm candidate count must be divisible by 64")

        frozen_bucket = raw_cell.get("frozen_bucket")
        prepared_bucket: dict[str, Any] | None = None
        if frozen_bucket is not None:
            if not isinstance(frozen_bucket, dict):
                raise SolveInputError(f"cells[{index}].frozen_bucket must be an object")

            def load_bucket_array(field: str) -> np.ndarray:
                payload, evidence = _bundle_payload(
                    source_root,
                    frozen_bucket.get(field),
                    field=f"cells[{index}].frozen_bucket.{field}",
                )
                input_evidence.append(
                    {"cell": cell, "field": f"frozen_bucket.{field}", **evidence}
                )
                value = np.load(io.BytesIO(payload), allow_pickle=False)
                if not isinstance(value, np.ndarray):
                    raise SolveInputError(
                        f"cells[{index}].frozen_bucket.{field} must be one NPY array"
                    )
                return np.ascontiguousarray(value)

            options = frozen_bucket.get("options")
            vector_width = frozen_bucket.get("vector_width")
            if not isinstance(options, list) or not options or not all(
                isinstance(option, str) and option for option in options
            ):
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket.options must be non-empty strings"
                )
            if len(set(options)) != len(options):
                raise SolveInputError(f"cells[{index}].frozen_bucket.options must be unique")
            if (
                not isinstance(vector_width, int)
                or isinstance(vector_width, bool)
                or vector_width < 1
            ):
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket.vector_width must be positive"
                )
            prepared_bucket = {
                "options": options,
                "vector_width": vector_width,
                "weights": load_bucket_array("weights"),
                "h": load_bucket_array("h"),
                "codes": load_bucket_array("codes"),
                "scales": load_bucket_array("scales"),
                "codebooks": load_bucket_array("codebooks"),
                "codebook_offsets": load_bucket_array("codebook_offsets"),
            }
            if int(prepared_bucket["codes"].shape[0]) != len(options):
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket options/codes count mismatch"
                )

        prepared_cells.append(
            {
                "cell": cell,
                "vectors": vectors_array,
                "codebook": codebook_array,
                "frozen_bucket": prepared_bucket,
            }
        )

    started = time.perf_counter()
    winners_by_cell: dict[str, np.ndarray] = {}
    bucket_scores_by_cell: dict[str, np.ndarray] = {}
    bucket_rows: list[dict[str, Any]] = []
    verbose_cells: list[dict[str, Any]] = []
    total_rows = 0

    for prepared in prepared_cells:
        cell = prepared["cell"]
        vectors = torch.from_numpy(prepared["vectors"]).to(device)
        codebook = torch.from_numpy(prepared["codebook"]).to(device)
        if reference_search:
            winners = exact_codebook.exhaustive_reference_winners(vectors, codebook)
            details: dict[str, Any] = {"rows": int(vectors.shape[0])}
        else:
            winners, details = exact_codebook.exact_codebook_winners(vectors, codebook)
        winners_by_cell[cell] = winners.detach().cpu().numpy().astype(np.int64, copy=False)

        prepared_bucket = prepared["frozen_bucket"]
        if prepared_bucket is not None:
            from . import frozen_score

            options = prepared_bucket["options"]
            weights = torch.from_numpy(prepared_bucket["weights"]).to(
                device=device, dtype=torch.bfloat16
            )
            h = torch.from_numpy(prepared_bucket["h"]).to(
                device=device, dtype=torch.float32
            )
            codes = torch.from_numpy(prepared_bucket["codes"]).to(device)
            scales = torch.from_numpy(prepared_bucket["scales"]).to(device)
            bucket_codebooks = torch.from_numpy(prepared_bucket["codebooks"]).to(
                device=device, dtype=torch.float32
            )
            offsets = torch.from_numpy(prepared_bucket["codebook_offsets"]).to(device)
            scorer = (
                frozen_score.reference_frozen_weighted_errors
                if reference_search
                else frozen_score.fused_frozen_weighted_errors
            )
            bucket_scores = scorer(
                weights,
                h,
                codes,
                scales,
                bucket_codebooks,
                offsets,
                vector_width=prepared_bucket["vector_width"],
            )
            bucket_score_array = (
                bucket_scores.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            winner_index = int(bucket_scores.argmin())
            bucket_scores_by_cell[cell] = bucket_score_array
            bucket_rows.append(
                {
                    "cell": cell,
                    "options": options,
                    "winner_index": winner_index,
                    "winner": options[winner_index],
                }
            )

        total_rows += int(vectors.shape[0])
        verbose_cells.append({"cell": cell, **details})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    artifact_name = "winners.npz"
    receipt_name = "SOLVE_RECEIPT.json"
    bucket_artifact_name = "bucket_scores.npz"
    try:
        artifact = temporary_output / artifact_name
        _atomic_npz(artifact, winners_by_cell)
        bucket_artifact: Path | None = None
        if bucket_scores_by_cell:
            bucket_artifact = temporary_output / bucket_artifact_name
            _atomic_npz(bucket_artifact, bucket_scores_by_cell)
        elapsed = time.perf_counter() - started
        receipt_path = temporary_output / receipt_name
        artifact_payload = artifact.read_bytes()
        receipt: dict[str, Any] = {
            "schema": "banana-smasher-solve-receipt-v1",
            "status": "PASS",
            "command": "solve",
            "backend": backend,
            "layer": int(manifest["layer"]),
            "shape": {
                "cells": len(winners_by_cell),
                "rows": total_rows,
                "candidates": int(candidate_count or 0),
            },
            "elapsed_seconds": elapsed,
            "artifact": artifact_name,
            "artifact_bytes": len(artifact_payload),
            "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "input_manifest": manifest_evidence,
            "inputs": input_evidence,
            "receipt": receipt_name,
        }
        if bucket_artifact is not None:
            bucket_payload = bucket_artifact.read_bytes()
            receipt["bucket_artifact"] = bucket_artifact_name
            receipt["bucket_artifact_bytes"] = len(bucket_payload)
            receipt["bucket_artifact_sha256"] = hashlib.sha256(bucket_payload).hexdigest()
            receipt["buckets"] = bucket_rows
        if verbose_receipts:
            receipt["verbose"] = {
                "cells": verbose_cells,
            }
        _atomic_json(receipt_path, receipt)
        _fsync_directory(temporary_output)
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary_output, output)
        _fsync_directory(output.parent)
    finally:
        shutil.rmtree(temporary_output, ignore_errors=True)
    return {
        "status": receipt["status"],
        "command": receipt["command"],
        "backend": receipt["backend"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "artifact": receipt["artifact"],
        "receipt": receipt["receipt"],
    }
