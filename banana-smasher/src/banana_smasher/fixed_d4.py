from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_SCHEMA = "banana-smasher-fixed-d4-materialization-v1"
_PROJECTIONS = ("down", "fused13")
_TIER_SPECS = {
    "d4_k2048": {"k": 2048, "bits": 11},
    "d4_k4096": {"k": 4096, "bits": 12},
}
_HEX = frozenset("0123456789abcdef")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _bound_array(root: Path, value: object, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a bound array object")
    relative = value.get("path")
    expected_bytes = value.get("bytes")
    expected_sha = value.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be a relative file")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label}.path escapes the manifest root")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not _is_sha256(expected_sha)
    ):
        raise ValueError(f"{label} requires bytes and lowercase SHA-256")
    path = root / relative_path
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte count mismatch")
    if _sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    value_array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value_array, np.ndarray):
        raise ValueError(f"{label} must be one NPY array")
    return value_array, {
        "path": relative_path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def _verify_basis_index(root: Path, value: object, *, basis_sha256: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("basis_index must be a bound source-model index")
    declared_sha = value.get("sha256")
    if declared_sha != basis_sha256:
        raise ValueError("basis_index SHA-256 must equal basis_sha256")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("basis_index.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        actual_sha = _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read basis_index {path}: {exc}") from exc
    if actual_sha != basis_sha256:
        raise ValueError(
            f"basis_index SHA-256 mismatch: expected {basis_sha256}, got {actual_sha}"
        )


def _packed_assignments(
    assignments: np.ndarray,
    *,
    label: str,
    k: int,
    bits_per_assignment: int,
) -> bytes:
    _validate_assignments(assignments, label=label, k=k)
    values = assignments.astype(np.int64, copy=False)
    rows = values.astype(np.uint16).reshape(256, -1)
    bits = (
        (rows[..., None] >> np.arange(bits_per_assignment, dtype=np.uint16)) & 1
    ).astype(np.uint8)
    packed = np.packbits(bits.reshape(256, -1), axis=1, bitorder="little")
    return packed.tobytes(order="C")


def _validate_assignments(assignments: np.ndarray, *, label: str, k: int) -> None:
    if assignments.ndim < 2 or assignments.shape[0] != 256:
        raise ValueError(f"{label} must have 256 expert rows")
    if assignments.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must have an integer dtype")
    rows = assignments.reshape(256, -1)
    for expert in range(256):
        row = rows[expert]
        for start in range(0, row.size, 1 << 20):
            chunk = row[start : start + (1 << 20)]
            if chunk.size and (int(chunk.min()) < 0 or int(chunk.max()) >= k):
                raise ValueError(f"{label} contains an assignment outside D4K{k}")


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
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


def persist_fixed_d4_solve(
    output_root: str | Path,
    *,
    tier: str,
    layer: int,
    basis_index: str | Path,
    basis_sha256: str,
    projections: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Persist exact in-memory solve winners before the solver releases them.

    Exact solvers call this once per completed layer while their winner arrays
    are still available.  The durable result is the ordinary public
    materialization manifest consumed by :func:`materialize_fixed_d4`.
    """

    if tier not in _TIER_SPECS:
        raise ValueError("fixed D4 solve persistence requires tier d4_k2048 or d4_k4096")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("fixed D4 solve persistence layer must be non-negative")
    if not _is_sha256(basis_sha256):
        raise ValueError("basis_sha256 must be a lowercase SHA-256")
    basis_index = Path(basis_index).expanduser().resolve()
    basis_payload = basis_index.read_bytes()
    if _sha256(basis_payload) != basis_sha256:
        raise ValueError("fixed D4 solve persistence basis_index SHA-256 mismatch")
    if set(projections) != set(_PROJECTIONS):
        raise ValueError("fixed D4 solve persistence requires down and fused13 projections")

    spec = _TIER_SPECS[tier]
    k = spec["k"]
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    assignment_count = 0
    try:
        basis_name = "model.safetensors.index.json"
        _atomic_write(staging / basis_name, basis_payload)
        persisted: dict[str, dict[str, dict[str, object]]] = {}
        for projection in _PROJECTIONS:
            row = projections[projection]
            if not isinstance(row, Mapping) or set(row) != {
                "assignments",
                "scales",
                "codebook",
            }:
                raise ValueError(
                    f"fixed D4 solve projection {projection} requires assignments, scales, and codebook"
                )
            assignments = np.asarray(row["assignments"])
            scales = np.asarray(row["scales"])
            codebook = np.asarray(row["codebook"])
            _validate_assignments(
                assignments, label=f"projections.{projection}.assignments", k=k
            )
            if scales.dtype != np.uint8 or scales.ndim < 2 or scales.shape[0] != 256:
                raise ValueError(
                    f"projections.{projection}.scales must be uint8 with 256 expert rows"
                )
            if codebook.shape != (k, 4) or codebook.dtype.kind != "f":
                raise ValueError(
                    f"projections.{projection}.codebook must be floating [{k}, 4]"
                )
            persisted[projection] = {}
            for field, value in (
                ("assignments", assignments),
                ("scales", scales),
                ("codebook", codebook),
            ):
                name = f"{projection}.{field}.npy"
                path = staging / name
                _atomic_save_npy(path, value)
                persisted[projection][field] = {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            assignment_count += int(assignments.size)

        manifest = {
            "schema": _SCHEMA,
            "tier": tier,
            "layer": layer,
            "basis_sha256": basis_sha256,
            "basis_index": {"path": basis_name, "sha256": basis_sha256},
            "projections": persisted,
        }
        _atomic_write(
            staging / "materialize.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.rename(staging, output_root)
        directory = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema": "banana-smasher-fixed-d4-solve-persistence-v1",
        "status": "PASS",
        "tier": tier,
        "layer": layer,
        "basis_sha256": basis_sha256,
        "assignment_count": assignment_count,
        "manifest": str(output_root / "materialize.json"),
    }


def _exact_nearest_assignments(
    normalized_vectors: np.ndarray,
    codebook: np.ndarray,
    *,
    chunk_vectors: int,
) -> np.ndarray:
    if (
        normalized_vectors.dtype.kind != "f"
        or normalized_vectors.ndim < 3
        or normalized_vectors.shape[0] != 256
        or normalized_vectors.shape[-1] != 4
    ):
        raise ValueError(
            "normalized_vectors must be floating [256, ..., 4] D4 objective vectors"
        )
    if not np.isfinite(codebook).all():
        raise ValueError("fixed D4 exact solve inputs must be finite")
    vectors = normalized_vectors.reshape(-1, 4)
    candidates = np.asarray(codebook, dtype=np.float64)
    candidate_norms = np.einsum("ij,ij->i", candidates, candidates)
    winner_dtype = np.uint16 if codebook.shape[0] > 256 else np.uint8
    winners = np.empty(vectors.shape[0], dtype=winner_dtype)
    for start in range(0, vectors.shape[0], chunk_vectors):
        chunk = np.asarray(
            vectors[start : start + chunk_vectors], dtype=np.float64
        )
        if not np.isfinite(chunk).all():
            raise ValueError("fixed D4 exact solve inputs must be finite")
        distances = (
            np.einsum("ij,ij->i", chunk, chunk)[:, None]
            + candidate_norms[None, :]
            - 2.0 * (chunk @ candidates.T)
        )
        winners[start : start + chunk.shape[0]] = np.argmin(distances, axis=1)
    return winners.reshape(normalized_vectors.shape[:-1])


def solve_fixed_d4_exact(
    config_path: str | Path,
    output_root: str | Path,
    *,
    basis_sha256: str,
) -> dict[str, Any]:
    """Exhaustively solve D4 codebook winners and persist them before release."""

    config_path = Path(config_path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fixed D4 exact solve config {config_path}: {exc}") from exc
    required = {
        "schema",
        "tier",
        "layer",
        "basis_index",
        "basis_sha256",
        "chunk_vectors",
        "projections",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError(
            "fixed D4 exact solve config fields mismatch: "
            f"expected={sorted(required)}"
        )
    if config.get("schema") != "banana-smasher-fixed-d4-exact-solve-v1":
        raise ValueError(
            "fixed D4 exact solve schema must be banana-smasher-fixed-d4-exact-solve-v1"
        )
    if config.get("basis_sha256") != basis_sha256:
        raise ValueError("fixed D4 exact solve basis mismatch")
    tier = config.get("tier")
    if tier not in _TIER_SPECS:
        raise ValueError("fixed D4 exact solve requires tier d4_k2048 or d4_k4096")
    layer = config.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("fixed D4 exact solve layer must be non-negative")
    chunk_vectors = config.get("chunk_vectors")
    if (
        isinstance(chunk_vectors, bool)
        or not isinstance(chunk_vectors, int)
        or chunk_vectors < 1
    ):
        raise ValueError("fixed D4 exact solve chunk_vectors must be positive")
    projections = config.get("projections")
    if not isinstance(projections, Mapping) or set(projections) != set(_PROJECTIONS):
        raise ValueError("fixed D4 exact solve requires down and fused13 projections")

    k = _TIER_SPECS[str(tier)]["k"]
    solved: dict[str, dict[str, np.ndarray]] = {}
    vector_count = 0
    for projection in _PROJECTIONS:
        row = projections[projection]
        expected_fields = {"normalized_vectors", "scales", "codebook"}
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(
                f"fixed D4 exact solve projection {projection} requires "
                "normalized_vectors, scales, and codebook"
            )
        vectors, _ = _bound_array(
            config_path.parent,
            row["normalized_vectors"],
            label=f"projections.{projection}.normalized_vectors",
        )
        scales, _ = _bound_array(
            config_path.parent,
            row["scales"],
            label=f"projections.{projection}.scales",
        )
        codebook, _ = _bound_array(
            config_path.parent,
            row["codebook"],
            label=f"projections.{projection}.codebook",
        )
        if codebook.shape != (k, 4) or codebook.dtype.kind != "f":
            raise ValueError(
                f"projections.{projection}.codebook must be floating [{k}, 4]"
            )
        if scales.dtype != np.uint8 or scales.ndim < 2 or scales.shape[0] != 256:
            raise ValueError(
                f"projections.{projection}.scales must be uint8 with 256 expert rows"
            )
        assignments = _exact_nearest_assignments(
            vectors, codebook, chunk_vectors=chunk_vectors
        )
        solved[projection] = {
            "assignments": assignments,
            "scales": scales,
            "codebook": codebook,
        }
        vector_count += int(assignments.size)

    basis_index = Path(str(config["basis_index"])).expanduser()
    if not basis_index.is_absolute():
        basis_index = config_path.parent / basis_index
    persisted = persist_fixed_d4_solve(
        output_root,
        tier=str(tier),
        layer=layer,
        basis_index=basis_index,
        basis_sha256=basis_sha256,
        projections=solved,
    )
    return {
        **persisted,
        "schema": "banana-smasher-fixed-d4-exact-solve-receipt-v1",
        "solver": "exhaustive-nearest-d4-v1",
        "vector_count": vector_count,
        "config_sha256": _sha256(config_path.read_bytes()),
    }


def verify_fixed_d4_model(
    model_root: str | Path, *, basis_sha256: str
) -> dict[str, Any]:
    """Verify a serveable pack and every fixed-D4 layer/basis binding."""

    from .contract import load_manifest, verify_pack

    model_root = Path(model_root).expanduser().resolve()
    verify_pack(model_root)
    manifest = load_manifest(model_root)
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not all(
        isinstance(layer, int) and not isinstance(layer, bool) for layer in layers
    ):
        raise ValueError("fixed D4 model pack has invalid layers")
    receipts = sorted((model_root / "provenance").glob("layer_*/LAYER_RECEIPT.json"))
    if not receipts:
        single = model_root / "provenance" / "LAYER_RECEIPT.json"
        if single.is_file():
            receipts = [single]
    receipt_layers: set[int] = set()
    for path in receipts:
        try:
            receipt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid fixed D4 layer receipt {path}: {exc}") from exc
        layer = receipt.get("layer")
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError(f"fixed D4 model has invalid layer receipt {path}")
        if receipt.get("tier") not in _TIER_SPECS or receipt.get(
            "basis_sha256"
        ) != basis_sha256:
            raise ValueError(f"fixed D4 model basis mismatch in {path}")
        receipt_layers.add(layer)
    if receipt_layers != set(layers):
        raise ValueError("fixed D4 receipts do not match the model pack layer set")
    return manifest


def _dense_logprobs(value: object, *, label: str) -> list[float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} did not return full-vocabulary log probabilities")
    parsed: dict[int, float] = {}
    for raw_token, raw_logprob in value.items():
        if isinstance(raw_token, bool):
            raise ValueError(f"{label} returned an invalid token id")
        try:
            token = int(raw_token)
            numeric = float(getattr(raw_logprob, "logprob", raw_logprob))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} returned an invalid log probability") from exc
        if token < 0 or not np.isfinite(numeric):
            raise ValueError(f"{label} returned an invalid log probability")
        parsed[token] = numeric
    if set(parsed) != set(range(len(parsed))):
        raise ValueError(
            f"{label} logprobs=-1 result is not a dense zero-based vocabulary"
        )
    return [parsed[token] for token in range(len(parsed))]


def produce_fixed_d4_logits(
    model_root: str | Path,
    producer_config: str | Path,
    bank_path: str | Path,
    output_path: str | Path,
    *,
    basis_sha256: str,
) -> dict[str, Any]:
    """Run the materialized fixed-D4 pack through public vLLM offline inference."""

    model_root = Path(model_root).expanduser().resolve()
    producer_config = Path(producer_config).expanduser().resolve()
    bank_path = Path(bank_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    verify_fixed_d4_model(model_root, basis_sha256=basis_sha256)
    try:
        config_payload = producer_config.read_bytes()
        config = json.loads(config_payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fixed D4 producer config {producer_config}: {exc}") from exc
    if (
        not isinstance(config, Mapping)
        or config.get("schema") != "banana-smasher-candidate-producer-v1"
        or config.get("producer") != "fixed-d4-vllm"
        or set(config) != {"schema", "producer", "parameters"}
    ):
        raise ValueError(
            "fixed D4 producer requires candidate-producer-v1 with producer fixed-d4-vllm"
        )
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "input_field",
        "batch_size",
        "engine",
    }:
        raise ValueError("fixed D4 producer parameters require input_field, batch_size, and engine")
    input_field = parameters.get("input_field")
    batch_size = parameters.get("batch_size")
    engine = parameters.get("engine")
    if not isinstance(input_field, str) or not input_field:
        raise ValueError("fixed D4 producer input_field must be a non-empty string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("fixed D4 producer batch_size must be positive")
    if not isinstance(engine, Mapping) or any(
        not isinstance(key, str) or not key for key in engine
    ):
        raise ValueError("fixed D4 producer engine must be an object")
    forbidden_engine = {"model", "runner", "task"} & set(engine)
    if forbidden_engine:
        raise ValueError(
            f"fixed D4 producer engine cannot override {sorted(forbidden_engine)}"
        )

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(bank_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid bank JSON at {bank_path}:{line_number}: {exc}") from exc
        tokens = row.get(input_field) if isinstance(row, Mapping) else None
        if (
            not isinstance(row, dict)
            or "window_id" not in row
            or not isinstance(tokens, list)
            or not tokens
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
        ):
            raise ValueError(
                f"bank row {line_number} requires window_id and non-empty integer {input_field}"
            )
        rows.append(row)
    if not rows:
        raise ValueError("fixed D4 producer bank is empty")

    try:
        vllm = importlib.import_module("vllm")
        LLM = vllm.LLM
        SamplingParams = vllm.SamplingParams
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            "fixed-d4-vllm producer requires the public banana-smasher vLLM runtime"
        ) from exc
    llm = LLM(model=str(model_root), **dict(engine))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=-1,
    )
    produced: list[dict[str, Any]] = []
    vocab_size: int | None = None
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        requests = [
            {"prompt_token_ids": row[input_field]}
            for row in batch
        ]
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        if len(outputs) != len(batch):
            raise ValueError("fixed D4 producer returned the wrong request count")
        for row, request_output in zip(batch, outputs):
            choices = getattr(request_output, "outputs", None)
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"window {row['window_id']!r} returned no model log probabilities"
                )
            sampled_logprobs = getattr(choices[0], "logprobs", None)
            if isinstance(sampled_logprobs, list):
                if len(sampled_logprobs) != 1:
                    raise ValueError(
                        f"window {row['window_id']!r} returned the wrong generated-token count"
                    )
                sampled_logprobs = sampled_logprobs[0]
            logits = _dense_logprobs(
                sampled_logprobs,
                label=f"window {row['window_id']!r}",
            )
            if vocab_size is None:
                vocab_size = len(logits)
            elif len(logits) != vocab_size:
                raise ValueError("fixed D4 producer vocabulary size drifted")
            produced.append(
                {
                    "window_id": row["window_id"],
                    "logits": logits,
                }
            )
    payload = b"".join(
        (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode()
        for row in produced
    )
    _atomic_write(output_path, payload)
    return {
        "schema": "banana-smasher-fixed-d4-producer-receipt-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "rows": len(produced),
        "vocab_size": vocab_size,
        "producer_config_sha256": _sha256(config_payload),
        "output_sha256": _sha256(payload),
        "output": str(output_path),
    }


def materialize_fixed_d4(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    basis_sha256: str,
) -> dict[str, Any]:
    """Materialize bound full fixed-D4 assignments as executable bs-pack wire planes."""

    if not _is_sha256(basis_sha256):
        raise ValueError("basis_sha256 must be a lowercase SHA-256")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
        raise ValueError(f"materialization manifest schema must be {_SCHEMA}")
    tier = manifest.get("tier")
    if tier not in _TIER_SPECS:
        raise ValueError("fixed D4 materialization requires tier d4_k2048 or d4_k4096")
    spec = _TIER_SPECS[str(tier)]
    k = spec["k"]
    bits_per_assignment = spec["bits"]
    if manifest.get("basis_sha256") != basis_sha256:
        raise ValueError("fixed D4 materialization basis mismatch")
    _verify_basis_index(
        manifest_path.parent,
        manifest.get("basis_index"),
        basis_sha256=basis_sha256,
    )
    layer = manifest.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("fixed D4 materialization layer must be non-negative")
    projections = manifest.get("projections")
    if not isinstance(projections, Mapping) or set(projections) != set(_PROJECTIONS):
        raise ValueError("fixed D4 materialization requires down and fused13 projections")

    output_root = Path(output_root).expanduser().resolve()
    layer_root = output_root / f"layer_{layer:03d}"
    if layer_root.exists():
        raise FileExistsError(layer_root)
    layer_root.mkdir(parents=True)

    files: list[dict[str, object]] = []
    sources: dict[str, Any] = {}
    assignment_count = 0
    for projection in _PROJECTIONS:
        row = projections[projection]
        if not isinstance(row, Mapping):
            raise ValueError(f"projection {projection} must be an object")
        assignments, assignment_source = _bound_array(
            manifest_path.parent,
            row.get("assignments"),
            label=f"projections.{projection}.assignments",
        )
        scales, scale_source = _bound_array(
            manifest_path.parent,
            row.get("scales"),
            label=f"projections.{projection}.scales",
        )
        codebook, codebook_source = _bound_array(
            manifest_path.parent,
            row.get("codebook"),
            label=f"projections.{projection}.codebook",
        )
        if scales.dtype != np.uint8 or scales.ndim < 2 or scales.shape[0] != 256:
            raise ValueError(f"projections.{projection}.scales must be uint8 with 256 expert rows")
        if codebook.shape != (k, 4) or codebook.dtype.kind != "f":
            raise ValueError(
                f"projections.{projection}.codebook must be floating [{k}, 4]"
            )

        payloads = {
            f"{tier}.{projection}.codebook.fp16.bin": np.ascontiguousarray(
                codebook, dtype="<f2"
            ).tobytes(),
            f"{tier}.{projection}.codes.le{bits_per_assignment}.bin": _packed_assignments(
                assignments,
                label=f"projections.{projection}.assignments",
                k=k,
                bits_per_assignment=bits_per_assignment,
            ),
            f"{tier}.{projection}.expert_ids.i16.bin": np.arange(
                256, dtype="<i2"
            ).tobytes(),
            f"{tier}.{projection}.scales.e8m0.bin": np.ascontiguousarray(
                scales, dtype=np.uint8
            ).tobytes(),
        }
        for name, payload in payloads.items():
            path = layer_root / name
            _atomic_write(path, payload)
            files.append(_file_record(path))
        assignment_count += int(assignments.size)
        sources[projection] = {
            "assignments": assignment_source,
            "scales": scale_source,
            "codebook": codebook_source,
            "assignment_count": int(assignments.size),
            "assignment_dtype": str(assignments.dtype),
        }

    receipt = {
        "schema": "banana_smasher-materialized-layer-v1",
        "status": "PASS",
        "layer": layer,
        "tier": tier,
        "basis_sha256": basis_sha256,
        "basis_index_sha256": basis_sha256,
        "materialization_manifest_sha256": _sha256(manifest_payload),
        "assignment_count": assignment_count,
        "assignment_dtype": f"source-bound-integer; wire=le{bits_per_assignment}",
        "sources": sources,
        "files": sorted(files, key=lambda row: str(row["path"])),
    }
    _atomic_write(
        layer_root / "LAYER_RECEIPT.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "schema": "banana-smasher-fixed-d4-materialization-receipt-v1",
        "status": "PASS",
        "layer": layer,
        "tier": tier,
        "basis_sha256": basis_sha256,
        "assignment_count": assignment_count,
        "output": str(layer_root),
        "receipt": str(layer_root / "LAYER_RECEIPT.json"),
    }
