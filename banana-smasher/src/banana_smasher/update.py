from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OS_MEMORY_FLOOR_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class TokenWindow:
    input_ids: np.ndarray[Any, Any]
    teacher_mask: np.ndarray[Any, Any]
    positions: np.ndarray[Any, Any]
    receipt: dict[str, Any]


def plan_token_window(
    *,
    requested_tokens: int | None,
    bytes_per_token: int,
    available_os_bytes: int | None,
    available_device_bytes: int | None,
) -> int:
    """Choose an exact physical token count before any tensor allocation."""
    if available_os_bytes is None:
        raise RuntimeError("cannot determine OS memory capacity")
    if available_device_bytes is None:
        raise RuntimeError("cannot determine accelerator memory capacity")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    usable_os = int(available_os_bytes) - OS_MEMORY_FLOOR_BYTES
    if usable_os <= 0:
        raise MemoryError("available memory cannot preserve the 4 GiB OS floor")
    capacity = min(usable_os // int(bytes_per_token), int(available_device_bytes) // int(bytes_per_token))
    if capacity < 1:
        raise MemoryError("capacity is insufficient for one physical token while preserving the 4 GiB OS floor")
    if requested_tokens is None:
        return int(capacity)
    if isinstance(requested_tokens, bool) or int(requested_tokens) <= 0:
        raise ValueError("requested tokens must be a positive integer")
    requested = int(requested_tokens)
    if requested > capacity:
        raise MemoryError(
            f"{requested} physical tokens exceed capacity {capacity} while preserving the 4 GiB OS floor"
        )
    return requested


def build_token_window(
    input_ids: Any,
    *,
    teacher_mask: Any,
    positions: Any,
    tokens: int,
) -> TokenWindow:
    """Build one batch-1 physical window without silently shrinking ``tokens``."""
    if isinstance(tokens, bool) or int(tokens) <= 0:
        raise ValueError("tokens must be a positive integer")
    tokens = int(tokens)
    ids = np.asarray(input_ids)
    mask = np.asarray(teacher_mask)
    position_values = np.asarray(positions)
    if ids.ndim != 1 or mask.ndim != 1 or position_values.ndim != 1:
        raise ValueError("input ids, teacher mask, and positions must be rank-1")
    if not (ids.shape == mask.shape == position_values.shape):
        raise ValueError("input ids, teacher mask, and positions must have identical geometry")
    if ids.shape[0] < tokens:
        raise ValueError(
            f"source cannot provide exactly {tokens} physical tokens; available={ids.shape[0]}"
        )
    ids = np.ascontiguousarray(ids[:tokens]).reshape(1, tokens)
    mask = np.ascontiguousarray(mask[:tokens], dtype=np.bool_).reshape(1, tokens)
    position_values = np.ascontiguousarray(position_values[:tokens]).reshape(1, tokens)
    if tokens > 1 and not np.array_equal(np.diff(position_values[0]), np.ones(tokens - 1, dtype=position_values.dtype)):
        raise ValueError("positions must be contiguous with unit stride")
    receipt = {
        "observed_tensor_shape": [1, tokens],
        "teacher_mask_shape": [1, tokens],
        "teacher_mask_true": int(mask.sum()),
        "position_shape": [1, tokens],
        "position_first": int(position_values[0, 0]),
        "position_last": int(position_values[0, -1]),
        "physical_tokens": tokens,
    }
    return TokenWindow(ids, mask, position_values, receipt)


def _available_os_memory() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        pages = 0
        page_size = 0
    available = pages * page_size
    if available > 0:
        return available
    try:
        completed = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    page_match = re.search(r"page size of (\d+) bytes", completed.stdout)
    if page_match is None:
        return None
    reusable_pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{label}:\s+(\d+)\.", completed.stdout, re.MULTILINE)
        if match is not None:
            reusable_pages += int(match.group(1))
    available = reusable_pages * int(page_match.group(1))
    return available if available > 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_file_update(
    *,
    source: str | Path,
    output: str | Path,
    identity: str | Path,
    tokens: int,
    segments: int,
    learning_rate: float,
    depth: int,
    reference: bool,
    restart: bool,
) -> dict[str, Any]:
    """Run one public file-backed update without an implicit backend fallback."""
    import torch

    from .production import ProductionTrainableSurface
    from .update_engine import run_segmented_update

    source_path = Path(source)
    identity_path = Path(identity)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"update input must be a regular NPZ file: {source_path}")
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError(f"update identity must be a regular JSON file: {identity_path}")
    identity_value = json.loads(identity_path.read_text())
    if identity_value.get("content_sha256") != _sha256(source_path):
        raise RuntimeError("update input content SHA-256 does not match identity")
    if segments <= 0 or segments > tokens:
        raise ValueError("segments must be between one and the physical token count")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not reference and not torch.cuda.is_available():
        raise RuntimeError(
            "accelerated update requires CUDA; pass --reference for the explicit debug implementation"
        )

    with np.load(source_path, allow_pickle=False) as archive:
        required = {"input_ids", "teacher_mask", "positions", "features", "targets"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"update input is missing arrays: {missing}")
        window = build_token_window(
            archive["input_ids"],
            teacher_mask=archive["teacher_mask"],
            positions=archive["positions"],
            tokens=tokens,
        )
        features = np.asarray(archive["features"])
        targets = np.asarray(archive["targets"])
    if features.ndim != 2 or targets.shape != features.shape:
        raise ValueError("features and targets must have identical rank-2 geometry")
    if features.shape[0] < tokens:
        raise ValueError(
            f"feature source cannot provide exactly {tokens} physical tokens; available={features.shape[0]}"
        )
    width = int(features.shape[1])
    if width <= 0:
        raise ValueError("feature width must be positive")
    bytes_per_token = int(features[:1].nbytes + targets[:1].nbytes + 24)
    available_os = _available_os_memory()
    plan_token_window(
        requested_tokens=tokens,
        bytes_per_token=bytes_per_token,
        available_os_bytes=available_os,
        available_device_bytes=(
            int(torch.cuda.mem_get_info()[0]) if not reference else available_os
        ),
    )

    device = torch.device("cpu" if reference else "cuda")
    surface = ProductionTrainableSurface(depth=depth, width=width)
    surface.to(device)
    feature_tensor = torch.as_tensor(
        np.ascontiguousarray(features[:tokens]), dtype=torch.float32, device=device
    ).reshape(1, tokens, width)
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(targets[:tokens]), dtype=torch.float32, device=device
    ).reshape(1, tokens, width)
    mask_tensor = torch.as_tensor(window.teacher_mask, device=device)
    index_chunks = [chunk for chunk in np.array_split(np.arange(tokens), segments) if len(chunk)]
    work = [
        (
            feature_tensor[:, chunk, :],
            target_tensor[:, chunk, :],
            mask_tensor[:, chunk],
        )
        for chunk in index_chunks
    ]
    optimizer = torch.optim.Adam(list(surface.parameters()), lr=float(learning_rate))

    def loss_sum(segment: tuple[Any, Any, Any]) -> Any:
        values, expected, teacher = segment
        selected = teacher.unsqueeze(-1).expand_as(expected)
        if not bool(selected.any()):
            raise RuntimeError("teacher mask selects no update targets")
        return (surface(values) - expected).square()[selected].sum()

    if not reference:
        torch.cuda.reset_peak_memory_stats()
    receipt_fields = {
        **window.receipt,
        "observed_tensor_shape": [1, tokens, width],
        "execution": {"forward": True, "backward": True, "optimizer": True},
        "memory_preflight": {
            "available_os_bytes": available_os,
            "os_floor_bytes": OS_MEMORY_FLOOR_BYTES,
            "bytes_per_token": bytes_per_token,
        },
    }
    result = run_segmented_update(
        parameters=list(surface.parameters()),
        optimizer=optimizer,
        segments=work,
        item_count=lambda segment: int(segment[2].sum()),
        loss_sum=loss_sum,
        output=output,
        identity=identity_value,
        backend="reference" if reference else "accelerated",
        restart=restart,
        receipt_fields=receipt_fields,
        synchronize=(lambda: None) if reference else torch.cuda.synchronize,
    )
    result["peak_memory_bytes"] = (
        0 if reference else int(torch.cuda.max_memory_allocated())
    )
    from .update_checkpoint import atomic_json

    atomic_json(
        Path(result["receipt"]),
        result,
    )
    return result
