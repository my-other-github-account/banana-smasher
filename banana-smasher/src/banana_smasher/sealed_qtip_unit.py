"""Read immutable producer wire; never regenerate its LUT or inverse transform."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

FORMAT = "banana-smasher-qtip-unit-v1"


def load_sealed_unit(root: Path, row: Mapping[str, Any]):
    """Verify and load one physical unit without allocating decoded weights."""
    import torch

    wire = row["wire"]
    binding = wire["unit"]
    path = (root / binding["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("sealed QTIP unit escapes artifact root")
    payload = path.read_bytes()
    if len(payload) != binding["bytes"] or hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError("sealed QTIP unit hash/size mismatch")
    import io
    unit = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if unit.get("schema") != FORMAT or unit.get("geometry") != wire["geometry"]:
        raise ValueError("sealed QTIP unit schema/geometry mismatch")
    shape = wire["unit_shape"]
    if unit.get("shape") != shape or len(shape) != 2:
        raise ValueError("sealed QTIP unit shape mismatch")
    m, k = shape
    geometry = unit["geometry"]
    if geometry["K"] not in (1, 2, 3, 4) or geometry["L"] != 16 or geometry["V"] != 2:
        raise ValueError("unsupported sealed QTIP unit geometry")
    if m <= 0 or k <= 0 or m % 32 or k % 32:
        raise ValueError("sealed QTIP unit requires 32-aligned shape")
    if (unit["trellis"].dtype != torch.uint16
            or unit["trellis"].numel() != geometry["K"] * m * k // 16
            or list(unit["SU"].shape) != [k] or list(unit["SV"].shape) != [m]
            or unit["Wscale"].numel() != 1
            or list(unit["tlut"].shape) != [1 << geometry["tlut_bits"], 2]):
        raise ValueError("sealed QTIP unit control/wire shape mismatch")
    bounds = wire["row_range"]
    if bounds not in ([0, m], [0, m // 2], [m // 2, m]):
        raise ValueError("sealed QTIP unit row range must be full or fused13 half")
    start, stop = bounds
    if list(row["shape"]) != [stop - start, k]:
        raise ValueError("sealed QTIP unit logical shape mismatch")
    if row.get("source_transform", {}).get("output_quantity") != "descaled_weight":
        raise ValueError("sealed QTIP unit must remain descaled")
    if not all(torch.isfinite(unit[key]).all() for key in ("tlut", "SU", "SV", "Wscale")):
        raise ValueError("sealed QTIP controls must be finite")
    return unit


def decode_sealed_unit(root: Path, row: Mapping[str, Any]):
    import torch
    from . import qtip_kernel_decompress, qtip_runner

    unit = load_sealed_unit(root, row)
    start, stop = row["wire"]["row_range"]
    # Same exact producer decoder, including stored TLUT, kernel swizzle and RHT.
    decoded = qtip_runner.decode_packed_weight(unit, qtip_kernel_decompress, torch.device("cpu"))
    return decoded[start:stop]
