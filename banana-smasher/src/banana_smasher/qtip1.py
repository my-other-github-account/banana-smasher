"""Small public QTIP bitshift-trellis primitive and declarative ring provider.

The transition, cyclic Viterbi, and bit packing contracts are independently
implemented from the public QTIP definition and checked against
Cornell-RelaxML/qtip commit e90c6688c8dfae326a3a81b5eb032db7c6680ec0.
No campaign source or generated model artifact is required.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_QTIP_PROVIDER_SCHEMA = "banana-smasher-qtip-provider-v1"
_QTIP_WIRE_SCHEMA = "banana-smasher-qtip-wire-v1"
Identity = tuple[int, int, str]


@dataclass(frozen=True)
class QtipGeometry:
    L: int
    K: int
    V: int
    tlut_bits: int = 9
    decode_mode: str = "quantlut"

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.L, self.K, self.V, self.tlut_bits)
        ):
            raise ValueError(f"invalid QTIP geometry: {self!r}")
        if self.L < 2 * self.branch_bits:
            raise ValueError(
                f"QTIP L={self.L} must retain two K*V={self.branch_bits} prefixes"
            )
        if self.tlut_bits > self.L:
            raise ValueError("QTIP tlut_bits cannot exceed L")
        if self.decode_mode not in {"lut", "quantlut", "quantlut_sym"}:
            raise ValueError(f"unsupported QTIP decode mode: {self.decode_mode!r}")
        if self.decode_mode == "quantlut_sym" and self.V != 2:
            raise ValueError("canonical quantlut_sym requires V=2")
        if self.decode_mode == "quantlut_sym" and self.tlut_bits >= self.L:
            raise ValueError("canonical quantlut_sym requires tlut_bits < L")
        if self.decode_mode != "lut" and self.L > 16:
            raise ValueError("canonical QTIP quantlut uses a 16-bit hash lane")

    @property
    def branch_bits(self) -> int:
        return self.K * self.V

    @property
    def states(self) -> int:
        return 1 << self.L

    @property
    def bpw(self) -> int:
        return self.K

    def as_mapping(self) -> dict[str, object]:
        return {
            "L": self.L,
            "K": self.K,
            "V": self.V,
            "tlut_bits": self.tlut_bits,
            "decode_mode": self.decode_mode,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "QtipGeometry":
        if not isinstance(value, Mapping):
            raise ValueError("QTIP geometry must be an object")
        required = {"L", "K", "V", "tlut_bits", "decode_mode"}
        if set(value) != required:
            raise ValueError(f"QTIP geometry requires exactly {sorted(required)}")
        return cls(
            L=value["L"],  # type: ignore[arg-type]
            K=value["K"],  # type: ignore[arg-type]
            V=value["V"],  # type: ignore[arg-type]
            tlut_bits=value["tlut_bits"],  # type: ignore[arg-type]
            decode_mode=value["decode_mode"],  # type: ignore[arg-type]
        )


QTIP1_GEOMETRY = QtipGeometry(L=16, K=1, V=1, tlut_bits=9, decode_mode="quantlut")
QTIP2_GEOMETRY = QtipGeometry(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym")


@dataclass(frozen=True)
class QtipProviderComponent:
    name: str
    geometry: QtipGeometry
    quarters: int
    backend: str

    def __post_init__(self) -> None:
        if not self.name or not self.backend:
            raise ValueError("QTIP provider component name/backend cannot be empty")
        if isinstance(self.quarters, bool) or not isinstance(self.quarters, int):
            raise ValueError("QTIP provider component quarters must be integral")
        if not 1 <= self.quarters <= 4:
            raise ValueError("QTIP provider component quarters must be in 1..4")

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "geometry": self.geometry.as_mapping(),
            "quarters": self.quarters,
            "backend": self.backend,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "QtipProviderComponent":
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "geometry",
            "quarters",
            "backend",
        }:
            raise ValueError("invalid QTIP provider component")
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            geometry=QtipGeometry.from_mapping(value["geometry"]),
            quarters=value["quarters"],  # type: ignore[arg-type]
            backend=value["backend"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class QtipProviderDeclaration:
    provider_id: str
    tier: str
    runtime_family: str
    components: tuple[QtipProviderComponent, ...]

    def __post_init__(self) -> None:
        if not self.provider_id or not self.tier or not self.runtime_family:
            raise ValueError("QTIP provider identity fields cannot be empty")
        if not self.components or sum(row.quarters for row in self.components) != 4:
            raise ValueError("QTIP provider components must total four quarters")
        names = [row.name for row in self.components]
        geometries = [row.geometry for row in self.components]
        if len(set(names)) != len(names) or len(set(geometries)) != len(geometries):
            raise ValueError("QTIP provider components must be unique")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": _QTIP_PROVIDER_SCHEMA,
            "id": self.provider_id,
            "tier": self.tier,
            "runtime_family": self.runtime_family,
            "components": [row.as_mapping() for row in self.components],
        }

    @classmethod
    def from_mapping(cls, value: object) -> "QtipProviderDeclaration":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "id",
            "tier",
            "runtime_family",
            "components",
        }:
            raise ValueError("invalid QTIP provider declaration")
        if value["schema"] != _QTIP_PROVIDER_SCHEMA:
            raise ValueError("unsupported QTIP provider schema")
        rows = value["components"]
        if not isinstance(rows, list):
            raise ValueError("QTIP provider components must be a list")
        return cls(
            provider_id=value["id"],  # type: ignore[arg-type]
            tier=value["tier"],  # type: ignore[arg-type]
            runtime_family=value["runtime_family"],  # type: ignore[arg-type]
            components=tuple(QtipProviderComponent.from_mapping(row) for row in rows),
        )


def qtip1_provider_declaration() -> QtipProviderDeclaration:
    return QtipProviderDeclaration(
        provider_id="qtip1",
        tier="qtip@1.00",
        runtime_family="qtip",
        components=(
            QtipProviderComponent(
                name="qtip1-k1v1",
                geometry=QTIP1_GEOMETRY,
                quarters=4,
                backend="canonical-bitshift-k1v1-v1",
            ),
        ),
    )


def qtip1_5_provider_declaration() -> QtipProviderDeclaration:
    """Declare 50/50 authentic QTIP1/QTIP2 without changing generic consumers."""
    return QtipProviderDeclaration(
        provider_id="qtip1_5",
        tier="qtip@1.50",
        runtime_family="qtip",
        components=(
            QtipProviderComponent(
                name="qtip1-k1v1",
                geometry=QTIP1_GEOMETRY,
                quarters=2,
                backend="canonical-bitshift-k1v1-v1",
            ),
            QtipProviderComponent(
                name="qtip2-k2v2",
                geometry=QTIP2_GEOMETRY,
                quarters=2,
                backend="qtip-trellis-v2-graph-replay-b256-chunked-batch-exact-v46",
            ),
        ),
    )


def _identity_sort_key(identity: Identity) -> bytes:
    layer, expert, projection = identity
    return hashlib.sha256(f"{layer}:{projection}:{expert}".encode()).digest()


def _validate_identity(identity: object) -> Identity:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 3
        or isinstance(identity[0], bool)
        or not isinstance(identity[0], int)
        or identity[0] < 0
        or isinstance(identity[1], bool)
        or not isinstance(identity[1], int)
        or identity[1] < 0
        or identity[2] not in {"down", "fused13"}
    ):
        raise ValueError(f"invalid QTIP provider identity: {identity!r}")
    return identity


def assign_qtip_provider_components(
    declaration: QtipProviderDeclaration,
    identities: Iterable[Identity],
) -> dict[Identity, QtipProviderComponent]:
    """Apply the same deterministic per-layer/projection quarter assignment as QTIP2.5."""
    rows = [_validate_identity(identity) for identity in identities]
    if len(set(rows)) != len(rows):
        raise ValueError("QTIP provider identities must be unique")
    groups: dict[tuple[int, str], list[Identity]] = {}
    for identity in rows:
        groups.setdefault((identity[0], identity[2]), []).append(identity)

    assigned: dict[Identity, QtipProviderComponent] = {}
    for group in groups.values():
        ordered = sorted(group, key=_identity_sort_key)
        start = 0
        remaining = len(ordered)
        for index, component in enumerate(declaration.components):
            count = (
                remaining
                if index == len(declaration.components) - 1
                else min((len(ordered) * component.quarters + 2) // 4, remaining)
            )
            for identity in ordered[start : start + count]:
                assigned[identity] = component
            start += count
            remaining -= count
        if remaining:
            raise AssertionError("validated QTIP provider assignment did not close")
    return assigned


def qtip_provider_counts(
    declaration: QtipProviderDeclaration,
    identities: Iterable[Identity],
) -> dict[str, int]:
    assigned = assign_qtip_provider_components(declaration, identities)
    return {
        component.name: sum(value == component for value in assigned.values())
        for component in declaration.components
    }


def gaussian_tlut(*, bits: int = 9, columns: int = 2) -> np.ndarray:
    """Return a deterministic standard-normal TLUT using the stdlib inverse CDF."""
    from statistics import NormalDist

    if bits < 1 or columns < 1:
        raise ValueError("TLUT bits and columns must be positive")
    count = 1 << bits
    values = np.asarray(
        [NormalDist().inv_cdf((index + 0.5) / count) for index in range(count)],
        dtype=np.float32,
    )
    if columns == 1:
        return values[:, None]
    table = np.empty((count, columns), dtype=np.float32)
    table[:, 0] = values
    for column in range(1, columns):
        table[:, column] = np.roll(values, int((column * count) // columns))
    return table


def _state_lut(geometry: QtipGeometry, tlut: np.ndarray) -> np.ndarray:
    table = np.asarray(tlut, dtype=np.float32)
    if table.ndim == 1:
        table = table[:, None]
    if geometry.decode_mode == "lut":
        if table.shape[0] != geometry.states or table.shape[1] < geometry.V:
            raise ValueError("pure QTIP LUT does not match L/V geometry")
        return table[:, : geometry.V].T.copy()
    expected = 1 << geometry.tlut_bits
    if table.shape[0] != expected or table.shape[1] < geometry.V:
        raise ValueError("QTIP TLUT does not match tlut_bits/V geometry")
    state = np.arange(geometry.states, dtype=np.uint64)
    hashed = (state + 1) * state
    if geometry.decode_mode == "quantlut":
        indices = (hashed >> (16 - geometry.tlut_bits)) & (expected - 1)
        return table[indices.astype(np.int64), : geometry.V].T.copy()
    sign = 1.0 - 2.0 * ((hashed >> 15) & 1).astype(np.float32)
    indices = (hashed >> (16 - geometry.tlut_bits - 1)) & (expected - 1)
    result = table[indices.astype(np.int64), : geometry.V].T.copy()
    result[0] *= sign
    return result


def _viterbi(
    matrix: np.ndarray,
    state_lut: np.ndarray,
    geometry: QtipGeometry,
    overlap: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % geometry.V:
        raise ValueError("QTIP input must be [sequences, steps*V]")
    batch, width = values.shape
    steps = width // geometry.V
    if steps * geometry.branch_bits < geometry.L:
        raise ValueError("QTIP sequence is too short to close the cyclic trellis")
    vectors = values.reshape(batch, steps, geometry.V)
    states = geometry.states
    shift = geometry.branch_bits
    prefixes = 1 << (geometry.L - shift)
    branches = 1 << shift
    prefix_ids = np.arange(prefixes, dtype=np.int32)
    predecessors = prefix_ids[:, None] + (
        np.arange(branches, dtype=np.int32)[None, :] << (geometry.L - shift)
    )

    def errors(step: int) -> np.ndarray:
        delta = state_lut.T[None, :, :] - vectors[:, step, None, :]
        return np.sum(delta * delta, axis=2, dtype=np.float32)

    cost = errors(0)
    if overlap is not None:
        overlap = np.asarray(overlap, dtype=np.int32)
        if overlap.shape != (batch,) or np.any(overlap < 0) or np.any(overlap >= prefixes):
            raise ValueError("invalid QTIP cyclic overlap prefixes")
        allowed = (overlap[:, None] << shift) + np.arange(branches, dtype=np.int32)
        masked = np.full_like(cost, np.inf)
        masked[np.arange(batch)[:, None], allowed] = cost[
            np.arange(batch)[:, None], allowed
        ]
        cost = masked

    backpointers = np.empty((steps, batch, prefixes), dtype=np.int32)
    backpointers[0] = 0
    for step in range(1, steps):
        options = cost[:, predecessors]
        choice = np.argmin(options, axis=2)
        best = np.take_along_axis(options, choice[:, :, None], axis=2)[:, :, 0]
        backpointers[step] = np.take_along_axis(
            np.broadcast_to(predecessors, (batch, prefixes, branches)),
            choice[:, :, None],
            axis=2,
        )[:, :, 0]
        cost = errors(step) + np.repeat(best, branches, axis=1)

    if overlap is None:
        final = np.argmin(cost, axis=1).astype(np.int32)
    else:
        allowed = overlap[:, None] + (
            np.arange(branches, dtype=np.int32)[None, :] << (geometry.L - shift)
        )
        final = allowed[
            np.arange(batch),
            np.argmin(cost[np.arange(batch)[:, None], allowed], axis=1),
        ]
    result = np.empty((batch, steps), dtype=np.int32)
    result[:, -1] = final
    for step in range(steps - 1, 0, -1):
        prefix = result[:, step] >> shift
        result[:, step - 1] = backpointers[step, np.arange(batch), prefix]
    if np.any(result < 0) or np.any(result >= states):
        raise AssertionError(f"QTIP state reconstruction escaped 0..{states - 1}")
    return result


def encode_qtip(
    matrix: np.ndarray,
    *,
    geometry: QtipGeometry,
    tlut: np.ndarray,
    scales: np.ndarray | Sequence[float] | None = None,
) -> "EncodedQtip":
    """Encode rows with canonical cyclic bitshift Viterbi and pack their states."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % geometry.V:
        raise ValueError("QTIP matrix must be [rows, steps*V]")
    state_lut = _state_lut(geometry, tlut)
    automatic_scales = scales is None
    if automatic_scales:
        source_rms = np.sqrt(np.mean(values * values, axis=1, dtype=np.float32))
        lut_rms = np.float32(np.sqrt(np.mean(state_lut * state_lut, dtype=np.float32)))
        row_scales = np.where(source_rms == 0, 1.0, source_rms / lut_rms).astype(
            np.float32
        )
    else:
        row_scales = np.asarray(scales, dtype=np.float32)
        if row_scales.ndim == 0:
            row_scales = np.full((values.shape[0],), row_scales, dtype=np.float32)
        if row_scales.shape != (values.shape[0],):
            raise ValueError("QTIP scales must contain one value per matrix row")
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0):
        raise ValueError("QTIP scales must be finite and positive")
    steps = values.shape[1] // geometry.V

    def encode_states(candidate_scales: np.ndarray) -> np.ndarray:
        normalized = values / candidate_scales[:, None]
        rolled = np.roll(normalized, (steps // 2) * geometry.V, axis=1)
        first = _viterbi(rolled, state_lut, geometry)
        overlap = first[:, steps // 2] >> geometry.branch_bits
        return _viterbi(normalized, state_lut, geometry, overlap=overlap)

    if automatic_scales and geometry.K == 1 and geometry.V == 1:
        # The cyclic K1 path samples a biased subset of the global LUT.  Search a
        # tiny fixed scale grid and keep the lowest-error path independently per row.
        best_error = np.full((values.shape[0],), np.inf, dtype=np.float32)
        best_scales = row_scales.copy()
        states = np.empty((values.shape[0], steps), dtype=np.int32)
        for multiplier in (0.5, 0.65, 0.8, 1.0):
            candidate_scales = row_scales * np.float32(multiplier)
            candidate_states = encode_states(candidate_scales)
            unit = state_lut[:, candidate_states].transpose(1, 2, 0).reshape(values.shape)
            decoded = unit * candidate_scales[:, None]
            error = np.asarray(
                np.mean((values - decoded) ** 2, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
            improved = error < best_error
            best_error[improved] = error[improved]
            best_scales[improved] = candidate_scales[improved]
            states[improved] = candidate_states[improved]
        row_scales = best_scales
    else:
        states = encode_states(row_scales)
    packed = pack_qtip_states(states, geometry)
    return EncodedQtip(
        geometry=geometry,
        shape=(int(values.shape[0]), int(values.shape[1])),
        states=states,
        packed=packed,
        scales=row_scales,
    )


def pack_qtip_states(states: np.ndarray, geometry: QtipGeometry) -> np.ndarray:
    """Pack cyclic state paths in the canonical MSB-first uint16 wire layout."""
    values = np.asarray(states, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] * geometry.branch_bits < geometry.L:
        raise ValueError("QTIP states must be [sequences, sufficient steps]")
    if np.any(values < 0) or np.any(values >= geometry.states):
        raise ValueError("QTIP state outside geometry")
    shift = geometry.branch_bits
    suffix_mask = (1 << (geometry.L - shift)) - 1
    if values.shape[1] > 1 and np.any(
        (values[:, :-1] & suffix_mask) != (values[:, 1:] >> shift)
    ):
        raise ValueError("QTIP state path violates bitshift transitions")
    if np.any((values[:, -1] & suffix_mask) != (values[:, 0] >> shift)):
        raise ValueError("QTIP state path does not close the cyclic trellis")
    bit_count = values.shape[1] * shift
    bits = np.empty((values.shape[0], bit_count), dtype=np.uint8)
    for row in range(values.shape[0]):
        stream = [
            (int(values[row, 0]) >> offset) & 1
            for offset in range(geometry.L - 1, -1, -1)
        ]
        for state in values[row, 1:]:
            stream.extend(
                (int(state) >> offset) & 1 for offset in range(shift - 1, -1, -1)
            )
        bits[row] = np.asarray(stream[:bit_count], dtype=np.uint8)
    pad = (-bit_count) % 16
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    words = np.zeros((bits.shape[0], bits.shape[1] // 16), dtype=np.uint16)
    for offset in range(16):
        words |= bits[:, offset::16].astype(np.uint16) << (15 - offset)
    return words


def unpack_qtip_states(
    packed: np.ndarray,
    *,
    steps: int,
    geometry: QtipGeometry,
) -> np.ndarray:
    """Unpack canonical cyclic QTIP words into exact state indices."""
    words = np.asarray(packed)
    if words.dtype != np.uint16 or words.ndim != 2 or steps < 1:
        raise ValueError("QTIP packed states must be a uint16 matrix")
    bit_count = steps * geometry.branch_bits
    expected_words = math.ceil(bit_count / 16)
    if words.shape[1] != expected_words or bit_count < geometry.L:
        raise ValueError("QTIP packed word shape does not match steps/geometry")
    bits = np.empty((words.shape[0], expected_words * 16), dtype=np.uint8)
    for offset in range(16):
        bits[:, offset::16] = ((words >> (15 - offset)) & 1).astype(np.uint8)
    bits = bits[:, :bit_count]
    shift = geometry.branch_bits
    tail = bits[:, : geometry.L - shift]
    stream = np.concatenate((bits, tail), axis=1)
    states = np.empty((words.shape[0], steps), dtype=np.int32)
    first = np.zeros((words.shape[0],), dtype=np.int32)
    for offset in range(geometry.L):
        first = (first << 1) | stream[:, offset].astype(np.int32)
    states[:, 0] = first
    mask = (1 << geometry.L) - 1
    for step in range(1, steps):
        branch = np.zeros((words.shape[0],), dtype=np.int32)
        start = geometry.L + (step - 1) * shift
        for offset in range(shift):
            branch = (branch << 1) | stream[:, start + offset].astype(np.int32)
        states[:, step] = ((states[:, step - 1] << shift) & mask) + branch
    return states


@dataclass(frozen=True)
class EncodedQtip:
    geometry: QtipGeometry
    shape: tuple[int, int]
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray

    @property
    def weights(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def code_bpw(self) -> float:
        return self.packed.nbytes * 8 / self.weights


def decode_qtip(encoded: EncodedQtip, *, tlut: np.ndarray) -> np.ndarray:
    steps = encoded.shape[1] // encoded.geometry.V
    states = unpack_qtip_states(
        encoded.packed,
        steps=steps,
        geometry=encoded.geometry,
    )
    state_lut = _state_lut(encoded.geometry, tlut)
    vectors = state_lut[:, states].transpose(1, 2, 0)
    decoded = vectors.reshape(encoded.shape).astype(np.float32)
    return decoded * encoded.scales[:, None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, root: Path, data_bytes: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "data_bytes": data_bytes,
        "sha256": _sha256(path),
    }


def _write_qtip_wire_into(
    root: str | Path,
    *,
    declaration: QtipProviderDeclaration,
    matrices: Mapping[Identity, np.ndarray],
    tlut: np.ndarray,
) -> dict[str, Any]:
    """Write one exact provider wire into a new staging directory."""
    if not matrices:
        raise ValueError("QTIP wire requires at least one matrix")
    output = Path(root).resolve()
    if output.exists():
        raise FileExistsError(f"QTIP wire output already exists: {output}")
    table = np.asarray(tlut, dtype=np.float32)
    if table.ndim != 2 or not table.size or not np.isfinite(table).all():
        raise ValueError("shared QTIP TLUT must be a finite non-empty matrix")
    assignments = assign_qtip_provider_components(declaration, matrices)
    for identity, raw_matrix in matrices.items():
        matrix = np.asarray(raw_matrix)
        component = assignments[identity]
        if (
            matrix.ndim != 2
            or matrix.size == 0
            or matrix.shape[1] % component.geometry.V
            or matrix.shape[1] // component.geometry.V * component.geometry.branch_bits
            < component.geometry.L
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                f"QTIP wire member {identity!r} must be a finite non-empty matrix "
                "with sufficient geometry-aligned columns"
            )
    output.mkdir(parents=True)
    tlut_path = output / "shared_tlut.npy"
    np.save(tlut_path, table, allow_pickle=False)
    tlut_artifact = _artifact(tlut_path, root=output, data_bytes=table.nbytes)
    members: list[dict[str, object]] = []
    total_weights = 0
    index_data_bytes = 0
    scale_data_bytes = 0
    member_physical_bytes = 0
    for identity in sorted(matrices):
        component = assignments[identity]
        encoded = encode_qtip(
            matrices[identity],
            geometry=component.geometry,
            tlut=table,
        )
        layer, expert, projection = identity
        stem = f"L{layer:03d}_E{expert:03d}_{projection}.{component.name}"
        packed_path = output / f"{stem}.trellis.npy"
        scale_path = output / f"{stem}.scales.npy"
        np.save(packed_path, encoded.packed, allow_pickle=False)
        np.save(scale_path, encoded.scales, allow_pickle=False)
        packed_artifact = _artifact(
            packed_path, root=output, data_bytes=encoded.packed.nbytes
        )
        scale_artifact = _artifact(
            scale_path, root=output, data_bytes=encoded.scales.nbytes
        )
        total_weights += encoded.weights
        index_data_bytes += encoded.packed.nbytes
        scale_data_bytes += encoded.scales.nbytes
        member_physical_bytes += int(packed_artifact["bytes"]) + int(
            scale_artifact["bytes"]
        )
        members.append(
            {
                "identity": {
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                },
                "component": component.name,
                "geometry": component.geometry.as_mapping(),
                "shape": list(encoded.shape),
                "weights": encoded.weights,
                "code_bpw": encoded.code_bpw,
                "trellis": packed_artifact,
                "scales": scale_artifact,
                "tlut": {
                    "path": tlut_artifact["path"],
                    "sha256": tlut_artifact["sha256"],
                    "columns": component.geometry.V,
                },
            }
        )
    counts = {
        component.name: sum(row["component"] == component.name for row in members)
        for component in declaration.components
    }
    data_wire_bytes = index_data_bytes + scale_data_bytes + table.nbytes
    physical_wire_bytes = member_physical_bytes + int(tlut_artifact["bytes"])
    receipt: dict[str, Any] = {
        "schema": _QTIP_WIRE_SCHEMA,
        "status": "PASS",
        "provider": declaration.as_mapping(),
        "counts": counts,
        "members": members,
        "shared_tlut": tlut_artifact,
        "accounting": {
            "weights": total_weights,
            "index_data_bytes": index_data_bytes,
            "scale_data_bytes": scale_data_bytes,
            "shared_tlut_data_bytes": table.nbytes,
            "wire_data_bytes": data_wire_bytes,
            "wire_file_bytes": physical_wire_bytes,
            "code_bpw": index_data_bytes * 8 / total_weights,
            "wire_data_bpw": data_wire_bytes * 8 / total_weights,
            "wire_file_bpw": physical_wire_bytes * 8 / total_weights,
        },
    }
    (output / "QTIP_WIRE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def write_qtip_wire(
    root: str | Path,
    *,
    declaration: QtipProviderDeclaration,
    matrices: Mapping[Identity, np.ndarray],
    tlut: np.ndarray,
) -> dict[str, Any]:
    """Materialize one provider wire transactionally with a shared TLUT."""
    output = Path(root).resolve()
    if output.exists():
        raise FileExistsError(f"QTIP wire output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    staging = staging_parent / "wire"
    try:
        receipt = _write_qtip_wire_into(
            staging,
            declaration=declaration,
            matrices=matrices,
            tlut=tlut,
        )
        os.replace(staging, output)
        return receipt
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


class QtipWireConsumer:
    """Generic runtime-side reader; dispatch is by declared component geometry."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_receipt_sha256: str | None = None,
    ):
        self.root = Path(root).resolve()
        receipt_path = self.root / "QTIP_WIRE.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid QTIP wire receipt: {self.root}") from exc
        if not isinstance(receipt, dict) or receipt.get("schema") != _QTIP_WIRE_SCHEMA:
            raise ValueError("unsupported QTIP wire schema")
        if receipt.get("status") != "PASS":
            raise ValueError("QTIP wire receipt status must be PASS")
        if expected_receipt_sha256 is not None and _sha256(receipt_path) != expected_receipt_sha256:
            raise ValueError("QTIP wire receipt SHA-256 drift")
        self.receipt = receipt
        self.declaration = QtipProviderDeclaration.from_mapping(receipt.get("provider"))
        self._components = {row.name: row for row in self.declaration.components}
        raw_members = receipt.get("members")
        if not isinstance(raw_members, list):
            raise ValueError("QTIP wire members must be a list")
        self._members: dict[Identity, dict[str, object]] = {}
        for row in raw_members:
            if not isinstance(row, dict) or not isinstance(row.get("identity"), dict):
                raise ValueError("invalid QTIP wire member")
            identity_row = row["identity"]
            identity = (
                identity_row.get("layer"),
                identity_row.get("expert"),
                identity_row.get("projection"),
            )
            identity = _validate_identity(identity)
            if identity in self._members:
                raise ValueError("duplicate QTIP wire identity")
            component = self._components.get(row.get("component"))
            if component is None or row.get("geometry") != component.geometry.as_mapping():
                raise ValueError("QTIP wire component/geometry drift")
            self._members[identity] = row  # type: ignore[index]
        if receipt.get("counts") != self.counts:
            raise ValueError("QTIP wire receipt component counts drift")
        self._tlut = self._load_array(
            receipt.get("shared_tlut"), dtype=np.dtype(np.float32)
        )

    def _load_array(self, artifact: object, *, dtype: np.dtype[Any]) -> np.ndarray:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("invalid QTIP wire artifact")
        path = (self.root / artifact["path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("missing QTIP wire artifact")
        if path.stat().st_size != artifact.get("bytes") or _sha256(path) != artifact.get(
            "sha256"
        ):
            raise ValueError("QTIP wire artifact hash/size drift")
        array = np.load(path, allow_pickle=False)
        if array.dtype != dtype or array.nbytes != artifact.get("data_bytes"):
            raise ValueError("QTIP wire artifact dtype/data-byte drift")
        return array

    @property
    def counts(self) -> dict[str, int]:
        return {
            component.name: sum(
                row["component"] == component.name for row in self._members.values()
            )
            for component in self.declaration.components
        }

    def decode(self, identity: Identity) -> np.ndarray:
        try:
            row = self._members[identity]
            component = self._components[row["component"]]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown QTIP wire identity: {identity!r}") from exc
        packed = self._load_array(row.get("trellis"), dtype=np.dtype(np.uint16))
        scales = self._load_array(row.get("scales"), dtype=np.dtype(np.float32))
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
            or any(value < 1 for value in shape)
            or shape[1] % component.geometry.V
        ):
            raise ValueError("invalid QTIP wire member shape")
        if packed.shape[0] != shape[0] or scales.shape != (shape[0],):
            raise ValueError("QTIP wire member row/scale shape drift")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("QTIP wire scales must be finite and positive")
        encoded = EncodedQtip(
            geometry=component.geometry,
            shape=(shape[0], shape[1]),
            states=np.empty((0, 0), dtype=np.int32),
            packed=packed,
            scales=scales,
        )
        return decode_qtip(encoded, tlut=self._tlut)


def verify_qtip_wire(
    root: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every declared artifact and runtime-decode every provider member."""
    try:
        consumer = QtipWireConsumer(
            root,
            expected_receipt_sha256=expected_receipt_sha256,
        )
        decoded = 0
        for identity in sorted(consumer._members):
            value = consumer.decode(identity)
            if not np.isfinite(value).all():
                raise ValueError(f"QTIP wire member {identity!r} decoded non-finite values")
            decoded += 1
        return {
            "schema": "banana-smasher-qtip-wire-verification-v1",
            "status": "PASS",
            "members": decoded,
            "counts": consumer.counts,
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "schema": "banana-smasher-qtip-wire-verification-v1",
            "status": "FAIL",
            "error": str(exc),
        }


__all__ = [
    "EncodedQtip",
    "QTIP1_GEOMETRY",
    "QTIP2_GEOMETRY",
    "QtipGeometry",
    "QtipProviderComponent",
    "QtipProviderDeclaration",
    "QtipWireConsumer",
    "assign_qtip_provider_components",
    "decode_qtip",
    "encode_qtip",
    "gaussian_tlut",
    "pack_qtip_states",
    "qtip1_5_provider_declaration",
    "qtip1_provider_declaration",
    "qtip_provider_counts",
    "unpack_qtip_states",
    "verify_qtip_wire",
    "write_qtip_wire",
]
