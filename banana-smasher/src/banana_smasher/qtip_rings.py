from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

QTIP_RINGS_FILENAME = "qtip_rings.json"
_TABLE_PATH = Path(__file__).with_name(QTIP_RINGS_FILENAME)

PERSISTENT_GENERIC_BACKEND = "persistent-prefix-generic-aot-v1"
PERSISTENT_V32_BACKEND = "persistent-soa-exact-prefix-dp-v32"
TRELLIS_V2_BACKEND = "qtip-trellis-v2-graph-replay-b256-chunked-batch-exact-v46"
PERSISTENT_BACKENDS = frozenset({PERSISTENT_GENERIC_BACKEND, PERSISTENT_V32_BACKEND})

Geometry = tuple[int, int, int]
Identity = tuple[int, int, str]


@dataclass(frozen=True)
class RingComponent:
    geometry: Geometry
    quarters: int
    backend: str


@dataclass(frozen=True)
class QtipRing:
    canonical_bpw: str
    components: tuple[RingComponent, ...]
    codebook: Mapping[str, object]
    aot: Mapping[str, object]

    @property
    def tier(self) -> str:
        return f"qtip@{self.canonical_bpw}"

    @property
    def geometries(self) -> tuple[Geometry, ...]:
        return tuple(component.geometry for component in self.components)

    def backend_for(self, geometry: Geometry) -> str:
        matches = {
            component.backend
            for component in self.components
            if component.geometry == geometry
        }
        if len(matches) != 1:
            raise ValueError(
                f"bpw {self.canonical_bpw} geometry {geometry!r} has "
                f"{len(matches)} backend recipes in {QTIP_RINGS_FILENAME}"
            )
        return matches.pop()


def qtip_ring_manifest(ring: QtipRing) -> dict[str, object]:
    """Return the complete run-manifest identity for one generated ring."""
    return {
        "schema": "banana-smasher-qtip-ring-identity-v1",
        "bpw": ring.canonical_bpw,
        "tier": ring.tier,
        "components": [
            {
                "geometry": {
                    key: value
                    for key, value in zip(("L", "K", "V"), component.geometry)
                },
                "quarters": component.quarters,
                "backend": component.backend,
            }
            for component in ring.components
        ],
        "codebook": dict(ring.codebook),
        "aot": dict(ring.aot),
        "producer": (
            f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
        ),
    }


def validate_qtip_ring_manifest(value: object, ring: QtipRing) -> None:
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    if value is None:
        raise ValueError(
            f"missing QTIP ring manifest for {ring.tier}; run `{producer}`"
        )
    if value != qtip_ring_manifest(ring):
        raise ValueError(
            f"QTIP ring manifest mismatch for {ring.tier}; run `{producer}`"
        )


def _canonical_bpw(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError(f"invalid QTIP bpw {value!r}")
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid QTIP bpw {value!r}") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError(f"invalid QTIP bpw {value!r}")
    hundredths = rate.quantize(Decimal("0.01"))
    if rate != hundredths:
        raise ValueError(
            f"bpw {value} is not a 0.25 increment in {QTIP_RINGS_FILENAME} — "
            "add geometry entry"
        )
    return f"{hundredths:.2f}"


def _geometry(value: object, *, bpw: str) -> Geometry:
    if not isinstance(value, dict) or set(value) != {"L", "K", "V"}:
        raise ValueError(f"{QTIP_RINGS_FILENAME} bpw {bpw} requires exact L/K/V geometry")
    result = tuple(value[key] for key in ("L", "K", "V"))
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in result):
        raise ValueError(f"{QTIP_RINGS_FILENAME} bpw {bpw} has invalid geometry {result!r}")
    return result  # type: ignore[return-value]


def load_qtip_rings(path: Path | None = None) -> dict[str, QtipRing]:
    """Generate the quarter-grid from one manifest-declared ring family."""
    table_path = (path or _TABLE_PATH).resolve()
    try:
        payload = json.loads(table_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid packaged QTIP geometry table: {table_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "banana-smasher-qtip-ring-family-v2"
    ):
        raise ValueError(f"invalid packaged QTIP geometry table: {table_path}")

    supported = payload.get("supported_bpw")
    family = payload.get("geometry_family")
    backend_rows = payload.get("backends")
    codebook = payload.get("codebook")
    aot_template = payload.get("aot")
    if (
        not isinstance(supported, dict)
        or set(supported) != {"min", "max", "step"}
        or not isinstance(family, dict)
        or set(family) != {"L", "V", "component_quarters"}
        or not isinstance(backend_rows, list)
        or not isinstance(codebook, dict)
        or not isinstance(aot_template, dict)
    ):
        raise ValueError(f"invalid packaged QTIP ring family: {table_path}")
    minimum = Decimal(_canonical_bpw(supported["min"]))
    maximum = Decimal(_canonical_bpw(supported["max"]))
    step = Decimal(str(supported["step"]))
    if (
        step != Decimal("0.25")
        or minimum > maximum
        or minimum * 4 != (minimum * 4).to_integral_value()
        or maximum * 4 != (maximum * 4).to_integral_value()
    ):
        raise ValueError(f"invalid supported_bpw range in {table_path}")
    L = family["L"]
    V = family["V"]
    component_quarters = family["component_quarters"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in (L, V)
    ) or component_quarters != 4:
        raise ValueError(f"invalid geometry_family in {table_path}")

    backends: dict[int, str] = {}
    for row in backend_rows:
        if not isinstance(row, dict) or set(row) != {"K", "backend"}:
            raise ValueError(f"invalid backend recipe in {table_path}")
        K = row["K"]
        backend = row["backend"]
        if (
            isinstance(K, bool)
            or not isinstance(K, int)
            or K < 1
            or not isinstance(backend, str)
            or not backend
            or K in backends
        ):
            raise ValueError(f"invalid backend recipe in {table_path}")
        backends[K] = backend
    producer_template = aot_template.get("producer")
    if not isinstance(producer_template, str) or "{bpw}" not in producer_template:
        raise ValueError(f"invalid AOT producer template in {table_path}")

    rings: dict[str, QtipRing] = {}
    for quarters in range(int(minimum * 4), int(maximum * 4) + 1):
        canonical = f"{Decimal(quarters) / 4:.2f}"
        lower, remainder = divmod(quarters, 4)
        weighted = [(lower, 4)] if remainder == 0 else [
            (lower, 4 - remainder),
            (lower + 1, remainder),
        ]
        components: list[RingComponent] = []
        for K, weight in weighted:
            backend = backends.get(K)
            if backend is None:
                raise ValueError(
                    f"missing backend recipe K={K} for bpw {canonical} in {table_path}"
                )
            components.append(
                RingComponent(
                    geometry=(L, K, V),
                    quarters=weight,
                    backend=backend,
                )
            )
        aot = dict(aot_template)
        aot["producer"] = producer_template.format(bpw=canonical)
        rings[canonical] = QtipRing(
            canonical_bpw=canonical,
            components=tuple(components),
            codebook=dict(codebook),
            aot=aot,
        )
    return rings


def resolve_qtip_ring(
    bpw: object,
    *,
    rings: Mapping[str, QtipRing] | None = None,
) -> QtipRing:
    canonical = _canonical_bpw(bpw)
    try:
        rate = Decimal(canonical)
    except InvalidOperation as exc:  # pragma: no cover - canonicalization already guards this
        raise ValueError(f"invalid QTIP bpw {bpw!r}") from exc
    if rate * 4 != (rate * 4).to_integral_value():
        raise ValueError(
            f"bpw {canonical} is not a 0.25 increment in {QTIP_RINGS_FILENAME} — "
            "add geometry entry"
        )
    table = load_qtip_rings() if rings is None else rings
    if canonical not in table:
        raise ValueError(
            f"bpw {canonical} not in {QTIP_RINGS_FILENAME} — add geometry entry"
        )
    return table[canonical]


def canonical_qtip_tier(bpw: object) -> str:
    return resolve_qtip_ring(bpw).tier


def known_qtip_geometries(
    rings: Mapping[str, QtipRing] | None = None,
) -> frozenset[Geometry]:
    table = load_qtip_rings() if rings is None else rings
    return frozenset(
        geometry
        for ring in table.values()
        for geometry in ring.geometries
    )


def known_qtip_backend_geometries(
    backend: str,
    *,
    rings: Mapping[str, QtipRing] | None = None,
) -> frozenset[Geometry]:
    table = load_qtip_rings() if rings is None else rings
    return frozenset(
        component.geometry
        for ring in table.values()
        for component in ring.components
        if component.backend == backend
    )


def backend_for_geometry(
    geometry: Geometry,
    *,
    rings: Mapping[str, QtipRing] | None = None,
) -> str:
    table = load_qtip_rings() if rings is None else rings
    matches = {
        component.backend
        for ring in table.values()
        for component in ring.components
        if component.geometry == geometry
    }
    if not matches:
        raise ValueError(
            f"geometry {geometry!r} not in {QTIP_RINGS_FILENAME} — add geometry entry"
        )
    if len(matches) != 1:
        raise ValueError(
            f"ambiguous backend recipes for geometry {geometry!r} in "
            f"{QTIP_RINGS_FILENAME}: {sorted(matches)}"
        )
    return matches.pop()


def qtip_workspace_bytes(*, steps: int, batch: int, prefixes: int) -> int:
    values = {"steps": steps, "batch": batch, "prefixes": prefixes}
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values.values()
    ):
        raise ValueError(f"invalid QTIP workspace dimensions: {values}")
    float_bytes = 4
    int32_bytes = 4
    return (
        2 * batch * prefixes * float_bytes
        + steps * batch * prefixes * int32_bytes
        + steps * batch * int32_bytes
    )


def qtip_cuda_allocation_bytes(
    requested_bytes: int,
    *,
    allocator_alignment: int = 2 << 20,
) -> int:
    """Return native CUDA caching-allocator segment bytes for one request.

    PyTorch's default native allocator first rounds requests to 512-byte blocks,
    then obtains 2 MiB segments for rounded requests at or below 1 MiB,
    20 MiB segments below 10 MiB, and otherwise rounds to a 2 MiB segment.
    Refusing a different alignment keeps the estimate bound to that exact public
    allocator contract rather than silently applying a campaign-local heuristic.
    """
    if (
        isinstance(requested_bytes, bool)
        or not isinstance(requested_bytes, int)
        or requested_bytes < 0
        or isinstance(allocator_alignment, bool)
        or not isinstance(allocator_alignment, int)
        or allocator_alignment != 2 << 20
    ):
        raise ValueError("invalid native QTIP CUDA allocation request")
    if requested_bytes == 0:
        return 0
    block_alignment = 512
    rounded_request = (
        (requested_bytes + block_alignment - 1) // block_alignment
    ) * block_alignment
    if rounded_request <= 1 << 20:
        return 2 << 20
    if rounded_request < 10 << 20:
        return 20 << 20
    return (
        (rounded_request + allocator_alignment - 1) // allocator_alignment
    ) * allocator_alignment


def qtip_peak_allocation_bytes(
    *,
    steps: int,
    batch: int,
    prefixes: int,
    x_bytes: int,
    lut_bytes: int,
    x_requires_copy: bool,
    lut_requires_copy: bool,
    overlap_copy_bytes: int,
    retained_state_storage_bytes: int,
    retained_output_bytes: int,
    final_concatenation_bytes: int,
    allocator_alignment: int = 2 << 20,
) -> dict[str, object]:
    """Return exact native-allocator peaks from the live Viterbi preflight.

    The public LDLQ builder preallocates one whole ``Qidxs`` tensor before any
    Viterbi call.  It is therefore resident at this preflight (and is shown in
    ``resident_phases``) but must not be rounded once per returned chunk or
    charged again against currently free bytes.  After the kernel loop, the
    contiguous quantized output is retained while the single final Qidxs
    contiguous copy is allocated; those two future allocations form the final
    additional phase.
    """
    dimensions = {
        "steps": steps,
        "batch": batch,
        "prefixes": prefixes,
        "x_bytes": x_bytes,
        "lut_bytes": lut_bytes,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in dimensions.values()
    ):
        raise ValueError(f"invalid QTIP peak dimensions: {dimensions}")
    allocation_inputs = {
        "overlap_copy_bytes": overlap_copy_bytes,
        "retained_state_storage_bytes": retained_state_storage_bytes,
        "retained_output_bytes": retained_output_bytes,
        "final_concatenation_bytes": final_concatenation_bytes,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in allocation_inputs.values()
    ):
        raise ValueError(f"invalid QTIP peak allocations: {allocation_inputs}")
    if not isinstance(x_requires_copy, bool) or not isinstance(lut_requires_copy, bool):
        raise ValueError("invalid QTIP contiguous-copy contract")
    if retained_state_storage_bytes != final_concatenation_bytes:
        raise ValueError(
            "QTIP preallocated state storage must exactly close final concatenation bytes"
        )
    if bool(retained_state_storage_bytes) != bool(retained_output_bytes):
        raise ValueError(
            "QTIP retained builder outputs must be either fully declared or absent"
        )

    raw_allocations = {
        "scratch": 2 * batch * prefixes * 4,
        "backpointer": steps * batch * prefixes * 4,
        "state_output": steps * batch * 4,
        "x_contiguous_copy": x_bytes if x_requires_copy else 0,
        "lut_contiguous_copy": lut_bytes if lut_requires_copy else 0,
        "overlap_storage": overlap_copy_bytes,
        "retained_state_storage": retained_state_storage_bytes,
        "retained_output": retained_output_bytes,
        "final_concatenation": final_concatenation_bytes,
    }
    allocations = {
        name: qtip_cuda_allocation_bytes(
            value, allocator_alignment=allocator_alignment
        )
        for name, value in raw_allocations.items()
    }
    additional_phases = {
        "kernel": sum(
            allocations[name]
            for name in (
                "scratch",
                "backpointer",
                "state_output",
                "x_contiguous_copy",
                "lut_contiguous_copy",
                "overlap_storage",
            )
        ),
        "final_concatenation": (
            allocations["state_output"]
            + allocations["retained_output"]
            + allocations["final_concatenation"]
        ),
    }
    resident_phases = {
        phase: additional + allocations["retained_state_storage"]
        for phase, additional in additional_phases.items()
    }
    return {
        "schema": "banana-smasher-qtip-cuda-peak-v2",
        "allocator_backend": "native",
        "allocator_alignment": allocator_alignment,
        "raw_allocations": raw_allocations,
        "allocations": allocations,
        "additional_phases": additional_phases,
        "resident_phases": resident_phases,
        "total_bytes": max(additional_phases.values()),
        "resident_peak_bytes": max(resident_phases.values()),
    }


def require_qtip_memory_capacity(
    *,
    effective_free: int,
    free_source: str = "unspecified",
    reserve: int,
    peak: dict[str, object],
    geometry: Geometry,
) -> dict[str, object]:
    """Require the exact rounded additional peak while preserving ``reserve``."""
    raw_allocations = peak.get("raw_allocations")
    allocations = peak.get("allocations")
    additional_phases = peak.get("additional_phases")
    resident_phases = peak.get("resident_phases")
    allocator_alignment = peak.get("allocator_alignment")
    total = peak.get("total_bytes")
    resident_peak = peak.get("resident_peak_bytes")
    expected_names = {
        "scratch",
        "backpointer",
        "state_output",
        "x_contiguous_copy",
        "lut_contiguous_copy",
        "overlap_storage",
        "retained_state_storage",
        "retained_output",
        "final_concatenation",
    }
    valid_scalars = (
        isinstance(effective_free, int)
        and not isinstance(effective_free, bool)
        and effective_free >= 0
        and isinstance(reserve, int)
        and not isinstance(reserve, bool)
        and reserve >= 4 << 30
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total >= 0
        and isinstance(resident_peak, int)
        and not isinstance(resident_peak, bool)
        and resident_peak >= 0
    )
    valid_maps = (
        isinstance(raw_allocations, dict)
        and set(raw_allocations) == expected_names
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in raw_allocations.values()
        )
        and allocator_alignment == 2 << 20
        and isinstance(allocations, dict)
        and set(allocations) == expected_names
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in allocations.values()
        )
        and isinstance(additional_phases, dict)
        and set(additional_phases) == {"kernel", "final_concatenation"}
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in additional_phases.values()
        )
        and isinstance(resident_phases, dict)
        and set(resident_phases) == set(additional_phases)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in resident_phases.values()
        )
    )
    if not valid_scalars or not valid_maps:
        raise ValueError("invalid QTIP memory capacity inputs")
    assert isinstance(total, int)
    assert isinstance(resident_peak, int)
    assert isinstance(allocator_alignment, int)
    assert isinstance(raw_allocations, dict)
    assert isinstance(allocations, dict)
    assert isinstance(additional_phases, dict)
    assert isinstance(resident_phases, dict)
    expected_allocations = {
        name: qtip_cuda_allocation_bytes(
            raw_value, allocator_alignment=allocator_alignment
        )
        for name, raw_value in raw_allocations.items()
    }
    expected_additional_phases = {
        "kernel": sum(
            expected_allocations[name]
            for name in (
                "scratch",
                "backpointer",
                "state_output",
                "x_contiguous_copy",
                "lut_contiguous_copy",
                "overlap_storage",
            )
        ),
        "final_concatenation": (
            expected_allocations["state_output"]
            + expected_allocations["retained_output"]
            + expected_allocations["final_concatenation"]
        ),
    }
    retained = expected_allocations["retained_state_storage"]
    expected_resident_phases = {
        phase: additional + retained
        for phase, additional in expected_additional_phases.items()
    }
    expected_total = max(expected_additional_phases.values())
    expected_resident_peak = max(expected_resident_phases.values())
    if (
        peak.get("schema") != "banana-smasher-qtip-cuda-peak-v2"
        or peak.get("allocator_backend") != "native"
        or allocations != expected_allocations
        or additional_phases != expected_additional_phases
        or resident_phases != expected_resident_phases
        or total != expected_total
        or resident_peak != expected_resident_peak
    ):
        raise ValueError("invalid QTIP memory peak closure")
    if effective_free - total < reserve:
        raise RuntimeError(
            f"QTIP memory preflight failed for L{geometry[0]}/K{geometry[1]}/V{geometry[2]}: "
            f"effective_free={effective_free} source={free_source} "
            f"reserve={reserve} exact_peak={total} "
            f"additional_phases={additional_phases} allocations={allocations}"
        )
    return peak


def canonical_qtip_packed_shape(
    *,
    codebook: Mapping[str, object],
    geometry: Geometry,
    matrix_shape: tuple[int, int],
) -> tuple[int, int]:
    """Derive the public runner's canonical wire shape from its codebook contract."""
    L, K, V = geometry
    key = f"L{L}-K{K}-V{V}"
    contract = codebook.get("pack_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"missing QTIP canonical pack contract for {key}")

    input_tile = contract.get("input_tile")
    packed_words_per_k = contract.get("packed_words_per_tile_per_k")
    output_rows_contract = contract.get("output_rows")
    dtype = contract.get("dtype")
    rows, columns = matrix_shape
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in matrix_shape
        )
        or not isinstance(input_tile, list)
        or len(input_tile) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in input_tile
        )
        or isinstance(packed_words_per_k, bool)
        or not isinstance(packed_words_per_k, int)
        or packed_words_per_k < 1
        or output_rows_contract != "input_tile_grid"
        or dtype != "uint16"
        or rows % input_tile[0]
        or columns % input_tile[1]
    ):
        raise ValueError(f"invalid QTIP canonical pack contract for {key}")
    total_words = (
        (rows // input_tile[0])
        * (columns // input_tile[1])
        * packed_words_per_k
        * K
    )
    output_rows = (rows // input_tile[0]) * (columns // input_tile[1])
    if total_words % output_rows:
        raise ValueError(f"invalid QTIP canonical pack contract for {key}")
    return output_rows, total_words // output_rows


def plan_qtip_streaming_batches(
    *,
    steps: int,
    batch: int,
    prefixes: int,
    available_workspace_bytes: int,
) -> dict[str, object]:
    """Plan exact sequence-independent slices for the same persistent kernel."""
    values = (steps, batch, prefixes, available_workspace_bytes)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError("invalid QTIP streaming plan inputs")
    chunk = 1 << (batch.bit_length() - 1)
    while chunk > 0 and qtip_workspace_bytes(
        steps=steps, batch=chunk, prefixes=prefixes
    ) > available_workspace_bytes:
        chunk //= 2
    if chunk < 1:
        raise RuntimeError(
            "QTIP streaming workspace cannot preserve the configured memory reserve"
        )
    slices = [
        (start, min(batch, start + chunk))
        for start in range(0, batch, chunk)
    ]
    return {
        "schema": "banana-smasher-qtip-streaming-plan-v1",
        "path": "same-accelerated-kernel",
        "batch": batch,
        "chunk_batch": chunk,
        "batch_slices": slices,
        "workspace_bytes": qtip_workspace_bytes(
            steps=steps, batch=chunk, prefixes=prefixes
        ),
    }


def effective_cuda_free_bytes(
    *,
    driver_free: int,
    reserved: int,
    allocated: int,
) -> int:
    """Include PyTorch cache that can satisfy a workspace allocation."""
    return driver_free + max(0, reserved - allocated)


def _identity_sort_key(identity: Identity) -> bytes:
    layer, expert, projection = identity
    return hashlib.sha256(f"{layer}:{projection}:{expert}".encode()).digest()


def assign_ring_geometries(
    ring: QtipRing,
    identities: Iterable[Identity],
) -> dict[Identity, Geometry]:
    """Assign exact component quotas per layer/projection, independent of input order."""
    rows = list(identities)
    if len(set(rows)) != len(rows):
        raise ValueError("QTIP ring assignment identities must be unique")
    groups: dict[tuple[int, str], list[Identity]] = {}
    for identity in rows:
        if (
            not isinstance(identity, tuple)
            or len(identity) != 3
            or isinstance(identity[0], bool)
            or not isinstance(identity[0], int)
            or identity[0] < 0
            or isinstance(identity[1], bool)
            or not isinstance(identity[1], int)
            or identity[1] < 0
            or not isinstance(identity[2], str)
            or not identity[2]
        ):
            raise ValueError(f"invalid QTIP ring identity: {identity!r}")
        groups.setdefault((identity[0], identity[2]), []).append(identity)

    assigned: dict[Identity, Geometry] = {}
    for group in groups.values():
        ordered = sorted(group, key=_identity_sort_key)
        start = 0
        remaining = len(ordered)
        remaining_quarters = 4
        for index, component in enumerate(ring.components):
            if index == len(ring.components) - 1:
                count = remaining
            else:
                count = (len(ordered) * component.quarters + 2) // 4
                count = min(count, remaining)
            for identity in ordered[start : start + count]:
                assigned[identity] = component.geometry
            start += count
            remaining -= count
            remaining_quarters -= component.quarters
        if remaining or remaining_quarters:
            raise AssertionError("validated ring allocation did not close")
    return assigned
