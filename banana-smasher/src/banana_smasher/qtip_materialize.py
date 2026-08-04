from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable
import uuid

from .qtip_rings import (
    QtipRing,
    assign_ring_geometries,
    qtip_ring_manifest,
    resolve_qtip_ring,
    validate_qtip_ring_manifest,
)

PRODUCER_VERB = "smash qtip-configs"
RUN_MANIFEST_NAME = "QTIP_RUN_MANIFEST.json"
OUTPUT_MANIFEST_NAME = "QTIP_CONFIG_MANIFEST.json"
EXPLICIT_RHT_SEED_POLICY = "qtip-rht-explicit-seed-v1"
_MATERIALIZE_LOCK = threading.Lock()
_QTIP_PROJECTIONS = frozenset({"fused13", "down"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_qtip_projection(value: object) -> str:
    """Return one public projection name, refusing path or dispatch control."""
    if not isinstance(value, str) or value not in _QTIP_PROJECTIONS:
        raise ValueError(f"unsupported QTIP projection: {value!r}")
    return value


def _relative_reference(path: Path, *, base: Path) -> str:
    """Serialize a local dependency without exposing its absolute host path."""
    return os.path.relpath(path.resolve(), start=base.resolve())


def _local_path(value: object, *, label: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path")
    if "://" in value:
        raise ValueError(f"{label} must be disk-local, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _artifact(record: object, *, label: str, base: Path) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} artifact record")
    path = _local_path(record.get("path"), label=label, base=base)
    if not path.is_file():
        raise ValueError(f"missing local {label}; run {PRODUCER_VERB}: {path}")
    size = record.get("bytes")
    sha = record.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha, str)
        or len(sha) != 64
    ):
        raise ValueError(f"invalid {label} hash/size record")
    observed_size = path.stat().st_size
    observed_sha = _sha256(path)
    if observed_size != size or observed_sha != sha:
        raise ValueError(
            f"{label} hash/size drift: {path}; "
            f"expected {size}/{sha}, observed {observed_size}/{observed_sha}"
        )
    return path


def _directory_binding(record: object, *, label: str, base: Path) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} directory binding")
    root = _local_path(record.get("path"), label=label, base=base)
    if not root.is_dir():
        raise ValueError(f"missing local {label}; run {PRODUCER_VERB}: {root}")
    seal = _artifact(record.get("manifest"), label=f"{label} manifest", base=base)
    if not seal.is_relative_to(root):
        raise ValueError(f"{label} manifest is outside its local root: {seal}")
    return root


def _geometry(value: object, *, tier: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"L", "K", "V"}:
        raise ValueError(f"manifest tier {tier!r} requires exact L/K/V geometry")
    result: dict[str, int] = {}
    for key in ("L", "K", "V"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"invalid manifest geometry {key}={item!r} for tier {tier!r}")
        result[key] = item
    return result


def _unique_row(rows: object, *, key: str, value: object, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"manifest {label} must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"manifest requires exactly one {label} for {key}={value!r}, got {len(matches)}")
    return matches[0]


def _ensure_ring_tier_row(
    manifest: dict[str, Any],
    ring: QtipRing,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a manifest and row for ``ring``, deriving one open-tier row if absent."""
    rows = manifest.get("tiers")
    if not isinstance(rows, list):
        raise ValueError("manifest tier row must be a list")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("name") == ring.tier
    ]
    if len(matches) == 1:
        row = matches[0]
        canonical_ring = qtip_ring_manifest(ring)
        existing_ring = row.get("ring")
        if existing_ring == canonical_ring or not _is_additive_legacy_value(
            existing_ring, canonical_ring
        ):
            return manifest, row
        canonical_row = dict(row)
        canonical_row["ring"] = canonical_ring
        produced = dict(manifest)
        produced["tiers"] = [
            canonical_row if candidate is row else candidate for candidate in rows
        ]
        return produced, canonical_row
    if matches:
        raise ValueError(
            f"manifest requires exactly one tier row for name={ring.tier!r}, got {len(matches)}"
        )

    templates = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("bindings"), dict)
        and isinstance(row.get("layers"), list)
        and row["layers"]
    ]
    identities = {
        json.dumps(
            {
                key: value
                for key, value in row.items()
                if key not in {"name", "geometry", "ring"}
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in templates
    }
    if len(identities) != 1:
        raise ValueError(
            f"manifest cannot derive {ring.tier!r} from {len(identities)} "
            "distinct hash-bound tier templates"
        )
    template = templates[0]
    generated = {
        key: value
        for key, value in template.items()
        if key not in {"name", "geometry", "ring"}
    }
    generated.update({"name": ring.tier, "ring": qtip_ring_manifest(ring)})
    produced = dict(manifest)
    produced["tiers"] = [*rows, generated]
    return produced, generated


def _is_additive_legacy_value(existing: object, replacement: object) -> bool:
    """Return whether canonical metadata only adds keys to a legacy value."""
    if isinstance(existing, dict) and isinstance(replacement, dict):
        return all(
            key in replacement and _is_additive_legacy_value(value, replacement[key])
            for key, value in existing.items()
        )
    return existing == replacement


def _ring_upgrade_config_preimage(
    existing_raw: bytes,
    replacement_raw: bytes,
    *,
    manifest_preimage_sha256: str,
) -> bool:
    """Accept only a hash-bound generated config with additive legacy metadata."""
    try:
        existing = json.loads(existing_raw)
        replacement = json.loads(replacement_raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(existing, dict) or not isinstance(replacement, dict):
        return False
    materialization = existing.get("materialization")
    replacement_materialization = replacement.get("materialization")
    if (
        not isinstance(materialization, dict)
        or not isinstance(replacement_materialization, dict)
        or materialization.get("run_manifest_sha256") != manifest_preimage_sha256
        or (
            "source_run_manifest_sha256" in materialization
            and materialization["source_run_manifest_sha256"]
            != manifest_preimage_sha256
        )
    ):
        return False
    normalized_materialization = dict(materialization)
    normalized_materialization["run_manifest_sha256"] = replacement_materialization.get(
        "run_manifest_sha256"
    )
    normalized_materialization["source_run_manifest_sha256"] = (
        replacement_materialization.get("source_run_manifest_sha256")
    )
    if normalized_materialization != replacement_materialization:
        return False
    if any(
        not _is_additive_legacy_value(existing.get(key), replacement.get(key))
        for key in ("codebook", "aot")
    ):
        return False
    normalized = dict(existing)
    normalized["codebook"] = replacement.get("codebook")
    normalized["aot"] = replacement.get("aot")
    normalized["materialization"] = replacement_materialization
    return normalized == replacement


def _ring_upgrade_receipt_preimage(
    existing_raw: bytes,
    replacement_raw: bytes,
    *,
    manifest_preimage_sha256: str,
    member_preimages: dict[Path, bytes],
) -> bool:
    """Validate the sealed generated index that will be resealed after upgrades."""
    try:
        existing = json.loads(existing_raw)
        replacement = json.loads(replacement_raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(existing, dict) or not isinstance(replacement, dict):
        return False
    if existing.get("run_manifest_sha256") != manifest_preimage_sha256 or (
        "source_run_manifest_sha256" in existing
        and existing["source_run_manifest_sha256"] != manifest_preimage_sha256
    ):
        return False
    stable_keys = (
        "schema",
        "status",
        "producer",
        "tier",
        "basis_sha256",
        "run_manifest",
        "source_run_manifest",
        "output_root",
        "layers",
        "members",
        "geometry",
        "ring",
    )
    if any(existing.get(key) != replacement.get(key) for key in stable_keys):
        return False
    existing_members = existing.get("member_records")
    replacement_members = replacement.get("member_records")
    if not isinstance(existing_members, list) or not isinstance(replacement_members, list):
        return False
    if len(existing_members) != len(replacement_members):
        return False
    stable_member_keys = (
        "layer",
        "expert",
        "projection",
        "geometry",
        "backend",
        "path",
        "source_sha256",
    )
    if [
        {key: row.get(key) for key in stable_member_keys}
        for row in existing_members
        if isinstance(row, dict)
    ] != [
        {key: row.get(key) for key in stable_member_keys}
        for row in replacement_members
        if isinstance(row, dict)
    ]:
        return False
    sealed_hashes: list[str] = []
    for row in existing_members:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            return False
        raw = member_preimages.get(Path(row["path"]))
        if raw is None:
            return False
        digest = hashlib.sha256(raw).hexdigest()
        if row.get("bytes") != len(raw) or row.get("sha256") != digest:
            return False
        sealed_hashes.append(digest)
    return existing.get("ordered_member_sha256") == hashlib.sha256(
        "".join(sealed_hashes).encode()
    ).hexdigest()


def _atomic_bytes(path: Path, raw: bytes) -> bool:
    """Write and fsync new bytes, or preserve an identical existing file byte-for-byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"materialized output is not a file: {path}")
        existing = path.read_bytes()
        if existing != raw:
            raise ValueError(f"refuse to rewrite divergent materialized config: {path}")
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _replace_validated_bytes_locked(
    path: Path, *, expected: bytes, replacement: bytes
) -> bool:
    """Replace a preimage while the caller holds the stable materialization lock."""
    if replacement == expected:
        return _atomic_bytes(path, replacement)
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"refuse to rewrite divergent materialized config: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_basis(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {label} basis SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {label} basis SHA-256") from exc
    return value


def _iter_selected_layers(tier_row: dict[str, Any], selected: list[int]) -> Iterable[dict[str, Any]]:
    rows = tier_row.get("layers")
    for layer in selected:
        yield _unique_row(rows, key="layer", value=layer, label="layer row")


def _kernel_plan_components(
    plan: object,
    ring: QtipRing,
) -> tuple[tuple[tuple[int, int, int], int, str], ...]:
    """Validate one verified cache plan and return its ordered ring geometry identity."""
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "banana-smasher-qtip-kernel-build-plan-v1"
        or plan.get("bpw") != ring.canonical_bpw
        or plan.get("tier") != ring.tier
        or plan.get("codebook") != dict(ring.codebook)
        or plan.get("aot") != dict(ring.aot)
    ):
        raise ValueError(
            f"QTIP kernel cache identity mismatch for {ring.tier}; run `{producer}`"
        )
    recipes = plan.get("recipes")
    if not isinstance(recipes, list):
        raise ValueError(
            f"QTIP kernel cache geometry mismatch for {ring.tier}; run `{producer}`"
        )
    result: list[tuple[tuple[int, int, int], int, str]] = []
    for recipe in recipes:
        if not isinstance(recipe, dict):
            raise ValueError(
                f"QTIP kernel cache geometry mismatch for {ring.tier}; run `{producer}`"
            )
        geometry = _geometry(recipe.get("geometry"), tier=ring.tier)
        backend = recipe.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ValueError(
                f"QTIP kernel cache geometry mismatch for {ring.tier}; run `{producer}`"
            )
        values = tuple(geometry[key] for key in ("L", "K", "V"))
        matching = [
            component
            for component in ring.components
            if component.geometry == values and component.backend == backend
        ]
        if len(matching) != 1:
            raise ValueError(
                f"QTIP kernel cache geometry mismatch for {ring.tier}; run `{producer}`"
            )
        result.append((values, matching[0].quarters, backend))
    return tuple(result)


def _manifest_ring_components(
    value: object,
    ring: QtipRing,
) -> tuple[tuple[tuple[int, int, int], int, str], ...]:
    """Read only the physical geometry identity that an older manifest declared."""
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    if (
        not isinstance(value, dict)
        or value.get("schema") != "banana-smasher-qtip-ring-identity-v1"
        or value.get("bpw") != ring.canonical_bpw
        or value.get("tier") != ring.tier
    ):
        raise ValueError(
            f"QTIP ring manifest mismatch for {ring.tier}; run `{producer}`"
        )
    components = value.get("components")
    if not isinstance(components, list):
        raise ValueError(
            f"QTIP ring manifest mismatch for {ring.tier}; run `{producer}`"
        )
    result: list[tuple[tuple[int, int, int], int, str]] = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError(
                f"QTIP ring manifest mismatch for {ring.tier}; run `{producer}`"
            )
        geometry = _geometry(component.get("geometry"), tier=ring.tier)
        quarters = component.get("quarters")
        backend = component.get("backend")
        if (
            isinstance(quarters, bool)
            or not isinstance(quarters, int)
            or quarters < 1
            or not isinstance(backend, str)
            or not backend
        ):
            raise ValueError(
                f"QTIP ring manifest mismatch for {ring.tier}; run `{producer}`"
            )
        result.append(
            (tuple(geometry[key] for key in ("L", "K", "V")), quarters, backend)
        )
    return tuple(result)


def reconcile_qtip_ring_manifest(
    source_root: Path,
    bpw: object,
    *,
    kernel_plan: object,
) -> dict[str, Any]:
    """CAS-upgrade matching ring metadata from one freshly verified cache plan.

    Geometry, backend, and quota identity must already agree.  This only repairs
    additive generic ring metadata (for example a newer packing contract); it
    cannot relabel a cache or manifest for a different physical ring.
    """
    source_root = source_root.resolve()
    manifest_path = source_root / RUN_MANIFEST_NAME
    ring = resolve_qtip_ring(bpw)
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    built_components = _kernel_plan_components(kernel_plan, ring)
    expected_components = tuple(
        (component.geometry, component.quarters, component.backend)
        for component in ring.components
    )
    if built_components != expected_components:
        raise ValueError(
            f"QTIP kernel cache geometry mismatch for {ring.tier}; run `{producer}`"
        )
    if not manifest_path.is_file():
        raise ValueError(
            f"missing QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
        )
    lock_path = source_root / ".QTIP_MATERIALIZE.lock"
    with _MATERIALIZE_LOCK, lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            raw = manifest_path.read_bytes()
            try:
                manifest = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
                ) from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != "banana-smasher-qtip-run-manifest-v1"
                or manifest.get("status") != "PASS"
            ):
                raise ValueError(
                    f"invalid QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
                )
            tier_row = _unique_row(
                manifest.get("tiers"), key="name", value=ring.tier, label="tier row"
            )
            observed_components = _manifest_ring_components(tier_row.get("ring"), ring)
            if observed_components != built_components:
                raise ValueError(
                    f"QTIP ring manifest geometry mismatch for {ring.tier}; run `{producer}`"
                )
            expected_ring = qtip_ring_manifest(ring)
            old_sha = hashlib.sha256(raw).hexdigest()
            if tier_row.get("ring") == expected_ring:
                return {
                    "schema": "banana-smasher-qtip-ring-cache-reconciliation-v1",
                    "status": "PASS",
                    "changed": False,
                    "bpw": ring.canonical_bpw,
                    "tier": ring.tier,
                    "manifest": str(manifest_path),
                    "manifest_preimage_sha256": old_sha,
                    "manifest_sha256": old_sha,
                    "geometry": [
                        {key: value for key, value in zip(("L", "K", "V"), item[0])}
                        for item in built_components
                    ],
                }
            tier_row["ring"] = expected_ring
            replacement = _json_bytes(manifest)
            _replace_validated_bytes_locked(
                manifest_path, expected=raw, replacement=replacement
            )
            return {
                "schema": "banana-smasher-qtip-ring-cache-reconciliation-v1",
                "status": "PASS",
                "changed": True,
                "bpw": ring.canonical_bpw,
                "tier": ring.tier,
                "manifest": str(manifest_path),
                "manifest_preimage_sha256": old_sha,
                "manifest_sha256": hashlib.sha256(replacement).hexdigest(),
                "geometry": [
                    {key: value for key, value in zip(("L", "K", "V"), item[0])}
                    for item in built_components
                ],
            }
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def require_qtip_ring_manifest(source_root: Path, bpw: object) -> str:
    """Resolve one public QTIP target from its run-manifest ring identity."""
    ring = resolve_qtip_ring(bpw)
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    manifest_path = source_root.resolve() / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"missing QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "banana-smasher-qtip-run-manifest-v1"
        or manifest.get("status") != "PASS"
    ):
        raise ValueError(
            f"invalid QTIP ring manifest for {ring.tier}: {manifest_path}; run `{producer}`"
        )
    try:
        tier_row = _unique_row(
            manifest.get("tiers"), key="name", value=ring.tier, label="tier row"
        )
    except ValueError as exc:
        raise ValueError(
            f"missing QTIP ring manifest for {ring.tier}; run `{producer}`"
        ) from exc
    validate_qtip_ring_manifest(tier_row.get("ring"), ring)
    return ring.tier


def _materialize_qtip_configs_locked(
    manifest_path: Path,
    *,
    tier: str,
    layers: list[int],
    output_root: Path,
    ring_upgrade_preimage_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize hash-bound QTIP configs from one open-tier run manifest.

    The manifest owns every tier name, geometry, layer, model, bank, and runtime path.
    This function contains no campaign tier menu, layer count, model default, or remote
    transport. All source bytes are validated before the first output is written.
    """
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    canonical_manifest_path = output_root / RUN_MANIFEST_NAME
    if not tier:
        raise ValueError("QTIP tier name must be non-empty")
    if not layers or len(set(layers)) != len(layers) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers
    ):
        raise ValueError(f"invalid materialization layers: {layers!r}")
    if not manifest_path.is_file():
        raise ValueError(f"missing QTIP run manifest; run {PRODUCER_VERB}: {manifest_path}")
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as exc:
        kind = "ring manifest" if tier.startswith("qtip@") else "run manifest"
        raise ValueError(f"invalid QTIP {kind} JSON: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "banana-smasher-qtip-run-manifest-v1"
        or manifest.get("status") != "PASS"
    ):
        raise ValueError(f"invalid QTIP run manifest: {manifest_path}")
    basis = _validate_basis(manifest.get("basis_sha256"), label="manifest")
    ring: QtipRing | None = None
    geometry: dict[str, int] | None = None
    if tier.startswith("qtip@"):
        ring = resolve_qtip_ring(tier.removeprefix("qtip@"))
        manifest, tier_row = _ensure_ring_tier_row(manifest, ring)
        validate_qtip_ring_manifest(tier_row.get("ring"), ring)
        declared_geometry = tier_row.get("geometry")
        if declared_geometry is not None:
            geometry = _geometry(declared_geometry, tier=tier)
            declared = tuple(geometry[key] for key in ("L", "K", "V"))
            if len(ring.geometries) != 1 or declared != ring.geometries[0]:
                raise ValueError(
                    f"manifest tier {tier!r} geometry differs from qtip_rings.json"
                )
    else:
        tier_row = _unique_row(
            manifest.get("tiers"), key="name", value=tier, label="tier row"
        )
        geometry = _geometry(tier_row.get("geometry"), tier=tier)
    canonical_manifest_raw = _json_bytes(manifest)
    if manifest == json.loads(manifest_raw):
        canonical_manifest_raw = manifest_raw
    canonical_manifest_sha = hashlib.sha256(canonical_manifest_raw).hexdigest()
    source_manifest_sha = (
        canonical_manifest_sha
        if canonical_manifest_path == manifest_path
        else hashlib.sha256(manifest_raw).hexdigest()
    )
    bindings = tier_row.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError(f"manifest tier {tier!r} requires bindings")

    model = bindings.get("model_root")
    if not isinstance(model, dict):
        raise ValueError(f"manifest tier {tier!r} requires model_root binding")
    model_root = _local_path(model.get("path"), label="model root", base=manifest_path.parent)
    if not model_root.is_dir():
        raise ValueError(f"missing local model root; run {PRODUCER_VERB}: {model_root}")
    model_index = _artifact(model.get("index"), label="model index", base=manifest_path.parent)
    if not model_index.is_relative_to(model_root):
        raise ValueError(f"model index is outside model root: {model_index}")
    if _sha256(model_index) != basis:
        raise ValueError(f"model basis mismatch: manifest={basis} model-index={_sha256(model_index)}")

    qtip_root = _directory_binding(
        bindings.get("qtip_root"), label="QTIP runtime root", base=manifest_path.parent
    )
    qtip_runner = _artifact(
        bindings.get("qtip_runner"), label="public QTIP runner", base=manifest_path.parent
    )
    reference = _artifact(
        bindings.get("reference_unit"), label="QTIP reference unit", base=manifest_path.parent
    )
    tlut = _artifact(bindings.get("tlut_source"), label="QTIP TLUT", base=manifest_path.parent)

    plan: list[tuple[Path, bytes, dict[str, Any]]] = []
    identities: set[tuple[int, int, str]] = set()
    for layer_row in _iter_selected_layers(tier_row, layers):
        layer = layer_row["layer"]
        capture_root = _directory_binding(
            layer_row.get("fit_capture_root"),
            label=f"L{layer:03d} capture bank",
            base=manifest_path.parent,
        )
        hessian = _artifact(
            layer_row.get("hessian_layer_manifest"),
            label=f"L{layer:03d} Hessian manifest",
            base=manifest_path.parent,
        )
        source_rows = layer_row.get("source_configs")
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"L{layer:03d} has no source configs; run {PRODUCER_VERB}")
        population = len(source_rows)
        source_inputs: list[tuple[Path, dict[str, Any], tuple[int, int, str]]] = []
        for index, record in enumerate(source_rows):
            source = _artifact(
                record,
                label=f"L{layer:03d} source config {index}",
                base=manifest_path.parent,
            )
            try:
                config = json.loads(source.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source config JSON: {source}") from exc
            if not isinstance(config, dict) or config.get("schema") != "banana-smasher-qtip-profile-config-v1":
                raise ValueError(f"invalid source QTIP config: {source}")
            if config.get("layer") != layer:
                raise ValueError(f"source config layer mismatch in {source}")
            source_basis = (
                config.get("input_identity", {}).get("model_index", {}).get("sha256")
                if isinstance(config.get("input_identity"), dict)
                else None
            )
            if source_basis != basis:
                raise ValueError(f"source config basis mismatch in {source}: {source_basis} != {basis}")
            expert = config.get("expert")
            projection = validate_qtip_projection(config.get("projection"))
            if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
                raise ValueError(f"invalid source config expert in {source}")
            identity = (layer, expert, projection)
            if identity in identities:
                raise ValueError(f"duplicate source config identity: {identity}")
            identities.add(identity)
            census = config.get("layer_census")
            if not isinstance(census, dict) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in census.values()
            ):
                raise ValueError(f"invalid source layer census in {source}")
            source_inputs.append((source, config, identity))

        ring_geometries = (
            assign_ring_geometries(
                ring,
                (identity for _source, _config, identity in source_inputs),
            )
            if ring is not None
            else {}
        )
        for source, config, identity in source_inputs:
            _layer, expert, projection = identity
            selected_geometry = (
                {
                    key: value
                    for key, value in zip(
                        ("L", "K", "V"), ring_geometries[identity]
                    )
                }
                if ring is not None
                else geometry
            )
            assert selected_geometry is not None
            selected_tuple = (
                selected_geometry["L"],
                selected_geometry["K"],
                selected_geometry["V"],
            )
            selected_backend = (
                ring.backend_for(selected_tuple) if ring is not None else None
            )
            census = config["layer_census"]
            materialized_census = {str(name): 0 for name in census}
            materialized_census[tier] = population
            output = output_root / f"L{layer:03d}" / f"E{expert:03d}_{projection}.json"
            if not output.resolve().is_relative_to(output_root):
                raise ValueError(f"materialized QTIP output escapes output root: {output}")
            config_base = output.parent
            materialized = dict(config)
            explicit_seed = materialized.get("rht_seed")
            if (
                type(explicit_seed) is int
                and 0 <= explicit_seed < (1 << 63)
                and materialized.get("rht_seed_policy") != "qtip-rht-manifest-v1"
            ):
                materialized["rht_seed_policy"] = EXPLICIT_RHT_SEED_POLICY
            materialized.update(
                {
                    "tier": tier,
                    "bpw": ring.canonical_bpw if ring is not None else None,
                    "geometry": selected_geometry,
                    "backend": selected_backend,
                    "codebook": dict(ring.codebook) if ring is not None else None,
                    "aot": dict(ring.aot) if ring is not None else None,
                    "model_root": _relative_reference(model_root, base=config_base),
                    "fit_capture_root": _relative_reference(capture_root, base=config_base),
                    "hessian_layer_manifest": _relative_reference(hessian, base=config_base),
                    "hessian_layer_manifest_sha256": _sha256(hessian),
                    "qtip_root": _relative_reference(qtip_root, base=config_base),
                    "qtip_runner": _relative_reference(qtip_runner, base=config_base),
                    "reference_unit": _relative_reference(reference, base=config_base),
                    "tlut_source": _relative_reference(tlut, base=config_base),
                    "layer_census": materialized_census,
                    "input_identity": {
                        "model_index": {
                            "path": _relative_reference(model_index, base=config_base),
                            "sha256": basis,
                        }
                    },
                    "materialization": {
                        "schema": "banana-smasher-qtip-config-materialization-v1",
                        "run_manifest": _relative_reference(
                            canonical_manifest_path, base=config_base
                        ),
                        "run_manifest_sha256": canonical_manifest_sha,
                        "source_run_manifest": _relative_reference(
                            manifest_path, base=config_base
                        ),
                        "source_run_manifest_sha256": source_manifest_sha,
                        "source_config": _relative_reference(source, base=config_base),
                        "source_config_sha256": _sha256(source),
                        "qtip_ring_bpw": (
                            ring.canonical_bpw if ring is not None else None
                        ),
                    },
                }
            )
            raw = _json_bytes(materialized)
            plan.append(
                (
                    output,
                    raw,
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "geometry": selected_geometry,
                        "backend": selected_backend,
                        "path": output.relative_to(output_root).as_posix(),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "source_sha256": _sha256(source),
                    },
                )
            )

    ordered_plan = sorted(plan, key=lambda item: item[0].as_posix())
    member_rows = [row for _, _, row in ordered_plan]
    ordered_member_sha = hashlib.sha256(
        "".join(str(row["sha256"]) for row in member_rows).encode()
    ).hexdigest()
    sealed_receipt: dict[str, Any] = {
        "schema": "banana-smasher-qtip-config-manifest-v1",
        "status": "PASS",
        "producer": PRODUCER_VERB,
        "tier": tier,
        "basis_sha256": basis,
        "run_manifest": _relative_reference(canonical_manifest_path, base=output_root),
        "run_manifest_sha256": canonical_manifest_sha,
        "source_run_manifest": _relative_reference(manifest_path, base=output_root),
        "source_run_manifest_sha256": source_manifest_sha,
        "output_root": ".",
        "layers": layers,
        "members": len(member_rows),
        "ordered_member_sha256": ordered_member_sha,
        "member_records": member_rows,
    }
    if ring is None:
        sealed_receipt["geometry"] = geometry
    else:
        sealed_receipt["ring"] = {
            "bpw": ring.canonical_bpw,
            "tier": ring.tier,
            "geometries": [
                {key: value for key, value in zip(("L", "K", "V"), item)}
                for item in ring.geometries
            ],
            "selection": "stable-exact-quota-per-layer-projection-v1",
        }
    receipt_path = output_root / OUTPUT_MANIFEST_NAME
    receipt_raw = _json_bytes(sealed_receipt)
    outputs = [(output, raw) for output, raw, _ in ordered_plan]
    outputs.append((receipt_path, receipt_raw))
    member_preimages = {
        target: target.read_bytes()
        for target, _raw in outputs[:-1]
        if target.is_file()
    }
    upgrade_preimages: dict[Path, bytes] = {}
    for target, raw in outputs:
        if not target.exists():
            continue
        if not target.is_file():
            raise ValueError(f"refuse to rewrite divergent materialized config: {target}")
        existing_raw = target.read_bytes()
        if existing_raw == raw:
            continue
        is_upgrade = False
        if ring_upgrade_preimage_sha256 is not None:
            if target == receipt_path:
                is_upgrade = _ring_upgrade_receipt_preimage(
                    existing_raw,
                    raw,
                    manifest_preimage_sha256=ring_upgrade_preimage_sha256,
                    member_preimages=member_preimages,
                )
            else:
                is_upgrade = _ring_upgrade_config_preimage(
                    existing_raw,
                    raw,
                    manifest_preimage_sha256=ring_upgrade_preimage_sha256,
                )
        if not is_upgrade:
            raise ValueError(f"refuse to rewrite divergent materialized config: {target}")
        upgrade_preimages[target] = existing_raw

    existing = sum(target.is_file() for target, _ in outputs[:-1])
    created = len(ordered_plan) - existing
    if canonical_manifest_path == manifest_path:
        _replace_validated_bytes_locked(
            canonical_manifest_path,
            expected=manifest_raw,
            replacement=canonical_manifest_raw,
        )
    else:
        _atomic_bytes(canonical_manifest_path, canonical_manifest_raw)
    for output, raw, _ in ordered_plan:
        if output in upgrade_preimages:
            _replace_validated_bytes_locked(
                output, expected=upgrade_preimages[output], replacement=raw
            )
        else:
            _atomic_bytes(output, raw)
    if receipt_path in upgrade_preimages:
        _replace_validated_bytes_locked(
            receipt_path,
            expected=upgrade_preimages[receipt_path],
            replacement=receipt_raw,
        )
    else:
        _atomic_bytes(receipt_path, receipt_raw)
    return {
        **sealed_receipt,
        "created_members": created,
        "existing_valid_members": existing,
    }


def materialize_qtip_configs(
    manifest_path: Path,
    *,
    tier: str,
    layers: list[int],
    output_root: Path,
    ring_upgrade_preimage_sha256: str | None = None,
) -> dict[str, Any]:
    """Serialize canonical manifest/config publication under a process-shared lock."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".QTIP_MATERIALIZE.lock"
    with _MATERIALIZE_LOCK, lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _materialize_qtip_configs_locked(
                manifest_path,
                tier=tier,
                layers=layers,
                output_root=output_root,
                ring_upgrade_preimage_sha256=ring_upgrade_preimage_sha256,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _validated_materialized_qtip_configs(
    source_root: Path,
    *,
    tier: str,
    layers: list[int],
) -> dict[str, Any] | None:
    """Verify and adopt a sealed config publication without regenerating its bytes."""
    receipt_path = source_root / OUTPUT_MANIFEST_NAME
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid sealed QTIP config manifest: {receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "banana-smasher-qtip-config-manifest-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("producer") != PRODUCER_VERB
        or receipt.get("tier") != tier
    ):
        raise ValueError(f"invalid sealed QTIP config manifest: {receipt_path}")
    declared_layers = receipt.get("layers")
    if not isinstance(declared_layers, list) or not set(layers).issubset(declared_layers):
        raise ValueError(
            f"sealed QTIP config manifest lacks selected layers {layers}: {receipt_path}"
        )
    try:
        declared_root = Path(str(receipt["output_root"]))
        if not declared_root.is_absolute():
            declared_root = source_root / declared_root
        declared_root = declared_root.resolve()
    except (KeyError, OSError) as exc:
        raise ValueError(f"invalid sealed QTIP output root: {receipt_path}") from exc
    if declared_root != source_root:
        raise ValueError(
            f"sealed QTIP output root differs from --source-root: "
            f"{declared_root} != {source_root}"
        )

    manifest_path = source_root / RUN_MANIFEST_NAME
    if (
        not manifest_path.is_file()
        or receipt.get("run_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError(f"sealed QTIP run manifest hash drift: {manifest_path}")
    source_manifest = Path(str(receipt.get("source_run_manifest", "")))
    if not source_manifest.is_absolute():
        source_manifest = source_root / source_manifest
    source_manifest = source_manifest.resolve()
    if (
        not source_manifest.is_file()
        or receipt.get("source_run_manifest_sha256") != _sha256(source_manifest)
    ):
        raise ValueError(f"sealed QTIP source run manifest hash drift: {source_manifest}")

    basis = _validate_basis(receipt.get("basis_sha256"), label="config manifest")
    records = receipt.get("member_records")
    if (
        not isinstance(records, list)
        or receipt.get("members") != len(records)
        or not records
    ):
        raise ValueError(f"invalid sealed QTIP member inventory: {receipt_path}")
    ordered_hashes: list[str] = []
    seen_layers: set[int] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"invalid sealed QTIP member record: {receipt_path}")
        member = Path(record["path"])
        if not member.is_absolute():
            member = source_root / member
        member = member.resolve()
        try:
            member.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"sealed QTIP member escapes --source-root: {member}") from exc
        if not member.is_file():
            raise ValueError(f"missing sealed QTIP materialized config: {member}")
        raw = member.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("bytes") != len(raw) or record.get("sha256") != digest:
            raise ValueError(f"refuse to rewrite divergent materialized config: {member}")
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid sealed QTIP materialized config: {member}") from exc
        layer = record.get("layer")
        materialization = config.get("materialization") if isinstance(config, dict) else None
        model_index = (
            config.get("input_identity", {}).get("model_index", {})
            if isinstance(config, dict)
            else {}
        )
        if (
            not isinstance(layer, int)
            or config.get("layer") != layer
            or config.get("tier") != tier
            or not isinstance(materialization, dict)
            or materialization.get("run_manifest_sha256")
            != receipt.get("run_manifest_sha256")
            or materialization.get("source_run_manifest_sha256")
            != receipt.get("source_run_manifest_sha256")
            or model_index.get("sha256") != basis
        ):
            raise ValueError(f"invalid sealed QTIP materialized config lineage: {member}")
        seen_layers.add(layer)
        ordered_hashes.append(digest)
    ordered_sha = hashlib.sha256("".join(ordered_hashes).encode()).hexdigest()
    if receipt.get("ordered_member_sha256") != ordered_sha:
        raise ValueError(f"sealed QTIP ordered member hash drift: {receipt_path}")
    missing_layers = sorted(set(layers) - seen_layers)
    if missing_layers:
        raise ValueError(
            f"sealed QTIP config manifest has no members for layers {missing_layers}: "
            f"{receipt_path}"
        )
    return {
        **receipt,
        "created_members": 0,
        "existing_valid_members": len(records),
        "config_root": str(source_root),
        "immutable_source": True,
    }


def ensure_qtip_configs(
    source_root: Path,
    *,
    tier: str,
    layers: list[int],
    ring_upgrade_preimage_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Adopt sealed configs or materialize missing inputs before solve dispatch."""
    source_root = source_root.resolve()
    if ring_upgrade_preimage_sha256 is None:
        sealed = _validated_materialized_qtip_configs(
            source_root,
            tier=tier,
            layers=layers,
        )
        if sealed is not None:
            return sealed
    manifest_path = source_root / RUN_MANIFEST_NAME
    if tier.startswith("qtip@") and manifest_path.is_file():
        # Unified ring configs are generated bytes, not merely tier-labelled
        # inputs. Re-running the idempotent materializer validates every
        # existing member against the current manifest/ring/AOT identity and
        # preserves valid files byte-for-byte and mtime-for-mtime.
        return materialize_qtip_configs(
            manifest_path,
            tier=tier,
            layers=layers,
            output_root=source_root,
            ring_upgrade_preimage_sha256=ring_upgrade_preimage_sha256,
        )
    selected = set(layers)
    existing_layers: set[int] = set()
    if source_root.is_dir():
        for path in source_root.rglob("E*_*.json"):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("tier") == tier:
                layer = value.get("layer")
                if isinstance(layer, int) and not isinstance(layer, bool):
                    existing_layers.add(layer)
    missing = sorted(selected - existing_layers)
    if not missing:
        return None
    if not manifest_path.is_file():
        # Preserve dispatch compatibility. The resident config gate emits the
        # producer-specific failure after its exact population scan.
        return None
    return materialize_qtip_configs(
        manifest_path,
        tier=tier,
        layers=layers,
        output_root=source_root,
        ring_upgrade_preimage_sha256=ring_upgrade_preimage_sha256,
    )
