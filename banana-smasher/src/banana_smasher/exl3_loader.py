from __future__ import annotations

from typing import Protocol


class TensorHeaderCollection(Protocol):
    def has_tensor(self, key: str) -> bool: ...


_REQUIRED_EXL3_GROUPS = (("su", "suh"), ("sv", "svh"), ("trellis",))
_KNOWN_EXL3_FIELDS = {"su", "suh", "sv", "svh", "trellis", "mcg", "mul1", "bias"}


def _complete(collection: TensorHeaderCollection, prefix: str) -> bool:
    return all(
        any(collection.has_tensor(f"{prefix}.{field}") for field in alternatives)
        for alternatives in _REQUIRED_EXL3_GROUPS
    )


def _present_fields(collection: TensorHeaderCollection, prefix: str) -> set[str]:
    return {
        field
        for field in _KNOWN_EXL3_FIELDS
        if collection.has_tensor(f"{prefix}.{field}")
    }


def resolve_exl3_storage_prefix(
    collection: TensorHeaderCollection,
    module_prefix: str,
    *,
    rank: int = 0,
) -> str | None:
    """Resolve canonical or single-rank EXL3 storage without changing module identity.

    Coalesced TP1 artifacts retain a ``.rank0`` storage namespace even though the
    runtime module key is rank-free. Canonical rank-free EXL3 remains preferred.
    A partially present rank namespace fails closed rather than falling through
    to a misleading unsupported-format error.
    """
    if not module_prefix or type(rank) is not int or rank < 0:
        raise ValueError("EXL3 storage resolution requires a module prefix and non-negative rank")
    if _complete(collection, module_prefix):
        return module_prefix
    ranked = f"{module_prefix}.rank{rank}"
    if _complete(collection, ranked):
        return ranked
    present = _present_fields(collection, ranked)
    if present:
        raise ValueError(
            f"incomplete EXL3 rank{rank} storage for {module_prefix}: "
            f"present={sorted(present)}"
        )
    return None


def admit_exl3_storage_alias(module: object, *, rank: int = 0) -> str | None:
    """Install a resolved storage-only alias on an EXL3 runtime module."""
    key = getattr(module, "key", None)
    config = getattr(module, "config", None)
    collection = getattr(config, "stc", None)
    if not isinstance(key, str) or not key or collection is None:
        return None
    resolved = resolve_exl3_storage_prefix(collection, key, rank=rank)
    if resolved is None or resolved == key:
        return resolved
    existing = getattr(module, "alt_key", None)
    if existing not in (None, resolved):
        raise ValueError(
            f"refusing to replace existing storage alias for {key}: {existing!r}"
        )
    setattr(module, "alt_key", resolved)
    return resolved
