from __future__ import annotations

import pytest

from banana_smasher.exl3_loader import (
    admit_exl3_storage_alias,
    resolve_exl3_storage_prefix,
)


class HeaderOnlyCollection:
    def __init__(self, keys: set[str]) -> None:
        self.keys = keys

    def has_tensor(self, key: str) -> bool:
        return key in self.keys


K216_PREFIX = "layers.0.ffn.experts.0.w3"
K216_FIELDS = {"trellis", "suh", "svh", "mcg"}


def _keys(prefix: str, fields: set[str] = K216_FIELDS) -> set[str]:
    return {f"{prefix}.{field}" for field in fields}


def test_known_k216_tp1_rank0_storage_is_admitted_without_renaming_module() -> None:
    collection = HeaderOnlyCollection(_keys(f"{K216_PREFIX}.rank0"))

    resolved = resolve_exl3_storage_prefix(collection, K216_PREFIX)

    assert resolved == f"{K216_PREFIX}.rank0"


def test_canonical_exl3_storage_remains_preferred() -> None:
    collection = HeaderOnlyCollection(
        _keys(K216_PREFIX) | _keys(f"{K216_PREFIX}.rank0")
    )

    assert resolve_exl3_storage_prefix(collection, K216_PREFIX) == K216_PREFIX


def test_partial_rank0_storage_is_rejected() -> None:
    collection = HeaderOnlyCollection(
        _keys(f"{K216_PREFIX}.rank0", {"trellis", "suh", "mcg"})
    )

    with pytest.raises(ValueError, match="incomplete EXL3 rank0 storage"):
        resolve_exl3_storage_prefix(collection, K216_PREFIX)


def test_known_k216_alias_is_installed_without_changing_runtime_module_key() -> None:
    class Config:
        stc = HeaderOnlyCollection(_keys(f"{K216_PREFIX}.rank0"))

    class Module:
        key = K216_PREFIX
        alt_key = None
        config = Config()

    module = Module()

    resolved = admit_exl3_storage_alias(module)

    assert resolved == f"{K216_PREFIX}.rank0"
    assert module.key == K216_PREFIX
    assert module.alt_key == f"{K216_PREFIX}.rank0"
