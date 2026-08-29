"""Fail-closed Banana V1 all-layer manifest adapter for resident providers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .banana_v1 import (
    BANANA_V1_GEOMETRY,
    materialize_banana_v1_candidate,
    predict_banana_v1_candidate,
    verify_banana_v1_candidate,
)


MANIFEST_SCHEMA = "banana-smasher-banana-v1-all43-runtime-adapter-v1"
_LAYER_COUNT = 43
_SHA_CHARS = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lower_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _verified_path(value: object, expected_sha: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a nonempty string")
    path = Path(value).expanduser().resolve()
    expected = _lower_sha(expected_sha, f"{label} SHA")
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} identity mismatch: {path}")
    return path


@dataclass(frozen=True)
class BananaV1MemberBinding:
    id: str
    layer: int
    expert: int
    projection: str
    row_start: int
    column_start: int
    member_root: Path
    receipt_sha256: str


class BananaV1All43Adapter:
    """Map one authenticated Banana V1 16x16 member into every physical layer."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        basis_sha256: str,
        terminal_path: Path,
        terminal_sha256: str,
        shared_codebook: np.ndarray,
        shared_codebook_sha256: str,
        members: tuple[BananaV1MemberBinding, ...],
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.basis_sha256 = basis_sha256
        self.terminal_path = terminal_path
        self.terminal_sha256 = terminal_sha256
        self.shared_codebook = shared_codebook
        self.shared_codebook_sha256 = shared_codebook_sha256
        self.members = members
        self.layers = tuple(row.layer for row in members)
        self._by_layer = {row.layer: row for row in members}
        self._decoded: dict[int, np.ndarray] = {}

    @classmethod
    def open(
        cls,
        manifest: str | Path,
        *,
        expected_basis_sha256: str,
        expected_terminal_sha256: str,
        expected_manifest_sha256: str | None = None,
    ) -> "BananaV1All43Adapter":
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(f"Banana V1 all43 manifest is missing: {manifest_path}")
        manifest_sha = _sha256(manifest_path)
        if expected_manifest_sha256 is not None and manifest_sha != _lower_sha(
            expected_manifest_sha256, "manifest SHA"
        ):
            raise ValueError("Banana V1 all43 manifest SHA mismatch")
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Banana V1 all43 manifest: {exc}") from exc
        if not isinstance(document, Mapping) or document.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"expected {MANIFEST_SCHEMA}")
        basis = _lower_sha(document.get("basis_sha256"), "manifest basis")
        expected_basis = _lower_sha(expected_basis_sha256, "expected basis")
        if basis != expected_basis:
            raise ValueError("Banana V1 all43 basis mismatch")

        terminal_row = document.get("terminal")
        if not isinstance(terminal_row, Mapping):
            raise ValueError("Banana V1 all43 terminal binding is missing")
        terminal_path = _verified_path(
            terminal_row.get("path"), terminal_row.get("sha256"), "parent terminal"
        )
        terminal_sha = _lower_sha(terminal_row.get("sha256"), "parent terminal SHA")
        if terminal_sha != _lower_sha(expected_terminal_sha256, "expected terminal SHA"):
            raise ValueError("Banana V1 all43 parent terminal SHA mismatch")
        terminal = json.loads(terminal_path.read_text())
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("basis_sha256") != basis
            or terminal.get("model_index_sha256") != basis
            or terminal.get("roster_count") != _LAYER_COUNT
            or terminal.get("layers") != list(range(_LAYER_COUNT))
        ):
            raise ValueError("Banana V1 all43 parent terminal contract drift")

        codebook_row = document.get("shared_codebook")
        if not isinstance(codebook_row, Mapping):
            raise ValueError("Banana V1 all43 shared codebook binding is missing")
        codebook_path = _verified_path(
            codebook_row.get("path"), codebook_row.get("sha256"), "shared codebook"
        )
        if codebook_row.get("dtype") != "float16" or codebook_row.get("shape") != [1024]:
            raise ValueError("Banana V1 all43 shared codebook geometry drift")
        compact = np.fromfile(codebook_path, dtype=np.float16)
        if compact.shape != (BANANA_V1_GEOMETRY.codebook_levels,) or not bool(
            np.isfinite(compact).all()
        ):
            raise ValueError("Banana V1 all43 shared codebook is not finite FP16[1024]")
        terminal_codebook = terminal.get("single_shared_codebook")
        if (
            not isinstance(terminal_codebook, Mapping)
            or terminal_codebook.get("sha256") != codebook_row.get("sha256")
            or terminal_codebook.get("shape") != [1024]
            or terminal_codebook.get("dtype") != "float16"
        ):
            raise ValueError("Banana V1 all43 terminal codebook binding drift")

        raw_members = document.get("members")
        if not isinstance(raw_members, list) or len(raw_members) != _LAYER_COUNT:
            raise ValueError("Banana V1 all43 manifest requires exactly 43 members")
        bindings: list[BananaV1MemberBinding] = []
        roots: set[Path] = set()
        for expected_layer, row in enumerate(raw_members):
            if not isinstance(row, Mapping):
                raise ValueError("Banana V1 all43 member row must be a mapping")
            layer = int(row.get("layer", -1))
            expected_id = f"L{expected_layer:03d}/E000/w1/tile-r000-r015-c000-c015"
            if (
                layer != expected_layer
                or row.get("id") != expected_id
                or row.get("expert") != 0
                or row.get("projection") != "w1"
                or row.get("row_start") != 0
                or row.get("column_start") != 0
            ):
                raise ValueError(f"Banana V1 all43 roster gap or mapping drift at L{expected_layer:03d}")
            root_value = row.get("member_root")
            if not isinstance(root_value, str) or not root_value:
                raise ValueError(f"L{layer:03d} Banana V1 member root is missing")
            root = Path(root_value).expanduser().resolve()
            if root in roots or not root.is_dir():
                raise ValueError(f"L{layer:03d} Banana V1 member root is duplicate or missing")
            roots.add(root)
            receipt_sha = _lower_sha(row.get("receipt_sha256"), "member receipt SHA")
            receipt_path = root / "BANANA_V1_RECEIPT.json"
            if _sha256(receipt_path) != receipt_sha or not verify_banana_v1_candidate(root):
                raise ValueError(f"L{layer:03d} Banana V1 candidate identity mismatch")
            materialized = materialize_banana_v1_candidate(root)
            member_codebook = np.ascontiguousarray(materialized["codebook"], dtype=np.float16)
            if not np.array_equal(member_codebook, compact):
                raise ValueError(f"L{layer:03d} does not use the one shared codebook")
            bindings.append(
                BananaV1MemberBinding(
                    id=expected_id,
                    layer=layer,
                    expert=0,
                    projection="w1",
                    row_start=0,
                    column_start=0,
                    member_root=root,
                    receipt_sha256=receipt_sha,
                )
            )
        return cls(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            basis_sha256=basis,
            terminal_path=terminal_path,
            terminal_sha256=terminal_sha,
            shared_codebook=np.ascontiguousarray(compact),
            shared_codebook_sha256=str(codebook_row["sha256"]),
            members=tuple(bindings),
        )

    def decode_member(self, layer: int) -> np.ndarray:
        selected = int(layer)
        if selected not in self._by_layer:
            raise ValueError(f"Banana V1 all43 has no L{selected:03d} member")
        if selected not in self._decoded:
            row = self._by_layer[selected]
            decoded = np.ascontiguousarray(
                predict_banana_v1_candidate(row.member_root), dtype=np.float32
            )
            stored = materialize_banana_v1_candidate(row.member_root)["decoded"]
            if decoded.shape != (16, 16) or not np.array_equal(decoded, stored):
                raise ValueError(f"L{selected:03d} Banana V1 decode is not exact")
            self._decoded[selected] = decoded
        return self._decoded[selected].copy()

    def patch_weight(
        self, layer: int, expert: int, projection: str, weight: Any
    ) -> Any:
        selected = self._by_layer.get(int(layer))
        if selected is None or int(expert) != selected.expert or projection != selected.projection:
            return weight
        if getattr(weight, "ndim", None) != 2 or min(map(int, weight.shape)) < 16:
            raise ValueError("Banana V1 physical weight must be a matrix at least 16x16")
        patched = weight.clone()
        tile = weight.new_tensor(self.decode_member(selected.layer))
        patched[
            selected.row_start : selected.row_start + 16,
            selected.column_start : selected.column_start + 16,
        ] = tile
        return patched

    def bind_plane_sources(self, sources: Mapping[int, Any]) -> None:
        if set(map(int, sources)) != set(range(_LAYER_COUNT)) or len(sources) != _LAYER_COUNT:
            raise ValueError("Banana V1 adapter requires exact PlaneSource layers 0..42")
        for layer in range(_LAYER_COUNT):
            source = sources[layer]
            if int(getattr(source, "layer", -1)) != layer:
                raise ValueError(f"Banana V1 PlaneSource identity drift at L{layer:03d}")
            source.__dict__["_banana_v1_all43_adapter"] = self


def bind_banana_v1_all43_from_env(sources: Mapping[int, Any]) -> BananaV1All43Adapter | None:
    manifest = os.environ.get("BANANA_V1_ALL43_MANIFEST")
    if manifest is None:
        return None
    adapter = BananaV1All43Adapter.open(
        manifest,
        expected_basis_sha256=os.environ["BANANA_V1_ALL43_BASIS_SHA256"],
        expected_terminal_sha256=os.environ["BANANA_V1_ALL43_TERMINAL_SHA256"],
        expected_manifest_sha256=os.environ["BANANA_V1_ALL43_MANIFEST_SHA256"],
    )
    adapter.bind_plane_sources(sources)
    return adapter


__all__ = [
    "BananaV1All43Adapter",
    "BananaV1MemberBinding",
    "MANIFEST_SCHEMA",
    "bind_banana_v1_all43_from_env",
]
