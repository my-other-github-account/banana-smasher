from __future__ import annotations

import hashlib
import runpy
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from safetensors import safe_open

from .contract import (
    PackValidationError,
    load_manifest,
    verify_mixed_v7_member_contract,
    verify_pack,
    verify_serve_compatibility,
)

Framework = Literal["np", "pt"]


class MixedV7MemberLoader:
    """Hash-verified reopen surface for identity-bound QTIP2/QTIP3 V7 members."""

    def __init__(self, contract: str | Path) -> None:
        self.contract = verify_mixed_v7_member_contract(contract)
        self.members: dict[tuple[int, int, str], dict[str, Any]] = {}
        for member in self.contract["members"]:
            layer_text, expert_text, projection = member["cell_id"].split(".")
            key = (int(layer_text[1:]), int(expert_text[1:]), projection)
            self.members[key] = member

    def reopen_hashes(self, layer: int, expert: int, tier: str) -> dict[str, str]:
        projections = ("w1", "w2", "w3") if tier == "qtip2" else ("down", "fused13")
        if tier not in {"qtip2", "qtip3"}:
            raise ValueError(f"unsupported mixed V7 tier: {tier}")
        hashes: dict[str, str] = {}
        for projection in projections:
            try:
                member = self.members[(layer, expert, projection)]
            except KeyError as exc:
                raise PackValidationError(
                    f"mixed V7 contract lacks L{layer:03d}.E{expert:03d}.{projection}"
                ) from exc
            if member["tier"] != tier:
                raise PackValidationError("mixed V7 tier selection mismatch")
            if tier == "qtip2":
                hashes[projection] = member["payload"]["sha256"]
                hashes[f"{projection}.tlut"] = member["unit_metadata"]["tlut"][
                    "sha256"
                ]
            else:
                hashes[f"{projection}.codes"] = member["payload"]["sha256"]
                for name in ("control", "tlut"):
                    hashes[f"{projection}.{name}"] = member["unit_metadata"][name]["sha256"]
        return hashes

    def member(self, layer: int, expert: int, projection: str) -> dict[str, Any]:
        try:
            return self.members[(layer, expert, projection)]
        except KeyError as exc:
            raise PackValidationError(
                f"mixed V7 contract lacks L{layer:03d}.E{expert:03d}.{projection}"
            ) from exc


class LayerTensorView:
    """A scoped layer view that keeps every backing mmap handle alive."""

    def __init__(
        self,
        *,
        root: Path,
        names: list[str],
        tensor_index: dict[str, dict[str, Any]],
        framework: Framework,
        handles: dict[str, Any],
    ) -> None:
        self.root = root
        self.names = names
        self.tensor_index = tensor_index
        self.framework = framework
        self._handles = handles
        self._loaded: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name not in self.names:
            raise KeyError(f"tensor {name!r} is not in this layer")
        if name in self._loaded:
            return self._loaded[name]
        metadata = self.tensor_index[name]
        storage = metadata.get("storage", {"kind": "npy", "path": metadata.get("path")})
        kind = storage.get("kind")
        if kind == "safetensors":
            tensor = self._handles[str(storage["path"])].get_tensor(name)
        elif kind == "npy":
            array = np.load(
                self.root / str(storage["path"]), mmap_mode="r", allow_pickle=False
            )
            if self.framework == "np":
                tensor = array
            else:
                tensor = self._to_torch(array)
        elif kind == "raw":
            try:
                dtype = np.dtype(metadata["dtype"])
                shape = tuple(int(value) for value in metadata["shape"])
                array = np.memmap(
                    self.root / str(storage["path"]),
                    dtype=dtype,
                    mode="r",
                    shape=shape,
                    order="C",
                )
            except Exception as exc:
                raise PackValidationError(
                    f"cannot mmap raw tensor {name}: {exc}"
                ) from exc
            if self.framework == "np":
                tensor = array
            else:
                tensor = self._to_torch(array)
        else:
            raise PackValidationError(
                f"unsupported tensor storage kind for {name}: {kind!r}"
            )
        self._loaded[name] = tensor
        return tensor

    @staticmethod
    def _to_torch(array: np.ndarray) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - only a vLLM runtime path
            raise PackValidationError(
                "framework='pt' requires torch to be installed"
            ) from exc
        return torch.from_numpy(array)

    def family(self, family: str) -> dict[str, Any]:
        marker = f".{family}."
        return {
            name.rsplit(".", 1)[-1]: self.get(name)
            for name in self.names
            if marker in name
        }


class PackLoader:
    """The single verified bs-pack reader shared by tools and vLLM."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify: bool = True,
        kernel_cache_root: str | Path | None = None,
        architecture: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if verify:
            self.receipt = verify_pack(self.root)
        else:
            self.receipt = None
        if (kernel_cache_root is None) != (architecture is None):
            raise PackValidationError(
                "kernel_cache_root and architecture must be supplied together"
            )
        self.serve_receipt = None
        self.kernel_cache_root: Path | None = None
        self._runtime_adapter_class: type[Any] | None = None
        if kernel_cache_root is not None and architecture is not None:
            self.kernel_cache_root = Path(kernel_cache_root).resolve()
            self.serve_receipt = verify_serve_compatibility(
                self.root, self.kernel_cache_root, architecture=architecture
            )
        self.manifest = load_manifest(self.root)
        self.tensor_index: dict[str, dict[str, Any]] = self.manifest["tensor_index"]
        self.layers = list(self.manifest["layers"])

    def runtime_adapter_class(self) -> type[Any]:
        """Import the hash-verified adapter directly, without changing PYTHONPATH."""
        if self._runtime_adapter_class is not None:
            return self._runtime_adapter_class
        if self.serve_receipt is None or self.kernel_cache_root is None:
            raise PackValidationError(
                "runtime adapter requires a successful kernel-cache serve check"
            )
        contract = self.serve_receipt["runtime_adapter"]
        path = self.kernel_cache_root / contract["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        module_name = f"_banana_smasher_runtime_{digest[:16]}"
        namespace = runpy.run_path(str(path), run_name=module_name)
        adapter_class = namespace.get(contract["class"])
        if (
            not isinstance(adapter_class, type)
            or getattr(adapter_class, "API_VERSION", None) != 1
            or not callable(getattr(adapter_class, "build_layer", None))
            or not callable(getattr(adapter_class, "forward", None))
        ):
            raise PackValidationError(
                f"runtime adapter class does not implement API v1: "
                f"{contract['class']} in {path}"
            )
        self._runtime_adapter_class = adapter_class
        return adapter_class

    def tensor_names(self, layer: int) -> list[str]:
        prefix = f"layers.{layer}."
        names = sorted(name for name in self.tensor_index if name.startswith(prefix))
        if not names:
            raise KeyError(f"pack has no layer {layer}")
        return names

    @contextmanager
    def open_layer(
        self, layer: int, *, framework: Framework = "pt"
    ) -> Iterator[LayerTensorView]:
        if framework not in ("np", "pt"):
            raise ValueError(f"unsupported framework: {framework!r}")
        names = self.tensor_names(layer)
        with ExitStack() as stack:
            handles: dict[str, Any] = {}
            for name in names:
                metadata = self.tensor_index[name]
                storage = metadata.get(
                    "storage", {"kind": "npy", "path": metadata.get("path")}
                )
                if storage.get("kind") != "safetensors":
                    continue
                relative = str(storage["path"])
                if relative not in handles:
                    handles[relative] = stack.enter_context(
                        safe_open(self.root / relative, framework=framework)
                    )
            yield LayerTensorView(
                root=self.root,
                names=names,
                tensor_index=self.tensor_index,
                framework=framework,
                handles=handles,
            )
