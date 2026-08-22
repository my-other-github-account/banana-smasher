"""Lifecycle-only residency for a SHA-pinned sealed builder.

The sealed source remains the sole owner of model construction, plane decode,
layer materialization, forward, and readout math.  This module only keeps the
constructed model alive and reuses the sealed ``build_layer_sd`` output for
checkpoint-bound ``load_state_dict`` swaps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SealedBuilderBinding:
    module: Any
    source_path: Path
    source_sha256: str
    plane_source: type
    build_layer_sd: Callable[..., Mapping[str, Any]]
    materialize_layer: Callable[..., Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_sealed_builder(
    source_path: str | Path, *, expected_sha256: str
) -> SealedBuilderBinding:
    """Import exact sealed lifecycle callables after checking source identity."""
    source = Path(source_path).expanduser().resolve(strict=True)
    observed = sha256_file(source)
    if observed != expected_sha256:
        raise RuntimeError(
            f"sealed builder SHA mismatch: {source}: {observed} != {expected_sha256}"
        )
    module_name = f"banana_smasher_sealed_builder_{observed[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed builder: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    required = ("PlaneSource", "build_layer_sd", "materialize_layer")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"sealed builder lifecycle seam missing: {missing}")
    return SealedBuilderBinding(
        module=module,
        source_path=source,
        source_sha256=observed,
        plane_source=module.PlaneSource,
        build_layer_sd=module.build_layer_sd,
        materialize_layer=module.materialize_layer,
    )


class ResidentSealedModel:
    """Keep one sealed model object resident and swap only declared state keys."""

    def __init__(
        self,
        *,
        binding: SealedBuilderBinding,
        model: Any,
        config: Any,
        weight_map: Mapping[str, str],
        get_tensor: Callable[[str], Any],
        layers: Sequence[int],
    ) -> None:
        self.binding = binding
        self.model = model
        self.config = config
        self.weight_map = weight_map
        self.get_tensor = get_tensor
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("resident sealed layers must be non-empty and unique")
        self.model_object_identity = id(model)
        self.model_build_count = 0
        self.swap_count = 0

    def materialize_once(self, plane_source: Any) -> dict[str, Any]:
        """Run the sealed planes -> state -> materialize closure exactly once."""
        if self.model_build_count:
            raise RuntimeError("sealed resident model was already materialized")
        for layer in self.layers:
            state = self.binding.build_layer_sd(
                layer,
                self.weight_map,
                self.get_tensor,
                "planes",
                plane_source,
            )
            self.binding.materialize_layer(self.model, layer, state, self.config)
        self.model_build_count = 1
        return self._receipt("materialize_once", loaded_keys=())

    def hot_swap(
        self,
        plane_source: Any,
        *,
        mutable_keys_by_layer: Mapping[int, Sequence[str]],
    ) -> dict[str, Any]:
        """Rebuild via sealed code, then copy only declared checkpoint surfaces."""
        if self.model_build_count != 1:
            raise RuntimeError("sealed resident model must be materialized before hot swap")
        loaded: list[str] = []
        for layer in self.layers:
            declared = tuple(str(key) for key in mutable_keys_by_layer.get(layer, ()))
            if not declared:
                continue
            full_state = self.binding.build_layer_sd(
                layer,
                self.weight_map,
                self.get_tensor,
                "planes",
                plane_source,
            )
            missing = [key for key in declared if key not in full_state]
            if missing:
                raise RuntimeError(f"sealed swap keys absent at layer {layer}: {missing}")
            selected = {key: full_state[key] for key in declared}
            resident_layer = self.model.model.layers[layer]
            incompatible = resident_layer.load_state_dict(
                selected, strict=False, assign=False
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"sealed swap unexpected keys at layer {layer}: "
                    f"{incompatible.unexpected_keys}"
                )
            loaded.extend(f"L{layer:03d}:{key}" for key in declared)
        self.swap_count += 1
        return self._receipt("hot_swap", loaded_keys=loaded)

    def _receipt(self, operation: str, *, loaded_keys: Sequence[str]) -> dict[str, Any]:
        loaded = tuple(loaded_keys)
        return {
            "operation": operation,
            "sealed_builder_source_path": str(self.binding.source_path),
            "sealed_builder_source_sha256": self.binding.source_sha256,
            "model_object_identity": self.model_object_identity,
            "model_build_count": self.model_build_count,
            "swap_count": self.swap_count,
            "loaded_key_count": len(loaded),
            "loaded_keys_sha256": hashlib.sha256(
                "\n".join(sorted(loaded)).encode()
            ).hexdigest(),
            "load_method": "torch.nn.Module.load_state_dict",
            "assign": False,
        }


__all__ = [
    "ResidentSealedModel",
    "SealedBuilderBinding",
    "bind_sealed_builder",
    "sha256_file",
]
