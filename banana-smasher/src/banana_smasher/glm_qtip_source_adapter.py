"""Canonical GLM FP8 -> QTIP source adapter (no external monkeypatch).

Uses the HF loader's dtype/scale contract, never the directory name.  Auxiliary
MTP and nonrouted tensors are not QTIP source units.  The caller's solver basis
gate still binds model_index_sha256 before loading any unit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hf_moe import (
    _load_safetensors_matrix,
    _routed_scale_binding,
    _safetensors_header,
    _sha256,
)


def load_glm_fp8_weight(
    model_root: Path, layer: int, expert: int, projection: str
) -> tuple[Any, dict[str, Any]]:
    import torch

    root = Path(model_root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    shape = config.get("text_config", config)
    if projection not in ("fused13", "down"):
        raise ValueError(f"unsupported GLM projection: {projection}")
    if not (
        int(shape.get("first_k_dense_replace", 0))
        <= layer
        < int(shape["num_hidden_layers"])
        and 0 <= expert < int(shape["n_routed_experts"])
    ):
        raise ValueError("GLM unit is outside main routed scope")
    index_path = root / "model.safetensors.index.json"
    mapping = json.loads(index_path.read_text())["weight_map"]
    names = ("gate", "up") if projection == "fused13" else ("down",)
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    matrices, sources = [], []
    headers: dict[str, dict[str, Any]] = {}
    for name in names:
        key = prefix + name + "_proj.weight"
        shard_name = mapping[key]
        # Scales may reside in a different safetensors shard.
        for filename in {shard_name, mapping.get(key + "_scale_inv", shard_name)}:
            if filename not in headers:
                headers[filename] = _safetensors_header(root / filename)
        row = {"name": key, **headers[shard_name][key]}
        scale_path, scale_row, transform = _routed_scale_binding(
            root, row, weight_map=mapping, headers=headers
        )
        shard = root / shard_name
        matrix = _load_safetensors_matrix(
            shard, row, scale_source=scale_path, scale_row=scale_row
        )
        matrices.append(torch.from_numpy(matrix.copy()))
        sources.append(
            {
                "path": str(shard),
                "bytes": shard.stat().st_size,
                "sha256": _sha256(shard),
                "weight_key": key,
                "dtype": row["dtype"],
                "transform": transform,
                "scale_source": None
                if scale_path is None or scale_row is None
                else {
                    "path": str(scale_path),
                    "bytes": scale_path.stat().st_size,
                    "sha256": _sha256(scale_path),
                    "weight_key": scale_row["name"],
                },
            }
        )
    value = torch.cat(matrices, dim=0) if projection == "fused13" else matrices[0]
    hidden, intermediate = (
        int(shape["hidden_size"]),
        int(shape["moe_intermediate_size"]),
    )
    expected = (
        (2 * intermediate, hidden)
        if projection == "fused13"
        else (hidden, intermediate)
    )
    if tuple(value.shape) != expected:
        raise ValueError(
            f"GLM source shape mismatch: {tuple(value.shape)} != {expected}"
        )
    return value.contiguous(), {
        "schema": "banana-smasher.glm-fp8-source.v1",
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "config_sha256": _sha256(config_path),
        "shards": sources,
    }


def capture_source_closure(runner, runtime_modules) -> dict[str, Any]:
    """Hash resolved imported files and interpreter/dependency identities.

    No inferred upstream commit: external runtime trees may not be git checkouts.
    This is an import-path/file attestation, not a hash of Python memory objects.
    """
    import hashlib
    import importlib
    import importlib.metadata
    import platform
    import sys

    if set(runtime_modules) != {"bitshift", "ldlq", "math_utils", "kernel_decompress"}:
        raise ValueError("GLM source closure requires all four loaded runtime modules")
    modules = {
        name: importlib.import_module(f"banana_smasher.{name}")
        for name in (
            "solver_qtip_profile",
            "qtip_batch_controller",
            "qtip_batch",
            "qtip_viterbi",
            "qtip_kernel_cache",
            "qtip_rings",
            "qtip1",
            "hf_moe",
            "glm_qtip_source_adapter",
            "glm_qtip_producers",
        )
    }
    solver = modules["solver_qtip_profile"]
    if solver._load_weight.__module__ != solver.__name__:
        raise ValueError("GLM closure refuses external weight loader monkeypatch")
    modules["qtip_runner"] = runner
    modules.update(
        {"runtime." + name: module for name, module in runtime_modules.items()}
    )
    paths = {}
    for name, module in modules.items():
        if not getattr(module, "__file__", None):
            raise ValueError(f"GLM source closure module has no file: {name}")
        paths[name] = Path(str(module.__file__)).resolve()
    paths["qtip_rings.json"] = Path(__file__).with_name("qtip_rings.json").resolve()
    files = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    dependencies = {}
    for name in ("torch", "triton", "numpy", "safetensors"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    result = {
        "schema": "banana-smasher.glm-import-closure.v1",
        "files": files,
        "interpreter": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "sha256": _sha256(Path(sys.executable).resolve()),
        },
        "dependencies": dependencies,
    }
    # Bind content by import role, not installation location. Shareable solve
    # receipts redact absolute paths; this signed identity remains reproducible.
    result["identity"] = {
        "schema": result["schema"],
        "files": {name: row["sha256"] for name, row in files.items()},
        "interpreter": {
            key: value
            for key, value in result["interpreter"].items()
            if key != "executable"
        },
        "dependencies": dependencies,
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result["identity"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def require_source_closure(expected_sha256, observed):
    if not expected_sha256 or expected_sha256 != observed["sha256"]:
        raise ValueError(
            "GLM launch source closure missing or mismatched; owner must bind real imports"
        )


def bind_source_closure(model_root, configs, runner, runtime_modules):
    mapping = json.loads(
        (Path(model_root) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    if not any(
        key.startswith("model.language_model.layers.") and ".mlp.experts." in key
        for key in mapping
    ):
        return None
    if not configs or any(
        not config.get("glm_source_closure_sha256") for config in configs
    ):
        raise ValueError("GLM launch source closure pin is required")
    closure = capture_source_closure(runner, runtime_modules)
    for config in configs:
        require_source_closure(config["glm_source_closure_sha256"], closure)
    return closure
