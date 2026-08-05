from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("banana_smasher_plugin")

RUNTIME_SCHEMA = "banana-smasher-vllm-runtime-v1"
RUNTIME_PROFILE = "sm121-single-gpu-v1"

RUNTIME_ENV_DEFAULTS = {
    "BANANA_SMASHER_REQUIRE_TRUE_C": "1",
    "BANANA_SMASHER_WARMUP_LEVEL": "light",
    "CUDA_MODULE_LOADING": "LAZY",
    "FLASHINFER_DISABLE_JIT": "1",
    "MALLOC_MMAP_THRESHOLD_": "65536",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_DISABLE_FLASHINFER_MLA": "0",
    "VLLM_FLASHINFER_MLA_DISABLE": "0",
    "VLLM_FLASHINFER_MLA_SKIP_WORKSPACE_BUFFER": "1",
    "VLLM_HAS_FLASHINFER_CUBIN": "1",
    "VLLM_MLA_DISABLE": "0",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_DEEP_GEMM": "1",
    "VLLM_USE_DEEP_GEMM_E8M0": "1",
}

ENGINE_DEFAULTS: dict[str, Any] = {
    "trust_remote_code": True,
    "tokenizer_mode": "deepseek_v4",
    "kv_cache_dtype": "fp8",
    "block_size": 256,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,
    "kv_cache_memory_bytes": 3221225472,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 16,
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
    "scheduler_reserve_full_isl": False,
    "generation_config": "vllm",
    "reasoning_parser": "deepseek_v4",
}

FRONTEND_DEFAULTS: dict[str, Any] = {
    "default_chat_template_kwargs": {"enable_thinking": True},
    "enable_auto_tool_choice": True,
    "tool_call_parser": "deepseek_v4",
}

_STOCK_DEFAULTS: dict[str, Any] = {
    "served_model_name": None,
    "trust_remote_code": False,
    "tokenizer_mode": "auto",
    "kv_cache_dtype": "auto",
    "block_size": None,
    "max_model_len": None,
    "gpu_memory_utilization": 0.92,
    "kv_cache_memory_bytes": None,
    "max_num_batched_tokens": None,
    "max_num_seqs": None,
    "cudagraph_capture_sizes": None,
    "scheduler_reserve_full_isl": True,
    "generation_config": "auto",
    "reasoning_parser": "",
    "default_chat_template_kwargs": None,
    "enable_auto_tool_choice": False,
    "tool_call_parser": None,
}


def configure_runtime_environment() -> dict[str, str]:
    """Install the plugin-owned process defaults before native registration."""
    applied: dict[str, str] = {}
    for name, value in RUNTIME_ENV_DEFAULTS.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_runtime_profile(model: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the artifact-scoped vLLM profile for a local Banana export."""
    root = Path(model).expanduser()
    if not root.is_dir():
        return None
    config = _read_json(root / "config.json")
    manifest = _read_json(root / "BANANA_PACK_MANIFEST.json")
    if config is None or manifest is None:
        return None
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return None
    if quantization.get("quant_method") != "banana_smasher":
        return None
    if manifest.get("schema") != "bs-pack" or manifest.get("quant_method") != "banana_smasher":
        return None

    configured = config.get("banana_smasher_runtime")
    if configured is not None:
        if not isinstance(configured, dict):
            raise RuntimeError("config.json banana_smasher_runtime must be an object")
        if configured.get("schema") != RUNTIME_SCHEMA:
            raise RuntimeError(
                "unsupported Banana Smasher runtime schema: "
                f"{configured.get('schema')!r}"
            )
        if configured.get("profile") != RUNTIME_PROFILE:
            raise RuntimeError(
                "unsupported Banana Smasher runtime profile: "
                f"{configured.get('profile')!r}"
            )

    engine = dict(ENGINE_DEFAULTS)
    frontend = dict(FRONTEND_DEFAULTS)
    if isinstance(configured, dict):
        configured_engine = configured.get("engine_args")
        configured_frontend = configured.get("frontend_args")
        if isinstance(configured_engine, dict):
            engine.update(
                {key: configured_engine[key] for key in ENGINE_DEFAULTS if key in configured_engine}
            )
        if isinstance(configured_frontend, dict):
            frontend.update(
                {
                    key: configured_frontend[key]
                    for key in FRONTEND_DEFAULTS
                    if key in configured_frontend
                }
            )

    model_name = (
        configured.get("served_model_name")
        if isinstance(configured, dict)
        else None
    ) or manifest.get("model_id") or manifest.get("instance_id") or "banana-smasher"
    return {
        "root": str(root.resolve()),
        "schema": RUNTIME_SCHEMA,
        "profile": RUNTIME_PROFILE,
        "served_model_name": str(model_name),
        "engine_args": engine,
        "frontend_args": frontend,
    }


def apply_runtime_profile(args: Namespace) -> dict[str, Any] | None:
    """Fill only untouched stock vLLM defaults for a Banana export."""
    positional_model = getattr(args, "model_tag", None)
    model = positional_model or getattr(args, "model", None)
    if model is None:
        return None
    profile = load_runtime_profile(model)
    if profile is None:
        return None
    if positional_model is not None:
        # Match ServeSubcommand.cmd: the positional model takes precedence.
        args.model = positional_model

    configure_runtime_environment()
    desired = {
        "served_model_name": [profile["served_model_name"]],
        **profile["engine_args"],
        **profile["frontend_args"],
    }
    applied: dict[str, Any] = {}
    for name, value in desired.items():
        if not hasattr(args, name):
            continue
        if getattr(args, name) == _STOCK_DEFAULTS[name]:
            setattr(args, name, value)
            applied[name] = value
    setattr(args, "_banana_smasher_runtime_profile", profile["profile"])
    _LOG.warning(
        "BANANA_SMASHER_VLLM_DEFAULTS model=%s profile=%s applied=%s",
        profile["root"],
        profile["profile"],
        sorted(applied),
    )
    return {**profile, "applied": applied}


def install_vllm_arg_defaults() -> bool:
    """Patch vLLM's normal Namespace-to-EngineArgs seam once per process."""
    from vllm.engine.arg_utils import EngineArgs

    descriptor = EngineArgs.__dict__["from_cli_args"]
    original = descriptor.__func__
    if getattr(original, "_banana_smasher_runtime_defaults", False):
        return False

    @classmethod
    def from_cli_args(cls, args: Namespace):
        apply_runtime_profile(args)
        return original(cls, args)

    from_cli_args.__func__._banana_smasher_runtime_defaults = True  # type: ignore[attr-defined]
    from_cli_args.__func__._banana_smasher_original = original  # type: ignore[attr-defined]
    EngineArgs.from_cli_args = from_cli_args  # type: ignore[method-assign]
    return True


__all__ = [
    "ENGINE_DEFAULTS",
    "FRONTEND_DEFAULTS",
    "RUNTIME_ENV_DEFAULTS",
    "RUNTIME_PROFILE",
    "RUNTIME_SCHEMA",
    "apply_runtime_profile",
    "configure_runtime_environment",
    "install_vllm_arg_defaults",
    "load_runtime_profile",
]
