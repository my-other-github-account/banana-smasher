from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


DEFAULT_MODEL_NAME = "banana-smasher-v5"
DEFAULT_FLASHINFER_CACHE_VOLUME = "banana-smasher-flashinfer-cache"
FLASHINFER_CACHE_ROOT = "/root/.cache/vllm/flashinfer_autotune_cache"

# These are process defaults, not a second serving implementation.  The final
# process is still stock ``vllm serve`` with the Banana Smasher general plugin.
RUNTIME_ENV_DEFAULTS = {
    "CUDA_MODULE_LOADING": "LAZY",
    "FLASHINFER_DISABLE_JIT": "1",
    "MALLOC_MMAP_THRESHOLD_": "65536",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_HAS_FLASHINFER_CUBIN": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_DEEP_GEMM": "1",
    "VLLM_USE_DEEP_GEMM_E8M0": "1",
}


def inspect_model_pack(model: Path) -> Path:
    """Perform the small launch-time identity check; the plugin validates bytes."""
    root = model.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"model artifact directory does not exist: {root}")
    manifest = root / "BANANA_PACK_MANIFEST.json"
    config_path = root / "config.json"
    if not manifest.is_file() or not config_path.is_file():
        raise ValueError(
            "model artifact must contain BANANA_PACK_MANIFEST.json and config.json"
        )
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config: {config_path}: {exc}") from exc
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "banana_smasher":
        raise ValueError(
            "model config quantization_config.quant_method must be banana_smasher"
        )
    return root


def vllm_serve_command(
    model: Path | str,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = DEFAULT_MODEL_NAME,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Return Boot10's known-working stock-vLLM command line."""
    return [
        "vllm",
        "serve",
        str(model),
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--tokenizer-mode",
        "deepseek_v4",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.80",
        "--kv-cache-memory-bytes",
        "3221225472",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "16",
        "--compilation-config",
        '{"cudagraph_capture_sizes":[1,2,4,8,16]}',
        "--no-scheduler-reserve-full-isl",
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "deepseek_v4",
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "deepseek_v4",
        "--host",
        host,
        "--port",
        str(port),
        *extra_args,
    ]


def container_serve_command(
    model: Path,
    *,
    image: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = DEFAULT_MODEL_NAME,
    cache_volume: str | None = DEFAULT_FLASHINFER_CACHE_VOLUME,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Run the same vLLM command in the dependency-complete PoC image."""
    root = inspect_model_pack(model)
    published_port = f"{port}:{port}" if host == "0.0.0.0" else f"{host}:{port}:{port}"
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-p",
        published_port,
        "-v",
        f"{root}:/model:ro",
    ]
    if cache_volume:
        command.extend(["-v", f"{cache_volume}:{FLASHINFER_CACHE_ROOT}"])
    command.extend(
        [
            image,
            *vllm_serve_command(
                "/model",
                host="0.0.0.0",
                port=port,
                served_model_name=served_model_name,
                extra_args=extra_args,
            ),
        ]
    )
    return command


def build_serve_command(
    model: Path,
    *,
    container_image: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = DEFAULT_MODEL_NAME,
    cache_volume: str | None = DEFAULT_FLASHINFER_CACHE_VOLUME,
    extra_args: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Build a local-pip or pinned-container launch transaction."""
    root = inspect_model_pack(model)
    environment = {**os.environ, **RUNTIME_ENV_DEFAULTS}
    if container_image:
        command = container_serve_command(
            root,
            image=container_image,
            host=host,
            port=port,
            served_model_name=served_model_name,
            cache_volume=cache_volume,
            extra_args=extra_args,
        )
    else:
        command = vllm_serve_command(
            root,
            host=host,
            port=port,
            served_model_name=served_model_name,
            extra_args=extra_args,
        )
    return command, environment


def serve(
    model: Path,
    *,
    container_image: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = DEFAULT_MODEL_NAME,
    cache_volume: str | None = DEFAULT_FLASHINFER_CACHE_VOLUME,
    extra_args: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> None:
    """Replace this process with Docker or stock ``vllm serve``."""
    command, default_environment = build_serve_command(
        model,
        container_image=container_image,
        host=host,
        port=port,
        served_model_name=served_model_name,
        cache_volume=cache_volume,
        extra_args=extra_args,
    )
    os.execvpe(command[0], command, dict(environment or default_environment))
