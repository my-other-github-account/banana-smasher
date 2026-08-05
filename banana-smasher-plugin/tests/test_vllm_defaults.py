from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from banana_smasher_plugin.vllm_defaults import (
    ENGINE_DEFAULTS,
    FRONTEND_DEFAULTS,
    RUNTIME_ENV_DEFAULTS,
    apply_runtime_profile,
    install_vllm_arg_defaults,
    load_runtime_profile,
)


def _pack(root: Path, *, profiled: bool = True) -> Path:
    root.mkdir()
    manifest = {
        "schema": "bs-pack",
        "schema_version": 1,
        "quant_method": "banana_smasher",
        "model_id": "DeepSeek-V4-Flash-BQ3",
        "instance_id": "u012-v5",
    }
    config: dict[str, object] = {
        "quantization_config": {"quant_method": "banana_smasher"}
    }
    if profiled:
        config["banana_smasher_runtime"] = {
            "schema": "banana-smasher-vllm-runtime-v1",
            "profile": "sm121-single-gpu-v1",
            "served_model_name": "banana-smasher-v5",
            "engine_args": dict(ENGINE_DEFAULTS),
            "frontend_args": dict(FRONTEND_DEFAULTS),
        }
    (root / "BANANA_PACK_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "config.json").write_text(json.dumps(config))
    return root


def _stock_args(model: Path) -> Namespace:
    return Namespace(
        model=str(model),
        served_model_name=None,
        trust_remote_code=False,
        tokenizer_mode="auto",
        kv_cache_dtype="auto",
        block_size=None,
        max_model_len=None,
        gpu_memory_utilization=0.92,
        kv_cache_memory_bytes=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        cudagraph_capture_sizes=None,
        compilation_config=None,
        scheduler_reserve_full_isl=True,
        generation_config="auto",
        reasoning_parser="",
        default_chat_template_kwargs=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
    )


def test_plain_vllm_namespace_receives_export_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _pack(tmp_path / "model")
    args = _stock_args(model)
    for name in RUNTIME_ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    result = apply_runtime_profile(args)

    assert result is not None
    assert args.served_model_name == ["banana-smasher-v5"]
    for name, value in ENGINE_DEFAULTS.items():
        assert getattr(args, name) == value
    for name, value in FRONTEND_DEFAULTS.items():
        assert getattr(args, name) == value
    assert args._banana_smasher_runtime_profile == "sm121-single-gpu-v1"
    assert {name: result["applied"][name] for name in ENGINE_DEFAULTS} == ENGINE_DEFAULTS
    for name, value in RUNTIME_ENV_DEFAULTS.items():
        assert __import__("os").environ[name] == value
    assert args.compilation_config == {"cudagraph_mode": "PIECEWISE"}
    assert __import__("os").environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
    # vLLM 0.24.0's shared-expert auxiliary stream is not capture-safe on
    # SM121: capture otherwise fails with cudaErrorStreamCaptureIsolation.
    assert __import__("os").environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"


def test_explicit_vllm_arguments_remain_authoritative(tmp_path: Path) -> None:
    args = _stock_args(_pack(tmp_path / "model"))
    args.max_model_len = 4096
    args.gpu_memory_utilization = 0.7
    args.served_model_name = ["custom-name"]
    args.enable_auto_tool_choice = True
    args.tool_call_parser = "custom-parser"
    args.compilation_config = {"cudagraph_mode": "FULL"}

    result = apply_runtime_profile(args)

    assert result is not None
    assert args.max_model_len == 4096
    assert args.gpu_memory_utilization == 0.7
    assert args.served_model_name == ["custom-name"]
    assert args.tool_call_parser == "custom-parser"
    assert args.compilation_config == {"cudagraph_mode": "FULL"}


def test_positional_vllm_model_takes_precedence(tmp_path: Path) -> None:
    model = _pack(tmp_path / "model")
    args = _stock_args(model)
    args.model = "Qwen/Qwen3-0.6B"
    args.model_tag = str(model)

    result = apply_runtime_profile(args)

    assert result is not None
    assert args.model == str(model)
    assert args.max_model_len == 8192


def test_legacy_export_without_profile_uses_plugin_defaults(tmp_path: Path) -> None:
    model = _pack(tmp_path / "model", profiled=False)
    profile = load_runtime_profile(model)

    assert profile is not None
    assert profile["served_model_name"] == "DeepSeek-V4-Flash-BQ3"
    assert profile["engine_args"] == ENGINE_DEFAULTS
    assert profile["frontend_args"] == FRONTEND_DEFAULTS


def test_unrelated_model_is_untouched(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}})
    )
    (model / "BANANA_PACK_MANIFEST.json").write_text(
        json.dumps({"schema": "bs-pack", "quant_method": "fp8"})
    )
    args = _stock_args(model)

    assert apply_runtime_profile(args) is None
    assert args.tokenizer_mode == "auto"
    assert args.max_model_len is None


def test_unknown_export_profile_fails_loud(tmp_path: Path) -> None:
    model = _pack(tmp_path / "model")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["banana_smasher_runtime"]["profile"] = "unknown"
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match="unsupported Banana Smasher runtime profile"):
        load_runtime_profile(model)


def test_engine_args_hook_applies_profile_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeEngineArgs:
        calls = 0

        @classmethod
        def from_cli_args(cls, args: Namespace):
            cls.calls += 1
            return {"model": args.model, "max_model_len": args.max_model_len}

    vllm = ModuleType("vllm")
    engine = ModuleType("vllm.engine")
    arg_utils = ModuleType("vllm.engine.arg_utils")
    setattr(arg_utils, "EngineArgs", FakeEngineArgs)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)

    assert install_vllm_arg_defaults() is True
    assert install_vllm_arg_defaults() is False
    args = _stock_args(_pack(tmp_path / "model"))
    result = FakeEngineArgs.from_cli_args(args)

    assert result["max_model_len"] == 8192
    assert FakeEngineArgs.calls == 1
