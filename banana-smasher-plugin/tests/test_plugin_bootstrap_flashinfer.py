from __future__ import annotations

import inspect

import banana_smasher_plugin
import pytest


def test_flashinfer_runtime_compat_is_installed_before_flashinfer_sparse_import() -> None:
    assert hasattr(banana_smasher_plugin, "configure_flashinfer_cuda_runtime")
    source = inspect.getsource(banana_smasher_plugin)
    assert source.index("configure_flashinfer_cuda_runtime()") < source.index(
        "import torch  # noqa: E402"
    )
    assert source.index("configure_flashinfer_cuda_runtime()") < source.index(
        "def configure_flashinfer_sparse_mla_signature_compat"
    )


def test_supported_vllm_version_is_checked_at_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        banana_smasher_plugin.importlib.metadata,
        "version",
        lambda name: "0.24.0" if name == "vllm" else "unexpected",
    )
    assert banana_smasher_plugin._require_supported_vllm_version() == "0.24.0"


def test_unsupported_vllm_version_fails_loudly(monkeypatch) -> None:
    monkeypatch.setattr(
        banana_smasher_plugin.importlib.metadata,
        "version",
        lambda name: "0.25.0" if name == "vllm" else "unexpected",
    )
    with pytest.raises(RuntimeError, match="requires stock vLLM 0.24.0"):
        banana_smasher_plugin._require_supported_vllm_version()
