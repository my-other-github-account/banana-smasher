from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import banana_smasher_plugin


def test_flashinfer_runtime_compat_is_installed_before_flashinfer_sparse_import() -> None:
    assert hasattr(banana_smasher_plugin, "configure_flashinfer_cuda_runtime")
    source = inspect.getsource(banana_smasher_plugin)
    assert source.index("configure_flashinfer_cuda_runtime()") < source.index(
        "import torch  # noqa: E402"
    )
    assert source.index("configure_flashinfer_cuda_runtime()") < source.index(
        "def configure_flashinfer_sparse_mla_signature_compat"
    )


def test_standalone_qtip_can_skip_absent_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banana_smasher_plugin, "_REAL_CUDART_PATH", Path(__file__))
    monkeypatch.setattr(banana_smasher_plugin, "_REAL_CUDA_RUNTIME", None)
    monkeypatch.setattr(banana_smasher_plugin.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    def absent(name: str):
        if name == "flashinfer.comm.cuda_ipc":
            error = ModuleNotFoundError("No module named 'flashinfer'")
            error.name = "flashinfer"
            raise error
        raise AssertionError(name)

    monkeypatch.setattr(banana_smasher_plugin.importlib, "import_module", absent)
    assert banana_smasher_plugin.configure_flashinfer_cuda_runtime(required=False) is False
