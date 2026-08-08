from __future__ import annotations

import inspect

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
