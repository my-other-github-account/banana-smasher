from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import banana_smasher_plugin


def test_flashinfer_comm_binds_real_cudart_before_later_tilelang_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = ModuleType("flashinfer.comm")
    cuda_ipc = ModuleType("flashinfer.comm.cuda_ipc")
    setattr(
        cuda_ipc,
        "cudart",
        SimpleNamespace(lib=SimpleNamespace(_name="/usr/local/cuda/lib64/libcudart.so.13")),
    )
    flashinfer = ModuleType("flashinfer")
    setattr(flashinfer, "comm", comm)
    setattr(comm, "cuda_ipc", cuda_ipc)
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.comm", comm)
    monkeypatch.setitem(sys.modules, "flashinfer.comm.cuda_ipc", cuda_ipc)

    bound = getattr(
        banana_smasher_plugin, "configure_flashinfer_cuda_runtime_binding"
    )()

    assert bound == "/usr/local/cuda/lib64/libcudart.so.13"


def test_flashinfer_comm_rejects_tilelang_cudart_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = ModuleType("flashinfer.comm")
    cuda_ipc = ModuleType("flashinfer.comm.cuda_ipc")
    setattr(
        cuda_ipc,
        "cudart",
        SimpleNamespace(
            lib=SimpleNamespace(
                _name="/usr/local/lib/python3.12/dist-packages/tilelang/lib/libcudart_stub.so"
            )
        ),
    )
    flashinfer = ModuleType("flashinfer")
    setattr(flashinfer, "comm", comm)
    setattr(comm, "cuda_ipc", cuda_ipc)
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.comm", comm)
    monkeypatch.setitem(sys.modules, "flashinfer.comm.cuda_ipc", cuda_ipc)

    with pytest.raises(RuntimeError, match="libcudart_stub"):
        getattr(
            banana_smasher_plugin, "configure_flashinfer_cuda_runtime_binding"
        )()
