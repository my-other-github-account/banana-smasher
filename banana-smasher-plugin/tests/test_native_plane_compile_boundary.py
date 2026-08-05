from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch

import banana_smasher_plugin.native_planes as native_planes
from test_native_plane_runtime import _tiny_pack


def test_native_plane_forward_registers_breakable_cudagraph_eager_boundary(
    tmp_path: Path, monkeypatch,
) -> None:
    registrations: list[
        tuple[str, object, object, list[str] | None, tuple[torch.Tag, ...]]
    ] = []
    calls: list[tuple[int, int]] = []
    dispatch_outputs: list[torch.Tensor] = []
    eager_decorations: list[object] = []

    torch_utils = ModuleType("vllm.utils.torch_utils")
    breakable = ModuleType("vllm.compilation.breakable_cudagraph")
    envs = ModuleType("vllm.envs")
    config = ModuleType("vllm.config")
    setattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", True)
    setattr(config, "CUDAGraphMode", SimpleNamespace(PIECEWISE="PIECEWISE"))
    setattr(
        config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode="PIECEWISE")
        ),
    )

    def eager_break_during_capture(fn):
        eager_decorations.append(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    def direct_register_custom_op(
        name, impl, *, mutates_args=None, fake_impl, tags=(),
    ):
        registrations.append((name, impl, fake_impl, mutates_args, tags))

        if mutates_args:

            def invoke_mutating(x, expert_ids, output, layer_key, projection_key):
                calls.append((layer_key, projection_key))
                return impl(x, expert_ids, output, layer_key, projection_key)

            invoke = invoke_mutating
        else:

            def invoke_functional(x, expert_ids, layer_key, projection_key):
                calls.append((layer_key, projection_key))
                return impl(x, expert_ids, layer_key, projection_key)

            invoke = invoke_functional
        monkeypatch.setattr(torch.ops.vllm, name, invoke, raising=False)

    torch_utils.direct_register_custom_op = direct_register_custom_op
    setattr(breakable, "eager_break_during_capture", eager_break_during_capture)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm.utils", ModuleType("vllm.utils"))
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setitem(sys.modules, "vllm.compilation", ModuleType("vllm.compilation"))
    monkeypatch.setitem(
        sys.modules, "vllm.compilation.breakable_cudagraph", breakable
    )
    monkeypatch.setattr(
        native_planes, "_NATIVE_PLANE_CUSTOM_OP_REGISTERED", False, raising=False
    )
    monkeypatch.setattr(
        native_planes, "_NATIVE_PLANE_CUSTOM_OP_AVAILABLE", False, raising=False
    )
    monkeypatch.setattr(native_planes, "_NATIVE_PLANE_LAYER_REGISTRY", {}, raising=False)
    monkeypatch.setattr(native_planes, "_NATIVE_PLANE_NEXT_KEY", 1, raising=False)

    pack = native_planes.NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))

    def dispatch(*, projection, x, expert_ids, state, output=None):
        del projection, expert_ids
        assert output is not None
        assert tuple(output.shape) == (x.shape[0], state.output_width)
        dispatch_outputs.append(output)
        output.fill_(7)
        return output

    layer = native_planes.NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    result = layer.forward(
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.tensor([0, 1]),
        "fused13",
    )

    assert registrations and registrations[0][0] == "banana_smasher_native_plane_forward"
    assert registrations[0][3] == ["output"], (
        "breakable cudagraph replay requires the custom op to mutate a "
        "caller-owned stable output buffer"
    )
    assert torch.Tag.cudagraph_unsafe in registrations[0][4]
    assert eager_decorations == [native_planes._native_plane_forward_op]
    assert calls == [(layer._custom_op_key, 0)]
    assert len(dispatch_outputs) == 1
    assert dispatch_outputs[0] is result
    assert result.shape == (2, 4)
    assert torch.all(result == 7)

    second = native_planes.NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    second.forward(torch.ones((1, 2)), torch.tensor([1]), "down")
    assert len(registrations) == 1
    assert len(eager_decorations) == 1
    assert calls[-1] == (second._custom_op_key, 1)
    assert second._custom_op_key != layer._custom_op_key


@pytest.mark.parametrize(
    ("breakable_enabled", "mode", "has_config", "message"),
    [
        (False, "PIECEWISE", True, "VLLM_USE_BREAKABLE_CUDAGRAPH=1"),
        (True, "FULL_AND_PIECEWISE", True, "cudagraph_mode=PIECEWISE"),
        (True, "PIECEWISE", False, "active vLLM config"),
    ],
)
def test_native_plane_cudagraph_boundary_fails_closed(
    monkeypatch, breakable_enabled, mode, has_config, message,
) -> None:
    envs = ModuleType("vllm.envs")
    config = ModuleType("vllm.config")
    setattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", breakable_enabled)
    setattr(config, "CUDAGraphMode", SimpleNamespace(PIECEWISE="PIECEWISE"))
    current = (
        SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))
        if has_config
        else None
    )
    setattr(config, "get_current_vllm_config_or_none", lambda: current)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    with pytest.raises(RuntimeError, match=message):
        native_planes._require_native_plane_breakable_cudagraph()


def test_cuda_native_plane_rejects_missing_breakable_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(native_planes, "_ensure_native_plane_custom_op", lambda: False)
    layer = cast(
        native_planes.NativePlaneLayer,
        SimpleNamespace(device=torch.device("cuda")),
    )

    with pytest.raises(RuntimeError, match="direct Python dispatch is not an accepted"):
        native_planes._register_native_plane_layer(layer)
