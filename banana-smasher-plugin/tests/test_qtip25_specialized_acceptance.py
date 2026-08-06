from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from banana_smasher_plugin.native_planes import _load_accelerated_dispatch


def test_qtip25_public_runtime_uses_specialized_native_dispatch_not_p1016_fallback(
    monkeypatch,
) -> None:
    fallback = ModuleType("banana_smasher_plugin.p1016_kernels")

    def forbidden_fallback(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forbidden mixed_exact_gemv fallback executed")

    fallback.mixed_exact_gemv = forbidden_fallback
    monkeypatch.setitem(sys.modules, fallback.__name__, fallback)

    specialized = ModuleType("banana_smasher_plugin.v4_acceleration")
    calls: list[tuple[str, tuple[int, ...]]] = []

    def specialized_dispatch(x, expert_ids, families, pointer_tables, codebook, vq_state, *, projection):
        del families, pointer_tables, codebook, vq_state
        calls.append((projection, tuple(expert_ids.tolist())))
        return x

    specialized.mixed_exact_native_gemv = specialized_dispatch
    specialized.runtime_sentinel = lambda: {"activated": True, "blocked": []}
    monkeypatch.setitem(sys.modules, specialized.__name__, specialized)

    state = SimpleNamespace(
        families=torch.tensor([0, 1], dtype=torch.int8),
        pointer_tables={
            "su": torch.ones((2, 4), dtype=torch.float32),
            "sv": torch.ones((2, 4), dtype=torch.float32),
            "wscale": torch.ones(2, dtype=torch.float32),
        },
        offsets2=torch.zeros(1, dtype=torch.int64),
        offsets3=torch.zeros(1, dtype=torch.int64),
        lut=torch.zeros((65536, 2), dtype=torch.float32),
        qtip_codebook=torch.zeros(1024, dtype=torch.float16),
        vq_state={},
    )
    x = torch.ones((2, 4), dtype=torch.float32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int64)

    result = _load_accelerated_dispatch()(
        projection="fused13",
        x=x,
        expert_ids=expert_ids,
        state=state,
    )

    assert torch.equal(result, x)
    assert calls == [("fused13", (0, 1))]
