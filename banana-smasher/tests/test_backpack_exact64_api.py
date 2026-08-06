from __future__ import annotations

import inspect
from types import SimpleNamespace

from banana_smasher import run_backpack_exact64
from banana_smasher.cli import _parser
from banana_smasher.hf_deepseek_v4_backpack_adapter import (
    _available_materialization_bytes,
)


def test_public_exact64_api_uses_single_host_full_layer_path() -> None:
    assert callable(run_backpack_exact64)

    parser = _parser()
    commands = next(
        action.choices
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    assert "backpack-exact64" in commands

    source = inspect.getsource(run_backpack_exact64)
    assert "with runtime.layer_stage(layer) as forward:" in source
    assert "mlp_chunk_stage" not in source
    assert "teacher_manifest_path, teacher = _revision_bind_teacher_manifest(" in source


def test_gb10_materialization_uses_reclaimable_unified_memory(tmp_path) -> None:
    gib = 1 << 30

    class FakeCuda:
        @staticmethod
        def mem_get_info():
            return 3 * gib, 120 * gib

        @staticmethod
        def get_device_properties(device):
            assert device == "cuda"
            return SimpleNamespace(name="NVIDIA GB10")

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   118000000 kB\n")

    assert _available_materialization_bytes(
        SimpleNamespace(cuda=FakeCuda), "cuda", meminfo_path=meminfo
    ) == 118000000 * 1024
