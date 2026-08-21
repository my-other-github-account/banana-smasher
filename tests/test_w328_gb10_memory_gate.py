from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools/w328_recovery/reconstruct_w328_exact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("w328_reconstruct_memory_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gb10_uma_uses_larger_host_available_witness() -> None:
    module = load_module()
    assert module.effective_free_bytes(
        cuda_free=71,
        host_available=96,
        device_name="NVIDIA GB10",
    ) == 96


def test_non_gb10_never_substitutes_host_available() -> None:
    module = load_module()
    assert module.effective_free_bytes(
        cuda_free=71,
        host_available=96,
        device_name="NVIDIA H100 80GB HBM3",
    ) == 71
