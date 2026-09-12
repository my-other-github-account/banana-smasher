"""Generic HF FP8 decoding remains usable with base dependencies only."""
import builtins
from pathlib import Path

import numpy as np

from banana_smasher import hf_moe


def test_generic_fp8_loader_does_not_require_torch(monkeypatch):
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("test blocks optional Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    row = dict(name="weight", dtype="F8_E4M3", shape=[2, 2])
    scale = dict(name="scale", dtype="F32", shape=[1, 1])

    def read(source, metadata):
        return bytes([56, 64, 72, 80]) if metadata["name"] == "weight" else np.array([2], dtype="<f4").tobytes()

    actual = hf_moe._load_safetensors_matrix(
        Path("weight"), row, scale_source=Path("scale"), scale_row=scale,
        tensor_payload_reader=read,
    )
    np.testing.assert_array_equal(actual, [[2, 4], [8, 16]])
