import random
import re
import sys
from types import ModuleType
from pathlib import Path

import torch

from repair_api.official_k2_resident_score import _canonical_causal_score_tokens
from repair_api.assets.static_w28_fast_k2_grouped import block_hadamard_128


ROOT = Path(__file__).parents[2]
PYTHON_SOURCE = ROOT / "repair_api" / "assets" / "fast_k2_grouped.py"
CUDA_SOURCE = ROOT / "repair_api" / "assets" / "fast_k2_grouped_kernel.cu"


def test_grouped_forward_stages_activations_once_at_proven_64_row_geometry():
    python = PYTHON_SOURCE.read_text()
    cuda = CUDA_SOURCE.read_text()

    assert "WORK_ROWS = 64" in python
    assert "counts + WORK_ROWS - 1" in python
    assert "rounding_mode=\"floor\"" in python
    assert "work_local * WORK_ROWS" in python

    assert "constexpr int kRowsPerWork = 64;" in cuda
    assert "constexpr int kRowsPerThread = kRowsPerWork / kTile;" in cuda
    assert "__shared__ float x_tile[kRowsPerWork * kTile];" in cuda
    assert "x_tile[linear] = row < end && row < rows" in cuda
    assert "x_tile[local_row * kTile + k]" in cuda
    assert "nvcuda::wmma::mma_sync" not in cuda
    assert "__float_to_tf32" not in cuda
    assert "#include <c10/cuda/CUDAStream.h>" in cuda
    assert "#include <ATen/cuda/CUDAContext.h>" not in cuda


def test_static_w28_wrapper_matches_the_cuda_kernel_work_stride():
    static = (ROOT / "repair_api" / "assets" / "static_w28_fast_k2_grouped.py").read_text()

    assert "WORK_ROWS = 64" in static
    assert "counts + WORK_ROWS - 1" in static
    assert "work_local * WORK_ROWS" in static
    assert "counts + 15" not in static
    assert "work_local * 16" not in static


def test_packed_expert_bounds_sealed_group2_work_to_two_cuda_streams():
    expert = (ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py").read_text()

    assert "sealed_group_tokens = 2 * 2048" in expert
    assert "group_count = hidden_states.shape[0] // sealed_group_tokens" in expert
    assert "stream_count = min(2, group_count)" in expert
    assert "torch.cuda.Stream(device=hidden_states.device)" in expert
    assert "stream = streams[group % stream_count]" in expert
    assert "outputs.append(self.forward(" in expert
    assert "return torch.cat(outputs, dim=0)" in expert


def test_grouped_parent_decode_uses_exact_two_word_rolling_window():
    cuda = CUDA_SOURCE.read_text()
    assert "const int code_group = position >> 3;" in cuda
    assert "tile_words[code_group ^ 1]" in cuda
    assert "tile_words[((code_group + 31) & 31) ^ 1]" in cuda
    assert "const int shift = 2 * (7 - (position & 7));" in cuda
    assert "for (int step = 0; step < 8; ++step)" not in cuda

    inverse_text = re.search(
        r"kInversePermutation\[256\] = \{(.*?)\};", cuda, re.DOTALL
    )
    assert inverse_text is not None
    inverse = [int(value) for value in re.findall(r"\d+", inverse_text.group(1))]
    assert len(inverse) == 256

    rng = random.Random(75)
    for _ in range(32):
        words = [rng.randrange(1 << 16) for _ in range(32)]
        for logical_index in range(256):
            position = inverse[logical_index]
            old_state = 0
            for step in range(8):
                code_position = (position + 249 + step) & 255
                pair = code_position >> 4
                within = code_position & 15
                word_index = pair * 2 + (1 if within < 8 else 0)
                code_shift = 14 - 2 * (within & 7)
                old_state = (old_state << 2) | ((words[word_index] >> code_shift) & 3)

            code_group = position >> 3
            previous = words[((code_group + 31) & 31) ^ 1]
            current = words[code_group ^ 1]
            combined = (previous << 16) | current
            shift = 2 * (7 - (position & 7))
            direct_state = (combined >> shift) & 0xFFFF
            assert direct_state == old_state


def test_packed_hadamard_is_exact_for_batch1_and_batch8_at_fixed_window_geometry(
    monkeypatch,
):
    """Attempt24 full/short rows must share one padded per-window route geometry."""
    full = _canonical_causal_score_tokens(list(range(2048)), real_len=2048, pad_token_id=1)
    short = _canonical_causal_score_tokens(list(range(1737)), real_len=1737, pad_token_id=1)
    assert len(full) == len(short) == 2048

    package = ModuleType("banana_smasher")
    qtip = ModuleType("banana_smasher.qtip_k2")
    qtip.normalized_hadamard_128 = lambda _device, dtype: torch.eye(128, dtype=dtype)
    monkeypatch.setitem(sys.modules, "banana_smasher", package)
    monkeypatch.setitem(sys.modules, "banana_smasher.qtip_k2", qtip)

    generator = torch.Generator().manual_seed(24)
    samples = [torch.randn((2, 128), generator=generator) for _ in range(8)]
    isolated = torch.cat([
        block_hadamard_128(sample, route_rows_per_sample=2) for sample in samples
    ])
    packed = block_hadamard_128(
        torch.cat(samples), route_rows_per_sample=2
    )
    assert torch.equal(packed, isolated)
