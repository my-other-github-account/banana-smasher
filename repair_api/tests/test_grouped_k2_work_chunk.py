import random
import re
from pathlib import Path


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
