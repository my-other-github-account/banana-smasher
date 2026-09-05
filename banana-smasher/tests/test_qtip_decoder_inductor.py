"""Actual default-backend regression for packed byte/word layout lowering."""
import hashlib

import pytest

torch = pytest.importorskip("torch")
from banana_smasher.qtip_kernel_decompress import decode_compressed


@pytest.mark.parametrize("rate", [1, 2, 3, 4])
def test_default_inductor_matches_packed_wire(rate):
    # Each codec rate is an independent production specialization. Reset only
    # Dynamo's test cache so pytest ordering does not trigger auto-dynamic R.
    torch._dynamo.reset()
    torch.set_num_threads(1)
    words = ((torch.arange(64 * rate, dtype=torch.int32) * 997 + 12345) % 65536).to(torch.uint16)
    lut = torch.arange(65536 * 2, dtype=torch.float32).reshape(65536, 2) / 1024
    args = (16, 9, rate, 1, 32, 32, words, lut)
    expected = decode_compressed.__wrapped__(*args)
    actual = decode_compressed(*args)  # production decorator: default Inductor
    assert actual.shape == (32, 32)
    assert torch.equal(actual, expected)
    # Frozen from the eager wire decoder at 56cf5aa, before the layout fix.
    # This detects shared eager/compiled semantic regressions (including signs).
    baseline = {
        1: "898c87ff584a3b766fc330469f9d0905a916b0e2483dc1250ee6ea30e8799a67",
        2: "772b0606df6894ac9d0438432fb186cb8b1f2b57fe039b0fde508fb080ff412c",
        3: "9837b108b72a2cdf1c64f4bf09b99d02c80418c878ed8662dfad8db0854037af",
        4: "7797c8939405be275a66a0c4347e05ed73e18fb50bd5e582cf40a1fa36b2c5e0",
    }
    assert hashlib.sha256(actual.numpy().tobytes()).hexdigest() == baseline[rate]
