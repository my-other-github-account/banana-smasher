"""Wire-only serving decode must not require a fitted reference tensor."""
import types

import torch

from banana_smasher import qtip_kernel_decompress, qtip_runner


def test_wire_only_decode_matches_existing_conformance_path():
    torch.manual_seed(19)
    candidate = {
        'geometry': {'L': 16, 'K': 2, 'V': 2, 'tlut_bits': 9},
        'shape': [32, 32],
        'tlut': torch.randn(512, 2),
        'trellis': torch.randint(0, 65536, (128,)).to(torch.uint16),
        'Wscale': torch.tensor(0.25),
        'SU': torch.ones(32),
        'SV': torch.ones(32),
    }
    kernel = types.SimpleNamespace(
        decode_compressed=qtip_kernel_decompress.decode_compressed._torchdynamo_orig_callable,
        __file__=qtip_kernel_decompress.__file__,
    )
    decode = getattr(qtip_runner, 'decode_packed_weight', None)
    assert callable(decode), 'missing reference-free canonical wire decoder'
    decoded = decode(candidate, kernel, torch.device('cpu'))
    assert 'reconstructed_weight' not in candidate
    expected, receipt = qtip_runner.decode_packed(
        dict(candidate, reconstructed_weight=decoded.half()), kernel, torch.device('cpu')
    )
    assert torch.equal(expected, decoded)
    assert receipt['fp16_bit_exact']
