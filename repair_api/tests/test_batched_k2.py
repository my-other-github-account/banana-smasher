from __future__ import annotations

import torch

from banana_smasher import qtip_k2
from repair_api.batched_k2 import decode_k2_matrix_batched, inverse_transform_batched


def test_batched_decode_and_inverse_match_official_memberwise_bytes() -> None:
    generator = torch.Generator().manual_seed(1701)
    packed = torch.randint(
        -(1 << 15),
        1 << 15,
        (3, 8, 8, 32),
        dtype=torch.int16,
        generator=generator,
    )
    lut = torch.randn(1024, dtype=torch.float16, generator=generator)
    su = torch.randn(3, 128, dtype=torch.float16, generator=generator)
    sv = torch.randn(3, 128, dtype=torch.float16, generator=generator)

    decoded = decode_k2_matrix_batched(packed, lut)
    physical = inverse_transform_batched(decoded, su.float(), sv.float())

    expected_decoded = torch.stack(
        [qtip_k2.decode_k2_matrix(member, lut) for member in packed]
    )
    expected_physical = torch.stack(
        [
            qtip_k2.inverse_transform(member, member_su.float(), member_sv.float())
            for member, member_su, member_sv in zip(expected_decoded, su, sv)
        ]
    )

    assert torch.equal(decoded, expected_decoded)
    assert torch.equal(
        physical.transpose(-2, -1).contiguous().to(torch.bfloat16),
        expected_physical.transpose(-2, -1).contiguous().to(torch.bfloat16),
    )
