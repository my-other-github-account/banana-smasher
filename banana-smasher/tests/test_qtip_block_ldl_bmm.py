"""Column-batched reference normalization, not a trellis layout experiment."""
import pytest
import torch
from banana_smasher import qtip_batch as qb


def test_column_bmm_preserves_reference_normalization():
    assert hasattr(qb, '_column_bmm_block_ldl'), 'missing column-batched reference factorizer'
    torch.manual_seed(712)
    for width, block in [(32, 8), (64, 16)]:
        x = torch.randn(width, width, dtype=torch.float64)
        h = x @ x.T + torch.eye(width, dtype=x.dtype)
        original = h.clone()
        lower = torch.linalg.cholesky(h)
        n = width // block
        diagonal = torch.diagonal(lower.reshape(n, block, n, block), dim1=0, dim2=2).permute(2, 0, 1)
        inverse = torch.linalg.inv(diagonal)
        expected = lower.view(width, n, block)
        for i in range(n):
            expected[:, i, :] = expected[:, i, :] @ inverse[i]
        expected = expected.reshape(width, width)
        expected.view(n, block, n, block).permute(0, 2, 1, 3)[torch.arange(n), torch.arange(n)] = torch.eye(block, dtype=h.dtype)
        actual = qb._column_bmm_block_ldl(h, block)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert torch.equal(h, original)


def test_column_bmm_rejects_invalid_geometry():
    assert hasattr(qb, '_column_bmm_block_ldl')
    for h, block in [(torch.eye(7), 4), (torch.ones(2, 3), 1), (torch.eye(4), 0)]:
        with pytest.raises(ValueError):
            qb._column_bmm_block_ldl(h, block)
    with pytest.raises(RuntimeError):
        qb._column_bmm_block_ldl(-torch.eye(4), 2)
