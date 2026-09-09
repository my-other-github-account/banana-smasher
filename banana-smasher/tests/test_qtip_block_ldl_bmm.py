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


def test_public_column_bmm_is_strict_opt_in_and_reaches_builder():
    import ast
    from pathlib import Path
    from test_qtip_reference_ldl import load
    common = load('_common', 'qtip_batch_controller.py')
    mode = load('_block_ldl_column_bmm', 'qtip_batch_controller.py', {'_common': common})
    assert mode([{}, {}]) is False
    assert mode([{'block_ldl_column_bmm': True, 'block_ldl_reference': True}]) is True
    for configs in [[{'block_ldl_column_bmm': 1}], [{'block_ldl_column_bmm': 'true'}], [{'block_ldl_column_bmm': True}], [{'block_ldl_column_bmm': True, 'block_ldl_reference': True}, {}]]:
        with pytest.raises(ValueError):
            mode(configs)
    src = Path(qb.__file__).parent
    tree = ast.parse((src / 'qtip_batch_controller.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_qtip_batch']
    assert any(k.arg == 'block_ldl_column_bmm' for k in calls[0].keywords)
    assert '\"block_ldl_column_bmm\": block_ldl_column_bmm' in (src / 'qtip_batch.py').read_text()


def test_column_bmm_rejects_invalid_geometry():
    assert hasattr(qb, '_column_bmm_block_ldl')
    for h, block in [(torch.eye(7), 4), (torch.ones(2, 3), 1), (torch.eye(4), 0)]:
        with pytest.raises(ValueError):
            qb._column_bmm_block_ldl(h, block)
    with pytest.raises(RuntimeError):
        qb._column_bmm_block_ldl(-torch.eye(4), 2)
