"""Unit-axis LDL specialization preserves canonical singleton factorization."""
import torch
from banana_smasher.qtip_batch import block_ldl_batch


def test_unitwise_ldl_matches_singleton_factorization_and_preserves_input():
    generator = torch.Generator().manual_seed(8290)
    a = torch.randn((3, 64, 64), generator=generator)
    h = a @ a.transpose(-1, -2) + torch.eye(64)
    before = h.clone()
    expected = torch.cat([block_ldl_batch(x.unsqueeze(0), 16) for x in h])
    actual = block_ldl_batch(h, 16, unitwise=True)
    assert torch.equal(actual, expected)
    assert torch.equal(h, before)


def test_unitwise_public_config_rejects_mixed_or_nonboolean_values():
    import pytest
    from banana_smasher.qtip_batch_controller import _block_ldl_unitwise
    assert _block_ldl_unitwise([{}, {}]) is False
    assert _block_ldl_unitwise([{'block_ldl_unitwise': True}] * 2) is True
    for configs in ([{}, {'block_ldl_unitwise': True}], [{'block_ldl_unitwise': 1}], [{'block_ldl_unitwise': 'false'}]):
        with pytest.raises(ValueError):
            _block_ldl_unitwise(configs)


def test_public_batch_routes_unitwise_option_and_receipts():
    import ast
    import inspect
    import banana_smasher.qtip_batch as batch
    import banana_smasher.qtip_batch_controller as controller
    tree = ast.parse(inspect.getsource(controller.main_batch))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_qtip_batch')
    assert any(k.arg == 'block_ldl_unitwise' for k in call.keywords)
    build = ast.parse(inspect.getsource(batch.build_qtip_batch))
    call = next(n for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'block_ldl_batch')
    assert any(k.arg == 'unitwise' and isinstance(k.value, ast.Name) and k.value.id == 'block_ldl_unitwise' for k in call.keywords)
    assert '"block_ldl_unitwise": block_ldl_unitwise' in inspect.getsource(batch.build_qtip_batch)
