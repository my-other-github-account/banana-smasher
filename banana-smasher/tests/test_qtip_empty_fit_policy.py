from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from banana_smasher.solver_qtip_profile import _prepare_fit_windows


def test_explicit_empty_fit_uses_only_original_capture_rows():
    x = torch.tensor([[1., 2.], [3., 4.]])
    captures = [dict(window=7, x=x)]
    empty = [dict(window=7, x=x[:0], weight=torch.empty(0))]
    runner = SimpleNamespace(expert_windows=lambda bank, expert: empty)
    windows, source = _prepare_fit_windows(
        runner, captures, model_root=Path('.'), layer=0, expert=1,
        projection='fused13', device=torch.device('cpu'),
        empty_fit_policy='clean-capture-unit-weight-v1')
    assert windows[0]['x'] is x
    assert torch.equal(windows[0]['weight'], torch.ones(2))
    assert source['empty_fit_policy'] == 'clean-capture-unit-weight-v1'
    assert source['original_routed_rows'] == 0
    assert source['original_routed_mass'] == 0
    assert source['fallback_rows'] == 2
    assert source['counterfactual_fit'] is True


def test_down_fallback_uses_original_expert_and_preserves_policy(monkeypatch):
    import banana_smasher.solver_qtip_profile as solver
    x = torch.tensor([[1., 2.]])
    captures = [dict(window=3, x=x)]
    empty = [dict(window=3, x=x[:0], weight=torch.empty(0))]
    source_weight = torch.tensor([[2., 0.], [0., 3.]])
    def load(root, layer, expert, projection):
        assert (layer, expert, projection) == (0, 1, 'fused13')
        return source_weight, {'source': 'test-native'}
    monkeypatch.setattr(solver, '_load_weight', load)
    def down(windows, weight, device):
        assert weight is source_weight
        assert windows[0]['x'] is x
        return [dict(windows[0], x=x @ weight)]
    runner = SimpleNamespace(expert_windows=lambda bank, expert: empty, down_windows=down)
    windows, source = solver._prepare_fit_windows(
        runner, captures, model_root=Path('.'), layer=0, expert=1,
        projection='down', device=torch.device('cpu'),
        empty_fit_policy='clean-capture-unit-weight-v1')
    assert torch.equal(windows[0]['x'], torch.tensor([[2., 6.]]))
    assert source['counterfactual_fit'] is True
    assert source['mode'] == 'source-fused13'


def test_default_and_nonempty_fit_are_unchanged():
    x = torch.tensor([[1., 2.]])
    captures = [dict(window=0, x=x)]
    for rows in (0, 1):
        original = [dict(window=0, x=x[:rows], weight=torch.ones(rows))]
        runner = SimpleNamespace(expert_windows=lambda bank, expert: original)
        policies = ('refuse',) if rows == 0 else ('refuse', 'clean-capture-unit-weight-v1')
        for policy in policies:
            windows, source = _prepare_fit_windows(
                runner, captures, model_root=Path('.'), layer=0, expert=1,
                projection='fused13', device=torch.device('cpu'), empty_fit_policy=policy)
            assert windows is original
            assert source == {'mode': 'routed-source-activation'}


def test_real_runner_hessian_is_same_bank_unit_weight_covariance():
    from banana_smasher import qtip_runner as runner
    x = torch.tensor([[1., 2.], [3., 4.]])
    captures = [dict(window=0, x=x, topk=torch.zeros((2, 1), dtype=torch.long),
                     route=torch.ones((2, 1)))]
    windows, source = _prepare_fit_windows(
        runner, captures, model_root=Path('.'), layer=0, expert=1,
        projection='fused13', device=torch.device('cpu'),
        empty_fit_policy='clean-capture-unit-weight-v1')
    hessian, rows, mass = runner.build_hessian(windows, torch.ones(2), torch.device('cpu'))
    z = runner.fwht(x)
    assert torch.allclose(hessian, z.T @ z / len(x))
    assert rows == mass == 2
    assert source['counterfactual_fit']
    refused, _ = _prepare_fit_windows(
        runner, captures, model_root=Path('.'), layer=0, expert=1,
        projection='fused13', device=torch.device('cpu'))
    with pytest.raises(RuntimeError, match='empty routed fit population'):
        runner.build_hessian(refused, torch.ones(2), torch.device('cpu'))


@pytest.mark.parametrize('policy', ['unknown', ''])
def test_unknown_policy_refuses(policy):
    with pytest.raises(ValueError, match='unknown empty fit policy'):
        _prepare_fit_windows(None, [], model_root=Path('.'), layer=0, expert=1,
                             projection='fused13', device=torch.device('cpu'),
                             empty_fit_policy=policy)


def test_empty_clean_bank_refuses():
    runner = SimpleNamespace(expert_windows=lambda bank, expert: [])
    with pytest.raises(ValueError, match='empty clean capture bank'):
        _prepare_fit_windows(runner, [], model_root=Path('.'), layer=0, expert=1,
                             projection='fused13', device=torch.device('cpu'),
                             empty_fit_policy='clean-capture-unit-weight-v1')


def test_production_call_binds_policy_from_hashed_config():
    import ast
    import inspect
    import banana_smasher.solver_qtip_profile as solver
    tree = ast.parse(inspect.getsource(solver))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_prepare_fit_windows']
    assert len(calls) == 1
    policy = next((k.value for k in calls[0].keywords if k.arg == 'empty_fit_policy'), None)
    assert policy is not None, 'production call does not bind explicit policy'
    expr = compile(ast.Expression(body=policy), '<policy-binding>', 'eval')
    assert eval(expr, {'config': {}}) == 'refuse'
    assert eval(expr, {'config': {'empty_fit_policy': 'clean-capture-unit-weight-v1'}}) == 'clean-capture-unit-weight-v1'
