"""Unsolved-prefix LDLQ product update; nonzero residual regression."""
import types
import torch
from banana_smasher.qtip_batch import ldlq_batch


def test_unsolved_prefix_preserves_nonzero_residual_quantization(monkeypatch):
    gen = torch.Generator().manual_seed(8445)
    weights = torch.randn((2, 32, 384), generator=gen)
    lower = torch.tril(torch.randn((2, 384, 384), generator=gen) * 0.025, diagonal=-1)
    args = types.SimpleNamespace(td_x=16, td_y=16, V=2)

    class RoundingCodebook:
        idx_dtype = torch.int32
        def quantize(self, values):
            rounded = values.round()
            return rounded, rounded[:, ::2].to(torch.int32)

    baseline = ldlq_batch(weights, lower, RoundingCodebook(), args)
    calls = []
    original = torch.bmm
    def observed(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return original(a, b)
    monkeypatch.setattr(torch, 'bmm', observed)
    candidate = ldlq_batch(weights, lower, RoundingCodebook(), args, update_unsolved_only=True)
    assert torch.equal(candidate[0], baseline[0])
    assert torch.equal(candidate[1], baseline[1])
    # Only the two still-unsolved prefixes need outer-buffer updates.
    outer = [a[1] for a, b in calls if a[2] == 128]
    assert outer == [256, 128]


def test_unsolved_update_config_is_explicit_and_batch_uniform():
    import pytest
    from banana_smasher.qtip_batch_controller import _ldlq_update_unsolved_only
    assert _ldlq_update_unsolved_only([{}]) is False
    assert _ldlq_update_unsolved_only([{'ldlq_update_unsolved_only': True}] * 2) is True
    for configs in ([{'ldlq_update_unsolved_only': 1}],
                    [{}, {'ldlq_update_unsolved_only': True}]):
        with pytest.raises(ValueError):
            _ldlq_update_unsolved_only(configs)
