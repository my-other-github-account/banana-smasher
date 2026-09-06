"""Installed byte and native-rest validation, using small CPU fixtures."""
import importlib

import pytest
import torch


def test_installed_canary_rejects_native_rest_mutation():
    api = importlib.import_module('banana_smasher.calibrated_layer_install')
    canary = getattr(api, 'checked_install_calibrated_layer', None)
    assert callable(canary), 'missing native-rest physical checker'
    layer = torch.nn.Module()
    layer.register_parameter('native', torch.nn.Parameter(torch.tensor([7.])))
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.register_parameter('gate_up_proj', torch.nn.Parameter(torch.zeros(1, 2, 2)))
    layer.mlp.experts.register_parameter('down_proj', torch.nn.Parameter(torch.zeros(1, 2, 1)))
    def installer(target, **kwargs):
        with torch.no_grad():
            target.mlp.experts.gate_up_proj.fill_(2)
        return {'installed_count': 2}
    result = canary(layer, installer=installer, layer_id=3)
    assert result['native_rest_unchanged'] is True
    assert result['native_rest_tensor_count'] == 1
    assert set(result['installed_tensor_sha256']) == {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}
    def bad(target, **kwargs):
        with torch.no_grad():
            target.native.add_(1)
        return {}
    with pytest.raises(ValueError, match='native rest changed'):
        canary(layer, installer=bad, layer_id=3)
