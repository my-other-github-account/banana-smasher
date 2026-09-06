"""CPU fixtures for the sealed producer-to-model installation boundary."""
import hashlib
import importlib
from types import SimpleNamespace

import pytest
import torch


def test_installs_both_projections_and_preserves_native_rest(tmp_path):
    api = importlib.import_module('banana_smasher.calibrated_layer_install')
    experts = SimpleNamespace(
        gate_up_proj=torch.zeros(2, 4, 3, dtype=torch.bfloat16),
        down_proj=torch.zeros(2, 3, 2, dtype=torch.bfloat16),
    )
    native = torch.tensor([7.0])
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=experts), native=native)
    members = []
    for expert in range(2):
        for projection, shape in [('fused13', (4, 3)), ('down', (3, 2))]:
            path = tmp_path / f'{expert}_{projection}.pt'
            torch.save({'weight': torch.full(shape, expert + 1.25),
                        'geometry': {'K': 2}, 'shape': list(shape)}, path)
            members.append({'cell': f'L003/E{expert:03d}_{projection}',
                            'bindings': {'artifact': {'path': str(path),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'bytes': path.stat().st_size}}})
    result = api.install_calibrated_layer(
        layer, layer_id=3, members=members, expected_experts=2, tier=2,
        decoder=lambda payload, device: payload['weight'].to(device),
    )
    assert result['installed_count'] == 4
    assert torch.equal(experts.gate_up_proj[1], torch.full((4, 3), 2.25, dtype=torch.bfloat16))
    assert torch.equal(experts.down_proj[0], torch.full((3, 2), 1.25, dtype=torch.bfloat16))
    assert layer.native is native and native.item() == 7
    for bad in (members[:-1], members + members[:1]):
        with pytest.raises(ValueError, match='inventory'):
            api.install_calibrated_layer(layer, layer_id=3, members=bad,
                                        expected_experts=2, tier=2, decoder=None)
    members[0]['bindings']['artifact']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='artifact bytes'):
        api.install_calibrated_layer(layer, layer_id=3, members=members,
                                    expected_experts=2, tier=2, decoder=None)


def test_installer_rejects_non_uniform_tier_before_reading_members():
    api = importlib.import_module('banana_smasher.calibrated_layer_install')
    for tier in (True, 0, 2.5, 5):
        with pytest.raises(ValueError, match='uniform tier'):
            api.install_calibrated_layer(None, layer_id=3, members=[],
                                        expected_experts=288, tier=tier, decoder=None)
