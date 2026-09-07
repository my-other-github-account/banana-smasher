"""Sealed producer wire is decoded before logical gate/up slicing."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from banana_smasher import qtip_kernel_decompress as kd, qtip_runner as qr
from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    # CPU arithmetic test uses the canonical eager body, not a solver fallback.
    monkeypatch.setattr(kd, 'decode_compressed', kd.decode_compressed._torchdynamo_orig_callable)
    unit = dict(schema='banana-smasher-qtip-unit-v1', shape=[64, 32],
                geometry=dict(L=16, K=2, V=2, tlut_bits=9),
                trellis=torch.arange(256, dtype=torch.int32).to(torch.uint16).reshape(8, 32),
                tlut=torch.arange(1024, dtype=torch.float32).reshape(512, 2) / 1024,
                SU=torch.where(torch.arange(32) % 2 == 0, 1., -1.).half(),
                SV=torch.where(torch.arange(64) % 3 == 0, -1., 1.).half(),
                Wscale=torch.tensor(0.75))
    path = tmp_path / 'QTIP_UNIT.pt'
    torch.save(unit, path)
    binding = dict(path=path.name, bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
    expected = qr.decode_packed_weight(unit, kd, torch.device('cpu'))
    rows = []
    for name, bounds in [('gate_proj', [0, 32]), ('up_proj', [32, 64])]:
        rows.append(dict(name=name, shape=[32, 32],
                         wire=dict(format='banana-smasher-qtip-unit-v1', unit=binding,
                                   geometry=unit['geometry'], unit_shape=[64, 32], row_range=bounds),
                         source_transform=dict(output_quantity='descaled_weight')))
    store = ArtifactTensorStore(dict(artifact_root=str(tmp_path), routed_tensors=rows,
                                    geometry=dict(routed_layer_ids=[3]),
                                    source=dict(model_root=str(tmp_path))))
    return unit, rows, store, expected


def test_sealed_unit_keeps_frozen_lut_and_inverse_transform(tmp_path, sealed):
    unit, rows, store, expected = sealed
    assert torch.equal(store.tensor('gate_proj'), expected[:32])
    assert torch.equal(store.tensor('up_proj'), expected[32:])
    assert store.model_reads == 0
    assert not store.requires_source_scale('gate_proj')

    # Admission must recognize the physical unit binding, not demand fake npy files.
    from banana_smasher.hf_moe import _verify_hf_moe_members
    receipt = dict(routed_tensors=rows, native_tensors=[], coverage=dict(duplicates=[], gaps=[]),
                   accounting=dict(planned_routed_tensor_count=2, routed_tensor_count=2,
                                   planned_native_tensor_count=0, native_tensor_count=0))
    _verify_hf_moe_members(tmp_path, receipt)


@pytest.mark.parametrize('mutation,match', [
    ('hash', 'hash/size'), ('bytes', 'hash/size'), ('shape', 'shape'),
    ('range', 'row range'), ('scale', 'descaled'), ('geometry', 'geometry'),
    ('nan', 'finite'),
])
def test_sealed_unit_refuses_corruption(tmp_path, sealed, mutation, match):
    unit, rows, store, _ = sealed
    row = rows[0]
    if mutation == 'hash':
        row['wire']['unit']['sha256'] = '0' * 64
    elif mutation == 'bytes':
        row['wire']['unit']['bytes'] += 1
    elif mutation == 'shape':
        row['shape'] = [16, 32]
    elif mutation == 'range':
        row['wire']['row_range'] = [1, 33]
    elif mutation == 'scale':
        row['source_transform'] = {}
    elif mutation == 'geometry':
        row['wire']['geometry'] = dict(L=16, K=1, V=2, tlut_bits=9)
    elif mutation == 'nan':
        unit['tlut'][0, 0] = float('nan')
        path = tmp_path / 'QTIP_UNIT.pt'
        torch.save(unit, path)
        row['wire']['unit'].update(bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=match):
        store.tensor('gate_proj')
