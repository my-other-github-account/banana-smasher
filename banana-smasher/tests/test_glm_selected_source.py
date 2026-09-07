import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.glm_qtip_source_adapter import load_glm_fp8_weight


def digest(b):
    return hashlib.sha256(b).hexdigest()


def fixture(root) -> dict:
    config = dict(num_hidden_layers=4, first_k_dense_replace=3,
                  n_routed_experts=1, hidden_size=2, moe_intermediate_size=2)
    (root / 'config.json').write_text(json.dumps(config))
    headers, data, mapping, rows = {}, {}, {}, []
    for projection in ('gate', 'up', 'down'):
        key = f'model.language_model.layers.3.mlp.experts.0.{projection}_proj.weight'
        for name, dtype, shape, raw, shard in (
            (key, 'F8_E4M3', [2, 2], bytes([56, 64, 72, 80]), 'weights.safetensors'),
            (key + '_scale_inv', 'F32', [1, 1], struct.pack('<f', 2), 'scales.safetensors'),
        ):
            blob = data.setdefault(shard, bytearray())
            headers.setdefault(shard, {})[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(blob), len(blob)+len(raw)])
            blob.extend(raw)
            mapping[name] = shard
    index = root / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': mapping}))
    parents = {}
    for shard, header in headers.items():
        raw_header = json.dumps(header).encode()
        prefix = struct.pack('<Q', len(raw_header)) + raw_header
        (root / shard).write_bytes(prefix + data[shard])
        hp = shard + '.header'
        (root / hp).write_bytes(prefix)
        parents[shard] = dict(header_path=hp, header_sha256=digest(raw_header), bytes=len(prefix)+len(data[shard]))
        for name, meta in header.items():
            a, b = meta['data_offsets']
            raw = bytes(data[shard][a:b])
            payload = digest(name.encode()) + '.payload'
            (root / payload).write_bytes(raw)
            rows.append(dict(key=name, parent=str(root / shard), parent_bytes=parents[shard]['bytes'],
                             header_sha256=digest(raw_header), offset=len(prefix)+a, bytes=b-a,
                             dtype=meta['dtype'], shape=meta['shape'], data_sha256=digest(raw)))
    desc = dict(source_basis=digest(index.read_bytes()), rows=rows)
    (root / 'descriptor.json').write_text(json.dumps(desc))
    manifest = dict(schema='banana-smasher.selected-tensor-source.v1',
                    index_sha256=digest(index.read_bytes()), config_sha256=digest((root/'config.json').read_bytes()),
                    descriptor_path='descriptor.json', descriptor_sha256=digest((root/'descriptor.json').read_bytes()),
                    parents=parents, payloads={r['key']:digest(r['key'].encode())+'.payload' for r in rows})
    return manifest


def enable(root, manifest):
    (root/'SELECTED_TENSORS.json').write_text(json.dumps(manifest))
    for name in manifest['parents']:
        (root/name).unlink()


def test_selected_real_fp8_matches_full_shards(tmp_path):
    manifest = fixture(tmp_path)
    expected = {p:load_glm_fp8_weight(tmp_path,3,0,p)[0] for p in ('down','fused13')}
    enable(tmp_path, manifest)
    for p in expected:
        value, receipt = load_glm_fp8_weight(tmp_path,3,0,p)
        assert value.equal(expected[p])
        assert receipt['selected_source']['coverage'] == 'selected-tensors-only'
        assert all(r['storage'] == 'selected-tensor-payload' for r in receipt['shards'])
        assert all('sha256' not in r for r in receipt['shards'])  # no full-shard claim


@pytest.mark.parametrize('damage', ['payload', 'header', 'index', 'config', 'offset', 'missing', 'escape'])
def test_selected_fails_closed(tmp_path, damage):
    m = fixture(tmp_path)
    enable(tmp_path,m)
    key = next(k for k in m['payloads'] if '.down_proj.weight' in k)
    if damage == 'payload':
        (tmp_path/m['payloads'][key]).write_bytes(b'xxxx')
    elif damage == 'header':
        (tmp_path/m['parents']['weights.safetensors']['header_path']).write_bytes(b'bad')
    elif damage in ('index','config'):
        (tmp_path/('model.safetensors.index.json' if damage=='index' else 'config.json')).write_text('{}')
    elif damage == 'offset':
        dp=tmp_path/'descriptor.json';d=json.loads(dp.read_text())
        next(r for r in d['rows'] if r['key']==key)['offset'] += 1
        dp.write_text(json.dumps(d));m['descriptor_sha256']=digest(dp.read_bytes())
        (tmp_path/'SELECTED_TENSORS.json').write_text(json.dumps(m))
    elif damage == 'missing':
        (tmp_path/m['payloads'][key]).unlink()
    else:
        m['payloads'][key] = '../outside.payload'
        (tmp_path/'SELECTED_TENSORS.json').write_text(json.dumps(m))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_glm_fp8_weight(tmp_path,3,0,'down')


def test_public_launch_closure_requires_selected_manifest_pin(tmp_path, monkeypatch):
    from banana_smasher import glm_qtip_source_adapter as adapter
    manifest = fixture(tmp_path)
    enable(tmp_path, manifest)
    monkeypatch.setattr(adapter, 'capture_source_closure', lambda *a: {'sha256': 'fixture-closure'})
    config = {'glm_source_closure_sha256': 'fixture-closure'}
    with pytest.raises(ValueError, match='manifest pin'):
        adapter.bind_source_closure(tmp_path, [config], None, None)
    config['selected_source_manifest_sha256'] = digest((tmp_path/'SELECTED_TENSORS.json').read_bytes())
    assert adapter.bind_source_closure(tmp_path, [config], None, None)['sha256'] == 'fixture-closure'
    (tmp_path/'SELECTED_TENSORS.json').unlink()
    with pytest.raises(ValueError, match='manifest pin'):
        adapter.bind_source_closure(tmp_path, [config], None, None)
