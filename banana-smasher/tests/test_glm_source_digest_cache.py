import hashlib
import json
from collections import Counter

import numpy as np

import pytest

from banana_smasher import glm_qtip_source_adapter as adapter


def test_source_digest_reads_immutable_file_once(tmp_path, monkeypatch):
    path = tmp_path / 'shard'
    path.write_bytes(b'actual fixture bytes')
    calls = []
    def digest(p):
        calls.append(p)
        return hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(adapter, '_sha256', digest)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert adapter._source_sha256(path) == expected
    assert adapter._source_sha256(path) == expected
    assert len(calls) == 1


def test_loader_reuses_weight_scale_digest_without_tensor_cache(tmp_path, monkeypatch):
    shard = tmp_path / 'weights.safetensors'
    shard.write_bytes(b'fixture source; tensor I/O stubbed')
    (tmp_path / 'config.json').write_text(json.dumps(dict(num_hidden_layers=45, n_routed_experts=288, first_k_dense_replace=3, hidden_size=2, moe_intermediate_size=2)))
    keys = [f'model.language_model.layers.3.mlp.experts.0.{name}_proj.weight' for name in ('gate', 'up')]
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': dict.fromkeys(keys, shard.name)}))
    monkeypatch.setattr(adapter, '_safetensors_header', lambda p: dict.fromkeys(keys, {'dtype':'F8_E4M3'}))
    monkeypatch.setattr(adapter, '_routed_scale_binding', lambda root, row, **kw: (shard, {'name':row['name']+'_scale_inv'}, {'kind':'fixture'}))
    matrix = np.arange(4, dtype=np.float32).reshape(2, 2)
    monkeypatch.setattr(adapter, '_load_safetensors_matrix', lambda *a, **kw: matrix.copy())
    calls = Counter()
    original = adapter._sha256
    def digest(path):
        calls[path] += 1
        return original(path)
    monkeypatch.setattr(adapter, '_sha256', digest)
    first, receipt = adapter.load_glm_fp8_weight(tmp_path, 3, 0, 'fused13')
    second, other = adapter.load_glm_fp8_weight(tmp_path, 3, 0, 'fused13')
    assert first.equal(second) and first.data_ptr() != second.data_ptr()
    assert receipt == other
    assert calls[shard] == 1


def test_source_digest_refuses_mutation(tmp_path):
    path = tmp_path / 'shard'
    path.write_bytes(b'original')
    adapter._source_sha256(path)
    path.write_bytes(b'modified')
    with pytest.raises(ValueError, match='source changed'):
        adapter._source_sha256(path)


def test_source_digest_refuses_change_during_hash(tmp_path, monkeypatch):
    path = tmp_path / 'racing'
    path.write_bytes(b'old')
    def mutate(p):
        value = hashlib.sha256(p.read_bytes()).hexdigest()
        p.write_bytes(b'new')
        return value
    monkeypatch.setattr(adapter, '_sha256', mutate)
    with pytest.raises(ValueError, match='changed during hashing'):
        adapter._source_sha256(path)


def test_source_digest_refuses_same_size_atomic_replacement(tmp_path):
    path = tmp_path / 'source'
    path.write_bytes(b'old')
    adapter._source_sha256(path)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'new')
    replacement.replace(path)
    with pytest.raises(ValueError, match='source changed'):
        adapter._source_sha256(path)
