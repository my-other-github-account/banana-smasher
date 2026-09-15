from pathlib import Path
import hashlib
import inspect
import pytest
from banana_smasher import glm_qtip_source_adapter as adapter


def test_default_and_prefetched_source_hash_keep_cache_and_identity(tmp_path, monkeypatch):
    p=tmp_path/'source';p.write_bytes(b'original bytes')
    calls=[]
    def fast(path):
        calls.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(adapter,'_SOURCE_DIGESTS',{})
    monkeypatch.setattr(adapter,'_sha256_prefetched',fast,raising=False)
    expected=hashlib.sha256(p.read_bytes()).hexdigest()
    assert adapter._source_sha256(p,prefetch=True)==expected
    assert adapter._source_sha256(p,prefetch=False)==expected
    assert calls==[p.resolve()]
    p.write_bytes(b'mutated bytes longer')
    with pytest.raises(ValueError,match='immutable source changed'):
        adapter._source_sha256(p,prefetch=True)


@pytest.mark.parametrize('value',[1,0,None,'true'])
def test_prefetch_option_rejects_non_boolean_before_io(value):
    with pytest.raises(ValueError,match='boolean'):
        adapter._source_sha256(Path('/not-a-source-fixture'),prefetch=value)


def test_mutation_during_prefetched_read_refuses_cache_commit(tmp_path,monkeypatch):
    p=tmp_path/'source';p.write_bytes(b'before')
    def changed(path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b'after is different')
        return digest
    monkeypatch.setattr(adapter,'_SOURCE_DIGESTS',{})
    monkeypatch.setattr(adapter,'_sha256_prefetched',changed,raising=False)
    with pytest.raises(ValueError,match='during hashing'):
        adapter._source_sha256(p,prefetch=True)
    assert not adapter._SOURCE_DIGESTS


def test_import_closure_binds_prefetch_implementation_file():
    from types import SimpleNamespace
    module=SimpleNamespace(__file__=__file__)
    closure=adapter.capture_source_closure(module,{name:module for name in ['bitshift','ldlq','math_utils','kernel_decompress']})
    row=closure['files']['source_hash_prefetch']
    assert row['sha256']==hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()


def test_public_batch_prefetch_flags_are_strict_and_consistent():
    from banana_smasher.qtip_batch_controller import _source_hash_prefetch
    assert _source_hash_prefetch([{},{}]) is False
    assert _source_hash_prefetch([{'source_hash_prefetch':True}]*2) is True
    for configs in [[{'source_hash_prefetch':True},{}],[{'source_hash_prefetch':1}],[{'source_hash_prefetch':None}]]:
        with pytest.raises(ValueError):_source_hash_prefetch(configs)


def test_non_glm_prefetch_refuses_instead_of_silently_ignoring(tmp_path,monkeypatch):
    from banana_smasher import solver_qtip_profile as sp
    index=(tmp_path/'model.safetensors.index.json').resolve()
    monkeypatch.setattr(sp,'_MODEL_INDEX_CACHE',{index:{'unrelated.tensor':'shard'}})
    with pytest.raises(ValueError,match='GLM'):
        sp._load_weight(tmp_path,40,47,'down',source_hash_prefetch=True)


def test_weight_and_down_fit_entry_accept_explicit_prefetch():
    from banana_smasher import solver_qtip_profile as sp
    assert inspect.signature(sp._load_weight).parameters['source_hash_prefetch'].default is False
    assert inspect.signature(sp._prepare_fit_windows).parameters['source_hash_prefetch'].default is False


@pytest.mark.parametrize('enabled',[False,True])
def test_real_weight_entry_forwards_flag_to_glm_only(tmp_path,monkeypatch,enabled):
    from banana_smasher import solver_qtip_profile as sp
    index=(tmp_path/'model.safetensors.index.json').resolve()
    monkeypatch.setattr(sp,'_MODEL_INDEX_CACHE',{index:{'model.language_model.layers.40.mlp.experts.47.down_proj.weight':'shard'}})
    calls=[]
    def load(*args,**kwargs):
        calls.append((args,kwargs));return 'tensor',{'original':True}
    monkeypatch.setattr(adapter,'load_glm_fp8_weight',load)
    assert sp._load_weight(tmp_path,40,47,'down',source_hash_prefetch=enabled)==('tensor',{'original':True})
    assert calls==[((tmp_path,40,47,'down'), {'source_hash_prefetch':True} if enabled else {})]


@pytest.mark.parametrize('enabled',[False,True])
def test_down_fit_forwards_flag_without_changing_windows(tmp_path,monkeypatch,enabled):
    from banana_smasher import solver_qtip_profile as sp
    from types import SimpleNamespace
    calls=[];routed=[object()];weight=object()
    def load(*args,**kwargs):
        calls.append((args,kwargs));return weight,{'original':True}
    def down(windows,w,device):
        assert windows is routed and w is weight and device=='cpu';return routed
    runner=SimpleNamespace(expert_windows=lambda c,e:routed,down_windows=down)
    monkeypatch.setattr(sp,'_load_weight',load)
    output,_=sp._prepare_fit_windows(runner,[],model_root=tmp_path,layer=40,expert=47,projection='down',device='cpu',source_hash_prefetch=enabled)
    assert output is routed
    assert calls==[((tmp_path,40,47,'fused13'),{'source_hash_prefetch':True} if enabled else {})]
