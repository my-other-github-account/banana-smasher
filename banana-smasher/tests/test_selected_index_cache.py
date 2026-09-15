import json
from pathlib import Path
import pytest
from banana_smasher import selected_tensor_source as selected
from test_glm_selected_source import fixture

@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(selected,'_INDEX_MAPPING_CACHE',None,raising=False)


def test_cached_mapping_is_fresh_and_parse_is_elided(monkeypatch):
    raw=b'{"weight_map":{"a":"one","b":"two"}}'
    original=selected.json.loads;calls=[]
    def loads(x):calls.append(x);return original(x)
    monkeypatch.setattr(selected.json,'loads',loads)
    first=selected._index_mapping(raw,True);first['a']='caller mutation'
    second=selected._index_mapping(raw,True)
    assert second=={'a':'one','b':'two'} and second is not first
    assert calls==[raw]


def test_default_reparses_and_nested_values_are_not_cached(monkeypatch):
    original=selected.json.loads;calls=[]
    def loads(x):calls.append(x);return original(x)
    monkeypatch.setattr(selected.json,'loads',loads)
    raw=b'{"weight_map":{"a":"one"}}'
    selected._index_mapping(raw,False);selected._index_mapping(raw,False)
    assert len(calls)==2 and selected._INDEX_MAPPING_CACHE is None
    nested=b'{"weight_map":{"a":[1]}}'
    one=selected._index_mapping(nested,True);one['a'].append(2)
    assert selected._index_mapping(nested,True)=={'a':[1]}
    assert selected._INDEX_MAPPING_CACHE is None


def test_changed_bytes_and_invalid_json_are_not_cached_hits():
    assert selected._index_mapping(b'{"weight_map":{"a":"one"}}',True)=={'a':'one'}
    assert selected._index_mapping(b'{"weight_map":{"a":"two"}}',True)=={'a':'two'}
    with pytest.raises(json.JSONDecodeError):selected._index_mapping(b'{',True)


def test_cache_budget_preserves_uncached_semantics(monkeypatch):
    monkeypatch.setattr(selected,'_INDEX_CACHE_MAX_BYTES',1,raising=False)
    assert selected._index_mapping(b'{"weight_map":{"a":"one"}}',True)=={'a':'one'}
    assert selected._INDEX_MAPPING_CACHE is None


def test_constructor_keeps_both_original_index_reads(tmp_path,monkeypatch):
    m=fixture(tmp_path);(tmp_path/'SELECTED_TENSORS.json').write_text(json.dumps(m))
    original=Path.read_bytes;calls=[]
    def read(p):
        if p.name=='model.safetensors.index.json':calls.append(str(p))
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',read)
    a=selected.SelectedTensorSource(tmp_path,cache_index=True)
    b=selected.SelectedTensorSource(tmp_path,cache_index=True)
    assert len(calls)==4 and a.mapping==b.mapping and a.mapping is not b.mapping
    (tmp_path/'model.safetensors.index.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='index_sha256 mismatch'):
        selected.SelectedTensorSource(tmp_path,cache_index=True)


@pytest.mark.parametrize('damage',['payload','header','config','descriptor','missing','escape'])
def test_warm_cache_does_not_bypass_other_original_gates(tmp_path,damage):
    m=fixture(tmp_path);mp=tmp_path/'SELECTED_TENSORS.json';mp.write_text(json.dumps(m))
    selected.SelectedTensorSource(tmp_path,cache_index=True)
    key=next(k for k in m['payloads'] if '.down_proj.weight' in k)
    if damage=='payload':(tmp_path/m['payloads'][key]).write_bytes(b'xxxx')
    elif damage=='header':(tmp_path/m['parents']['weights.safetensors']['header_path']).write_bytes(b'bad')
    elif damage=='config':(tmp_path/'config.json').write_bytes(b'{}')
    elif damage=='descriptor':(tmp_path/'descriptor.json').write_bytes(b'{}')
    else:
        m['payloads'][key]='missing.payload' if damage=='missing' else '../outside.payload'
        mp.write_text(json.dumps(m))
    with pytest.raises((ValueError,FileNotFoundError)):
        source=selected.SelectedTensorSource(tmp_path,cache_index=True)
        source.read(tmp_path/'weights.safetensors',dict(name=key,dtype='F8_E4M3',shape=[2,2]))


@pytest.mark.parametrize('value',[1,None,'true'])
def test_cache_flag_is_strict(value,tmp_path):
    with pytest.raises(ValueError,match='boolean'):
        selected.SelectedTensorSource(tmp_path,cache_index=value)
