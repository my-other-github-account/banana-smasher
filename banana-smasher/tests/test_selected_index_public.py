from pathlib import Path
import json
from types import SimpleNamespace
import pytest
from banana_smasher import glm_qtip_source_adapter as adapter
from banana_smasher import selected_tensor_source as selected
from banana_smasher import solver_qtip_profile as sp
from test_glm_selected_source import fixture


def test_selected_flag_reaches_real_reader_and_preserves_weights(tmp_path,monkeypatch):
    manifest=fixture(tmp_path)
    (tmp_path/'SELECTED_TENSORS.json').write_text(json.dumps(manifest))
    expected=adapter.load_glm_fp8_weight(tmp_path,3,0,'down')[0]
    calls=[];original=selected._index_mapping
    def mapping(raw,enabled):calls.append(enabled);return original(raw,enabled)
    monkeypatch.setattr(selected,'_index_mapping',mapping)
    actual,_=adapter.load_glm_fp8_weight(tmp_path,3,0,'down',selected_index_cache=True)
    assert actual.equal(expected) and calls==[True]


def test_selected_flag_refuses_native_parent_lane(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError,match='selected'):
        adapter.load_glm_fp8_weight(tmp_path,3,0,'down',selected_index_cache=True)


@pytest.mark.parametrize('bad',[1,None,'true'])
def test_selected_flag_is_strict_at_weight_entry(tmp_path,bad):
    with pytest.raises(ValueError,match='boolean'):
        sp._load_weight(tmp_path,3,0,'down',selected_index_cache=bad)


def test_batch_flags_strict_consistent_and_default_off():
    from banana_smasher.qtip_batch_controller import _selected_index_cache
    assert _selected_index_cache([{},{}]) is False
    assert _selected_index_cache([{'selected_index_cache':True}]*2) is True
    for c in [[{}, {'selected_index_cache':True}],[{'selected_index_cache':1}],[{'selected_index_cache':None}]]:
        with pytest.raises(ValueError):_selected_index_cache(c)


def test_down_fit_forwards_selected_flag(tmp_path,monkeypatch):
    calls=[];weight=object();routed=[object()]
    def load(*args,**kwargs):calls.append(kwargs);return weight,{}
    monkeypatch.setattr(sp,'_load_weight',load)
    runner=SimpleNamespace(expert_windows=lambda c,e:routed,down_windows=lambda r,w,d:r)
    sp._prepare_fit_windows(runner,[],model_root=tmp_path,layer=3,expert=0,projection='down',device='cpu',selected_index_cache=True)
    assert calls==[{'selected_index_cache':True}]
