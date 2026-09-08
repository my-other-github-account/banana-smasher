import math
import pytest
import torch
from banana_smasher.qtip_runner import fwht


def test_explicit_rounded_normalization_preserves_cpu():
    x = torch.arange(64, dtype=torch.float32).reshape(2,32) / 7
    assert torch.equal(fwht(x, normalization='rounded'), fwht(x))
    with pytest.raises(ValueError, match='normalization'):
        fwht(x, normalization='invalid')


def test_sealed_public_normalization_selection(tmp_path):
    from test_sealed_qtip_execution import unit_row
    from banana_smasher.sealed_qtip_unit import decode_sealed_unit
    _, row=unit_row(tmp_path)
    a=decode_sealed_unit(tmp_path,row,execution='eager')
    b=decode_sealed_unit(tmp_path,row,execution='eager',normalization='rounded')
    assert torch.equal(a,b)
    with pytest.raises(ValueError,match='normalization'):
        decode_sealed_unit(tmp_path,row,normalization='invalid')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires claimed CUDA')
@pytest.mark.parametrize('n',[32,64,2048,4096])
def test_rounded_cuda_fwht_matches_cpu(n):
    g=torch.Generator().manual_seed(73)
    x=torch.randn((13,n), generator=g)
    got=fwht(x.cuda(), normalization='rounded').cpu()
    assert torch.equal(got,fwht(x))


def test_artifact_selects_normalization(tmp_path,monkeypatch):
    import json
    from test_sealed_qtip_execution import unit_row
    from banana_smasher import sealed_qtip_unit
    from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore
    _,row=unit_row(tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'expert.weight':'absent.safetensors'}}))
    artifact=dict(artifact_root=str(tmp_path),source=dict(model_root=str(tmp_path)),geometry=dict(routed_layer_ids=[3]),routed_tensors=[row],native_tensors=[],packed_decode_execution='eager',packed_decode_normalization='rounded')
    original=sealed_qtip_unit.decode_sealed_unit;calls=[]
    def capture(*args,**kwargs):
        calls.append(kwargs.get('normalization','default'));return original(*args,**kwargs)
    monkeypatch.setattr(sealed_qtip_unit,'decode_sealed_unit',capture)
    store=ArtifactTensorStore(artifact);store.tensor('expert.weight')
    assert calls==['rounded'] and store.model_reads==0
    with pytest.raises(ValueError,match='normalization'):
        ArtifactTensorStore(dict(artifact,packed_decode_normalization='invalid'))
