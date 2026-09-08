"""Consumer opt-in selects the existing decoder; wire checks are never skipped."""
import hashlib
import json
import torch
import pytest
from banana_smasher import sealed_qtip_unit, qtip_runner, qtip_kernel_decompress


def unit_row(tmp_path):
    torch.manual_seed(19)
    unit=dict(schema=sealed_qtip_unit.FORMAT, geometry=dict(L=16,K=2,V=2,tlut_bits=9),
              shape=[32,32], tlut=torch.randn(512,2),
              trellis=torch.randint(0,65536,(128,)).to(torch.uint16),
              Wscale=torch.tensor(.25),SU=torch.ones(32),SV=torch.ones(32))
    p=tmp_path/'unit.pt';torch.save(unit,p)
    row=dict(name='expert.weight',shape=[32,32],source_transform=dict(output_quantity='descaled_weight'),
             wire=dict(format=sealed_qtip_unit.FORMAT,geometry=unit['geometry'],unit_shape=[32,32],row_range=[0,32],
                       unit=dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
    return unit,row


def test_sealed_consumer_eager_preserves_real_wire_decode(tmp_path):
    unit,row=unit_row(tmp_path)
    expected=qtip_runner.decode_packed_weight(unit,qtip_kernel_decompress.select_decoder('eager'),torch.device('cpu'))
    actual=sealed_qtip_unit.decode_sealed_unit(tmp_path,row,execution='eager')
    assert torch.equal(actual,expected)
    with pytest.raises(ValueError,match='decoder execution'):
        sealed_qtip_unit.decode_sealed_unit(tmp_path,row,execution='fallback')
    (tmp_path/'unit.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash/size'):
        sealed_qtip_unit.decode_sealed_unit(tmp_path,row,execution='eager')


def test_artifact_consumer_selects_eager_without_source_fallback(tmp_path):
    from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore
    unit,row=unit_row(tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'expert.weight':'absent.safetensors'}}))
    artifact=dict(artifact_root=str(tmp_path),source=dict(model_root=str(tmp_path)),
                  geometry=dict(routed_layer_ids=[3]),routed_tensors=[row],native_tensors=[],
                  packed_decode_execution='eager')
    store=ArtifactTensorStore(artifact)
    assert getattr(store,'packed_decode_execution',None)=='eager'
    assert torch.equal(store.tensor('expert.weight'),sealed_qtip_unit.decode_sealed_unit(tmp_path,row,execution='eager'))
    assert store.model_reads==0 and store.payload_reads==1
    artifact.pop('packed_decode_execution')
    assert ArtifactTensorStore(artifact).packed_decode_execution=='compiled'
    for value in ['fallback',None,True]:
        with pytest.raises(ValueError,match='decoder execution'):
            ArtifactTensorStore(dict(artifact,packed_decode_execution=value))
