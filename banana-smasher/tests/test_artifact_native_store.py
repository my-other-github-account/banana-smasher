import hashlib
import json
import pytest
import torch
from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore


def fixture(tmp_path):
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'control':'missing.safetensors'}}))
    value=torch.tensor([[1.,-2.],[3.,4.]],dtype=torch.bfloat16)
    raw=value.view(torch.uint8).numpy().tobytes()
    (tmp_path/'control.bin').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    row=dict(name='control',path='control.bin',dtype='BF16',shape=[2,2],source_bytes=len(raw),representation='exact-source-data-bytes',storage_root='primary',source_sha256=digest,artifact_sha256=digest)
    artifact=dict(native_payload_reads=True,artifact_root=str(tmp_path),source={'model_root':str(tmp_path)},routed_tensors=[],native_tensors=[row],geometry={'routed_layer_ids':[5]})
    return artifact,value


def test_native_store_reads_sealed_bytes_without_source_shard(tmp_path):
    artifact,value=fixture(tmp_path)
    store=ArtifactTensorStore(artifact)
    assert torch.equal(store.tensor('control'),value)
    assert store.payload_reads==1
    assert store.model_reads==0


@pytest.mark.parametrize('field,value', [('artifact_sha256','0'*64),('source_sha256','0'*64),('source_bytes',9),('representation','untrusted'),('storage_root','native'),('path','../escape.bin'),('dtype','UNKNOWN')])
def test_native_store_refuses_invalid_binding(tmp_path,field,value):
    artifact,_=fixture(tmp_path)
    artifact['native_tensors'][0][field]=value
    store=ArtifactTensorStore(artifact)
    with pytest.raises(ValueError):
        store.tensor('control')
    assert store.model_reads==0


def test_native_store_refuses_corrupt_bytes(tmp_path):
    artifact,_=fixture(tmp_path)
    (tmp_path/'control.bin').write_bytes(b'badbytes')
    with pytest.raises(ValueError,match='hash mismatch'):
        ArtifactTensorStore(artifact).tensor('control')


def test_native_store_requires_explicit_opt_in(tmp_path):
    artifact,_=fixture(tmp_path)
    artifact.pop('native_payload_reads')
    with pytest.raises(FileNotFoundError):
        ArtifactTensorStore(artifact).tensor('control')
