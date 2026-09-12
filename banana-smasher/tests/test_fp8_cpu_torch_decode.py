from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from banana_smasher import hf_moe as hf
from banana_smasher import glm_qtip_source_adapter as adapter
from test_glm_selected_source import fixture, enable

@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('projection',['down','fused13'])
def test_public_torchdecode(tmp_path,monkeypatch,selected,projection):
    m=fixture(tmp_path)
    if selected:enable(tmp_path,m)
    old_from=torch.from_numpy;old_view=torch.Tensor.view;seen=[];views=[]
    def from_numpy(array):
        if array.dtype==np.uint8:
            assert array.flags.owndata and array.flags.writeable
            seen.append(array.copy())
        return old_from(array)
    def view(tensor,*args,**kwargs):
        if args==(torch.float8_e4m3fn,):
            assert tensor.dtype==torch.uint8 and tensor.device.type=='cpu'
            views.append(tensor.numel())
        return old_view(tensor,*args,**kwargs)
    monkeypatch.setattr(torch,'from_numpy',from_numpy)
    monkeypatch.setattr(torch.Tensor,'view',view)
    actual,_=adapter.load_glm_fp8_weight(tmp_path,3,0,projection)
    expected=torch.tensor([[2.,4.],[8.,16.]])
    if projection=='fused13':expected=torch.cat([expected,expected])
    assert actual.equal(expected)
    assert len(seen)==len(views)==(2 if projection=='fused13' else 1)
    assert views==[4]*len(views)

@pytest.mark.parametrize('symbol',range(256))
def test_all_fp8_symbols_actual_loader(symbol):
    row=dict(name='weight',dtype='F8_E4M3',shape=[2,2])
    scale=dict(name='scale',dtype='F32',shape=[1,1])
    def read(source,r):
        return bytes([symbol]*4) if r['name']=='weight' else np.array([2.],dtype='<f4').tobytes()
    exponent=(symbol>>3)&15;mantissa=symbol&7
    magnitude=np.ldexp(float(mantissa)/8,-6) if exponent==0 else np.ldexp(1+float(mantissa)/8,exponent-7)
    expected=np.nan if exponent==15 and mantissa==7 else (-magnitude if symbol&128 else magnitude)
    if not np.isfinite(expected):
        with pytest.raises(ValueError,match='non-finite'):
            hf._load_safetensors_matrix(Path('weight'),row,scale_source=Path('scale'),scale_row=scale,fp8_decode_with_torch=True,tensor_payload_reader=read)
    else:
        actual=hf._load_safetensors_matrix(Path('weight'),row,scale_source=Path('scale'),scale_row=scale,fp8_decode_with_torch=True,tensor_payload_reader=read)
        np.testing.assert_array_equal(actual,np.full((2,2),expected*2,dtype=np.float32))
        assert np.all(np.signbit(actual)==np.signbit(expected))

@pytest.mark.parametrize('symbol',[127,255])
@pytest.mark.parametrize('position',[0,1,2,3])
def test_nan_position_precedes_scale_admission(symbol,position):
    raw=bytearray([56]*4);raw[position]=symbol
    with pytest.raises(ValueError,match='non-finite'):
        hf._load_safetensors_matrix(Path('weight'),dict(name='weight',dtype='F8_E4M3',shape=[2,2]),fp8_decode_with_torch=True,tensor_payload_reader=lambda *a:bytes(raw))

def test_empty_array_preserves_missing_scale_error():
    with pytest.raises(ValueError,match='requires weight_scale_inv'):
        hf._load_safetensors_matrix(Path('weight'),dict(name='weight',dtype='F8_E4M3',shape=[0,2]),fp8_decode_with_torch=True,tensor_payload_reader=lambda *a:b'')

def test_nonmatrix_error_order():
    with pytest.raises(ValueError,match='must be a matrix'):
        hf._load_safetensors_matrix(Path('weight'),dict(name='weight',dtype='F8_E4M3',shape=[4]),fp8_decode_with_torch=True,tensor_payload_reader=lambda *a:bytes([56]*4))
