import hashlib
import json
import threading
from pathlib import Path
import pytest
import torch
from banana_smasher import solver_qtip_profile as sp


def bank(root, count=4):
    for i in range(count):
        p=root/f'xmoe_L004_win{i:04d}.pt'
        torch.save(dict(layer=4,win=i,x=torch.arange(12).reshape(3,4).float()+i,topk=torch.ones(3,2,dtype=torch.long),w=torch.ones(3,2)),p)
        p.with_suffix('.pt.DONE.json').write_text(json.dumps(dict(md5=hashlib.md5(p.read_bytes()).hexdigest())))


def test_bounded_parallel_hashes_keep_captures_identical(tmp_path, monkeypatch):
    bank(tmp_path)
    serial=sp._load_captures(tmp_path,4,4)
    expected=[{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in r.items()} for r in serial]
    sp._release_capture_bank(tmp_path,4,4,serial)
    original=sp._md5
    barrier=threading.Barrier(4)
    names=set()
    def observed(path):
        names.add(threading.current_thread().name)
        barrier.wait(timeout=5)
        return original(path)
    monkeypatch.setattr(sp,'_md5',observed)
    result=sp._load_captures(tmp_path,4,4,hash_workers=4)
    assert len(names)==4
    assert len(result)==len(expected)
    for a,b in zip(result,expected):
        assert a.keys()==b.keys()
        for k in a:
            assert torch.equal(a[k],b[k]) if isinstance(a[k],torch.Tensor) else a[k]==b[k]
    sp._release_capture_bank(tmp_path,4,4,result)
    assert not result and not sp._CAPTURE_CACHE


@pytest.mark.parametrize('workers',[1,2,4])
def test_corruption_refused_without_cached_bank(tmp_path,workers):
    bank(tmp_path)
    path=tmp_path/'xmoe_L004_win0002.pt'
    with path.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(RuntimeError,match='capture MD5 mismatch'):
        sp._load_captures(tmp_path,4,4,hash_workers=workers)
    assert (tmp_path.resolve(),4,4) not in sp._CAPTURE_CACHE


@pytest.mark.parametrize('workers',[True,0,3,8,'4'])
def test_invalid_parallelism_refused(tmp_path,workers):
    with pytest.raises(ValueError,match='capture hash workers'):
        sp._load_captures(tmp_path,4,4,hash_workers=workers)


def test_parallel_missing_receipt_refused(tmp_path):
    bank(tmp_path)
    (tmp_path/'xmoe_L004_win0001.pt.DONE.json').unlink()
    with pytest.raises(FileNotFoundError,match='missing fit capture or receipt'):
        sp._load_captures(tmp_path,4,4,hash_workers=4)
    assert (tmp_path.resolve(),4,4) not in sp._CAPTURE_CACHE
