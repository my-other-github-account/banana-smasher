import hashlib
from pathlib import Path
import pytest
from banana_smasher.source_hash_prefetch import sha256_prefetched

@pytest.mark.parametrize('size',[0,1,8191,8<<20,(8<<20)+1,16<<20,(16<<20)+17])
def test_complete_bytes_and_chunk_boundaries(size,tmp_path):
    root=tmp_path
    data=(bytes(range(256))*(size//256+1))[:size]
    p=root/str(size)
    if not p.exists():p.write_bytes(data)
    assert sha256_prefetched(p)==hashlib.sha256(data).hexdigest()
    assert p.read_bytes()==data

def test_missing_path_raises_original_error():
    with pytest.raises(FileNotFoundError):sha256_prefetched(Path(__file__).with_name('MISSING_PREFETCH_FIXTURE'))

def test_one_read_ahead_overlaps_first_hash_update(monkeypatch):
    import threading
    from banana_smasher import source_hash_prefetch as implementation
    second_read=threading.Event();release=threading.Event();seen=[]
    real_hash=hashlib.sha256
    class Stream:
        calls=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):
            self.calls+=1
            if self.calls==1:return b'first'
            if self.calls==2:
                second_read.set();return b'second'
            return b''
        def fileno(self):return -1
    stream=Stream()
    class P:
        def open(self,mode):return stream
    class Digest:
        def __init__(self):self.inner=real_hash()
        def update(self,block):
            if not seen:
                assert second_read.wait(2), 'read must overlap hash update'
                assert stream.calls==2, 'only one pending read allowed'
            seen.append(block);self.inner.update(block)
        def hexdigest(self):return self.inner.hexdigest()
    monkeypatch.setattr(implementation.hashlib,'sha256',Digest)
    assert implementation.sha256_prefetched(P())==real_hash(b'firstsecond').hexdigest()
    assert seen==[b'first',b'second'] and stream.calls==3


class FailedStream:
    def __init__(self):self.calls=0;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def read(self,n):
        self.calls+=1
        if self.calls==2:raise OSError('read failure marker')
        return b'abc'
    def fileno(self):return -1
class FailedPath:
    def __init__(self):self.stream=FailedStream()
    def open(self,mode):assert mode=='rb';return self.stream

def test_read_error_is_propagated_and_stream_closed():
    p=FailedPath()
    with pytest.raises(OSError,match='read failure marker'):sha256_prefetched(p)
    assert p.stream.closed and p.stream.calls==2
