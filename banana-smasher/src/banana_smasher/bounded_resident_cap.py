"""Explicit stricter resident resource envelope; never selected by default."""
from pathlib import Path
import hashlib,json,re
G=1<<30

def admit(available,free,planned_write_bytes,witness_path,witness_sha256):
    raw=Path(witness_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=witness_sha256:
        raise ValueError('working-set witness mismatch')
    w=json.loads(raw)
    match=re.search(r'^VmPeak:\s+(\d+) kB$',w['proc_status'],re.M)
    if match is None:
        raise ValueError('missing VmPeak witness')
    peak=int(match.group(1))*1024
    if peak>=32*G or w['cuda_peak_reserved']>=4*G or w['ru_maxrss_bytes']>=32*G:
        raise ValueError('witness exceeds stricter caps')
    if type(planned_write_bytes) is not int or planned_write_bytes<=0:
        raise ValueError('positive aggregate output budget required')
    if available<=48*G+planned_write_bytes:
        raise RuntimeError('HOST_8G_HEADROOM_REFUSED')
    if free<=4*G+planned_write_bytes:
        raise RuntimeError('ORIGINAL_STORAGE_RESERVE_REFUSED')
    return dict(cpu_address_limit=32*G,cuda_allocator_limit=4*G,overhead_bytes=4*G,estimated_peak_bytes=40*G,host_reserve_bytes=8*G,storage_reserve_bytes=4*G,witness_vmpeak=peak,witness_sha256=witness_sha256)

def enforce(resource,cuda):
    resource.setrlimit(resource.RLIMIT_AS,(32*G,32*G))
    cuda.set_per_process_memory_fraction(4*G/cuda.get_device_properties(0).total_memory,0)
    return dict(cpu_address_limit=32*G,cuda_allocator_limit=4*G)
