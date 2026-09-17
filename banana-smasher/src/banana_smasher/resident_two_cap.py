"""Original singleton policy for two (default) or opt-in four/eight sequential cells.

Use verify for deployment admission and apply_cap for the child. Default
singleton caps stay unchanged. Explicit sequential30 qualification enforces a
smaller address-space ceiling, with authenticated working-set evidence and
unchanged CUDA, overhead, host and storage reserves; it is not adoption.
"""
import hashlib
import json
import re
from pathlib import Path

G = 1 << 30


def _envelope(spec):
    policy = spec.get('resource_envelope', 'original')
    if policy == 'original':
        return 48 * G
    if policy != 'sequential30':
        raise ValueError('unknown resource envelope')
    raw = Path(spec['working_set_witness']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['working_set_sha256']:
        raise ValueError('working-set witness mismatch')
    witness = json.loads(raw)
    match = re.search(r'^VmPeak:\s+(\d+) kB$', witness['proc_status'], re.M)
    if match is None or int(match.group(1)) * 1024 >= 30 * G:
        raise ValueError('witness exceeds sequential address cap')
    if not 0 <= witness['ru_maxrss_bytes'] < 30 * G or not 0 <= witness['cuda_peak_reserved'] < 4 * G:
        raise ValueError('witness exceeds sequential resource caps')
    return 30 * G
import fleet_cap_run8805 as original

EXPECTED_ORIGINAL_SHA256 = '7a9e1269c8d7f2a2752db819889e7fd022eeef28780ee4c112c92aa74eac8168'


def _check(spec):
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != EXPECTED_ORIGINAL_SHA256:
        raise ValueError('original fleet cap source mismatch')
    cells = spec['cells']
    count = spec.get('resident_cell_limit', 2)
    if type(count) is not int or count not in (2, 4, 8):
        raise ValueError('resident_cell_limit must be 2 or explicitly authorized 4/8')
    if count == 8 and spec.get('resource_envelope') != 'sequential30':
        raise ValueError('eight cells require the enforced sequential30 envelope')
    if spec['K'] != 4 or not isinstance(cells, list) or len(cells) != count or len(set(cells)) != count:
        raise ValueError('requires exactly the admitted count of distinct K4 cells')
    values = [original.budget([cell]) for cell in cells]
    peak, output = _envelope(spec) + 8 * G, sum(v[1] for v in values)
    if spec['estimated_peak_bytes'] != peak or spec['planned_write_bytes'] < output:
        raise ValueError('enforced singleton peak and aggregate output budget required')
    return peak, output


def _one(spec, cell):
    if cell not in spec['cells']:
        raise ValueError('cell outside admitted pair')
    return dict(spec, cells=[cell])


def verify(spec, available, free):
    limits = _check(spec)
    if spec.get('resource_envelope', 'original') == 'sequential30':
        assert available > limits[0] + 8 * G + spec['planned_write_bytes'], 'HOST_8G_HEADROOM_REFUSED'
        assert free > 4 * G + spec['planned_write_bytes'], 'ORIGINAL_STORAGE_RESERVE_REFUSED'
    else:
        for cell in spec['cells']:
            original.verify(_one(spec, cell), available, free)
    return limits


def apply_cap(spec, resource, cuda):
    peak, _ = _check(spec)
    if spec.get('resource_envelope', 'original') == 'sequential30':
        resource.setrlimit(resource.RLIMIT_AS, (30 * G, 30 * G))
        cuda.set_per_process_memory_fraction(4 * G / cuda.get_device_properties(0).total_memory, 0)
        return dict(cpu_address_limit=30 * G, cuda_allocator_limit=4 * G,
                    overhead_bytes=4 * G, estimated_peak_bytes=peak,
                    numerical_change=False, resource_envelope='sequential30',
                    witness_sha256=spec['working_set_sha256'])
    return original.apply_cap(_one(spec, spec['cells'][0]), resource, cuda)
