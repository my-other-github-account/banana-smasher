"""Original singleton resource policy projected onto two sequential cells.

Use verify for deployment admission and apply_cap for the child. Numerical
modules, singleton resource caps and original reserves remain unchanged.
"""
import hashlib
from pathlib import Path
import fleet_cap_run8805 as original

EXPECTED_ORIGINAL_SHA256 = '7a9e1269c8d7f2a2752db819889e7fd022eeef28780ee4c112c92aa74eac8168'


def _check(spec):
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != EXPECTED_ORIGINAL_SHA256:
        raise ValueError('original fleet cap source mismatch')
    cells = spec['cells']
    if spec['K'] != 4 or not isinstance(cells, list) or len(cells) != 2 or len(set(cells)) != 2:
        raise ValueError('requires exactly two distinct K4 cells')
    values = [original.budget([cell]) for cell in cells]
    peak, output = max(v[0] for v in values), sum(v[1] for v in values)
    if spec['estimated_peak_bytes'] != peak or spec['planned_write_bytes'] < output:
        raise ValueError('unchanged singleton peak and aggregate output budget required')
    return peak, output


def _one(spec, cell):
    if cell not in spec['cells']:
        raise ValueError('cell outside admitted pair')
    return dict(spec, cells=[cell])


def verify(spec, available, free):
    limits = _check(spec)
    # Preserve aggregate planned writes while applying the original reserves.
    for cell in spec['cells']:
        original.verify(_one(spec, cell), available, free)
    return limits


def apply_cap(spec, resource, cuda):
    _check(spec)
    return original.apply_cap(_one(spec, spec['cells'][0]), resource, cuda)
