"""Focused source-provider gates; fixture bytes are not scientific outputs."""
import importlib
import torch
import pytest


def provider():
    try:
        return importlib.import_module('banana_smasher.q1_pre_source')
    except ModuleNotFoundError:
        pytest.fail('K1 PRE source provider is missing')


def test_k1_decode_reuses_canonical_wire_and_inverse_transform():
    api = provider()
    from banana_smasher import qtip_kernel_decompress as kernel
    from banana_smasher.fwht import bounded_fwht
    torch.manual_seed(17)
    # A small valid packed-wire fixture, not an admitted production artifact.
    unit = dict(shape=[32, 32], geometry=dict(L=16, K=1, V=2, tlut_bits=9,
                decode_mode='quantlut_sym', td_x=16, td_y=16),
                trellis=torch.randint(0, 65536, (64,), dtype=torch.int32).to(torch.uint16),
                tlut=torch.randn(512, 2), Wscale=torch.tensor(0.25),
                SU=torch.ones(32), SV=-torch.ones(32))
    index = torch.arange(65536)
    quadratic = (index + 1) * index
    expanded = unit['tlut'][(quadratic >> 6) & 511].clone()
    expanded[:, 0] *= 1 - ((quadratic >> 15) & 1) * 2
    eager = kernel.decode_compressed._torchdynamo_orig_callable
    raw = eager(16, 9, 1, 1, 32, 32, unit['trellis'], expanded)
    expected = bounded_fwht(bounded_fwht((raw * unit['Wscale']).T).T * unit['SV'][:, None]) * unit['SU']
    actual = api.decode_unit(unit, device='cpu', kernel_decode=eager)
    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_partial_clean_inventory_cannot_be_a_uniform_source(tmp_path):
    import json, hashlib
    api = provider()
    p = tmp_path / 'inventory.json'
    p.write_text(json.dumps(dict(basis=api.BASIS, accepted_clean=[])))
    with pytest.raises(ValueError, match='full routed-expert coverage'):
        api.CleanK1Source(p, hashlib.sha256(p.read_bytes()).hexdigest(), tmp_path)


def test_inventory_pin_rejected_before_payload_loading(tmp_path):
    api = provider()
    p = tmp_path / 'inventory.json'
    p.write_text('{}')
    with pytest.raises(ValueError, match='inventory bytes'):
        api.CleanK1Source(p, '0' * 64, tmp_path)


@pytest.mark.parametrize('key,value', [('K', 4), ('decode_mode', 'learned'), ('V', 1)])
def test_decode_refuses_non_k1_geometry(key, value):
    api = provider()
    geom = dict(api.GEOMETRY)
    geom[key] = value
    with pytest.raises(ValueError, match='frozen K1 geometry'):
        api.decode_unit(dict(geometry=geom), device='cpu')


def test_full_grid_admitted_but_duplicate_cell_rejected(tmp_path):
    import json, hashlib
    api = provider()
    rows = [dict(cell=[l, e, p]) for l in range(43) for e in range(256)
            for p in ('fused13', 'down')]
    p = tmp_path / 'inventory.json'
    def source():
        p.write_text(json.dumps(dict(basis=api.BASIS, accepted_clean=rows)))
        return api.CleanK1Source(p, hashlib.sha256(p.read_bytes()).hexdigest(), tmp_path)
    assert len(source().rows) == 22016
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='full routed-expert coverage'):
        source()


def test_fill_layer_preserves_full_projection_order_and_progress(tmp_path):
    api = provider()
    # Exercise the forward callback separately from numerical wire decoding.
    class Probe(api.CleanK1Source):
        def __init__(self):
            self.output, self.decoded_units, self.calls = tmp_path, 0, []
        def decode(self, layer, expert, projection):
            self.calls.append((layer, expert, projection))
            self.decoded_units += 1
            return self.decoded_units
    source = Probe()
    gate_up, down = [None] * 256, [None] * 256
    source.fill_layer(7, gate_up, down)
    assert gate_up == list(range(1, 513, 2))
    assert down == list(range(2, 513, 2))
    assert source.calls[-1] == (7, 255, 'down')
    import json
    assert json.loads((tmp_path / 'DECODE_PROGRESS.json').read_text())['decoded_units'] == 512


