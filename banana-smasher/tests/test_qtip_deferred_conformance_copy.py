"""Execute the builder's actual reference/conformance/store boundary."""
import ast
from pathlib import Path
import pytest


@pytest.mark.parametrize('on_device', [False, True])
def test_reference_stays_on_device_until_packed_conformance(on_device):
    src = Path(__file__).parents[1] / 'src/banana_smasher/qtip_batch.py'
    tree = ast.parse(src.read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_qtip_batch')
    assignment = next(n for n in ast.walk(builder) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "candidate['reconstructed_weight']" for t in n.targets))
    loop = next(n for n in ast.walk(builder) if isinstance(n, ast.For) and assignment in n.body)
    body = loop.body[loop.body.index(assignment):]
    events = []

    class Tensor:
        def __init__(self, device):
            self.device = device
        def half(self):
            events.append('half')
            return self
        def cpu(self):
            events.append('cpu')
            return Tensor('cpu')

    def decode(runner, candidate, codebook, decoder, device, *, compare_on_device):
        events.append(('validate', candidate['reconstructed_weight'].device))
        assert compare_on_device is on_device
        return {'fp16_bit_exact': True}

    env = dict(candidate={}, reconstructed=Tensor('cuda'), raw=object(), candidates=[],
               packed_decode_receipts=[], runner=None, codebook=None, kernel_decode=None,
               device='cuda', packed_conformance_on_device=on_device, _decode_candidate=decode)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(src), 'exec'), env)
    assert events == (['half', ('validate', 'cuda'), 'cpu'] if on_device else ['half', 'cpu', ('validate', 'cpu')])
    assert env['candidates'][0]['reconstructed_weight'].device == 'cpu'
    assert env['packed_decode_receipts'] == [{'fp16_bit_exact': True}]
