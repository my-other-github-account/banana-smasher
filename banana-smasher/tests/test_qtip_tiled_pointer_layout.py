"""CPU address-level regression for internal K1 pointer tiling."""
import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def offset_fn():
    tree = ast.parse(SOURCE.read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tiled_pointer_offset']
    assert found, 'K1 tiled pointer address helper missing'
    fn = found[0]
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns[fn.name]

def test_four_step_width64_mapping_is_bijective():
    offset = offset_fn()
    prefixes, steps, batch = 16384, 4, 3
    seen = set()
    for t in range(steps):
        for b in range(batch):
            for j in range(prefixes):
                got = offset(t, b, j, batch, prefixes, steps)
                expected = (((t // 4 * batch + b) * (prefixes // 64) + j // 64) * 4 + t % 4) * 64 + j % 64
                assert got == expected
                seen.add(got)
    assert seen == set(range(steps * batch * prefixes))

def test_non_k1_and_unaligned_steps_keep_original_addresses():
    offset = offset_fn()
    for prefixes, steps in [(1, 4), (16, 4), (4096, 128), (1024, 128), (256, 128), (16384, 3)]:
        for t in range(steps):
            for b in range(3):
                for j in [0, prefixes // 2, prefixes - 1]:
                    assert offset(t, b, j, 3, prefixes, steps) == (t * 3 + b) * prefixes + j
