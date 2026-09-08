"""Compact internal K1 pointer layout, never a new packed model format."""
import ast
from pathlib import Path

def test_uint16_pairs_reduce_private_prefix_storage():
    source = (Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py').read_text()
    assert 'PACK_BRANCH_PAIRS=pack_branch_pairs' in source
    assert 'storage_prefixes = prefixes // 2 if pack_branch_pairs else prefixes' in source
    tree=ast.parse(source)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_store_prefix_backpointers')
    text=ast.unparse(helper)
    assert 'tl.split' in text and 'odd << 8' in text
    assert 'PACK_BRANCH_PAIRS' in text
