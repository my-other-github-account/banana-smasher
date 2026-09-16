"""Exact-source resident adapter (two default, four opt-in); numerical code unchanged."""
import ast
import hashlib
from pathlib import Path

SOURCE_SHA256 = '04e54e9f747f027c0274b0f187e2f9b39024eae20fc3002989879473a22fa83d'
PIN = 'aee895c2abeda737081501a197c493794391b131'


def render(source, *, cell_limit=2):
    if type(cell_limit) is not int or cell_limit not in (2, 4):
        raise ValueError('only default two or explicitly authorized four cells')
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('producer source mismatch: review changed source before adaptation')
    source = source.replace("assert len(prepared['rows'])==1 and", "assert len(prepared['rows'])==2 and", 1)
    source = source.replace("fresh_manifest=r/('K4_'+manifest_path.name)", "fresh_manifest=r/('K4_'+row['cell'].replace('/','_')+'_'+manifest_path.name)", 1)
    source = source.replace("for row in prepared['rows']:", "assert len(set(s['cells']))==2 and all(cell.endswith('_down') for cell in s['cells'])\nassert not set(s['cells']).intersection(s['prior_complete_k4_cells'])\nassert len({Path(row['config']).name for row in prepared['rows']})==2\nassert min(s['planned_write_bytes'],s['planned_output_bytes'])>=12582912\nfor row in prepared['rows']:", 1)
    source = source.replace('from fleet_cap_run8805 import apply_cap', 'from resident_two_cap import apply_cap', 1)
    if cell_limit == 4:
        source = source.replace("len(prepared['rows'])==2", "len(prepared['rows'])==4", 1)
        source = source.replace("len(set(s['cells']))==2 and all(cell.endswith('_down') for cell in s['cells'])", "s.get('resident_cell_limit')==4 and len(set(s['cells']))==4 and all(cell.rsplit('_',1)[-1] in ('down','fused13') for cell in s['cells'])", 1)
        source = source.replace("len({Path(row['config']).name for row in prepared['rows']})==2", "len({Path(row['config']).name for row in prepared['rows']})==4", 1)
        source = source.replace('>=12582912', ">=sum((10 if cell.endswith('fused13') else 6)*(1<<20) for cell in s['cells'])", 1)
    return source


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_producer', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cell-limit', type=int, choices=(2, 4), default=2,
                        help='four requires an explicit task-local integration grant; not production adoption')
    args = parser.parse_args()
    rendered = render(args.original_producer.read_text(), cell_limit=args.cell_limit)
    ast.parse(rendered)
    with args.output.open('x') as stream:
        stream.write(rendered)
        stream.flush()
        __import__('os').fsync(stream.fileno())


if __name__ == '__main__':
    main()
