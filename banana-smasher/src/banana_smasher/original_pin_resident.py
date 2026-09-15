"""Exact-source two-cell adapter. No imports or modifications to numerical code."""
import ast
import hashlib
from pathlib import Path

SOURCE_SHA256 = '04e54e9f747f027c0274b0f187e2f9b39024eae20fc3002989879473a22fa83d'
PIN = 'aee895c2abeda737081501a197c493794391b131'


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('producer source mismatch: review changed source before adaptation')
    source = source.replace("assert len(prepared['rows'])==1 and", "assert len(prepared['rows'])==2 and", 1)
    source = source.replace("fresh_manifest=r/('K4_'+manifest_path.name)", "fresh_manifest=r/('K4_'+row['cell'].replace('/','_')+'_'+manifest_path.name)", 1)
    source = source.replace("for row in prepared['rows']:", "assert len(set(s['cells']))==2 and all(cell.endswith('_down') for cell in s['cells'])\nassert not set(s['cells']).intersection(s['prior_complete_k4_cells'])\nassert len({Path(row['config']).name for row in prepared['rows']})==2\nassert min(s['planned_write_bytes'],s['planned_output_bytes'])>=12582912\nfor row in prepared['rows']:", 1)
    return source


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_producer', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rendered = render(args.original_producer.read_text())
    ast.parse(rendered)
    with args.output.open('x') as stream:
        stream.write(rendered)
        stream.flush()
        __import__('os').fsync(stream.fileno())


if __name__ == '__main__':
    main()
