"""Authenticated local safetensors ranges, never a partial pretend model shard.

The opt-in SELECTED_TENSORS.json sidecar preserves original index/config and
owner descriptor bytes. Only selected FP8 weights/F32 scales are supported.
A caller must pin the sidecar/descriptor in its immutable launch closure.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


class SelectedTensorSource:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        raw = self._path('SELECTED_TENSORS.json').read_bytes()
        self.manifest = m = json.loads(raw)
        if m['schema'] != 'banana-smasher.selected-tensor-source.v1':
            raise ValueError('unsupported selected tensor source schema')
        for name, field in [('model.safetensors.index.json', 'index_sha256'), ('config.json', 'config_sha256')]:
            if _digest(self._path(name).read_bytes()) != m[field]:
                raise ValueError(f'selected source {field} mismatch')
        self.mapping = json.loads(self._path('model.safetensors.index.json').read_bytes())['weight_map']
        descriptor_bytes = self._path(m['descriptor_path']).read_bytes()
        if _digest(descriptor_bytes) != m['descriptor_sha256']:
            raise ValueError('selected source descriptor digest mismatch')
        descriptor = json.loads(descriptor_bytes)
        if descriptor['source_basis'] != m['index_sha256']:
            raise ValueError('selected source descriptor basis mismatch')
        self.rows = {r['key']: r for r in descriptor['rows']}
        if len(self.rows) != len(descriptor['rows']):
            raise ValueError('duplicate selected source keys')
        if set(self.rows) != set(m['payloads']):
            raise ValueError('selected source payload coverage mismatch')
        self.headers = {}
        self.header_lengths = {}
        for shard, parent in m['parents'].items():
            raw_header = self._path(parent['header_path']).read_bytes()
            if len(raw_header) < 8:
                raise ValueError('truncated selected source header')
            size = struct.unpack('<Q', raw_header[:8])[0]
            if len(raw_header) != 8 + size or _digest(raw_header[8:]) != parent['header_sha256']:
                raise ValueError('selected source header digest/length mismatch')
            self.headers[shard] = json.loads(raw_header[8:])
            self.header_lengths[shard] = len(raw_header)
        self.receipt = dict(schema=m['schema'], manifest_sha256=_digest(raw),
                            descriptor_sha256=m['descriptor_sha256'],
                            coverage='selected-tensors-only', index_sha256=m['index_sha256'])

    def _path(self, relative):
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('selected source path escapes root')
        resolved = (self.root / p).resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ValueError('selected source path escapes root')
        return resolved

    def reference(self, key):
        r = self.rows[key]
        return dict(storage='selected-tensor-payload', weight_key=key,
                    parent_path=r['parent'], parent_bytes=r['parent_bytes'],
                    header_sha256=r['header_sha256'], offset=r['offset'],
                    payload_path=str(self._path(self.manifest['payloads'][key])),
                    payload_bytes=r['bytes'], payload_sha256=r['data_sha256'],
                    dtype=r['dtype'], shape=r['shape'])

    def read(self, source, row):
        key = row['name']
        if key not in self.rows:
            raise ValueError(f'uncovered selected source tensor: {key}')
        r = self.rows[key]
        shard = self.mapping[key]
        if Path(source) != self.root / shard or Path(r['parent']).name != shard:
            raise ValueError('selected source parent/index mismatch')
        parent = self.manifest['parents'][shard]
        original = self.headers[shard][key]
        start, end = original['data_offsets']
        size = {'F8_E4M3': 1, 'F32': 4}.get(r['dtype'])
        if (size is None or r['shape'] != original['shape'] or r['dtype'] != original['dtype']
            or row['shape'] != original['shape'] or row['dtype'] != original['dtype']
            or r['header_sha256'] != parent['header_sha256']
            or r['parent_bytes'] != parent['bytes']
            or r['offset'] != self.header_lengths[shard] + start
            or start < 0 or end < start or r['bytes'] != end-start
            or r['offset'] + r['bytes'] > parent['bytes']
            or any(type(n) is not int or n <= 0 for n in r['shape'])
            or math.prod(r['shape']) * size != r['bytes']):
            raise ValueError('selected source header/offset/geometry mismatch')
        raw = self._path(self.manifest['payloads'][key]).read_bytes()
        if len(raw) != r['bytes'] or _digest(raw) != r['data_sha256']:
            raise ValueError('selected source payload length/digest mismatch')
        return raw
