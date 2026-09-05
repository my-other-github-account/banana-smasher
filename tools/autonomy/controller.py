#!/usr/bin/env python3
"""Bounded control observations; scoring is explicit and owner-local only."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import os
import tempfile


def read_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        ident, owner, payload = line.split('\t', 2)
        row = json.loads(payload)
        if row.get('id') != ident or row.get('owner') != owner:
            raise ValueError('row identity mismatch')
        rows.append(row)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate registered IDs')
    return rows


def retain_gates(path, gates):
    """Single dispatcher writes all rows, including failures; never delete a gate."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    with os.fdopen(fd, 'w') as f:
        f.write('# id\towner\tjson\n')
        for g in gates:
            f.write(g['id'] + '\t' + g['owner'] + '\t' + json.dumps(g, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def select_work(queue, claims, host):
    """Proposal only. Dispatcher must re-read HOST_CLAIM + SHARDS and CAS."""
    active = [c for c in claims if c.get('active', True)]
    if any(c.get('host') == host for c in active):
        return None
    for q in sorted(queue, key=lambda r: r.get('priority', 999)):
        if q.get('status') != 'ready' or not q.get('owner'):
            continue
        if q.get('hosts') and host not in q['hosts']:
            continue
        if not all(k in q for k in ('resource', 'start', 'end')) or q['start'] >= q['end']:
            continue
        if any(c.get('resource') == q['resource'] and
               (not all(k in c for k in ('start', 'end')) or
                max(c['start'], q['start']) < min(c['end'], q['end'])) for c in active):
            continue
        return q
    return None

IDENTITY = ('model_index_sha256', 'artifact_sha256', 'teacher_sha256',
            'scorer_sha256', 'window_ids', 'support_sha256', 'position_policy', 'reference')

class LocalReader:
    def exists(self, host, path):
        if host != 'local':
            raise ValueError('local scorer cannot mutate a remote host')
        return Path(path).is_file()

    def read(self, host, path):
        if host != 'local':
            raise ValueError('local reader requires local host')
        return Path(path).read_bytes()


def verify_metric(raw, expected):
    d = json.loads(raw)
    if not all(expected.get(k) for k in IDENTITY):
        raise ValueError('owner must register full frozen evaluation identity')
    if d.get('identity') != expected:
        raise ValueError('native/window/teacher/artifact identity mismatch')
    ids = expected['window_ids']
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError('expected exact frozen 64 unique window IDs')
    if d.get('metric') != 'forward_kl':
        raise ValueError('requires forward KL, not NLL subtraction or weight MSE')
    values = d.get('window_values')
    if not isinstance(values, list) or len(values) != 64:
        raise ValueError('requires 64 per-window metrics')
    for v in values + [d.get('value')]:
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('invalid KL metric')
    mean = math.fsum(values) / 64
    if not math.isclose(mean, d['value'], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('aggregate does not match per-window readback')
    return dict(value=mean, result_sha256=hashlib.sha256(raw).hexdigest())


def evaluate_gate(gate, reader, execute=False, owner=None):
    result = dict(gate)
    result.pop('value', None)
    result.pop('result_sha256', None)
    result['status'] = 'pending'
    try:
        if gate.get('registration_pending'):
            raise ValueError('registration incomplete: ' + gate['next_action'])
        if not gate.get('owner'):
            raise ValueError('missing decision owner')
        if reader.exists(gate['host'], gate['result']):
            result.update(verify_metric(reader.read(gate['host'], gate['result']), gate['expected']))
            result['status'] = 'verified'
            result['reason'] = 'metric readback verified; quality decision remains with owner'
            return result
        if not reader.exists(gate['host'], gate['marker']):
            raise ValueError('await completed artifact; owner verify producer frontier')
        if not execute:
            raise ValueError('completed but unscored: owner execute registered scorer then read back')
        if gate['host'] != 'local' or owner != gate['owner']:
            raise ValueError('only matching owner may score locally; sensor never scores')
        argv = gate.get('score_argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(s, str) for s in argv):
            raise ValueError('owner must register scorer argv')
        proc = subprocess.run(argv, capture_output=True, timeout=60, check=False)
        if proc.returncode:
            raise ValueError('scorer exit %d: %s' % (proc.returncode, proc.stderr.decode(errors='replace')[-500:]))
        result.update(verify_metric(reader.read(gate['host'], gate['result']), gate['expected']))
        result['status'] = 'verified'
        result['reason'] = 'scorer completed and metric independently read back; not automatic GREEN'
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        result['reason'] = str(exc)
    return result


PROBE = '''import json,pathlib,sys,shutil,subprocess
r=json.loads(sys.argv[1]); p=pathlib.Path(r['root'])
if not p.is_dir(): raise RuntimeError('frontier root missing: '+str(p))
n=sum(1 for _ in p.rglob(r['pattern']))
d=shutil.disk_usage(p)
try:
 g=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=5,check=True).stdout.split()
except (OSError,subprocess.SubprocessError): g=None
print(json.dumps(dict(count=n,gpu_pids=g,disk_used_pct=100*d.used/d.total)))
'''


def remote_read(host, program, argument):
    import shlex
    argv = ['python3', '-c', program, argument]
    if host != 'local':
        argv = ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, shlex.join(argv)]
    return subprocess.run(argv, capture_output=True, timeout=20, check=True).stdout


class RemoteReader(LocalReader):
    def read(self, host, path):
        return remote_read(host, 'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())', path)

    def exists(self, host, path):
        return remote_read(host, 'import pathlib,sys;print(int(pathlib.Path(sys.argv[1]).is_file()))', path).strip() == b'1'


def probe_frontier(row):
    try:
        return dict(row, **json.loads(remote_read(row['host'], PROBE, json.dumps(row))))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(row, count=None, error=str(exc), action='owner investigate; never kill based on count alone')


def tick(state, previous=None):
    from concurrent.futures import ThreadPoolExecutor
    import time
    start = time.monotonic()
    state = Path(state)
    gates = read_rows(state / 'GATES.tsv')
    queue = read_rows(state / 'QUEUE.tsv')
    rows = read_rows(state / 'FRONTIERS.tsv')
    unique = {}
    for row in rows:
        key = (row['host'], row['root'], row['pattern'])
        if key in unique and unique[key]['owner'] != row['owner']:
            raise ValueError('duplicate frontier has conflicting owners')
        unique.setdefault(key, row)
    def observe(g):
        try:
            return evaluate_gate(g, RemoteReader())
        except subprocess.SubprocessError as exc:
            return dict(g, status='pending', reason=str(exc))
    with ThreadPoolExecutor(max_workers=8) as pool:
        observed_gates = list(pool.map(observe, gates))
        frontiers = list(pool.map(probe_frontier, unique.values()))
    old = {r['id']: r for r in (previous or {}).get('frontiers', [])}
    for f in frontiers:
        p = old.get(f['id'], {})
        if f.get('count') is not None and f.get('count') == p.get('count'):
            f['unchanged_ticks'] = p.get('unchanged_ticks', 0) + 1
            if f['unchanged_ticks'] >= 2:
                f['action'] = 'owner inspect PID/startticks/log; unchanged count is not proof of death'
        if f.get('disk_used_pct', 0) >= 95:
            f['disk_action'] = 'owner propose named rebuildable paths; preserve artifacts; no automatic deletion'
        if f.get('gpu_pids') == []:
            f['seat_action'] = 'dispatcher reconcile live claim and SHARDS; GPU empty does not mean unowned'
    warnings = []
    if not queue:
        warnings.append('QUEUE_EMPTY')
    if not gates:
        warnings.append('GATES_EMPTY')
    return dict(schema='banana-smasher.control-observation.v1', role='sensor',
                gates=observed_gates, queue=queue, frontiers=frontiers,
                warnings=warnings, elapsed_seconds=time.monotonic()-start,
                authority='observations only; sole dispatcher owns decisions; no metric automatically GREEN')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    previous = None
    if args.receipt and args.receipt.exists():
        previous = json.loads(args.receipt.read_text())
    result = tick(args.state, previous)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.receipt:
        fd, tmp = tempfile.mkstemp(dir=args.receipt.parent, prefix='sensor-observation-')
        with os.fdopen(fd, 'w') as f:
            f.write(text + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.receipt)
    print(text)


if __name__ == '__main__':
    main()
