"""Regression checks for the sealed run8553 research non-promotion decision."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT=Path(__file__).parents[1]/'receipts/glm_ldlq_continuation_run8553'

def test_manifest_and_physical_output_inventory():
    m=json.loads((ROOT/'MANIFEST.json').read_text())
    assert m['count']==len(m['files'])
    assert len({r['path'] for r in m['files']})==m['count']
    for r in m['files']:
        p=ROOT/r['path']
        assert p.stat().st_size==r['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    outputs=json.loads((ROOT/'PHYSICAL_OUTPUT_ACK.json').read_text())
    assert len(outputs)==48 and len({r['path'] for r in outputs})==48
    assert sum(r['bytes'] for r in outputs)==68091056

def test_no_promotion_from_neutral_speed_or_failed_quality():
    s=json.loads((ROOT/'SUMMARY.json').read_text())
    assert s['status']=='CONTINUATION_NOT_COMPLETE'
    assert not s['adoption'] and not s['production_mutations']
    assert s['quality_count']==48 and s['quality_pass']==40
    for c in s['candidates']:
        assert c['adopted'] is False
        q=json.loads((ROOT/c['candidate']/'QUALITY.json').read_text())
        assert c['quality_gate']['passed']==sum(r['passed'] for r in q['rows'])
        for phase,t in c['timing'].items():
            assert t['ratio']==statistics.mean(t['baseline_seconds'])/statistics.mean(t['candidate_seconds'])
        if c['candidate']=='distance':
            assert c['decision']=='REJECT_SPEED_AND_QUALITY' and c['quality_gate']['passed']==4
        else:
            assert c['quality_gate']['passed']==12
        assert 'cold-JIT' in s['timing_scope']

def test_each_actual_output_matches_its_sealed_quality_receipt():
    outputs=json.loads((ROOT/'PHYSICAL_OUTPUT_ACK.json').read_text())
    for o in outputs:
        parts=Path(o['path']).parts
        arm,phase,group=parts[1:4]
        cell=parts[-2]
        q=json.loads((ROOT/o['candidate']/'QUALITY.json').read_text())
        matches=[r for r in q['rows'] if (r['arm'],r['phase'],r['cell'])==(arm,phase,cell)]
        assert len(matches)==1 and matches[0]['artifact_sha256']==o['sha256']
