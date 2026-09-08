import json,hashlib,math
from pathlib import Path
r=Path('receipts_run8482')
collection=json.loads((r/'COLLECTION.json').read_text())
for rel,digest in collection['files'].items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==digest,rel
result=json.loads((r/'RESULT.json').read_text());validation=result['validation'];frozen=json.loads((r/'FROZEN_LIMITS.json').read_text())
assert result['state']=='SEALED_HISTORICAL_OWNER_BRIDGE'
assert len(validation['rows'])==12 and len(frozen['rows'])==12
expected={(e,a,p) for e in (90,91,92) for a in ('singleton','batch') for p in ('cold','warm')}
assert {(x['expert'],x['arm'],x['phase']) for x in validation['rows']}==expected
for row in validation['rows']:
 assert row['pass_numerical']==(row['nmse']<=frozen['limits'][str(row['expert'])])
assert validation['all_pass']==all(x['pass_numerical'] for x in validation['rows'])
for arm in ('old1','old2','singleton','batch'):
 for e in (90,91,92):
  cfg=json.loads((r/arm/f'E{e}.json').read_text());source=json.loads(Path(f'publication-run8461/accel/receipts/ds4_current_k3_sixcell_run8478/candidate_L009_E{e:03d}.json').read_text())
  changed={k for k in cfg.keys()|source.keys() if cfg.get(k)!=source.get(k)}
  assert changed<={'qtip_runner','exact_solver','materialization','block_ldl_unitwise'}
  for phase in ('cold','warm'):
   receipt=json.loads((r/arm/phase/f'solve/L009/E{e:03d}_down/QTIP_SOLVE_RECEIPT.json').read_text())
   assert receipt['status']=='PASS' and receipt['rht_seed']==source['rht_seed']
   assert receipt['build']['packed_decode']['runtime_check_performed'] is True
   assert receipt['build']['packed_decode']['fp16_bit_exact'] is True
   assert receipt['config_sha256']==hashlib.sha256((r/arm/f'E{e}.json').read_bytes()).hexdigest()
phases={}
for phase in ('cold','warm'):
 def metric(arm):return next(x for x in result['arms'][arm]['phases'] if x['phase']==phase)
 baseline=(metric('old1')['wall']+metric('old2')['wall'])/2
 singleton=metric('singleton')['wall'];batch=metric('batch')['wall']
 phases[phase]=dict(owner_mean_seconds=baseline,proposed_singleton_seconds=singleton,proposed_batch_seconds=batch,owner_to_proposed_singleton=baseline/singleton,owner_to_batch=baseline/batch,proposed_singleton_to_batch=singleton/batch,peak_allocated={a:metric(a)['peak_allocated'] for a in result['arms']},maxrss_kib={a:metric(a)['maxrss_kib'] for a in result['arms']})
summary=dict(state='PASS_HISTORICAL_OWNER_COMPATIBILITY' if validation['all_pass'] else 'FAIL_HISTORICAL_OWNER_COMPATIBILITY',scope='Three historical L009 down cells; old4921456b to new39da421e singleton and batch; no new production admissions, no forward replay, no fullmodel equivalence',numerical_rows=len(validation['rows']),producer_seals=24,phases=phases,baseline_validation_seconds=frozen['validation_seconds'],candidate_validation_seconds=validation['validation_seconds'],quality_rows=validation['rows'],production_adoption=False,source_result_sha256=hashlib.sha256((r/'RESULT.json').read_bytes()).hexdigest(),collection_sha256=hashlib.sha256((r/'COLLECTION.json').read_bytes()).hexdigest(),repeat_scope='Two old-owner process repeats; one new-singleton and one new-batch process, each cold+warm; prior current-main repeated speed gate remains separately sealed')
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
