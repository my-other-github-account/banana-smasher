from pathlib import Path
import json,hashlib,statistics,math,copy
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence_run8503'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reduce(e):
 rows=[]
 for arm in ('old1','new1','new2','old2'):
  r=json.loads((e/'whole'/f'{arm}_RESULT.json').read_text());assert len(r['rows'])==2
  for row in r['rows']:
   assert row['arm']==arm and row['phase'] in ('cold','warm')
   p=e/'whole'/f"{arm}_{row['phase']}.pt";assert sha(p)==row['artifact_sha256'] and p.stat().st_size==row['bytes']
   b=row['build'];assert b['canonical_pack']['canonical_pack_roundtrip_exact'] and b['packed_decode']['fp16_bit_exact']
   assert b['rht_seed']==5413548984517567745 and b['fit_rows']==65 and b['fit_route_mass']==17.0009765625
   assert row['calls']==256 and row['sequences']==65536 and row['solver']['branch_sampling']=='full'
   rows.append(row)
 assert len({(x['arm'],x['phase']) for x in rows})==8
 q=json.loads((e/'original_quality/ORIGINAL_QUALITY.json').read_text())['rows'];assert len(q)==8
 assert {(x['arm'],x['phase']) for x in q}=={(x['arm'],x['phase']) for x in rows}
 for x in q:
  assert x['artifact_sha256']==next(r['artifact_sha256'] for r in rows if r['arm']==x['arm'] and r['phase']==x['phase'])
  assert math.isfinite(x['nmse']) and x['nmse']<=x['original_nmse']*1.0001+1e-12
  assert x['metrics']['qtip_hyb']['sse_ratio_vs_true_vq']<=1.0001+1e-12
  assert x['original_artifact_sha256']=='a202fc9d67cd75e7db8d1f5db95d2de502ef0e67f9bf0a69c6ddd1a6e753ee0a'
 wall=json.loads((e/'whole/PROCESS_WALL.json').read_text());assert len(wall)==5 and all(x['returncode']==0 for x in wall)
 old=statistics.mean(x['whole_seconds'] for x in wall if x['arm'].startswith('old'));new=statistics.mean(x['whole_seconds'] for x in wall if x['arm'].startswith('new'));assert new<old
 phase={}
 for k in ('old','new'):
  rr=[r for r in rows if r['arm'].startswith(k) and r['phase']=='warm']
  phase[k]=dict(ldlq_seconds=statistics.mean(r['build']['phase_seconds']['ldlq'] for r in rr),build_seconds=statistics.mean(r['build_wall'] for r in rr),whole_iteration_seconds=statistics.mean(r['whole_iteration_seconds'] for r in rr))
 assert phase['new']['ldlq_seconds']<phase['old']['ldlq_seconds']
 return dict(status='REPRESENTATIVE_SPEED_QUALITY_PASS_OWNER_ROLLOUT_PENDING',tested_code_pin='eb3b97314416440bd835fb8327d60b52b0826bdb',candidate_config_patch={'viterbi_num_warps':8,'viterbi_branch_unroll':True},whole_process=dict(baseline_mean_seconds=old,candidate_mean_seconds=new,speedup=old/new,scope='fresh process including setup, cold+warm full-build pair, outputs and shutdown; shared compiler cache with compile costs retained; ABBA'),warm=phase,dominant_phase_speedup=phase['old']['ldlq_seconds']/phase['new']['ldlq_seconds'],warm_whole_build_speedup=phase['old']['build_seconds']/phase['new']['build_seconds'],quality=dict(verified_rows=len(q),original_nmse=q[0]['original_nmse'],max_nmse=max(x['nmse'] for x in q),max_output_sse_ratio=max(x['metrics']['qtip_hyb']['sse_ratio_vs_true_vq'] for x in q),all_decoded_equal_original=all(x['same_decoded'] for x in q),scope='original owner packed artifact, authentic clean-fit 16 windows/65 routed rows, not heldout/frozen/full-model'),scientific_outputs_newly_accepted=0,production_adoption=False,frozen_evaluation_touched=False,original_d1_cells_per_hour=322.56732008115506,production_504_cells_per_hour_demonstrated=False,baseline_caveat='Canonical bounded builder shared across both arms to isolate original-owner Viterbi vs current-main scheduling. Baseline output independently equals retained original owner artifact; original s7 throughput is not compared to s6 timing.',wall_records=wall,artifact_seals=[dict(arm=r['arm'],phase=r['phase'],sha256=r['artifact_sha256'],bytes=r['bytes']) for r in rows])
if __name__=='__main__':
 report=reduce(E);(ROOT/'RESULT_run8503.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
