"""Runtime-only canonical schedule adoption; no fit/evaluation mutation."""
from copy import deepcopy
PIN='aa455253fa0cd8944a4148112d44c537938d40c4'
def rebind(cfg,manifest,*,runner,runner_sha,closure_sha,manifest_path,selected_sha):
 assert cfg['geometry']['K']==1
 assert cfg['geometry'].get('L')==16 and cfg['geometry'].get('V')==2
 c=deepcopy(cfg);m=deepcopy(manifest)
 c.update(block_ldl_unitwise=True,block_ldl_reference=True,qtip_runner=runner,glm_source_closure_sha256=closure_sha,viterbi_num_warps=16,viterbi_lut_l1_retention=True,viterbi_structured_gather=True,viterbi_branch_unroll=True,capture_hash_workers=4,viterbi_backpointer_dtype="uint16")
 if selected_sha is None:c.pop('selected_source_manifest_sha256',None)
 else:c['selected_source_manifest_sha256']=selected_sha
 m['canonical_commit']=PIN
 assert len(m['tiers'])==1
 m['tiers'][0]['bindings']['qtip_runner']=dict(path=runner,sha256=runner_sha)
 c['materialization']['run_manifest']=manifest_path
 c['materialization'].pop('run_manifest_sha256',None)
 return c,m
