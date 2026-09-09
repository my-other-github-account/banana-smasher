import ast
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def test_actual_rebase_keeps_unreachable_costs():
 tree=ast.parse(SOURCE.read_text());kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
 guards=[n for n in ast.walk(kernel) if isinstance(n,ast.If) and ast.unparse(n.test)=='DISTANCE_ALPHABET and HAS_OVERLAP and CONDITIONED_DISTANCE_SUM' and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='minimum_cost' for t in a.targets) for a in n.body)]
 assert len(guards)==1
 class TL:
  min=staticmethod(np.min);where=staticmethod(np.where)
 for values in [np.array([3,7,np.inf],dtype=np.float32),np.full(4,np.inf,dtype=np.float32)]:
  env={'tl':TL,'previous_costs':values.copy()}
  exec(compile(ast.Module(body=guards[0].body,type_ignores=[]),str(SOURCE),'exec'),env)
  result=env['previous_costs'];assert not np.isnan(result).any()
  if np.isfinite(values).any():assert result[0]==0 and result[1]==4 and np.isinf(result[2])
  else:assert np.isinf(result).all()
