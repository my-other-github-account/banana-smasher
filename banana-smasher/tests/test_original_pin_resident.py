import ast,importlib.util,pathlib,unittest
ROOT=pathlib.Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('adapter',ROOT/'src/banana_smasher/original_pin_resident.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
SOURCE=(ROOT/'tests/fixtures/original_pin_producer.txt').read_text()
class Check(unittest.TestCase):
 def test_two_original_rows_admitted_by_real_guard(self):
  tree=ast.parse(adapter.render(SOURCE));guard=next(n for n in tree.body if isinstance(n,ast.Assert) and 'len(prepared' in ast.unparse(n))
  eval(compile(ast.Expression(guard.test),'guard','eval'),{'prepared':{'rows':[1,2]},'s':{'canonical_commit':'aee895c2abeda737081501a197c493794391b131'}}) or self.fail('two rows refused')
 def test_unique_manifest_for_same_basename(self):
  tree=ast.parse(adapter.render(SOURCE));loop=next(n for n in tree.body if isinstance(n,ast.For) and ast.unparse(n.target)=='row')
  assignment=next(n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fresh_manifest' for t in n.targets))
  code=compile(ast.Expression(assignment.value),'path','eval')
  env={'r':pathlib.Path('/output'),'manifest_path':pathlib.Path('/source/MANIFEST.json')}
  a=eval(code,dict(env,row={'cell':'L014/E180_down'}));b=eval(code,dict(env,row={'cell':'L014/E181_down'}));self.assertNotEqual(a,b)
 def test_source_drift_refused(self):
  with self.assertRaises(ValueError):adapter.render(SOURCE+'\n')
 def test_singleton_dispatch_unchanged(self):
  original=ast.parse(SOURCE);candidate=ast.parse(adapter.render(SOURCE))
  loops=lambda t:[ast.dump(n) for n in t.body if isinstance(n,ast.For) and 'pair_index' in ast.unparse(n.target)]
  self.assertEqual(loops(original),loops(candidate))
 def test_missing_only_and_capacity_guards(self):
  tree=ast.parse(adapter.render(SOURCE));nodes=[n for n in tree.body if isinstance(n,ast.Assert) and ('prior_complete_k4_cells' in ast.unparse(n) or 'planned_write_bytes' in ast.unparse(n) or 'set(s[' in ast.unparse(n) or 'Path(row[' in ast.unparse(n))]
  self.assertEqual(len(nodes),4)
  code=compile(ast.Module(body=nodes,type_ignores=[]),'admission','exec')
  prepared={'rows':[{'config':'/a/a.json'},{'config':'/b/b.json'}]}
  base={'cells':['L014/E180_down','L014/E181_down'],'prior_complete_k4_cells':[],'planned_write_bytes':12582912,'planned_output_bytes':12582912}
  exec(code,{'s':base,'prepared':prepared,'Path':pathlib.Path})
  for override in [{'prior_complete_k4_cells':['L014/E180_down']},{'cells':['L014/E180_down']*2},{'planned_write_bytes':6291456}]:
   with self.assertRaises(AssertionError):exec(code,{'s':dict(base,**override),'prepared':prepared,'Path':pathlib.Path})
 def test_singleton_guard_refuses_one_and_three(self):
  tree=ast.parse(adapter.render(SOURCE));guard=next(n for n in tree.body if isinstance(n,ast.Assert) and 'len(prepared' in ast.unparse(n))
  code=compile(ast.Expression(guard.test),'guard','eval')
  for n in [1,3]:self.assertFalse(eval(code,{'prepared':{'rows':[1]*n},'s':{'canonical_commit':adapter.PIN}}))
if __name__=='__main__':unittest.main()
