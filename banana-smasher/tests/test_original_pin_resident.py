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
 def test_explicit_four_cell_template(self):
  rendered=adapter.render(SOURCE,cell_limit=4)
  tree=ast.parse(rendered)
  guards=[n for n in tree.body if isinstance(n,ast.Assert) and ('len(prepared' in ast.unparse(n) or 'prior_complete_k4_cells' in ast.unparse(n) or 'planned_write_bytes' in ast.unparse(n) or 'set(s[' in ast.unparse(n) or 'Path(row[' in ast.unparse(n) or 'resident_cell_limit' in ast.unparse(n))]
  code=compile(ast.Module(body=guards,type_ignores=[]),'admission','exec')
  cells=['L040/E047_down','L020/E000_fused13','L026/E096_fused13','L026/E097_fused13']
  s=dict(cells=cells,resident_cell_limit=4,canonical_commit=adapter.PIN,prior_complete_k4_cells=[],planned_write_bytes=36<<20,planned_output_bytes=36<<20)
  env=dict(s=s,prepared={'rows':[dict(config=f'/in/{i}.json') for i in range(4)]},Path=pathlib.Path)
  exec(code,env)
  for override in [dict(resident_cell_limit=2),dict(prior_complete_k4_cells=cells[:1]),dict(planned_write_bytes=35<<20)]:
   with self.assertRaises(AssertionError):exec(code,dict(env,s=dict(s,**override)))
  loops=lambda text:[ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.For) and 'pair_index' in ast.unparse(n.target)]
  self.assertEqual(loops(SOURCE),loops(rendered))
 def test_relocated_panel_preserves_science_and_singleton_loop(self):
  rendered=adapter.render(SOURCE,cell_limit=4,relocated_panel=True)
  self.assertIn("cfg['model_root']=row['model_root']",rendered)
  self.assertIn("cfg['qtip_root']=row['qtip_root']",rendered)
  self.assertIn("cfg['geometry']['K']==row['input_k']",rendered)
  self.assertIn("assert row['input_k'] in (1,4)",rendered)
  self.assertIn("H(Path(row['model_root'])/'model.safetensors.index.json')==s['intended_basis']",rendered)
  loops=lambda text:[ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.For) and 'pair_index' in ast.unparse(n.target)]
  self.assertEqual(loops(SOURCE),loops(rendered))
  pair=adapter.render(SOURCE,cell_limit=2,relocated_panel=True)
  self.assertNotIn("all(cell.endswith('_down')",pair)
  self.assertIn("sum((10 if cell.endswith('fused13') else 6)",pair)
 def test_qualification_singleton_dispatch(self):
  rows=[pathlib.Path('/one'),pathlib.Path('/two')]
  self.assertEqual(adapter.singleton_groups(rows),[[rows[0]],[rows[1]]])
  calls=[]
  result=adapter.run_singleton([rows[0]],pathlib.Path('/out'),40,lambda *args:calls.append(args) or 'result')
  self.assertEqual(result,'result');self.assertEqual(calls,[([rows[0]],pathlib.Path('/out'),40)])
  with self.assertRaises(ValueError):adapter.run_singleton(rows,pathlib.Path('/out'),40,lambda *args:None)
  self.assertIn('from original_pin_resident import run_singleton as run_pair',adapter.render(SOURCE,cell_limit=4,relocated_panel=True))
 def test_cli_four_cell_option(self):
  import subprocess,sys,tempfile
  with tempfile.TemporaryDirectory() as temp:
   out=pathlib.Path(temp)/'producer.py'
   result=subprocess.run([sys.executable,str(ROOT/'src/banana_smasher/original_pin_resident.py'),str(ROOT/'tests/fixtures/original_pin_producer.txt'),str(out),'--cell-limit','4'],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
   self.assertEqual(out.read_text(),adapter.render(SOURCE,cell_limit=4))
if __name__=='__main__':unittest.main()
