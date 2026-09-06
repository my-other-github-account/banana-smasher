import ast, copy, json, unittest
from pathlib import Path
D=Path(__file__).resolve().parent
source=ast.parse((D/'resume_fleet.py').read_text())
program=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROGRAM' for t in n.targets))
fixture=json.loads((Path(__file__).parent/'closure_fixture.json').read_text())
def check(mods):
 tree=ast.parse(program)
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_imports']
 if funcs:
  ns={'Path':Path};exec(compile(ast.Module(body=funcs,type_ignores=[]),'verifier','exec'),ns)
  return ns['verify_imports'](mods,json.loads((D/'IMPORT_CLOSURE.json').read_text()))
 statement=next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and "qtip_validation_bitshift" in ast.unparse(n))
 exec(compile(ast.Module(body=[statement],type_ignores=[]),'verifier','exec'),{'mods':mods})
class ImportClosureTests(unittest.TestCase):
 def test_real_closure(self):check(fixture)
 def test_wrong_digest_rejected(self):
  bad=copy.deepcopy(fixture);bad['qtip_validation_bitshift']['sha256']='0'*64
  with self.assertRaises((ValueError,AssertionError)):check(bad)
 def test_missing_alias_rejected(self):
  bad=copy.deepcopy(fixture);del bad['qtip_validation_ldlq']
  with self.assertRaises((ValueError,AssertionError)):check(bad)
if __name__=='__main__':unittest.main()
