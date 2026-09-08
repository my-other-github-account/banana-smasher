import pathlib, tarfile, hashlib, json
src=pathlib.Path('/Users/macmini/.hermes/kanban/boards/banana-smasher/workspaces/t_bd7914e6/accel_closure_run8499.tar')
assert hashlib.sha256(src.read_bytes()).hexdigest()=='a08ab89be8af74c98b1a17110eafbd5a0b40269916f706475b53bd2bbd8a8e34'
out=pathlib.Path('closure_run8503');out.mkdir(exist_ok=True)
with tarfile.open(src) as t:
 m=json.load(t.extractfile('MANIFEST.json'))
 print('manifest keys',list(m))
 rows=m['files']+m.get('components',[])
 print('rows',len(rows))
 for r in rows:
  name=r['archive_path']; p=pathlib.PurePosixPath(name)
  assert not p.is_absolute() and '..' not in p.parts
  b=t.extractfile(name).read()
  assert hashlib.sha256(b).hexdigest()==r['sha256'],name
  assert len(b)==r['bytes'],name
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
 print(json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)[:14000])
 print('VERIFIED',len(rows))
