import ast
from pathlib import Path


def test_packed_decoder_uses_static_geometry_specializations():
    path=Path(__file__).parents[1]/'src/banana_smasher/qtip_kernel_decompress.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode_compressed')
    class Torch:
        def compile(self,func=None,**kw):
            self.options=kw
            return func if func is not None else lambda f:f
    torch=Torch()
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),{'torch':torch})
    assert torch.options.get('dynamic') is False, 'Mixed K2/K3 must not generalize packed geometry into symbolic strides'
