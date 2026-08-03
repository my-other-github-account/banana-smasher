from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/banana_smasher_plugin/p1016_kernels.py"
)


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _calls(node: ast.AST) -> set[str]:
    result = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            result.add(target.id)
        elif isinstance(target, ast.Attribute):
            result.add(target.attr)
    return result


def test_public_dispatch_uses_shape_policy_not_generic_fused_kernel() -> None:
    public = _function("mixed_exact_gemv")
    calls = _calls(public)
    assert "mixed_shape_aware_gemv" in calls
    assert "_mixed_exact_gemv" not in calls


def test_shape_dispatch_has_static_family_and_chunked_vector_paths() -> None:
    source = SOURCE.read_text()
    assert "from .dispatch_policy import shape_policy" in source
    shape_dispatch = _function("mixed_shape_aware_gemv")
    calls = _calls(shape_dispatch)
    assert "shape_policy" in calls
    assert "mixed_static_gemv" in calls
    rendered = ast.unparse(shape_dispatch)
    subscripts = {
        node.slice.value
        for node in ast.walk(shape_dispatch)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert {"chunks", "chunk_tokens"} <= subscripts
    assert "output[start:stop]" in rendered


def test_static_family_dispatch_writes_one_shared_output_without_dequantization() -> None:
    static_dispatch = _function("mixed_static_gemv")
    calls = _calls(static_dispatch)
    assert {
        "qtip_raw_gemv_dynamic",
        "d4_gemv_dynamic",
        "native_mxfp4_gemv_dynamic",
    } <= calls
    source = ast.unparse(static_dispatch)
    assert "dequant" not in source.lower()
    assert "torch.empty" in source
