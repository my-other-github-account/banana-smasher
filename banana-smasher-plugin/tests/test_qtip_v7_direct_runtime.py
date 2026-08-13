from __future__ import annotations

import ast
from pathlib import Path


PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = PLUGIN / "src/banana_smasher_plugin"
RUNTIME = PACKAGE / "qtip_v7_runtime.py"


def test_qtip_v7_runtime_selects_only_direct_kernel_and_zero_forbidden_state() -> None:
    source = RUNTIME.read_text()
    tree = ast.parse(source)
    runtime_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QtipV7DirectLayer"
    )
    assignments = {
        target.id: node.value.value
        for node in runtime_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert assignments["duplicate_packed_bytes"] == 0
    assert assignments["persistent_decoded_state_bytes"] == 0
    assert assignments["persistent_dense_weight_bytes"] == 0
    assert assignments["generic_fallback_calls"] == 0
    assert "qtip2_v7_direct" in source
    assert "_fault_mapped_pages" in source
    assert "direct_dispatch_calls" in source
    assert "capture_qtip_v7_layer_smoke" in source
    assert "GB10 coherent host memory" in source
    assert "_qtip2_decode_states" not in source
    assert "dequant" not in source.lower()
    assert "packed_view(expert, projection).to(" not in source


def test_cuda_extension_exposes_v7_shapes_and_embedded_lut_alias_path() -> None:
    wrapper = (PACKAGE / "csrc/qtip/wrapper.cpp").read_text()
    qtip = (PACKAGE / "csrc/qtip/qtip_dynamic_torch.cu").read_text()
    transforms = (PACKAGE / "csrc/qtip_transforms.cu").read_text()
    boundary = (PACKAGE / "native_extensions.py").read_text()

    assert "qtip2_v7_direct" in wrapper
    assert "qtip2_v7_direct" in qtip
    assert "HostCodebook" in qtip
    assert "V7_VARIANTS(2048, 4096)" in qtip
    assert "V7_VARIANTS(4096, 2048)" in qtip
    assert "width == 2048 || width == 4096" in transforms
    assert '("qtip2_v7_direct",)' in boundary


def test_v7_counter_indices_cover_every_shape_variant() -> None:
    source = RUNTIME.read_text()
    assert "128 if variant == 8 else 32 + variant" in source
    assert "129 if variant == 8 else 40 + variant" in source