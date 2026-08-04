from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, cast

import pytest


PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = PLUGIN / "src/banana_smasher_plugin"
CSRC = PACKAGE / "csrc"
INVENTORY = PACKAGE / "acceleration_inventory.json"
SETUP = PLUGIN / "setup.py"
PYPROJECT = PLUGIN / "pyproject.toml"
NATIVE_EXTENSIONS = PACKAGE / "native_extensions.py"
ACCELERATION = PACKAGE / "v4_acceleration.py"
NATIVE_PLANES = PACKAGE / "native_planes.py"
POLICY = PACKAGE / "dispatch_policy.py"


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"missing function {name} in {path}")


def test_inventory_is_truthful_about_proved_port_and_historical_blockers() -> None:
    inventory = json.loads(INVENTORY.read_text())
    assert inventory["schema"] == "banana-smasher-v4-acceleration-port-v2"
    assert inventory["basis_commit"] == "038c0d8d169d68756fc309fd8c9aa2eb77169ee9"
    items = {item["id"]: item for item in inventory["items"]}
    assert all(item["status"] in {"ported", "preserved", "blocked"} for item in items.values())
    assert all(item["historical_sources"] for item in items.values())
    assert all(item["test"] for item in items.values())
    assert all(item["sentinel"] for item in items.values())
    assert {name for name, item in items.items() if item["status"] == "blocked"} == {
        "c2_c16_graph_routes",
        "device_family_compaction",
        "direct_packed_large_prefill",
        "norm_shared_expert_fusion",
        "packed_mblock16_prefill",
        "qtip_transform_fusion",
        "routed_fp8_scheduler_fusion",
        "same_image_f521_p1321_acceptance",
        "single_platform_extension",
    }
    assert inventory["fusions"] == [
        "packed_projection",
        "qtip_su_fwht_lookup_wscale_fwht_sv",
        "fused13_projection",
        "silu_and_mul",
        "down_projection",
        "fixed_topk_weighted_reduce",
        "stock_dense_norm_quant",
        "stock_dense_activation_quant",
    ]
    assert "mc2" not in INVENTORY.read_text()
    assert "mc4afrag" not in INVENTORY.read_text()


def test_one_platform_extension_owns_one_canonical_source_tree() -> None:
    setup = SETUP.read_text()
    pyproject = PYPROJECT.read_text()
    assert setup.count("CUDAExtension(") == 1
    assert 'CSRC = Path("src")' in setup
    assert "CSRC = ROOT /" not in setup
    assert '"banana_smasher_plugin._v4_moe"' in setup
    assert '"12.0;12.1+PTX"' in setup
    for relative in (
        "csrc/route_compaction.cu",
        "csrc/qtip_transforms.cu",
        "csrc/vq_warp_gemv.cu",
        "csrc/qtip/inference_dynamic.cu",
        "csrc/qtip/qtip_dynamic_torch.cu",
        "csrc/qtip/wrapper.cpp",
    ):
        assert (PACKAGE / relative).is_file(), relative
    assert '"csrc/*.cu"' in pyproject
    assert '"csrc/qtip/*.cu"' in pyproject
    assert not (PACKAGE / "native/vq_warp_gemv.cu").exists()
    assert not (PACKAGE / "native/qtip").exists()
    assert "native/**/*.cu" not in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject
    assert '"torch==2.11.0"' in pyproject
    assert not (PACKAGE / "vq_warp.py").exists()


def test_prebuilt_boundary_has_no_jit_or_generic_fallback() -> None:
    source = NATIVE_EXTENSIONS.read_text()
    assert 'import_module("banana_smasher_plugin._v4_moe")' in source
    assert "_vq_warp_gemv" not in source
    assert "_qtip_dynamic_kernels" not in source
    assert "torch.utils.cpp_extension" not in source
    assert "fallback" not in source.lower()
    dispatch = _function_source(NATIVE_PLANES, "_load_accelerated_dispatch")
    assert "p1016_kernels" not in dispatch
    assert "reference_dispatch" not in dispatch
    assert "sentinel.get('activated', False)" in dispatch
    sentinel = _function_source(ACCELERATION, "runtime_sentinel")
    assert "'activated': True" in sentinel
    assert "'graph_reuse': True" in sentinel


def test_unified_module_registers_qtip_vq_and_mxfp4_without_process_abort() -> None:
    wrapper = (CSRC / "qtip/wrapper.cpp").read_text()
    qtip = (CSRC / "qtip/qtip_dynamic_torch.cu").read_text()
    inference = (CSRC / "qtip/inference_dynamic.cu").read_text()
    vq = (CSRC / "vq_warp_gemv.cu").read_text()
    assert wrapper.count("PYBIND11_MODULE") == 1
    for symbol in (
        "decompress_matvec_dynamic_2_4096",
        "decompress_matvec_dynamic_3_4096",
        "decompress_matvec_dynamic_2_2048",
        "decompress_matvec_dynamic_3_2048",
    ):
        compact_symbol = symbol.replace("dynamic", "compact")
        assert compact_symbol in wrapper
    assert "DEFINE_COMPACT_QTIP" in qtip
    assert "PYBIND11_MODULE" not in vq
    assert "TORCH_LIBRARY_FRAGMENT(banana_smasher_v4" in vq
    assert "mxfp4_compact" in vq
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in inference
    assert "exit(" not in "\n".join((wrapper, qtip, inference, vq))


def test_physical_route_policy_names_only_evidenced_symbols() -> None:
    namespace: dict[str, object] = {}
    exec(compile(POLICY.read_text(), str(POLICY), "exec"), namespace)
    shape_policy = cast(Any, namespace["shape_policy"])
    expected = {
        6: ("vq_scalar_m1", "vq_warp_gemv_kernel", 1, 4),
        12: ("vq_scalar_m2", "vq_warp_gemv_kernel", 2, 4),
        24: ("vq_vector_m4", "vq_warp_gemm_m4_kernel", 4, 4),
        48: ("vq_vector_m4_x2", "vq_warp_gemm_m4_kernel", 4, 4),
        96: ("vq_vector_m4_x4", "vq_warp_gemm_m4_kernel", 4, 4),
        192: (
            "mixed_packed_mblock16",
            "vq_warp_gemm_m4_kernel",
            16,
            16,
        ),
    }
    for rows, (route, symbol, valid_m, mblock) in expected.items():
        decision = shape_policy(rows)
        assert decision["kernel"] == route
        assert decision["physical_symbol"] == symbol
        assert decision["valid_m"] == valid_m
        assert decision["mblock"] == mblock
        assert decision["activation"] == "active"
        assert decision["graph_reuse"] is (rows <= 96)
    for tokens in (63, 64, 65, 2000, 8192):
        decision = shape_policy(tokens * 6)
        assert decision["kernel"] == "mixed_packed_mblock16"
        assert decision["mblock"] == 16
        assert decision["zero_dequant"] is True
    with pytest.raises(NotImplementedError, match="at most 8192"):
        shape_policy(8193 * 6)


def test_runtime_source_does_not_claim_unsupported_fusions_or_python_compaction() -> None:
    native = NATIVE_PLANES.read_text()
    acceleration = ACCELERATION.read_text()
    assert "BANANA_SMASHER_V4_QTIP_FUSION=true" not in native
    assert "compact_family_routes" not in acceleration
    assert "torch.nonzero" not in acceleration
    assert ".numel():" not in acceleration
    assert "gate_up_down_fusion" not in acceleration
    assert "weighted_scatter_add" not in acceleration


def test_cuda_source_exposes_scalar_vector_and_current_stream_without_split_k() -> None:
    source = (CSRC / "vq_warp_gemv.cu").read_text()
    for marker in (
        "vq_warp_gemv_kernel",
        "vq_warp_gemm_m4_kernel",
        "route_stride >= kRowStride",
        "getCurrentCUDAStream",
        "kWarpsPerBlock = 16",
        "kCodesPerLaneItem = 8",
        "kPadBf16PerItem = 2",
        "ld.global.cg.u32",
        "__ldg",
    ):
        assert marker in source
    for forbidden in ("atomicAdd", "split_k", "cudaStreamDefault"):
        assert forbidden not in source


def test_device_compactor_is_static_graph_safe_and_owns_inactive_rows() -> None:
    source = (CSRC / "route_compaction.cu").read_text()
    assert "compact_routes_cuda" in source
    assert "TORCH_LIBRARY_FRAGMENT(banana_smasher_v4" in source
    assert "expert_id == -1" in source
    assert "out[route * output_width + column] =" in source
    assert "family_block_counts" in source
    assert "block_route_rows" in source
    assert "cudaMemcpy" not in source
    assert "cudaDeviceSynchronize" not in source
    assert "item<" not in source


def test_dispatch_uses_persistent_compaction_buffers_without_python_shape_control() -> None:
    source = _function_source(ACCELERATION, "mixed_exact_native_gemv")
    assert "allocate_compaction_state" in source
    assert "torch.ops.banana_smasher_v4.compact_routes" in source
    assert "family_block_counts" in source
    assert "block_route_rows" in source
    for forbidden in ("torch.nonzero", ".item(", "tolist()", "unique_consecutive"):
        assert forbidden not in source


def test_physical_counters_are_written_by_the_family_kernels() -> None:
    acceleration = ACCELERATION.read_text()
    compactor = (CSRC / "route_compaction.cu").read_text()
    qtip = (CSRC / "qtip/inference_dynamic.cu").read_text()
    vq = (CSRC / "vq_warp_gemv.cu").read_text()

    assert '"physical_counters": torch.zeros(24' in acceleration
    assert '"physical_launches": (10, 14)' in acceleration
    assert '"physical_blocks": (14, 18)' in acceleration
    assert '"physical_rows": (18, 22)' in acceleration
    assert "physical_counters[10 + family]" not in compactor
    assert "physical_counters[22]" in compactor
    assert "record_physical_dispatch" in qtip
    assert qtip.count("physical_counters") >= 8
    assert "record_physical_dispatch" in vq
    assert vq.count("physical_counters") >= 12
    assert acceleration.count('compact["physical_counters"]') == 4


def test_compaction_tensor_ranges_have_explicit_const_pointer_types() -> None:
    source = (CSRC / "route_compaction.cu").read_text()
    assert "#include <array>" in source
    assert source.count("std::array<const at::Tensor*,") == 2
    assert "for (const at::Tensor* tensor : {" not in source
