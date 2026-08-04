from __future__ import annotations

import importlib.util
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = PLUGIN / "src/banana_smasher_plugin"
MATRIX = PACKAGE / "specialized_kernel_matrix.json"
VARIANTS = PACKAGE / "specialized_variants.py"

TIERS = {
    "qtip2_2.0117": ("qtip2", None),
    "qtip3_3.0117": ("qtip3", None),
    "d4_k1024": ("d4", 10),
    "d4_k2048": ("d4", 11),
    "d4_k4096": ("d4", 12),
    "native_mxfp4": ("native_mxfp4", None),
}
PROJECTIONS = {"fused13": 4096, "down": 2048}
VARIANT_TOKENS = {
    "decode_c1": 1,
    "decode_c2": 2,
    "decode_c4": 4,
    "decode_c8": 8,
    "decode_c16": 16,
    "prefill_bm16": 32,
    "prefill_large": 64,
    "prefill_exact_2k": 2048,
}
REQUIRED_FIELDS = {
    "tier",
    "family",
    "index_bits",
    "projection",
    "input_k",
    "output_n",
    "variant",
    "tokens",
    "route_rows",
    "graph_replay",
    "source",
    "source_symbol",
    "build_target",
    "package_member",
    "operator_schema",
    "registration_order",
    "flags_defaults",
    "transforms_reformats",
    "workspace",
    "tuning",
    "warmup",
    "counter",
    "expected_physical_proof",
}


def _load_variants() -> Any:
    spec = importlib.util.spec_from_file_location("banana_smasher_specialized_variants", VARIANTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_exhaustively_binds_every_admitted_tier_projection_and_shape() -> None:
    matrix = json.loads(MATRIX.read_text())
    assert matrix["schema"] == "banana-smasher-specialized-kernel-matrix-v1"
    rows = matrix["rows"]
    expected_keys = set(product(TIERS, PROJECTIONS, VARIANT_TOKENS))
    actual_keys = {(row["tier"], row["projection"], row["variant"]) for row in rows}
    assert actual_keys == expected_keys
    assert len(rows) == len(expected_keys) == 96

    counters: set[str] = set()
    for row in rows:
        assert REQUIRED_FIELDS <= row.keys()
        family, bits = TIERS[row["tier"]]
        assert row["family"] == family
        assert row["index_bits"] == bits
        assert row["input_k"] == PROJECTIONS[row["projection"]]
        assert row["output_n"] == 4096
        assert row["tokens"] == VARIANT_TOKENS[row["variant"]]
        assert row["route_rows"] == row["tokens"] * 6
        assert row["graph_replay"] is row["variant"].startswith("decode_")
        assert row["build_target"] == "banana_smasher_plugin._v4_moe"
        assert row["package_member"] == "banana_smasher_plugin/_v4_moe*.so"
        assert row["counter"]["name"] not in counters
        counters.add(row["counter"]["name"])
        assert row["counter"]["index"] >= 32
        assert row["expected_physical_proof"]["counter_nonzero"] == row["counter"]["name"]
        assert row["expected_physical_proof"]["forbidden_counters_zero"] == [
            "mixed_exact_gemv",
            "p1016_generic",
            "triton_fallback",
        ]
        source_symbol = row["source_symbol"]
        if row["family"].startswith("qtip"):
            assert row["family"] in source_symbol
            assert f"k{row['input_k']}" in source_symbol
            assert row["variant"] in source_symbol
        elif row["family"] == "d4":
            assert source_symbol.startswith("d4_specialized_")
            assert f"<{row['index_bits']},{row['variant_id']},{row['input_k']}>" in source_symbol
        else:
            assert source_symbol == (
                f"mxfp4_specialized_kernel<{row['variant_id']},{row['input_k']}>"
            )


def test_specialization_selector_is_exact_for_decode_bm16_large_and_2k() -> None:
    variants = _load_variants()
    for tier, projection, variant in product(TIERS, PROJECTIONS, VARIANT_TOKENS):
        row = variants.specialization_for(tier, projection, VARIANT_TOKENS[variant])
        assert row["variant"] == variant
        assert row["tier"] == tier
        assert row["projection"] == projection
    assert variants.specialization_for("qtip2_2.0117", "fused13", 8192)["variant"] == "prefill_large"


def test_product_sources_do_not_use_forbidden_generic_or_zero_offset_routes() -> None:
    native_planes = (PACKAGE / "native_planes.py").read_text()
    acceleration = (PACKAGE / "v4_acceleration.py").read_text()
    extensions = (PACKAGE / "native_extensions.py").read_text()
    product_source = "\n".join((native_planes, acceleration, extensions))
    assert "p1016_kernels" not in product_source
    assert "torch.zeros(256" not in product_source
    assert "torch.zeros(384" not in product_source
    assert "mixed_exact_gemv" not in product_source
    assert "specialization_for" in acceleration
    assert "specialized_qtip_gemv" in extensions
    assert "specialized_d4_gemm" in extensions
    assert "specialized_mxfp4_gemm" in extensions


def test_prefill_workspaces_are_ephemeral_instead_of_retained_by_every_layer() -> None:
    acceleration = (PACKAGE / "v4_acceleration.py").read_text()
    builder = acceleration.split("def build_device_resident_planes(", 1)[1].split(
        "def mixed_exact_native_gemv(", 1
    )[0]
    dispatch = acceleration.split("def mixed_exact_native_gemv(", 1)[1]

    # Only graph-replayed decode shapes belong in every layer's resident state.
    for shape in ("(6, 1)", "(12, 2)", "(24, 4)", "(48, 4)", "(96, 4)"):
        assert shape in builder
    for route_rows in (192, 384, 12288, 49152):
        assert f"({route_rows}, 16)" not in builder

    # Non-graph prefill shapes allocate for the current call but are not retained
    # by all 43 layers. The CUDA allocator can reuse their storage after outputs
    # leave scope; only graph-replayed decode buffers need stable Python owners.
    assert "compact = compaction.get(key)" in dispatch
    assert "if compact is None:" in dispatch
    assert 'if bool(qtip2_row["graph_replay"]):' in dispatch
    assert "compaction[key] = compact" in dispatch
    assert '"physical_counter_tensors": {}' in builder
    assert 'vq_state["physical_counter_tensors"][key] = physical_counters' in dispatch

    counter_reader = acceleration.split("def physical_counter_tensor(", 1)[1].split(
        "def runtime_sentinel(", 1
    )[0]
    assert 'vq_state["physical_counter_tensors"]' in counter_reader


def test_every_matrix_source_symbol_is_owned_by_compiled_specialized_source() -> None:
    matrix = json.loads(MATRIX.read_text())
    qtip = (PACKAGE / "csrc/qtip/qtip_dynamic_torch.cu").read_text()
    qtip_kernel = (PACKAGE / "csrc/qtip/inference_dynamic.cu").read_text()
    wrapper = (PACKAGE / "csrc/qtip/wrapper.cpp").read_text()
    vq = (PACKAGE / "csrc/vq_warp_gemv.cu").read_text()
    acceleration = (PACKAGE / "v4_acceleration.py").read_text()

    for row in matrix["rows"]:
        source = qtip if row["family"].startswith("qtip") else vq
        if row["family"].startswith("qtip"):
            assert row["source_symbol"] in source, row["source_symbol"]
            assert row["source_symbol"] in wrapper
        elif row["family"] == "d4":
            assert row["source_symbol"].split("<", 1)[0] in source
        else:
            assert "mxfp4_specialized_kernel" in source

    assert "qtip_trellis_tlut_kernel" in qtip_kernel
    assert "d4_specialized" in vq
    assert "mxfp4_specialized" in vq
    assert "physical_counters.numel() >= 128" in vq
    assert '"physical_counters": torch.zeros(128' in acceleration


def test_d4_tier_specialization_does_not_change_mxfp4_launch_arity() -> None:
    vq = (PACKAGE / "csrc/vq_warp_gemv.cu").read_text()
    mxfp4 = vq.split("at::Tensor mxfp4_specialized(", 1)[1]

    assert "auto launch = [&](auto variant_tag, auto expected_k_tag)" in mxfp4
    assert "auto launch = [&](auto index_bits_tag" not in mxfp4


def test_native_extension_preflight_rejects_partial_registered_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = PACKAGE / "native_extensions.py"
    spec = importlib.util.spec_from_file_location("banana_smasher_native_extensions", path)
    assert spec is not None and spec.loader is not None
    native_extensions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_extensions)

    partial_extension = SimpleNamespace(qtip2_k4096_decode_c1=lambda: None)
    partial_ops = SimpleNamespace(compact_routes=lambda: None)
    monkeypatch.setattr(native_extensions, "_module", lambda: partial_extension)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(banana_smasher_v4=partial_ops)),
    )

    with pytest.raises(RuntimeError, match="missing required exports"):
        native_extensions.preflight_native_extensions()
