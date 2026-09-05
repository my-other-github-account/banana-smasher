"""CPU source/receipt regressions; these do not certify quantized quality."""

import types

import pytest
import torch

from banana_smasher.qtip_batch import build_qtip_batch


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_batch_receipt_uses_actual_codebook_geometry(k):
    class Codebook:
        L, K, V = 16, k, 2
        tlut_bits, decode_mode = 9, "test"
        idx_dtype = torch.int32
        lut = torch.ones(2, 2)
        tlut = torch.ones(2, 2)

        def quantize(self, values, **kwargs):
            return values.clone(), torch.zeros(
                (values.shape[0], 128), dtype=torch.int32
            )

    runner = types.SimpleNamespace(
        fwht=lambda x: x,
        build_hessian=lambda windows, su, device: (torch.eye(len(su)), 1, 1.0),
        pack_kernel_layout=lambda cb, states, rows, width: (
            torch.zeros(16, dtype=torch.int16),
            {"canonical_pack_roundtrip_exact": True},
        ),
    )
    kernel = types.SimpleNamespace(
        decode_compressed=lambda L, bits, K, V, rows, width, packed, lut: torch.ones(
            rows, width
        )
    )
    candidates, receipt = build_qtip_batch(
        runner,
        [torch.ones(16, 128)],
        [[]],
        Codebook(),
        kernel,
        torch.device("cpu"),
        [1],
    )
    assert receipt["solver_geometry"]["K"] == k
    assert (
        receipt["implementation"] == f"current-k{k}-full16-cross-unit-batched-ldlq-v1"
    )
    assert all(row["geometry"]["K"] == k for row in receipt["packed_decode"])
    assert candidates[0]["geometry"]["K"] == k


@pytest.mark.parametrize("projection", ["fused13", "down"])
def test_public_solver_loads_glm_suffix_with_split_fp8_scales(tmp_path, projection):
    import hashlib
    import json
    from safetensors.torch import save_file
    from banana_smasher.solver_qtip_profile import _load_weight

    config = {
        "text_config": {
            "hidden_size": 128,
            "moe_intermediate_size": 128,
            "n_routed_experts": 288,
            "num_hidden_layers": 45,
            "first_k_dense_replace": 3,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    prefix = "model.language_model.layers.3.mlp.experts.256."
    weights, scales, mapping = {}, {}, {}
    for name, value in [("gate", 1.0), ("up", 2.0), ("down", 3.0)]:
        key = prefix + name + "_proj.weight"
        weights[key] = torch.full((128, 128), value).to(torch.float8_e4m3fn)
        scales[key + "_scale_inv"] = torch.full((1, 1), 2.0)
        mapping[key] = "weights.safetensors"
        mapping[key + "_scale_inv"] = "scales.safetensors"
    save_file(weights, str(tmp_path / "weights.safetensors"))
    save_file(scales, str(tmp_path / "scales.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    matrix, receipt = _load_weight(tmp_path, 3, 256, projection)
    expected = (
        torch.cat([torch.full((128, 128), 2.0), torch.full((128, 128), 4.0)])
        if projection == "fused13"
        else torch.full((128, 128), 6.0)
    )
    assert torch.equal(matrix, expected)
    assert matrix.dtype == torch.float32 and matrix.is_contiguous()
    for shard in receipt["shards"]:
        assert (
            shard["scale_source"]["sha256"]
            == hashlib.sha256(
                (tmp_path / "scales.safetensors").read_bytes()
            ).hexdigest()
        )


def test_glm_launch_closure_includes_real_runtime_paths_and_rejects_drift(tmp_path):
    from banana_smasher import qtip_runner
    from banana_smasher.glm_qtip_source_adapter import (
        capture_source_closure,
        require_source_closure,
    )

    modules = {}
    for name in ("bitshift", "ldlq", "math_utils", "kernel_decompress"):
        path = tmp_path / (name + ".py")
        path.write_text("# bounded fixture, not deployed runtime\n")
        modules[name] = types.SimpleNamespace(__file__=str(path))
    closure = capture_source_closure(qtip_runner, modules)
    files = closure["files"]
    for name in (
        "solver_qtip_profile",
        "qtip_batch_controller",
        "qtip_batch",
        "qtip_runner",
        "qtip_viterbi",
        "qtip_kernel_cache",
        "hf_moe",
        "glm_qtip_source_adapter",
        "qtip_rings",
        "qtip1",
    ):
        assert name in files
    assert "qtip_rings.json" in files
    assert all("runtime." + name in files for name in modules)
    assert closure["interpreter"]["executable"]
    assert closure["dependencies"]["torch"]
    # Public receipts redact absolute paths; the identity digest must survive it.
    import hashlib
    import json
    from banana_smasher.solver_qtip_profile import _public_receipt

    public = _public_receipt(closure)
    assert (
        public["sha256"]
        == hashlib.sha256(
            json.dumps(
                public["identity"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    require_source_closure(closure["sha256"], closure)
    with pytest.raises(ValueError, match="closure"):
        require_source_closure(None, closure)
    (tmp_path / "bitshift.py").write_text("# changed fixture\n")
    changed = capture_source_closure(qtip_runner, modules)
    with pytest.raises(ValueError, match="closure"):
        require_source_closure(closure["sha256"], changed)


def test_both_solver_entries_enforce_glm_closure_before_reference_load():
    import ast
    import inspect
    from banana_smasher import qtip_batch_controller, solver_qtip_profile

    for function in (solver_qtip_profile.main, qtip_batch_controller.main_batch):
        tree = ast.parse(inspect.getsource(function))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        bind = [
            node.lineno
            for node in calls
            if ast.unparse(node.func).endswith("bind_source_closure")
        ]
        load = [node.lineno for node in calls if ast.unparse(node.func) == "torch.load"]
        assert bind and min(bind) < min(load)


def test_closure_binding_requires_pin_only_for_glm(tmp_path):
    import json
    from banana_smasher.glm_qtip_source_adapter import bind_source_closure

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"layers.3.ffn.experts.0.w1.weight": "f"}})
    )
    assert bind_source_closure(tmp_path, [{}], None, {}) is None
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": "f"
                }
            }
        )
    )
    with pytest.raises(ValueError, match="closure"):
        bind_source_closure(tmp_path, [{}], None, {})


def test_all_solve_receipt_paths_carry_launch_closure():
    import ast
    import inspect
    from banana_smasher import qtip_batch_controller, solver_qtip_profile

    for function in (solver_qtip_profile.main, qtip_batch_controller.main_batch):
        tree = ast.parse(inspect.getsource(function))
        receipts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            and {"solver", "source_weight"}
            <= {key.value for key in node.keys if isinstance(key, ast.Constant)}
        ]
        assert receipts
        for receipt in receipts:
            assert "glm_source_closure" in {
                key.value for key in receipt.keys if isinstance(key, ast.Constant)
            }


def test_closure_rejects_external_weight_loader_monkeypatch(monkeypatch):
    from banana_smasher import solver_qtip_profile, qtip_runner
    from banana_smasher.glm_qtip_source_adapter import capture_source_closure

    monkeypatch.setattr(solver_qtip_profile, "_load_weight", lambda *args: None)
    with pytest.raises(ValueError, match="external.*loader"):
        capture_source_closure(
            qtip_runner,
            {
                name: qtip_runner
                for name in ("bitshift", "ldlq", "math_utils", "kernel_decompress")
            },
        )
