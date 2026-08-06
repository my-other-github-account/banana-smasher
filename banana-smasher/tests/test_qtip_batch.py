from __future__ import annotations

import ast
from pathlib import Path
import types

import torch

from banana_smasher.qtip_batch import block_ldl_batch, ldlq_batch


def test_current_k2_cross_unit_cpu_shapes_and_unit_axis_match() -> None:
    generator = torch.Generator().manual_seed(20260804)
    factors = torch.randn((2, 16, 16), generator=generator)
    hessians = factors @ factors.transpose(1, 2) + torch.eye(16) * 0.25
    batched_lower = block_ldl_batch(hessians, 16)
    serial_lower = torch.stack(
        [block_ldl_batch(hessian.unsqueeze(0), 16)[0] for hessian in hessians]
    )
    assert batched_lower.shape == (2, 16, 16)
    assert torch.allclose(batched_lower, serial_lower, atol=2e-5, rtol=2e-5)

    class IdentityCodebook:
        idx_dtype = torch.int32

        def quantize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            states = torch.zeros((values.shape[0], 128), dtype=torch.int32)
            return values.clone(), states

    weights = torch.randn((2, 16, 128), generator=generator)
    lower = torch.zeros((2, 128, 128))
    args = types.SimpleNamespace(td_x=16, td_y=16, V=2)
    quantized, states = ldlq_batch(
        weights, lower, IdentityCodebook(), args, buf_cols=128
    )
    serial = [
        ldlq_batch(
            weights[unit : unit + 1],
            lower[unit : unit + 1],
            IdentityCodebook(),
            args,
            buf_cols=128,
        )
        for unit in range(2)
    ]

    assert quantized.shape == weights.shape
    assert states.shape == (2, 16, 64)
    assert torch.equal(quantized, torch.cat([row[0] for row in serial]))
    assert torch.equal(states, torch.cat([row[1] for row in serial]))


def test_ldlq_batch_advances_multi_buffer_width_without_cross_chunk_bmm() -> None:
    class IdentityCodebook:
        idx_dtype = torch.int32

        def quantize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            states = torch.zeros((values.shape[0], 128), dtype=torch.int32)
            return values.clone(), states

    generator = torch.Generator().manual_seed(821)
    weights = torch.randn((2, 16, 256), generator=generator)
    lower = torch.zeros((2, 256, 256))
    args = types.SimpleNamespace(td_x=16, td_y=16, V=2)

    quantized, states = ldlq_batch(
        weights, lower, IdentityCodebook(), args, buf_cols=128
    )

    assert torch.equal(quantized, weights)
    assert states.shape == (2, 16, 128)


def test_batch_builder_releases_full_matrices_before_canonical_pack() -> None:
    import banana_smasher.qtip_batch as batch

    tree = ast.parse(Path(batch.__file__).read_text())
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_qtip_batch"
    )
    deletes = {
        target.id: node.lineno
        for node in ast.walk(build)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    pack_branch = next(
        node.lineno
        for node in ast.walk(build)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "callable"
    )

    assert deletes["lower"] < pack_branch
    assert deletes["transformed"] < pack_branch
    assert deletes["quantized"] < pack_branch


def test_batch_matrix_lifetime_receipt_is_batch_aware() -> None:
    from banana_smasher.qtip_batch import _BatchMatrixLifetime

    lifetime = _BatchMatrixLifetime((2, 16, 128))
    transformed = torch.zeros((2, 16, 128), dtype=torch.float32)
    lower = torch.zeros((2, 128, 128), dtype=torch.float32)
    quantized = torch.zeros_like(transformed)
    lifetime.observe("ldl_ready", transformed=transformed, lower=lower)
    lifetime.observe("ldlq_ready", quantized=quantized)
    lifetime.observe("quantized_released_before_pack")
    receipt = lifetime.receipt()

    assert receipt["schema"] == "banana-smasher-qtip-batch-matrix-lifetime-v1"
    assert receipt["batch_units"] == 2
    assert receipt["released_before_pack"] == ["lower", "transformed", "quantized"]
    assert receipt["max_live_fp32_matrix_equivalents"] == 18.0
    assert receipt["events"][-1]["phase"] == "quantized_released_before_pack"


def test_accelerated_main_many_contains_no_serial_fallback() -> None:
    import inspect

    from banana_smasher.solver_qtip_profile import main_many

    tree = ast.parse(inspect.getsource(main_many))
    accelerated = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "batch_size > 1"
    )
    accelerated_source = "\n".join(ast.unparse(node) for node in accelerated.body)
    assert "main(path, root, layer" not in accelerated_source
    assert "len(chunk) == 1" not in accelerated_source


def test_aggregate_receipt_names_every_active_batch_acceleration() -> None:
    from banana_smasher.qtip_batch_controller import _ACTIVE_BUILD_ACCELERATIONS

    assert _ACTIVE_BUILD_ACCELERATIONS == (
        "persistent-prefix-full16",
        "kernel-cache",
        "shared-capture/single-process-staging",
        "batched-block-LDL",
        "cross-unit-LDLQ",
        "FWHT",
        "bounded-batch-matrix-lifetime",
        "canonical-pack-from-states",
        "packed-byte-reconstruction",
    )
