from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest


def _package_root() -> Path:
    spec = importlib.util.find_spec("banana_smasher")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations)))


def test_k2_exact_contract_enumerates_all_canonical_branches() -> None:
    exact_source = (_package_root() / "trellis_v2" / "exact.py").read_text()
    module = ast.parse(exact_source)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"BRANCHES", "PREFIXES", "STATES", "STEPS", "MAX_CHUNK"}
    }
    assert assignments == {
        "BRANCHES": 16,
        "PREFIXES": 4096,
        "STATES": 65536,
        "STEPS": 128,
        "MAX_CHUNK": 8192,
    }
    assert '"branch_sampling": "full"' in exact_source
    assert '"backpointer_dtype": "packed-uint4-q"' in exact_source
    assert '"minimum_ctas_per_sm": 2' in exact_source
    assert '"fallback": 0' in exact_source
    assert "alternating-parity-full" not in exact_source
    assert "triton" not in exact_source
    assert "prepare_exact_cuda" in exact_source
    cuda_source = (
        _package_root() / "trellis_v2" / "csrc" / "trellis_v2_exact.cu"
    ).read_text()
    assert "__launch_bounds__(THREADS, 2)" in cuda_source


@pytest.mark.skipif(
    not os.environ.get("BANANA_SMASHER_QTIP_CANONICAL_ROOT"),
    reason="set BANANA_SMASHER_QTIP_CANONICAL_ROOT for Cornell parity gate",
)
def test_full16_assignments_match_cornell_canonical_with_and_without_overlap() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("Cornell K2 assignment parity is a required CUDA gate, not a skip")

    canonical_root = Path(os.environ["BANANA_SMASHER_QTIP_CANONICAL_ROOT"])
    codebook_root = canonical_root / "lib" / "codebook"

    def unused_dependency(*_args, **_kwargs):
        raise AssertionError("Cornell Viterbi parity unexpectedly used a decoder dependency")

    stubs = {
        "lib": types.ModuleType("lib"),
        "lib.codebook": types.ModuleType("lib.codebook"),
        "lib.utils": types.ModuleType("lib.utils"),
        "lib.utils.kernel_check": types.ModuleType("lib.utils.kernel_check"),
        "lib.utils.kernel_decompress": types.ModuleType("lib.utils.kernel_decompress"),
        "lib.utils.matmul_had": types.ModuleType("lib.utils.matmul_had"),
    }
    setattr(stubs["lib.codebook"], "kdict", {})
    setattr(stubs["lib.utils.kernel_check"], "has_kernel", unused_dependency)
    setattr(stubs["lib.utils.kernel_decompress"], "decode_compressed", unused_dependency)
    setattr(stubs["lib.utils.matmul_had"], "matmul_hadU_cuda", unused_dependency)
    setattr(stubs["lib.utils.matmul_had"], "matmul_hadUt_cuda", unused_dependency)
    setattr(stubs["lib"], "codebook", stubs["lib.codebook"])
    setattr(stubs["lib"], "utils", stubs["lib.utils"])
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        bitshift_spec = importlib.util.spec_from_file_location(
            "cornell_qtip_bitshift", codebook_root / "bitshift.py"
        )
        assert bitshift_spec is not None and bitshift_spec.loader is not None
        bitshift = importlib.util.module_from_spec(bitshift_spec)
        bitshift_spec.loader.exec_module(bitshift)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    from banana_smasher.trellis_v2.exact import trellis_v2_exact

    generator = torch.Generator(device="cuda").manual_seed(20260803)
    tlut = torch.randn((1 << 16, 2), generator=generator, device="cuda")
    codebook = bitshift.bitshift_codebook(L=16, K=2, V=2, tlut=tlut).to(device="cuda")
    x = torch.randn((256, 1), generator=generator, device="cuda", dtype=torch.float16)

    parity_rows = []
    for overlap in (None, torch.tensor([1731], device="cuda", dtype=torch.int32)):
        expected = codebook.viterbi(x, overlap=overlap)
        actual = trellis_v2_exact(codebook, x, overlap=overlap)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assignment_sha256 = hashlib.sha256(
            actual.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        parity_rows.append(
            {
                "overlap": None if overlap is None else int(overlap.item()),
                "assignment_sha256": assignment_sha256,
                "steps": int(actual.shape[0]),
                "sequences": int(actual.shape[1]),
            }
        )

    receipt_path = os.environ.get("BANANA_SMASHER_QTIP_PARITY_RECEIPT")
    if receipt_path:
        receipt = {
            "schema": "banana-smasher-qtip2-k2-cornell-parity-v1",
            "status": "PASS",
            "geometry": {"L": 16, "K": 2, "V": 2},
            "retained_prefix_costs": 4096,
            "branches_per_prefix": 16,
            "branch_sampling": "full",
            "cases": parity_rows,
        }
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
