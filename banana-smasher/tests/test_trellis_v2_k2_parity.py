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
        and target.id in {"BRANCHES", "PREFIXES", "STATES", "STEPS"}
    }
    assert assignments == {
        "BRANCHES": 16,
        "PREFIXES": 4096,
        "STATES": 65536,
        "STEPS": 128,
    }
    assert '"branch_sampling": "full"' in exact_source
    assert "alternating-parity-full" not in exact_source


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
    kdict_spec = importlib.util.spec_from_file_location(
        "cornell_qtip_kdict", codebook_root / "kdict.py"
    )
    assert kdict_spec is not None and kdict_spec.loader is not None
    kdict = importlib.util.module_from_spec(kdict_spec)
    kdict_spec.loader.exec_module(kdict)
    lib_stub = types.ModuleType("lib")
    codebook_stub = types.ModuleType("lib.codebook")
    setattr(codebook_stub, "kdict", kdict)
    setattr(lib_stub, "codebook", codebook_stub)
    saved = {name: sys.modules.get(name) for name in ("lib", "lib.codebook")}
    sys.modules["lib"] = lib_stub
    sys.modules["lib.codebook"] = codebook_stub
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
    codebook = bitshift.bitshift_codebook(L=16, K=2, V=2, tlut=tlut)
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
