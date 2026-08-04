from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from banana_smasher import cli


def test_qtip_solver_sources_ship_in_public_package() -> None:
    package = Path(cli.__file__).resolve().parent
    expected = {
        "qtip_kernel_cache.py",
        "qtip_materialize.py",
        "qtip_rings.json",
        "qtip_rings.py",
        "qtip_viterbi.py",
        "solver_qtip_profile.py",
    }
    assert expected <= {path.name for path in package.iterdir()}

    solver = ast.parse((package / "solver_qtip_profile.py").read_text())
    functions = {
        node.name for node in solver.body if isinstance(node, ast.FunctionDef)
    }
    assert {"_bind_builder_memory_contract", "_release_capture_bank", "main", "main_many"} <= functions


def test_qtip_extension_builder_declares_runtime_setuptools_dependency() -> None:
    project = Path(__file__).resolve().parents[1]
    metadata = (project / "pyproject.toml").read_text()
    assert '"setuptools==' in metadata


def test_public_qtip_runner_trusts_manifest_sha_not_package_anchor(tmp_path: Path) -> None:
    from hashlib import sha256

    from banana_smasher.solver_qtip_profile import _load_public_qtip_runner

    runner_path = tmp_path / "qtip2_adapter.py"
    runner_path.write_text(
        """from types import ModuleType


def _legacy_pack(cb, states, m, n):
    return states, []


_rate = ModuleType("qtip2_rate")
_rate.pack_kernel_layout_batch = _legacy_pack


def build_qtip(cb, states, m, n):
    return _rate.pack_kernel_layout_batch(cb, states, m, n)
"""
    )
    runner_sha = sha256(runner_path.read_bytes()).hexdigest()

    runner = _load_public_qtip_runner(runner_path, runner_sha)
    assert runner.__file__ is not None
    assert Path(runner.__file__).resolve() == runner_path.resolve()
    assert runner._rate.pack_kernel_layout_batch is not runner._legacy_pack

    with pytest.raises(ValueError, match="public QTIP runner SHA mismatch"):
        _load_public_qtip_runner(runner_path, "0" * 64)


def test_historical_qtip_config_uses_canonical_ring_codebook() -> None:
    from banana_smasher.qtip_rings import resolve_qtip_ring
    from banana_smasher.solver_qtip_profile import _resolve_config_codebook

    config = {"geometry": {"L": 16, "K": 2, "V": 2}}
    codebook = _resolve_config_codebook(config, config["geometry"])

    assert codebook == dict(resolve_qtip_ring("2.00").codebook)
    assert config["codebook"] == codebook
    assert codebook["pack_contract"]["dtype"] == "uint16"


def test_qtip_accelerator_entrypoint_fails_explicitly_without_triton() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["triton"] = None
sys.modules["triton.language"] = None
from banana_smasher import qtip_viterbi
for entrypoint, args in (
    (qtip_viterbi.exact_prefix_viterbi, (object(), object())),
    (qtip_viterbi.install_exact_prefix_viterbi, (object(),)),
):
    try:
        entrypoint(*args)
    except RuntimeError as exc:
        assert "requires the solve extra on a supported platform" in str(exc)
    else:
        raise AssertionError(f"accelerator entrypoint {entrypoint.__name__} did not fail closed")
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_trellis_v2_package_import_is_safe_without_compiled_extension() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ.pop("BANANA_SMASHER_TRELLIS_V2_EXTENSION", None)
os.environ.pop("BANANA_SMASHER_TRELLIS_V2_EXTENSION_SHA256", None)
from banana_smasher import trellis_v2
try:
    trellis_v2.install_trellis_v2(object())
except RuntimeError as exc:
    assert "could not load the SHA-pinned" in str(exc)
else:
    raise AssertionError("unconfigured accelerator entrypoint did not fail closed")
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_streaming_viterbi_recursion_preserves_public_argument_order() -> None:
    package = Path(cli.__file__).resolve().parent
    tree = ast.parse((package / "qtip_viterbi.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "exact_prefix_viterbi"
    )
    recursive_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exact_prefix_viterbi"
    ]
    assert len(recursive_calls) == 1
    call = recursive_calls[0]
    assert ast.unparse(call.args[0]) == "cb"
    assert ast.unparse(call.args[1]) == "x[:, start:end]"


def test_matrix_lifetime_releases_capture_bank_before_source_dequantization() -> None:
    package = Path(cli.__file__).resolve().parent
    tree = ast.parse((package / "solver_qtip_profile.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    capture_deletes = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "captures"
            for target in node.targets
        )
    ]
    assert calls["_prepare_fit_windows"] < calls["_release_capture_bank"]
    assert calls["_release_capture_bank"] < calls["_load_weight"]
    assert capture_deletes == [calls["_release_capture_bank"] + 1]


def test_matrix_lifetime_helpers_are_geometry_bound_and_drop_cached_rows() -> None:
    torch = pytest.importorskip("torch")
    from banana_smasher import solver_qtip_profile as solver

    source = torch.zeros((32, 49), dtype=torch.float32)

    class Codebook:
        K = 3
        V = 2
        idx_dtype = torch.int32

    codebook = Codebook()
    contract = solver._bind_builder_memory_contract(codebook, source)
    expected_states = source.numel() // codebook.V
    assert contract["state_elements"] == expected_states
    assert contract["state_storage_bytes"] == expected_states * 4
    assert contract["retained_output_bytes"] == source.numel() * 4

    capture_root = Path("/tmp/banana-smasher-qtip-memory-lifetime")
    captures = [{"window": 0, "x": object()}]
    cache_key = (capture_root.resolve(), 35, 32)
    solver._CAPTURE_CACHE[cache_key] = captures
    solver._release_capture_bank(capture_root, 35, 32, captures)
    assert cache_key not in solver._CAPTURE_CACHE
    assert captures == []


def test_solve_help_exposes_unified_qtip_surface() -> None:
    parser = cli._parser()
    args = parser.parse_args(
        [
            "solve",
            "--source-root",
            "/tmp/source",
            "--root",
            "/tmp/run",
            "--tier",
            "qtip",
            "--bpw",
            "1.75",
            "--all-cells",
            "--layers",
            "35",
            "--resume",
        ]
    )
    assert args.command == "solve"
    assert args.tier == "qtip"
    assert args.bpw == "1.75"
    assert args.all_cells is True
    assert args.resume is True


def test_public_qtip_solve_routes_materialization_and_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rings = ModuleType("banana_smasher.qtip_rings")
    setattr(rings, "canonical_qtip_tier", lambda bpw: f"qtip@{bpw}")

    materialize = ModuleType("banana_smasher.qtip_materialize")
    setattr(
        materialize,
        "ensure_qtip_configs",
        lambda source, *, tier, layers: {
            "status": "PASS",
            "tier": tier,
            "layers": layers,
        },
    )
    setattr(
        materialize,
        "require_qtip_ring_manifest",
        lambda source, bpw: f"qtip@{bpw}",
    )

    calls: list[dict[str, object]] = []
    solver = ModuleType("banana_smasher.solver_qtip_profile")

    def main_many(source, root, layer, **kwargs):
        calls.append(
            {
                "source": str(source),
                "root": str(root),
                "layer": layer,
                **kwargs,
            }
        )
        return {"status": "PASS", "layer": layer, "resumed_units": 1}

    setattr(solver, "main_many", main_many)
    monkeypatch.setitem(__import__("sys").modules, rings.__name__, rings)
    monkeypatch.setitem(__import__("sys").modules, materialize.__name__, materialize)
    monkeypatch.setitem(__import__("sys").modules, solver.__name__, solver)

    rc = cli.main(
        [
            "solve",
            "--source-root",
            "/tmp/source",
            "--root",
            "/tmp/run",
            "--tier",
            "qtip",
            "--bpw",
            "1.75",
            "--all-cells",
            "--layers",
            "35",
            "--resume",
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "source": "/tmp/source",
            "root": "/tmp/run",
            "layer": 35,
            "tier": "qtip@1.75",
            "all_cells": True,
            "profile_mode": False,
            "resume": True,
            "resume_flag_explicit": True,
        }
    ]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["bpw"] == "1.75"
    assert receipt["layer_receipts"][0]["resumed_units"] == 1
