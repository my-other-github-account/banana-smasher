from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest

import banana_smasher
from banana_smasher import backpack_runtime_exact64 as exact64
from banana_smasher.cli import _parser
from banana_smasher.hf_deepseek_v4_backpack_adapter import (
    _available_materialization_bytes,
)


def test_exact64_oracle_is_private_and_has_no_cli_route() -> None:
    assert not hasattr(banana_smasher, "run_backpack_exact64")

    parser = _parser()
    commands = next(
        action.choices for action in parser._actions if getattr(action, "choices", None)
    )
    assert "backpack-exact64" not in commands

    source = inspect.getsource(exact64._run_backpack_exact64)
    assert "with runtime.layer_stage(layer) as forward:" in source
    assert "mlp_chunk_stage" not in source
    assert "teacher_manifest_path, teacher = _revision_bind_teacher_manifest(" in source
    assert "allow_over_cap=diagnostic_nonshipping" in source
    signature = inspect.signature(exact64._run_backpack_exact64)
    assert "qtip2_v7_root_map_path" in signature.parameters
    assert "qtip2_v7_member_roster_path" in signature.parameters
    assert "mixed_v7_member_contract_path" in signature.parameters
    assert "checkpoint_path" in signature.parameters
    assert "checkpoint_sha256" in signature.parameters
    assert signature.parameters["diagnostic_nonshipping"].default is False
    assert signature.bind_partial(diagnostic_nonshipping=True).arguments == {
        "diagnostic_nonshipping": True
    }
    assert 'binding_inputs["mixed_v7_member_contract"]' in source
    assert "exact64 mixed V7 member contract identity mismatch" in source
    assert "exact64 checkpoint bytes do not match explicit checkpoint SHA" in source


def test_exact64_accepts_only_the_known_run6322_virtual_product(
    tmp_path, monkeypatch
) -> None:
    basis = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
    manifest_sha = "18613b80dbc5c610d2ba252fa169fdca9ec1b5a248c90502ee1515a34d6abc05"
    terminal_sha = "f7b5fab1269ef61881f37fced720e4b21d2dcbbb6c14cad87efcbdc36eb33370"
    virtual = tmp_path / "exact102-virtual"
    virtual.mkdir()
    manifest = virtual / "BACKPACK_VIRTUAL_MANIFEST.json"
    terminal = tmp_path / "EXACT102_VIRTUAL_TERMINAL.json"
    accounting = {"whole_shipping_bytes": 102_000_000_000}
    manifest.write_text(
        json.dumps(
            {
                "status": "PASS_LOGICAL_FULL_WIRE",
                "basis_sha256": basis,
                "whole_model_accounting": accounting,
            }
        )
    )
    terminal.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-mixed-exact102-virtual-terminal-v1",
                "status": "PASS",
                "basis_sha256": basis,
                "virtual_manifest_path": str(manifest),
                "virtual_manifest_sha256": manifest_sha,
                "whole_model_accounting": accounting,
            }
        )
    )
    identities = {manifest: manifest_sha, terminal: terminal_sha}
    monkeypatch.setattr(exact64, "_sha256_file", lambda path: identities[path])

    accepted_manifest, accepted_terminal = exact64._validate_virtual_product_identity(
        virtual_manifest_path=manifest,
        virtual_manifest_sha256=manifest_sha,
        virtual_terminal_path=terminal,
        virtual_terminal_sha256=terminal_sha,
        basis_sha256=basis,
    )

    assert accepted_manifest["whole_model_accounting"] == accounting
    assert accepted_terminal["virtual_manifest_sha256"] == manifest_sha
    rejected_terminal = json.loads(terminal.read_text())
    rejected_terminal["virtual_manifest_sha256"] = "0" * 64
    terminal.write_text(json.dumps(rejected_terminal))
    with pytest.raises(ValueError, match="terminal does not bind"):
        exact64._validate_virtual_product_identity(
            virtual_manifest_path=manifest,
            virtual_manifest_sha256=manifest_sha,
            virtual_terminal_path=terminal,
            virtual_terminal_sha256=terminal_sha,
            basis_sha256=basis,
        )


def test_exact64_rejects_unbound_historical_teacher_manifest(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "teacher.v1.json"
    source_path.write_text(
        json.dumps({"schema": "banana-smasher-anchor-teacher-sidecars-v1"})
    )
    sidecar = tmp_path / "window.pt"
    sidecar.write_bytes(b"old")
    historical = {
        "support_width": 8192,
        "window_ids": ["window-0"],
        "identities": {
            "bank_sha256": "old-bank",
            "teacher_sha256": "old-teacher",
        },
        "windows": [
            {
                "path": sidecar.name,
                "bytes": sidecar.stat().st_size,
                "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(
        exact64,
        "load_teacher_support_manifest",
        lambda *args, **kwargs: historical,
    )

    with pytest.raises(ValueError, match="revision-bound"):
        exact64._revision_bind_teacher_manifest(
            source_path,
            tmp_path / "output",
            bank_sha256="current-bank",
            basis_sha256="current-basis",
            model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        )


def test_exact64_rejects_expert_only_virtual_accounting() -> None:
    with pytest.raises(ValueError, match="standardized whole-model accounting"):
        exact64._validate_whole_model_accounting(
            {
                "byte_accounting": {
                    "assigned_expert_bytes": 101_334_321_540,
                    "fixed_nonexpert_bytes": 8_121_467,
                    "assigned_package_bytes": 101_342_443_007,
                }
            }
        )


def test_exact64_rejects_inconsistent_whole_model_byte_equations() -> None:
    accounting = {
        "expert_physical_wire_bytes": 92_967_887_386,
        "dense_nonrouted_bytes": 9_017_358_064,
        "repair_bytes": 0,
        "metadata_bytes": 14_754_550,
        "fixed_nonexpert_bytes": 9_032_112_614,
        "whole_shipping_bytes": 102_000_000_001,
        "shipping_bytes_cap": 102_000_000_000,
        "shipping_slack_bytes": 0,
        "logical_base_parameters": 284_334_567_511,
        "whole_model_bpw_numerator_bits": 816_000_000_008,
        "whole_model_bpw_exact_ratio": "816000000008/284334567511",
    }

    with pytest.raises(ValueError, match="whole_shipping_bytes"):
        exact64._validate_whole_model_accounting({"whole_model_accounting": accounting})


def test_exact64_rejects_false_whole_model_bpw_decimal() -> None:
    accounting = {
        "expert_physical_wire_bytes": 92_967_887_386,
        "dense_nonrouted_bytes": 9_017_358_064,
        "repair_bytes": 0,
        "metadata_bytes": 14_754_550,
        "fixed_nonexpert_bytes": 9_032_112_614,
        "whole_shipping_bytes": 102_000_000_000,
        "shipping_bytes_cap": 102_000_000_000,
        "shipping_slack_bytes": 0,
        "logical_base_parameters": 284_334_567_511,
        "whole_model_bpw_numerator_bits": 816_000_000_000,
        "whole_model_bpw_exact_ratio": "816000000000/284334567511",
        "whole_model_bpw_decimal": "3.0",
    }

    with pytest.raises(ValueError, match="BPW decimal"):
        exact64._validate_whole_model_accounting({"whole_model_accounting": accounting})


def test_gb10_materialization_uses_reclaimable_unified_memory(tmp_path) -> None:
    gib = 1 << 30

    class FakeCuda:
        @staticmethod
        def mem_get_info():
            return 3 * gib, 120 * gib

        @staticmethod
        def get_device_properties(device):
            assert device == "cuda"
            return SimpleNamespace(name="NVIDIA GB10")

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   118000000 kB\n")

    assert (
        _available_materialization_bytes(
            SimpleNamespace(cuda=FakeCuda), "cuda", meminfo_path=meminfo
        )
        == 118000000 * 1024
    )
