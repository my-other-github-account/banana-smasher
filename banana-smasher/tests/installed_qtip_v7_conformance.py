#!/usr/bin/env python3
"""Installed-wheel QTIP2-V7 API/CLI conformance smoke.

The caller supplies a schema-valid fresh fixture plan and the built wheel. This
script stubs only the unavailable physical producer/wire/runtime boundary; every
public plan, stage, solve, materialization, receipt, resume, and stop-boundary
call is loaded from the installed wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from zipfile import ZipFile

import numpy as np


def _write_cli_process_hook(path: Path, counter: Path) -> None:
    """Inject only unavailable hardware seams into a real installed ``smash`` process."""
    path.write_text(
        textwrap.dedent(
            f'''\
            import hashlib
            import json
            from pathlib import Path

            import numpy as np
            import banana_smasher.backpack_qtip_v7 as v7

            _COUNTER = Path({str(counter)!r})

            def _bump(name):
                values = json.loads(_COUNTER.read_text())
                values[name] += 1
                _COUNTER.write_text(json.dumps(values, sort_keys=True) + "\\n")

            def producer(units, parent_lut, *, output_root):
                _bump("producer")
                results = [{{
                    "layer": unit["layer"],
                    "expert": unit["expert"],
                    "projection": unit["projection"],
                    "packed_codes": b"\\x01\\x02",
                    "suh": np.ones(2, dtype=np.float16),
                    "svh": np.ones(2, dtype=np.float16),
                    "global_scale": 1.0,
                    "decoded": np.asarray(unit["weight"], dtype=np.float32).reshape(-1),
                }} for unit in units]
                return results, {{
                    "method": "qtip2-v7-installed-cli-process-sentinel",
                    "qfn_calls": len(results),
                    "extension_calls": len(results),
                    "cuda_tiles": len(results),
                    "generic_fallback_calls": 0,
                }}

            def wire(*, source_root, lut, layer, output, receipt):
                _bump("wire")
                output.write_bytes(b"0123456789abcdef")
                payload = {{
                    "schema": "banana-smasher-qtip-v7-wire-v1",
                    "status": "PASS",
                    "layer": layer,
                    "wire": str(output),
                    "complete_wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "generic_fallback_calls": 0,
                }}
                receipt.write_text(json.dumps(payload, sort_keys=True) + "\\n")
                return payload

            def account(*, receipts, output, weight_denominator, weight_denominator_label):
                payload = {{
                    "schema": "banana-smasher-qtip-v7-model-accounting-v1",
                    "status": "PASS",
                    "verified_layer_receipts": len(receipts),
                    "qtip_routed_stored_bytes": 16,
                    "stored_wire_bpw": {{"weight_denominator": weight_denominator}},
                }}
                output.write_text(json.dumps(payload, sort_keys=True) + "\\n")
                return payload

            def runtime_decode(*, selected_layers, cells):
                _bump("runtime_decode")
                selected_wire = Path(selected_layers[0]["path"])
                assert selected_wire.read_bytes() == b"0123456789abcdef"
                return np.concatenate([
                    np.asarray(cell["weights"], dtype=np.float32).reshape(-1)
                    for cell in cells
                ])

            def legacy(*_args, **_kwargs):
                _bump("legacy")
                raise AssertionError("legacy packaged loader called")

            v7._produce_native_v7_batch = producer
            v7._materialize_native_v7_layer = wire
            v7._account_native_v7_model = account
            v7.decode_selected_qtip_v7_backpack_weights = runtime_decode
            v7._load_legacy_packaged_unit = legacy
            '''
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.wheel = args.wheel.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()

    required = {
        "banana_smasher/QTIP_V7_DEFAULT_API_SPEC.md",
        "banana_smasher/README.md",
        "banana_smasher/examples/fresh-flash-qtip2-v7.json",
        "banana_smasher/schema/banana-smasher-backpack-plan-v1.schema.json",
    }
    with ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        if required - names:
            raise AssertionError(f"wheel missing {sorted(required - names)}")

    import banana_smasher
    import banana_smasher.backpack_qtip_v7 as v7
    from banana_smasher import BackpackPlan, build_backpack

    package = Path(banana_smasher.__file__).resolve().parent
    if not (package / "QTIP_V7_DEFAULT_API_SPEC.md").is_file():
        raise AssertionError("installed normative spec missing")
    if not (package / "examples/fresh-flash-qtip2-v7.json").is_file():
        raise AssertionError("installed example missing")

    calls = {"producer": 0, "wire": 0, "runtime_decode": 0, "legacy": 0}

    def producer(units, parent_lut, *, output_root):
        calls["producer"] += 1
        results = [
            {
                "layer": unit["layer"],
                "expert": unit["expert"],
                "projection": unit["projection"],
                "packed_codes": b"\x01\x02",
                "suh": np.ones(2, dtype=np.float16),
                "svh": np.ones(2, dtype=np.float16),
                "global_scale": 1.0,
                "decoded": np.asarray(unit["weight"], dtype=np.float32).reshape(-1),
            }
            for unit in units
        ]
        return results, {
            "method": "qtip2-v7-installed-conformance-sentinel",
            "qfn_calls": len(results),
            "extension_calls": len(results),
            "cuda_tiles": len(results),
            "generic_fallback_calls": 0,
        }

    def wire(*, source_root, lut, layer, output, receipt):
        calls["wire"] += 1
        output.write_bytes(b"0123456789abcdef")
        payload = {
            "schema": "banana-smasher-qtip-v7-wire-v1",
            "status": "PASS",
            "layer": layer,
            "wire": str(output),
            "complete_wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "generic_fallback_calls": 0,
        }
        receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def legacy(*_args, **_kwargs):
        calls["legacy"] += 1
        raise AssertionError("legacy packaged loader called")

    def account(*, receipts, output, weight_denominator, weight_denominator_label):
        payload = {
            "schema": "banana-smasher-qtip-v7-model-accounting-v1",
            "status": "PASS",
            "verified_layer_receipts": len(receipts),
            "qtip_routed_stored_bytes": 16,
            "stored_wire_bpw": {"weight_denominator": weight_denominator},
        }
        output.write_text(json.dumps(payload) + "\n")
        return payload

    def runtime_decode(*, selected_layers, cells):
        calls["runtime_decode"] += 1
        selected_wire = Path(selected_layers[0]["path"])
        assert selected_wire.read_bytes() == b"0123456789abcdef"
        return np.concatenate(
            [np.asarray(cell["weights"], dtype=np.float32).reshape(-1) for cell in cells]
        )

    v7._produce_native_v7_batch = producer
    v7._materialize_native_v7_layer = wire
    v7._account_native_v7_model = account
    v7.decode_selected_qtip_v7_backpack_weights = runtime_decode
    v7._load_legacy_packaged_unit = legacy

    plan_document = json.loads(args.plan.read_text())
    parsed = BackpackPlan.from_mapping(plan_document, base_dir=args.plan.parent)
    python_root = args.run_root / "python"
    cli_root = args.run_root / "cli"
    if args.run_root.exists():
        shutil.rmtree(args.run_root)
    python_result = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    hook_root = args.run_root / "cli-process-hook"
    hook_root.mkdir(parents=True)
    cli_calls_path = hook_root / "calls.json"
    cli_calls_path.write_text(
        json.dumps({"producer": 0, "wire": 0, "runtime_decode": 0, "legacy": 0})
        + "\n"
    )
    _write_cli_process_hook(hook_root / "sitecustomize.py", cli_calls_path)
    smash = Path(sys.executable).with_name("smash")
    if not smash.is_file():
        raise AssertionError(f"installed smash executable missing: {smash}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(hook_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    cli_command = [
        str(smash), "backpack", "build", "--plan", str(args.plan),
        "--run-root", str(cli_root), "--through", "pre-repair-anchor",
    ]
    cli_process = subprocess.run(
        cli_command,
        cwd=args.run_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if cli_process.returncode != 0:
        raise AssertionError(
            f"installed smash process returned {cli_process.returncode}: {cli_process.stderr}"
        )
    cli_result = json.loads(cli_process.stdout)
    for result in (python_result, cli_result):
        assert result["method"] == "qtip2-v7-native"
        assert result["runtime_family"] == "qtip2_v7"
        assert result["producer_calls"] == 1 and result["wire_calls"] == 1
        assert min(result["qfn_calls"], result["extension_calls"], result["cuda_tiles"]) > 0
        assert result["legacy_packaged_loader_calls"] == 0
        assert result["generic_fallback_calls"] == 0
        assert result["runtime_wire_decode_calls"] == 1
        assert result["through"] == "pre_repair_anchor"
        assert result["completed_stage"] == "pre_repair_anchor"
        assert result["repair_executed"] is False
        assert result["plan_sha256"]
        assert result["selected_assignment_sha256"]
        assert result["selected_pack_manifest_sha256"]
        assert result["stored_wire_bytes"] > 0
        assert result["weight_denominator"] > 0
    cli_calls = json.loads(cli_calls_path.read_text())
    assert calls == {"producer": 1, "wire": 1, "runtime_decode": 1, "legacy": 0}
    assert cli_calls == {"producer": 1, "wire": 1, "runtime_decode": 1, "legacy": 0}
    combined_calls = {
        name: calls[name] + cli_calls[name]
        for name in ("producer", "wire", "runtime_decode", "legacy")
    }
    assert not (python_root / "stages/07-repair.json").exists()
    assert not (cli_root / "stages/07-repair.json").exists()
    status_process = subprocess.run(
        [str(smash), "backpack", "status", "--run-root", str(cli_root)],
        cwd=args.run_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if status_process.returncode != 0:
        raise AssertionError(
            f"installed smash status returned {status_process.returncode}: {status_process.stderr}"
        )
    status_result = json.loads(status_process.stdout)
    assert status_result["completed_stages"] == 6
    assert status_result["first_incomplete_stage"] == "repair"
    print(json.dumps({
        "status": "PASS",
        "installed_module": str(package / "__init__.py"),
        "wheel": str(args.wheel.resolve()),
        "method": python_result["method"],
        "python_cli_identity_equal": python_result["method"] == cli_result["method"],
        "calls": combined_calls,
        "installed_cli_command": cli_command,
        "repair_executed": False,
        "completed_stages": status_result["completed_stages"],
        "first_incomplete_stage": status_result["first_incomplete_stage"],
        "plan_sha256": python_result["plan_sha256"],
        "selected_assignment_sha256": python_result["selected_assignment_sha256"],
        "selected_pack_manifest_sha256": python_result["selected_pack_manifest_sha256"],
        "stored_wire_bytes": python_result["stored_wire_bytes"],
        "weight_denominator": python_result["weight_denominator"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
