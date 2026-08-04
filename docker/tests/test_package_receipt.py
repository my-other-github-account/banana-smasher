from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_receipt_module():
    script = Path(__file__).parents[1] / "scripts" / "write_package_receipt.py"
    scripts = str(script.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("write_package_receipt_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_receipt_verifies_the_build_source_commit(monkeypatch, tmp_path):
    receipt = _load_receipt_module()
    source_commit = "1" * 40
    observed = {}

    monkeypatch.setenv("BANANA_SMASHER_SOURCE_COMMIT", source_commit)
    monkeypatch.setattr(
        receipt.importlib.util,
        "find_spec",
        lambda name: type("Spec", (), {"origin": "/tmp/banana_smasher_plugin/__init__.py"})(),
    )
    monkeypatch.setattr(
        receipt.importlib.metadata,
        "version",
        lambda name: receipt.EXPECTED_PACKAGES.get(name, "test-version"),
    )

    def verify_provenance(root, actual_source_commit):
        observed["root"] = root
        observed["source_commit"] = actual_source_commit
        return {"manifest_sha256": {}}

    monkeypatch.setattr(receipt, "verify_provenance_manifests", verify_provenance)
    monkeypatch.setattr(receipt, "verify_asset_set", lambda *args: {"status": "PASS"})
    output = tmp_path / "package-sbom.json"
    monkeypatch.setattr(sys, "argv", ["write_package_receipt.py", str(output)])

    receipt.main()

    assert observed["source_commit"] == source_commit
    assert json.loads(output.read_text())["status"] == "PASS"
