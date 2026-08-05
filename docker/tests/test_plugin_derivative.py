from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker/scripts/apply_plugin_derivative.py"
BUILDER = ROOT / "docker/scripts/build_plugin_derivative.py"
DOCKERFILE = ROOT / "docker/Dockerfile.plugin-derivative"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Derivative Test",
            "-c",
            "user.email=derivative@example.invalid",
            "commit",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return git(repo, "rev-parse", "HEAD")


def test_change_classifier_has_an_explicit_pure_plugin_allowlist() -> None:
    module = load_module(SCRIPT, "apply_plugin_derivative_classifier")
    assert module.classify_change(
        "M", "banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py"
    ) == ("runtime", "banana_smasher_plugin/native_planes.py")
    assert module.classify_change(
        "A", "banana-smasher-plugin/src/banana_smasher_plugin/runtime_defaults.json"
    ) == ("runtime", "banana_smasher_plugin/runtime_defaults.json")
    assert module.classify_change(
        "M", "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy"
    ) == ("runtime", "banana_smasher_plugin/qtip_tlut.npy")
    assert module.classify_change(
        "M", "banana-smasher-plugin/tests/test_native_plane_runtime.py"
    ) == ("test", None)
    assert module.classify_change("M", "provenance/SOURCE_INVENTORY.json") == (
        "provenance",
        None,
    )

    forbidden = (
        "banana-smasher-plugin/src/banana_smasher_plugin/csrc/route_compaction.cu",
        "banana-smasher-plugin/src/banana_smasher_plugin/_v4_moe.so",
        "banana-smasher-plugin/pyproject.toml",
        "banana-smasher-plugin/setup.py",
        "docker/Dockerfile",
        "runtime/ASSET_MANIFEST.json",
        "banana-smasher/kernels/cubins-sm120/qtip.cubin",
    )
    for path in forbidden:
        with pytest.raises(RuntimeError, match="DERIVATIVE_CHANGE_REFUSE"):
            module.classify_change("M", path)
    with pytest.raises(RuntimeError, match="DERIVATIVE_STATUS_REFUSE"):
        module.classify_change("D", "banana-smasher-plugin/src/banana_smasher_plugin/old.py")


def test_manifest_is_git_object_bound_and_rejects_native_changes(tmp_path: Path) -> None:
    module = load_module(SCRIPT, "apply_plugin_derivative_manifest")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    runtime = repo / "banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py"
    test = repo / "banana-smasher-plugin/tests/test_native_plane_runtime.py"
    inventory = repo / "provenance/SOURCE_INVENTORY.json"
    for path, data in (
        (runtime, b"VALUE = 1\n"),
        (test, b"def test_value(): assert True\n"),
        (inventory, b'{"schema":"inventory"}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    base = commit_all(repo, "base")
    runtime.write_bytes(b"VALUE = 2\n")
    test.write_bytes(b"def test_value(): assert 2 == 2\n")
    inventory.write_bytes(b'{"schema":"inventory","updated":true}\n')
    target = commit_all(repo, "pure plugin")
    parent = "sha256:" + "1" * 64
    manifest, payloads = module.build_manifest(repo, base, target, parent)

    assert manifest["schema"] == "banana-smasher-plugin-derivative-v1"
    assert manifest["base_commit"] == base
    assert manifest["source_commit"] == target
    assert manifest["source_tree"] == git(repo, "rev-parse", f"{target}^{{tree}}")
    assert manifest["parent_image_id"] == parent
    assert [row["role"] for row in manifest["changes"]] == [
        "runtime",
        "test",
        "provenance",
    ]
    runtime_row = manifest["runtime_assets"][0]
    assert runtime_row["distribution_path"] == "banana_smasher_plugin/native_planes.py"
    assert runtime_row["sha256"] == hashlib.sha256(b"VALUE = 2\n").hexdigest()
    assert payloads[runtime_row["source_path"]] == b"VALUE = 2\n"

    native = repo / "banana-smasher-plugin/src/banana_smasher_plugin/csrc/new.cu"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text("// native change\n")
    forbidden_target = commit_all(repo, "native")
    with pytest.raises(RuntimeError, match="DERIVATIVE_CHANGE_REFUSE"):
        module.build_manifest(repo, target, forbidden_target, parent)


class FakeDistribution:
    def __init__(self, site: Path, dist_info: Path):
        self.site = site
        self.dist_info = dist_info
        self.version = "0.2.0"

    def locate_file(self, member: object) -> Path:
        return self.site / str(member)

    @property
    def files(self):
        return [Path(f"{self.dist_info.name}/RECORD")]


def test_apply_rewrites_record_provenance_and_removes_pycache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module(SCRIPT, "apply_plugin_derivative_apply")
    site = tmp_path / "site"
    package = site / "banana_smasher_plugin"
    dist_info = site / "banana_smasher_plugin-0.2.0.dist-info"
    provenance = tmp_path / "provenance"
    staged = tmp_path / "assets"
    package.mkdir(parents=True)
    dist_info.mkdir()
    provenance.mkdir()
    (package / "__init__.py").write_bytes(b"\n")
    (package / "native_planes.py").write_bytes(b"VALUE = 1\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/native_planes.pyc").write_bytes(b"stale")
    (dist_info / "RECORD").write_text(
        "banana_smasher_plugin/native_planes.py,,\n"
        "banana_smasher_plugin-0.2.0.dist-info/RECORD,,\n"
    )
    (provenance / "source.json").write_text('{"base_image":"sealed"}\n')
    (provenance / "SOURCE_INVENTORY.json").write_text('{"old":true}\n')
    data = b"VALUE = 2\n"
    source = staged / "banana_smasher_plugin/native_planes.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    inventory = staged / "provenance/SOURCE_INVENTORY.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_bytes(b'{"updated":true}\n')
    manifest = {
        "schema": "banana-smasher-plugin-derivative-v1",
        "base_commit": "1" * 40,
        "source_commit": "2" * 40,
        "source_tree": "3" * 40,
        "parent_image_id": "sha256:" + "4" * 64,
        "runtime_assets": [
            {
                "status": "M",
                "role": "runtime",
                "source_path": "banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py",
                "distribution_path": "banana_smasher_plugin/native_planes.py",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        "source_inventory": {
            "source_path": "provenance/SOURCE_INVENTORY.json",
            "staged_path": "provenance/SOURCE_INVENTORY.json",
            "bytes": len(inventory.read_bytes()),
            "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        },
    }
    manifest["manifest_sha256"] = module.manifest_digest(manifest)
    monkeypatch.syspath_prepend(str(site))
    for imported in tuple(sys.modules):
        if imported == "banana_smasher_plugin" or imported.startswith(
            "banana_smasher_plugin."
        ):
            sys.modules.pop(imported)
    original_import_module = module.importlib.import_module

    def import_module_and_materialize_cache(name: str):
        imported = original_import_module(name)
        cache = package / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "import-created.pyc").write_bytes(b"fresh")
        return imported

    monkeypatch.setattr(module.importlib, "import_module", import_module_and_materialize_cache)
    result = module.apply_derivative(
        manifest,
        staged,
        provenance,
        distribution=FakeDistribution(site, dist_info),
        verify_imports=True,
    )

    assert result["status"] == "PASS_PLUGIN_DERIVATIVE_APPLIED"
    assert (package / "native_planes.py").read_bytes() == data
    assert not (package / "__pycache__").exists()
    rows = list(csv.reader((dist_info / "RECORD").read_text().splitlines()))
    runtime_row = next(row for row in rows if row[0] == "banana_smasher_plugin/native_planes.py")
    expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    assert runtime_row == [
        "banana_smasher_plugin/native_planes.py",
        f"sha256={expected}",
        str(len(data)),
    ]
    source_receipt = json.loads((provenance / "source.json").read_text())
    assert source_receipt["banana_smasher_source_commit"] == "2" * 40
    assert source_receipt["banana_smasher_source_tree"] == "3" * 40
    assert source_receipt["derived_from_image"] == "sha256:" + "4" * 64
    assert source_receipt["plugin_derivative_manifest_sha256"] == manifest[
        "manifest_sha256"
    ]
    assert (provenance / "SOURCE_INVENTORY.json").read_bytes() == inventory.read_bytes()


def test_materialize_context_accepts_temporary_directory_created_by_caller(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    builder = load_module(BUILDER, "build_plugin_derivative_materialize")
    context = tmp_path / "already-created-context"
    context.mkdir()
    dockerfile = tmp_path / "Dockerfile"
    apply_script = tmp_path / "apply.py"
    dockerfile.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    apply_script.write_text("print('ok')\n")
    source_path = "banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py"
    payload = b"VALUE = 2\n"
    manifest = {
        "runtime_assets": [
            {
                "source_path": source_path,
                "distribution_path": "banana_smasher_plugin/native_planes.py",
            }
        ],
        "source_inventory": None,
    }
    builder.materialize_context(
        context,
        tmp_path,
        manifest,
        {source_path: payload},
        dockerfile,
        apply_script,
    )
    assert (context / "assets/banana_smasher_plugin/native_planes.py").read_bytes() == payload


def test_dockerfile_and_builder_preserve_fast_tier_contract() -> None:
    dockerfile = DOCKERFILE.read_text()
    builder = BUILDER.read_text()
    assert "ARG BASE_IMAGE" in dockerfile and "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY assets /tmp/banana-smasher-plugin-derivative/assets" in dockerfile
    assert "COPY manifest.json" in dockerfile
    assert "COPY apply_plugin_derivative.py" in dockerfile
    assert "banana-smasher-plugin/src" not in dockerfile
    assert "--manifest-sha256" in dockerfile
    assert "io.banana-smasher.parent.image" in dockerfile
    assert "io.banana-smasher.source.commit" in dockerfile
    assert "io.banana-smasher.source.tree" in dockerfile
    assert "--no-cache" in builder and "--load" in builder
    assert "docker" in builder and "image" in builder and "inspect" in builder
    assert "build_manifest" in builder
    assert "elapsed_seconds" in builder
    assert "O_EXCL" in builder
    assert "pure plugin" in builder.lower()
    assert "plugin native wheel" in builder.lower()
    assert "full aot" in builder.lower()
