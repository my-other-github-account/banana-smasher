from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "notes/GOLDEN_CLOSURE_REUSE_LEDGER.json"
BUILD = ROOT / "banana-smasher/kernels/rebuild.py"
INVENTORY = ROOT / "provenance/SOURCE_INVENTORY.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_kernel_closure_covers_every_checked_in_binary() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert ledger["schema"] == "banana-smasher-golden-source-closure-v1"
    assert set(ledger["families"]) == {"M4", "W2", "W3", "W4", "MLA"}
    assert set(ledger["default_rebuild_families"]) == {"M4", "W2", "W3", "W4"}

    expected = {
        path.relative_to(ROOT).as_posix()
        for cubin_dir in (
            ROOT / "banana-smasher/kernels/cubins-sm120",
            ROOT / "banana-smasher/kernels/cubins-e43",
        )
        for path in cubin_dir.glob("*.cubin")
    }
    artifacts = {entry["path"]: entry for entry in ledger["artifacts"]}
    assert set(artifacts) == expected

    for path_text, entry in artifacts.items():
        path = ROOT / path_text
        source = ROOT / entry["source"]
        family = ledger["families"][entry["family"]]
        assert source.is_file(), f"missing source for {path_text}: {source}"
        assert _sha256(path) == entry["sha256"]
        assert _sha256(source) == entry["source_sha256"]
        assert family["build_command"]
        assert family["toolchain"]

    cubit = "c139df8b34f1dcab607f8ccb685fdea948f3ae4d"
    for name in ("W2", "W3", "W4", "MLA"):
        assert ledger["families"][name]["toolchain"]["cubit_commit"] == cubit
    mla = artifacts["banana-smasher/kernels/cubins-sm120/mla_prefill_state.cubin"]
    assert mla["source_role"] == "binary_derived_comparison_oracle"
    assert ledger["families"]["MLA"]["source_closure"]["status"] == "BLOCKED"
    assert "producing_source" not in ledger["families"]["MLA"]["source_closure"]
    m4 = ledger["families"]["M4"]["toolchain"]
    assert m4["cuda_arch"] == "12.0a"
    assert m4["builder"] == (
        "torch.utils.cpp_extension.CUDAExtension + BuildExtension(use_ninja=True)"
    )


def test_closure_sources_are_public_and_have_exact_origin_bindings() -> None:
    ledger = json.loads(LEDGER.read_text())
    encoded = LEDGER.read_text()
    assert "/Use" + "rs/" not in encoded and "/ho" + "me/" not in encoded
    assert "SPDX" + "-" not in encoded

    for source in (ROOT / "banana-smasher/kernels/sources").rglob("*"):
        if source.is_file():
            text = source.read_text(errors="ignore")
            assert "/work" + "space/" not in text, source

    expected_origins = {
        "M4": {
            "csrc/vq_warp_gemv.cu": "4ff296f7e42e2a906543aa4563fc50e1e04d58e6d41de46cda9bce4ae686fef5",
            "setup.py": "7b921d5b0c7eddb2663c032746a6da1eb49746077a772623e9711f731614e097",
        },
        "W3": {
            "gen_moe_w3.py": "57f34116990d0d2d891e9740434016c500f7137457f994734679c5646363a451",
            "build.sh": "a92e4bb2ad66b580705346a5b8ffcf07832cc47e24995a51f4b198270ebc2d84",
            "qmma_e4m3.merc.stub": "80ae5c0b657a67061d9370d7ce4d9957a94ee54e61bf6643dd82c8352ee005f7",
        },
    }
    for family, origins in expected_origins.items():
        assert ledger["families"][family]["origin_hashes"] == origins

    for family in ledger["families"].values():
        path_text = family["source_root"]
        assert not Path(path_text).is_absolute()
        source_root = ROOT / path_text
        assert source_root.is_dir()
        for source in source_root.rglob("*"):
            if source.is_file() and source.suffix not in {".cubin", ".stub"}:
                assert "SPDX" + "-" not in source.read_text(errors="ignore")

    for family in ledger["families"].values():
        source_root = ROOT / family["source_root"]
        for relative, expected in family.get("retained_hashes", {}).items():
            assert _sha256(source_root / relative) == expected


def test_upstream_native_dependencies_have_source_rebuild_bindings() -> None:
    dependencies = json.loads(LEDGER.read_text())["upstream_native_dependencies"]
    assert dependencies["stock_vllm"]["image"] == (
        "vllm/vllm-openai:v0.24.0@sha256:"
        "32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b"
    )
    assert dependencies["stock_vllm"]["source_commit"] == (
        "ee0da84ab9e04ac7610e28580af62c365e898389"
    )
    assert dependencies["deepgemm"]["source_commit"] == (
        "f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca"
    )
    flashinfer = dependencies["flashinfer"]
    assert flashinfer["version"] == "0.6.17"
    assert flashinfer["source_commit"] == "d020372b068f335e2fe427372e134977a2235c49"
    assert flashinfer["source_build_path"] == "docker/Dockerfile:29-74"

    registration = json.loads(LEDGER.read_text())["runtime_registration"]
    assert registration["family_mapping"] == {
        "D4": "native learned-VQ",
        "MXFP4": "native MXFP4",
        "QTIP2": "native QTIP2 trellis+TLUT",
        "QTIP3": "native QTIP3 trellis+TLUT",
        "scalar W2": "Cubit fragment-major 2-bit planes",
        "scalar W3": "Cubit fragment-major 3-bit planes",
    }
    assert "linux_aarch64" in registration["platform_wheel_example"]
    assert "importlib.resources" in registration["resource_registration_example"]


def test_source_inventory_covers_retained_kernel_sources() -> None:
    inventory = json.loads(INVENTORY.read_text())
    entries = {
        entry["path"]: entry
        for section in ("files", "generated_files")
        for entry in inventory[section]
    }
    retained = ROOT / "banana-smasher/kernels/sources"
    expected = {path.relative_to(ROOT).as_posix() for path in retained.rglob("*") if path.is_file()}
    assert expected <= set(entries)
    expected.add(BUILD.relative_to(ROOT).as_posix())
    for relative in expected:
        path = ROOT / relative
        assert entries[relative]["bytes"] == path.stat().st_size
        assert entries[relative]["output_sha256"] == _sha256(path)


def test_w3_generator_matches_checked_in_reference_sass(tmp_path: Path) -> None:
    family = ROOT / "banana-smasher/kernels/sources/w3"
    generated = tmp_path / "moe_w3_mm_e43_base_k2048.sass"
    env = {
        "MOEW3_MC": "1",
        "MOEW3_AFRAG": "0",
        "MOEW3_LUT_LO": "0xb6bfc6cd",
        "MOEW3_LUT_HI": "0x4d463c21",
    }
    subprocess.run(
        [sys.executable, str(family / "gen_moe_w3.py"), str(generated), "2048"],
        check=True,
        env={**__import__("os").environ, **env},
    )
    reference = family / "reference-sass/moe_w3_mm_e43_base_k2048.sass"
    assert generated.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize(
    "variant,mc,afrag,k,reference_name",
    [
        (entry["variant"], {"base": "1", "mc2": "2", "mc4": "4", "mc4afrag": "4"}[entry["variant"]], "1" if entry["variant"] == "mc4afrag" else "0", entry["k"], Path(entry["reference_sass"]).name)
        for entry in json.loads(LEDGER.read_text())["artifacts"]
        if entry["family"] == "W2"
        and entry.get("reference_sass")
        and entry.get("source_mode") != "retained_reference_sass"
    ],
)
def test_w2_generator_matches_every_checked_in_reference_sass(
    tmp_path: Path,
    variant: str,
    mc: str,
    afrag: str,
    k: int,
    reference_name: str,
) -> None:
    family = ROOT / "banana-smasher/kernels/sources/w2"
    generated = tmp_path / f"{variant}-{k}.sass"
    env = {"MOEW2_MC": mc, "MOEW2_AFRAG": afrag}
    subprocess.run(
        [sys.executable, str(family / "gen/gen_moe_w2.py"), str(generated), str(k)],
        check=True,
        env={**os.environ, **env},
    )
    reference = family / "sass" / reference_name
    assert generated.read_bytes() == reference.read_bytes()


def test_w2_legacy_k2048_uses_retained_sass_without_false_generator_claim() -> None:
    artifacts = json.loads(LEDGER.read_text())["artifacts"]
    retained = {
        Path(entry["path"]).name
        for entry in artifacts
        if entry.get("source_mode") == "retained_reference_sass"
    }
    assert retained == {"moe_w2_mm_k2048.cubin", "moe_w2_mm_mc2_k2048.cubin"}


def test_rebuild_refuses_unbound_m4_and_validates_external_cubit() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert ledger["families"]["M4"]["expected_output"] == {
        "sha256": None,
        "status": "PENDING_TARGET_BUILD",
    }
    assert ledger["status"] == "SOURCE_CANDIDATE_HARDWARE_GATES_OUTSTANDING"
    text = BUILD.read_text()
    assert "validate_external_cubit" in text
    assert "external Cubit git revision" in text
    assert "expected M4 output SHA-256 is not bound" in text
    assert 'if "M4" in selected_families and not requested' not in text


def test_plugin_build_dependencies_are_exactly_pinned() -> None:
    pyproject = ROOT / "banana-smasher-plugin/pyproject.toml"
    assert '"torch==2.9.0"' in pyproject.read_text()
    assert '"quack-kernels==0.4.1"' in pyproject.read_text()


def test_rebuild_entry_point_is_separate_output_only() -> None:
    text = BUILD.read_text()
    assert "--output" in text
    assert "cubins-sm120" not in re.sub(r"admitted.*", "", text)
    help_result = subprocess.run(
        [sys.executable, str(BUILD), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--family" in help_result.stdout
    assert "--output" in help_result.stdout

    rejected = ROOT / "banana-smasher/kernels/rejected-build-output"
    result = subprocess.run(
        [sys.executable, str(BUILD), "--output", str(rejected), "--family", "W3"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "refusing output inside the source repository" in result.stderr
    assert not rejected.exists()
