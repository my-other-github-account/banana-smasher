from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch

import launch_joint_all43_official_k2 as binding


def test_build_exec_seals_official_source_and_binds_trainer(tmp_path: Path, monkeypatch):
    source = tmp_path / "official.py"
    trainer = tmp_path / "trainer.py"
    source.write_bytes(b"sealed official source")
    trainer.write_text("pass\n")
    monkeypatch.setattr(binding, "OFFICIAL_SOURCE_SHA256", binding.sha256_path(source))

    executable, argv, env = binding.build_exec(source, trainer, ["--run-root", "/run"])

    assert executable == binding.sys.executable
    assert argv == [binding.sys.executable, str(trainer.resolve()), "--run-root", "/run"]
    assert env["JOINT_V7_EXPERT_BASE"] == f"{source.resolve()}:JointV7ExpertBase"
    assert env["JOINT_ALL43_MECHANISM"] == "official-qtip-k2-1d1024-joint-all43"
    assert env["JOINT_ALL43_CANONICAL_REFERENCE"].startswith("419790fad2cc5370")


def test_build_exec_refuses_source_identity_drift(tmp_path: Path):
    source = tmp_path / "official.py"
    trainer = tmp_path / "trainer.py"
    source.write_bytes(b"wrong")
    trainer.write_text("pass\n")

    with pytest.raises(RuntimeError, match="source mismatch"):
        binding.build_exec(source, trainer, [])


def test_usage_seam_preserves_exact_forward_and_accounts_all43(tmp_path: Path, monkeypatch):
    """Accounting observes every PlaneSource without entering the numerical path."""
    source_path = Path(__file__).parents[2] / "runtime" / "v7" / "runner" / "joint_v7_expert_base.py"
    repaired_text = source_path.read_text()
    seam = "        source.wire_lut()  # accounting only; official numerical path remains source.master\n"
    assert repaired_text.count(seam) == 1
    baseline_path = tmp_path / "baseline_official_k2.py"
    baseline_path.write_text(repaired_text.replace(seam, ""))

    repair_stub = types.ModuleType("qtip_v7_repair")
    repair_stub._parse_v7_member = lambda path, projection: {"shape": [2, 2]}
    package_stub = types.ModuleType("banana_smasher")
    qtip_stub = types.ModuleType("banana_smasher.qtip_k2")
    qtip_stub.decode_k2_matrix = lambda packed, lut: packed.float() + lut[:4].reshape(2, 2).float()
    qtip_stub.inverse_transform = lambda decoded, su, sv: decoded
    package_stub.qtip_k2 = qtip_stub
    monkeypatch.setitem(sys.modules, "qtip_v7_repair", repair_stub)
    monkeypatch.setitem(sys.modules, "banana_smasher", package_stub)
    monkeypatch.setitem(sys.modules, "banana_smasher.qtip_k2", qtip_stub)
    monkeypatch.setenv("BANANA_SMASHER_PUBLIC_SRC", str(tmp_path))

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.packed_k2_from_member = lambda path, projection, device: {
            "packed": torch.tensor([[1, 2], [3, 4]], dtype=torch.int16),
            "su": torch.ones(2),
            "sv": torch.ones(2),
            "m": 2,
            "k": 2,
        }
        return module

    baseline = load("official_k2_baseline", baseline_path)
    repaired = load("official_k2_repaired", source_path)
    used: set[int] = set()
    calls = 0

    class PlaneSource:
        def __init__(self, layer: int):
            self.layer = layer
            self.master = torch.linspace(-1, 1, 1024, dtype=torch.float32)

        def member_path(self, expert: int, projection: str) -> Path:
            return tmp_path / f"L{self.layer:03d}_{expert}_{projection}"

        def wire_lut(self) -> torch.Tensor:
            nonlocal calls
            calls += 1
            used.add(self.layer)
            return self.master + 999  # must be ignored by the official numerical path

    hidden = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    top_k_index = torch.zeros((1, 1), dtype=torch.int64)
    top_k_weights = torch.ones((1, 1), dtype=torch.float32)
    for layer in range(43):
        plane = PlaneSource(layer)
        expected = baseline.JointV7ExpertBase(layer, plane_source=plane)(
            hidden, top_k_index, top_k_weights
        )
        observed = repaired.JointV7ExpertBase(layer, plane_source=plane)(
            hidden, top_k_index, top_k_weights
        )
        assert torch.equal(observed, expected)

    assert used == set(range(43))
    assert calls == 43 * 3
    assert binding.sha256_path(source_path) == binding.OFFICIAL_SOURCE_SHA256
