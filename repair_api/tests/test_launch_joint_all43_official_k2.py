from pathlib import Path

import pytest

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
