from __future__ import annotations

import json

from repair_api import cli


def test_continue_training_command_uses_repair_api_public_facade(
    monkeypatch, tmp_path, capsys
) -> None:
    calls = []

    def continue_training(
        cls, artifact_root, start_checkpoint, milestones, *, config, receipt_path
    ):
        calls.append(
            (artifact_root, start_checkpoint, tuple(milestones), config, receipt_path)
        )
        return {
            "status": "PASS",
            "public_api": "repair_api.ResidentRepairAPI.continue_training",
            "resident_state_persisted": True,
        }

    monkeypatch.setattr(
        cli.ResidentRepairAPI,
        "continue_training",
        classmethod(continue_training),
        raising=False,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"expert_plane_expansion": {"surface": "expert_planes_l028_su_sv"}}))
    receipt_path = tmp_path / "receipt.json"
    artifact_root = tmp_path / "artifact"

    assert (
        cli.main(
            [
                "continue-training",
                "--artifact-root",
                str(artifact_root),
                "--start-checkpoint",
                "UPDATE_000",
                "--milestones",
                "1",
                "--config",
                str(config_path),
                "--receipt",
                str(receipt_path),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["public_api"] == "repair_api.ResidentRepairAPI.continue_training"
    assert calls[0][0] == artifact_root.resolve()
    assert calls[0][1:3] == ("UPDATE_000", (1,))
    assert calls[0][3]["expert_plane_expansion"]["surface"] == "expert_planes_l028_su_sv"
    assert calls[0][4] == receipt_path.resolve()