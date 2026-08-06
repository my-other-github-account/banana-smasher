from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.backpack_selection import select_measured_nonworse
from banana_smasher.locality import require_local_path
from banana_smasher.staging import stage_qsfp_manifest


def test_locality_rejects_sshfs(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    payload = remote / "payload.bin"
    payload.write_bytes(b"x")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"1 0 0:1 / {tmp_path} rw - ext4 /dev/nvme0n1 rw\n"
        f"2 1 0:2 / {remote} rw - fuse.sshfs dnola@192.168.200.1:/home/dnola/missions rw\n"
    )
    assert require_local_path(
        local, label="local", mountinfo_path=mountinfo
    ) == local.resolve()
    with pytest.raises(ValueError, match="explicit QSFP staging API"):
        require_local_path(payload, label="payload", mountinfo_path=mountinfo)


def test_qsfp_stage_is_explicit_and_rejects_aliases(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qsfp-stage-v1",
                "status": "READY",
                "items": [
                    {
                        "source_host": "spark-1",
                        "source_path": "/source/layer0",
                        "destination": "qtip2/layer0",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="direct QSFP address"):
        stage_qsfp_manifest(bad, tmp_path / "bad-output")

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qsfp-stage-v1",
                "status": "READY",
                "items": [
                    {
                        "source_host": "dnola@192.168.200.1",
                        "source_path": "/source/layer0",
                        "destination": "qtip2/layer0",
                        "bytes": 1,
                    }
                ],
            }
        )
    )

    def fake_transfer(item: dict[str, object], output: Path) -> dict[str, object]:
        target = output / str(item["destination"])
        target.mkdir(parents=True)
        (target / "payload").write_bytes(b"x")
        return {**item, "actual_bytes": 1, "elapsed_seconds": 0.01, "status": "PASS"}

    receipt = stage_qsfp_manifest(
        good, tmp_path / "good-output", transfer=fake_transfer
    )
    assert receipt["status"] == "PASS"
    assert receipt["transport"] == "direct-qsfp-ssh-rsync"
    assert receipt["bytes"] == 1


def test_batched_stage_groups_relative_files(tmp_path: Path) -> None:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qsfp-stage-v2",
                "status": "READY",
                "items": [
                    {
                        "source_host": "dnola@192.168.200.1",
                        "source_root": "/source0",
                        "destination": "qtip2",
                        "relative_paths": ["L000/E000_down/QTIP_UNIT.pt"],
                        "bytes": 1,
                    },
                    {
                        "source_host": "dnola@192.168.200.3",
                        "source_root": "/source1",
                        "destination": "qtip2",
                        "relative_paths": ["L001/E000_down/QTIP_UNIT.pt"],
                        "bytes": 1,
                    },
                ],
            }
        )
    )

    def fake_transfer(item: dict[str, object], root: Path) -> dict[str, object]:
        relative_paths = item["relative_paths"]
        assert isinstance(relative_paths, list)
        for relative in relative_paths:
            target = root / str(item["destination"]) / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return {**item, "actual_bytes": len(relative_paths), "status": "PASS"}

    receipt = stage_qsfp_manifest(
        manifest, tmp_path / "batch-output", transfer=fake_transfer
    )
    assert receipt["status"] == "PASS"
    assert receipt["manifest_schema"] == "banana-smasher-qsfp-stage-v2"
    assert receipt["bytes"] == 2


def test_measured_selection_retains_baseline_on_proxy_reversal(tmp_path: Path) -> None:
    basis = "a" * 64
    solve = {
        "basis_sha256": basis,
        "arms": {
            "without_qtip2_5": {
                "tiers": ["native_mxfp4", "qtip2", "qtip3"],
                "assignment_map_sha256": "b" * 64,
                "objective": {"value": 0.037037603438611934},
            },
            "with_qtip2_5": {
                "tiers": ["native_mxfp4", "qtip2", "qtip2_5", "qtip3"],
                "assignment_map_sha256": "c" * 64,
                "objective": {"value": 0.033664764878413196},
            },
        },
    }
    common = {
        "status": "PASS",
        "basis_sha256": basis,
        "bank_sha256": "d" * 64,
        "windows": 64,
        "positions": 65536,
        "support_width": 8192,
    }
    baseline = {**common, "mean_kld": 0.10588817638713062, "top1_matches": 45192}
    expanded = {**common, "mean_kld": 0.11906004925872879, "top1_matches": 44835}
    solve_path = tmp_path / "solve.json"
    baseline_path = tmp_path / "baseline.json"
    expanded_path = tmp_path / "expanded.json"
    solve_path.write_text(json.dumps(solve))
    baseline_path.write_text(json.dumps(baseline))
    expanded_path.write_text(json.dumps(expanded))
    receipt = select_measured_nonworse(
        solve_path,
        baseline_path,
        expanded_path,
        tmp_path / "selection.json",
        baseline_arm="without_qtip2_5",
        expanded_arm="with_qtip2_5",
    )
    assert receipt["decision"] == "RETAIN_BASELINE"
    assert receipt["chosen_assignment_map_sha256"] == "b" * 64
    assert receipt["measured"]["expanded_minus_baseline"]["mean_kld"] > 0
    assert receipt["measured"]["expanded_minus_baseline"]["top1_matches"] == -357
    assert receipt["proxy"]["ordering_agrees_with_measurement"] is False
