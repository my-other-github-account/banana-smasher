from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.contract import PackValidationError
from banana_smasher.backpack_virtual import materialize_mixed_v7_virtual_backpack
from banana_smasher.loader import MixedV7MemberLoader
from banana_smasher.repack import bind_mixed_v7_member_contract


BASIS = "a" * 64
ASSIGNMENT_SHA = "b" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _identity(root: Path, assignment: dict[str, str]) -> Path:
    path = root / "identity.json"
    _write_json(
        path,
        {
            "schema": "banana-smasher-mixed-backpack-identity-v1",
            "status": "PRE_REPAIR_SOLVED",
            "basis_sha256": BASIS,
            "assignment_sha256": ASSIGNMENT_SHA,
            "assignment": assignment,
        },
    )
    return path


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def test_qtip2_only_codes_contract_reopens_exact_wire_hashes(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized"
    tlut = materialized / "qtip2/shared/qtip_tlut.f16"
    tlut.parent.mkdir(parents=True)
    tlut.write_bytes(b"shared-q2-tlut")
    members = []
    for projection, payload in zip(("w1", "w2", "w3"), (b"gate", b"down", b"up")):
        wire = materialized / "qtip2" / "L000" / "E000" / f"{projection}.q2v7wire"
        wire.parent.mkdir(parents=True, exist_ok=True)
        wire.write_bytes(payload)
        members.append(
            {
                "cell_id": f"L000.E000.{projection}",
                "tier": "qtip2",
                "payload": _descriptor(wire, materialized),
                "unit_metadata": {"tlut": _descriptor(tlut, materialized)},
            }
        )
    index = tmp_path / "members.json"
    _write_json(
        index,
        {
            "schema": "banana-smasher-mixed-v7-materialized-members-v1",
            "status": "SEALED",
            "basis_sha256": BASIS,
            "assignment_sha256": ASSIGNMENT_SHA,
            "materialized_root": str(materialized),
            "members": members,
        },
    )
    output = tmp_path / "contract.json"

    receipt = bind_mixed_v7_member_contract(
        _identity(tmp_path / "solve", {"L000.E000": "qtip2"}), index, output=output
    )
    loader = MixedV7MemberLoader(output)

    assert receipt["status"] == "PASS_ADMISSION_READY"
    assert loader.reopen_hashes(0, 0, "qtip2") == {
        "w1": "c974e17b8e7321ce8c12983de3d0ed4a289821f579bbe0925b0181a4bc8e8d80",
        "w1.tlut": "7ed5b3a9783e0298128e42a10979f5a7170b5e2f82767913e26b497743918970",
        "w2": "908aec4512d80ff4fefb1970899091e9de8e734b36b8fdb7678e77dc092f6959",
        "w2.tlut": "7ed5b3a9783e0298128e42a10979f5a7170b5e2f82767913e26b497743918970",
        "w3": "75a288c0d6898c5f7b054590845978a82a3ad79fcce3d43ff68a7501e5a91ee9",
        "w3.tlut": "7ed5b3a9783e0298128e42a10979f5a7170b5e2f82767913e26b497743918970",
    }


def test_qtip3_only_codes_contract_reopens_codes_and_per_unit_metadata(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "materialized"
    members = []
    for projection in ("down", "fused13"):
        unit = materialized / "qtip3" / "L023" / f"E007_{projection}"
        unit.mkdir(parents=True, exist_ok=True)
        codes = unit / "codes.npy"
        control = unit / "CONTROL.pt"
        tlut = materialized / "qtip3/shared/qtip_tlut.npy"
        tlut.parent.mkdir(parents=True, exist_ok=True)
        codes.write_bytes(f"codes-{projection}".encode())
        control.write_bytes(f"control-{projection}".encode())
        if not tlut.exists():
            tlut.write_bytes(b"shared-tlut")
        members.append(
            {
                "cell_id": f"L023.E007.{projection}",
                "tier": "qtip3",
                "payload": _descriptor(codes, materialized),
                "unit_metadata": {
                    "control": _descriptor(control, materialized),
                    "tlut": _descriptor(tlut, materialized),
                },
            }
        )
    index = tmp_path / "members.json"
    _write_json(
        index,
        {
            "schema": "banana-smasher-mixed-v7-materialized-members-v1",
            "status": "SEALED",
            "basis_sha256": BASIS,
            "assignment_sha256": ASSIGNMENT_SHA,
            "materialized_root": str(materialized),
            "members": members,
        },
    )
    output = tmp_path / "contract.json"

    bind_mixed_v7_member_contract(
        _identity(tmp_path / "solve", {"L023.E007": "qtip3"}), index, output=output
    )
    loader = MixedV7MemberLoader(output)

    reopened = loader.reopen_hashes(23, 7, "qtip3")
    assert reopened["down.codes"] == "0181710422c3eeda2ecbc6ae20139a14912054181e55c57a83b7da5d2fc683ec"
    assert reopened["fused13.control"] == "307290617bbd8f62ee47bc105dc7bdf7ffa1ed267ecab71057851d4ae4536b52"
    assert reopened["down.tlut"] == "58bfbd29f52b67c7e2f8b179cb4fe1676aab73e911ef9100b5a587274d40f20a"
    assert reopened["down.tlut"] == reopened["fused13.tlut"]


def test_contract_fails_closed_on_payload_drift(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized"
    wire = materialized / "w1.q2v7wire"
    wire.parent.mkdir(parents=True)
    wire.write_bytes(b"sealed")
    index = tmp_path / "members.json"
    members = []
    tlut = materialized / "qtip_tlut.f16"
    tlut.write_bytes(b"tlut")
    for projection in ("w1", "w2", "w3"):
        path = materialized / f"{projection}.q2v7wire"
        path.write_bytes(b"sealed")
        members.append(
            {
                "cell_id": f"L000.E000.{projection}",
                "tier": "qtip2",
                "payload": _descriptor(path, materialized),
                "unit_metadata": {"tlut": _descriptor(tlut, materialized)},
            }
        )
    _write_json(
        index,
        {
            "schema": "banana-smasher-mixed-v7-materialized-members-v1",
            "status": "SEALED",
            "basis_sha256": BASIS,
            "assignment_sha256": ASSIGNMENT_SHA,
            "materialized_root": str(materialized),
            "members": members,
        },
    )
    output = tmp_path / "contract.json"
    bind_mixed_v7_member_contract(
        _identity(tmp_path / "solve", {"L000.E000": "qtip2"}), index, output=output
    )
    wire.write_bytes(b"drifte")

    with pytest.raises(PackValidationError, match="SHA-256 drift"):
        MixedV7MemberLoader(output)


def test_mixed_virtualizer_projects_expert_tiers_to_runtime_cells(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized"
    q2_tlut = materialized / "q2.tlut"
    q3_tlut = materialized / "q3.tlut"
    q2_tlut.parent.mkdir(parents=True)
    q2_tlut.write_bytes(b"q2-tlut")
    q3_tlut.write_bytes(b"q3-tlut")
    members = []
    for projection in ("w1", "w2", "w3"):
        path = materialized / f"q2-{projection}.wire"
        path.write_bytes(projection.encode())
        members.append({
            "cell_id": f"L000.E000.{projection}",
            "tier": "qtip2",
            "payload": _descriptor(path, materialized),
            "unit_metadata": {"tlut": _descriptor(q2_tlut, materialized)},
        })
    for projection in ("down", "fused13"):
        codes = materialized / f"q3-{projection}.npy"
        control = materialized / f"q3-{projection}.pt"
        codes.write_bytes(f"codes-{projection}".encode())
        control.write_bytes(f"control-{projection}".encode())
        members.append({
            "cell_id": f"L000.E001.{projection}",
            "tier": "qtip3",
            "payload": _descriptor(codes, materialized),
            "unit_metadata": {
                "control": _descriptor(control, materialized),
                "tlut": _descriptor(q3_tlut, materialized),
            },
        })
    solve = tmp_path / "solve"
    _identity(solve, {"L000.E000": "qtip2", "L000.E001": "qtip3"})
    index = tmp_path / "members.json"
    _write_json(index, {
        "schema": "banana-smasher-mixed-v7-materialized-members-v1",
        "status": "SEALED",
        "basis_sha256": BASIS,
        "assignment_sha256": ASSIGNMENT_SHA,
        "materialized_root": str(materialized),
        "members": members,
    })

    receipt = materialize_mixed_v7_virtual_backpack(solve, index, tmp_path / "virtual")
    rows = [
        json.loads(line)
        for line in (tmp_path / "virtual/MATERIALIZATION_INDEX.jsonl").read_text().splitlines()
    ]

    assert receipt["status"] == "PASS"
    assert [(row["cell_id"], row["source_key"]) for row in rows] == [
        ("L0:E0:down", "qtip2_v7"),
        ("L0:E0:fused13", "qtip2_v7"),
        ("L0:E1:down", "qtip3_v7"),
        ("L0:E1:fused13", "qtip3_v7"),
    ]
