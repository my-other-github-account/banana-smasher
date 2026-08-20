from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.contract import PackValidationError
from banana_smasher.qtip_v7_routes import QtipV7RouteCensus

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
KINDS = (
    "nas_sftp",
    "ssh",
    "nas_shell",
    "nas_shell_stream",
    "nas_sftp_tranches",
    "local",
    "dense_roster",
    "ssh_transfer_manifest_full_tree",
    "split",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _route(kind: str, layer: int) -> dict:
    base = {"kind": kind}
    if kind in {"nas_sftp", "ssh", "nas_shell", "local"}:
        base.update({"source": f"/providers/L{layer:03d}", "layout": "flat", "ext": "q2v7wire"})
    elif kind in {"nas_shell_stream", "nas_sftp_tranches"}:
        base.update({"root": f"/providers/L{layer:03d}"})
    elif kind == "ssh_transfer_manifest_full_tree":
        base.update(
            {
                "root": f"/providers/L{layer:03d}",
                "transfer_manifest_path": f"/providers/L{layer:03d}/TRANSFER.json",
                "transfer_manifest_sha256": _sha(f"transfer-{layer}"),
            }
        )
    elif kind == "split":
        base["parts"] = [
            {"root": f"/providers/L{layer:03d}/a", "files": 384, "bytes": 384 * 2_109_444},
            {"root": f"/providers/L{layer:03d}/b", "files": 384, "bytes": 384 * 2_109_444},
        ]
    else:
        base.update({"roster_sha256": _sha(f"roster-{layer}"), "hosts": {"spark-1": "local"}})
    return base


def _document() -> dict:
    layers = []
    for layer in range(43):
        kind = KINDS[layer % len(KINDS)]
        layers.append(
            {
                "layer": layer,
                "producer_task_id": f"producer-{layer}",
                "terminal_sha256": _sha(f"terminal-{layer}"),
                "route": _route(kind, layer),
                "physical_census": {
                    "file_count": 768,
                    "wire_bytes": 768 * 2_109_444,
                    "sentinels": [],
                },
            }
        )
    return {
        "schema": "banana-smasher.qtip2_v7.final_43_route_census.v1",
        "status": "PASS_43_OF_43",
        "basis_sha256": BASIS,
        "frozen_layer_count": 43,
        "frozen_layers": list(range(43)),
        "complete_layers": 43,
        "complete_members": 43 * 768,
        "complete_wire_bytes": 43 * 768 * 2_109_444,
        "gaps": 0,
        "duplicates": 0,
        "fallback_calls": 0,
        "layers": layers,
    }


def test_current_all43_route_census_loads_without_inventing_one_layout(tmp_path: Path) -> None:
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(_document()))
    census = QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    assert census.layers == tuple(range(43))
    assert census.complete_members == 33_024
    assert {row.kind for row in census.routes} == set(KINDS)
    assert census.route(34).layer == 34
    assert census.route(34).terminal_sha256 == _sha("terminal-34")
    assert census.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_route_census_fails_closed_on_basis_and_coverage(tmp_path: Path) -> None:
    value = _document()
    value["layers"].pop()
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PackValidationError, match="exactly layers 0..42"):
        QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    value = _document()
    path.write_text(json.dumps(value))
    with pytest.raises(PackValidationError, match="basis"):
        QtipV7RouteCensus.load(path, expected_basis_sha256=_sha("wrong"))


def test_dense_roster_is_explicitly_not_relabelled_as_qtip_wire(tmp_path: Path) -> None:
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(_document()))
    census = QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    dense = next(row for row in census.routes if row.kind == "dense_roster")
    assert dense.wire_format == "dense_bf16_roster"
    assert dense.member_count == 768
    ordinary = next(row for row in census.routes if row.kind == "nas_sftp")
    assert ordinary.wire_format == "qtip2_v7_fixed_wire"
