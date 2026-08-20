"""Fail-closed loader for the current all-43 QTIP2-V7 route artifact.

The route census is an identity/composition manifest, not a promise that every
provider uses one path layout.  This module validates and normalizes all route
kinds observed in the sealed all-43 artifact without performing transport or
scoring.  Materialization remains a separate, explicit runtime phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import PackValidationError

ROUTE_CENSUS_SCHEMA = "banana-smasher.qtip2_v7.final_43_route_census.v1"
ROUTE_CENSUS_STATUS = "PASS_43_OF_43"
LAYERS = tuple(range(43))
EXPERTS = 256
PROJECTIONS = ("w1", "w2", "w3")
MEMBERS_PER_LAYER = EXPERTS * len(PROJECTIONS)
QTIP_V7_MEMBER_BYTES = 2_109_444
ROUTE_KINDS = frozenset(
    {
        "nas_sftp",
        "ssh",
        "nas_shell",
        "nas_shell_stream",
        "nas_sftp_tranches",
        "local",
        "dense_roster",
        "ssh_transfer_manifest_full_tree",
        "split",
    }
)


def load_qtip2_v7_wire(path: str | Path, *, projection: str) -> dict[str, Any]:
    """Read one current V7 member using its complete packed/SU/SV geometry."""

    member = Path(path).expanduser().resolve()
    if projection in {"w1", "w3"}:
        packed_shape = (256, 128, 32)
        su_count, sv_count = 4096, 2048
        weight_shape = (2048, 4096)
    elif projection == "w2":
        packed_shape = (128, 256, 32)
        su_count, sv_count = 2048, 4096
        weight_shape = (4096, 2048)
    else:
        raise PackValidationError(f"unsupported QTIP V7 projection: {projection!r}")
    try:
        size = member.stat().st_size
    except OSError as exc:
        raise PackValidationError(f"cannot read QTIP V7 member {member}: {exc}") from exc
    if size != QTIP_V7_MEMBER_BYTES:
        raise PackValidationError(
            f"QTIP V7 member byte geometry mismatch: {size} != {QTIP_V7_MEMBER_BYTES}"
        )
    packed_bytes = 2_097_152
    su_end = packed_bytes + su_count * 2
    sv_end = su_end + sv_count * 2
    if sv_end + 4 != QTIP_V7_MEMBER_BYTES:
        raise AssertionError("internal QTIP V7 geometry accounting drift")
    raw = np.memmap(member, mode="r", dtype=np.uint8, shape=(size,))
    return {
        "packed": raw[:packed_bytes].view("<i2").reshape(packed_shape),
        "SU": raw[packed_bytes:su_end].view("<f2"),
        "SV": raw[su_end:sv_end].view("<f2"),
        "Wscale": raw[sv_end:].view("<f4").reshape(()),
        "weight_shape": weight_shape,
        "path": member,
    }


def _sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PackValidationError(f"QTIP V7 {field} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PackValidationError(f"QTIP V7 {field} must be a positive integer")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackValidationError(f"QTIP V7 {field} must be an object")
    return value


@dataclass(frozen=True)
class QtipV7LayerRoute:
    layer: int
    kind: str
    wire_format: str
    layout: str | None
    member_count: int
    wire_bytes: int
    producer_task_id: str
    terminal_sha256: str
    route: Mapping[str, Any]
    physical_census: Mapping[str, Any]


@dataclass(frozen=True)
class QtipV7RouteCensus:
    path: Path
    sha256: str
    basis_sha256: str
    complete_members: int
    complete_wire_bytes: int
    routes: tuple[QtipV7LayerRoute, ...]
    document: Mapping[str, Any]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(row.layer for row in self.routes)

    def route(self, layer: int) -> QtipV7LayerRoute:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in LAYERS:
            raise PackValidationError(f"QTIP V7 layer is outside 0..42: {layer!r}")
        return self.routes[layer]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_basis_sha256: str,
    ) -> "QtipV7RouteCensus":
        route_path = Path(path).expanduser().resolve()
        try:
            raw = route_path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackValidationError(
                f"cannot read QTIP V7 route census {route_path}: {exc}"
            ) from exc
        document = _mapping(value, "route census")
        if document.get("schema") != ROUTE_CENSUS_SCHEMA:
            raise PackValidationError(
                f"QTIP V7 route census schema must be {ROUTE_CENSUS_SCHEMA}"
            )
        if document.get("status") != ROUTE_CENSUS_STATUS:
            raise PackValidationError("QTIP V7 route census is not PASS_43_OF_43")
        expected_basis = _sha(expected_basis_sha256, "expected basis")
        basis = _sha(document.get("basis_sha256"), "basis")
        if basis != expected_basis:
            raise PackValidationError(
                f"QTIP V7 route census basis mismatch: {basis} != {expected_basis}"
            )
        if (
            document.get("frozen_layer_count") != len(LAYERS)
            or document.get("frozen_layers") != list(LAYERS)
            or document.get("complete_layers") != len(LAYERS)
            or document.get("gaps") != 0
            or document.get("duplicates") != 0
            or document.get("fallback_calls") != 0
        ):
            raise PackValidationError("QTIP V7 route census all-43 closure gates failed")
        rows = document.get("layers")
        if not isinstance(rows, list) or len(rows) != len(LAYERS):
            raise PackValidationError("QTIP V7 route census must contain exactly layers 0..42")
        if [row.get("layer") if isinstance(row, Mapping) else None for row in rows] != list(LAYERS):
            raise PackValidationError("QTIP V7 route census must contain exactly layers 0..42")

        normalized: list[QtipV7LayerRoute] = []
        for expected_layer, raw_row in enumerate(rows):
            row = _mapping(raw_row, f"layer {expected_layer}")
            route = _mapping(row.get("route"), f"layer {expected_layer} route")
            census = _mapping(
                row.get("physical_census"), f"layer {expected_layer} physical census"
            )
            kind = route.get("kind")
            if kind not in ROUTE_KINDS:
                raise PackValidationError(
                    f"QTIP V7 layer {expected_layer} unsupported route kind: {kind!r}"
                )
            producer = row.get("producer_task_id")
            if not isinstance(producer, str) or not producer:
                raise PackValidationError(
                    f"QTIP V7 layer {expected_layer} producer identity is missing"
                )
            member_count = _positive_int(
                census.get("file_count"), f"layer {expected_layer} file_count"
            )
            if member_count != MEMBERS_PER_LAYER:
                raise PackValidationError(
                    f"QTIP V7 layer {expected_layer} member coverage is {member_count}, "
                    f"expected {MEMBERS_PER_LAYER}"
                )
            if kind == "dense_roster":
                wire_format = "dense_bf16_roster"
                wire_bytes_value = census.get("physical_bytes", census.get("wire_bytes"))
                _sha(route.get("roster_sha256"), f"layer {expected_layer} roster")
            else:
                wire_format = "qtip2_v7_fixed_wire"
                wire_bytes_value = census.get("wire_bytes")
                if kind in {"nas_sftp", "nas_shell"}:
                    if not isinstance(route.get("source"), str) or not route["source"]:
                        raise PackValidationError(
                            f"QTIP V7 layer {expected_layer} route source is missing"
                        )
                elif kind in {"ssh", "local"}:
                    location = route.get("source", route.get("root"))
                    if not isinstance(location, str) or not location:
                        raise PackValidationError(
                            f"QTIP V7 layer {expected_layer} route location is missing"
                        )
                elif kind in {
                    "nas_shell_stream",
                    "nas_sftp_tranches",
                    "ssh_transfer_manifest_full_tree",
                }:
                    if not isinstance(route.get("root"), str) or not route["root"]:
                        raise PackValidationError(
                            f"QTIP V7 layer {expected_layer} route root is missing"
                        )
                elif kind == "split":
                    parts = route.get("parts")
                    if not isinstance(parts, list) or len(parts) < 2:
                        raise PackValidationError(
                            f"QTIP V7 layer {expected_layer} split route is incomplete"
                        )
                    if any(
                        not isinstance(part, Mapping)
                        or not isinstance(part.get("root", part.get("source")), str)
                        for part in parts
                    ):
                        raise PackValidationError(
                            f"QTIP V7 layer {expected_layer} split part is malformed"
                        )
            wire_bytes = _positive_int(
                wire_bytes_value, f"layer {expected_layer} physical bytes"
            )
            normalized.append(
                QtipV7LayerRoute(
                    layer=expected_layer,
                    kind=str(kind),
                    wire_format=wire_format,
                    layout=(
                        str(route["layout"])
                        if isinstance(route.get("layout"), str)
                        else None
                    ),
                    member_count=member_count,
                    wire_bytes=wire_bytes,
                    producer_task_id=producer,
                    terminal_sha256=_sha(
                        row.get("terminal_sha256"),
                        f"layer {expected_layer} terminal",
                    ),
                    route=dict(route),
                    physical_census=dict(census),
                )
            )
        complete_members = _positive_int(
            document.get("complete_members"), "complete_members"
        )
        complete_wire_bytes = _positive_int(
            document.get("complete_wire_bytes"), "complete_wire_bytes"
        )
        if complete_members != len(LAYERS) * MEMBERS_PER_LAYER:
            raise PackValidationError("QTIP V7 complete member count drift")
        if sum(row.member_count for row in normalized) != complete_members:
            raise PackValidationError("QTIP V7 per-layer member accounting drift")
        return cls(
            path=route_path,
            sha256=hashlib.sha256(raw).hexdigest(),
            basis_sha256=basis,
            complete_members=complete_members,
            complete_wire_bytes=complete_wire_bytes,
            routes=tuple(normalized),
            document=dict(document),
        )


__all__ = [
    "LAYERS",
    "MEMBERS_PER_LAYER",
    "QTIP_V7_MEMBER_BYTES",
    "QtipV7LayerRoute",
    "QtipV7RouteCensus",
    "load_qtip2_v7_wire",
]
