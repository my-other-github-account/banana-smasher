"""Public, resumable QTIP3 V7 Banana Smasher producer.

This module is deliberately the only producer seam for the missing QTIP3 V7
scope.  It binds immutable plan/config data before touching a claim, validates
the model-index basis, and calls the supported ``qtip25_native_v4`` public
cell API.  The host transaction is injected so unit tests can exercise all
state transitions without pretending to own a Spark host.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .qtip25_native_v4_api import build_qtip_native_cell

SCHEMA = "banana-smasher-qtip3-v7-public-api-producer-v1"
ADMISSION_SCHEMA = f"{SCHEMA}-admission"
PROGRESS_SCHEMA = f"{SCHEMA}-progress"
TERMINAL_SCHEMA = f"{SCHEMA}-terminal"
RELEASE_SCHEMA = f"{SCHEMA}-release"
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
LAYERS = (2, 4, 5, 12, 13, 14, 21, 22, 38, 40, 41)
PROJECTIONS = ("fused13", "down")
EXPERTS = tuple(range(256))
EXPECTED_CELLS = len(LAYERS) * len(EXPERTS) * len(PROJECTIONS)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cell_roster(
    path: str | Path, *, intended_basis_sha256: str, expected_count: int | None = None
) -> tuple[tuple[int, int, str], ...]:
    """Load the exact missing-cell roster from physical admission evidence."""
    payload = json.loads(Path(path).read_text())
    if payload.get("basis_sha256") != intended_basis_sha256:
        raise RuntimeError("CELL_ROSTER_BASIS_MISMATCH")
    rows = []
    for item in payload.get("missing", ()):
        match = re.fullmatch(r"L(\d{3}):E(\d{3}):(down|fused13)", str(item.get("cell_id", "")))
        if match is None:
            raise ValueError("invalid physical-preflight cell_id")
        rows.append((int(match.group(1)), int(match.group(2)), match.group(3)))
    roster = tuple(sorted(rows, key=lambda row: (row[0], row[1], PROJECTIONS.index(row[2]))))
    if len(set(roster)) != len(roster):
        raise ValueError("physical-preflight roster contains duplicates")
    if expected_count is not None and len(roster) != expected_count:
        raise ValueError(f"cell roster count mismatch expected={expected_count} actual={len(roster)}")
    return roster


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return _sha_bytes(data)


def _write_progress_monotone(path: Path, payload: dict[str, Any]) -> str:
    """Never replace a durable checkpoint with an earlier counter/frontier."""
    if path.is_file():
        current = json.loads(path.read_text())
        current_count = int(current.get("accepted_cells", current.get("cells_passed", -1)))
        candidate_count = int(payload["accepted_cells"])
        if current_count > candidate_count or (
            current_count == candidate_count
            and current.get("last_cell") != payload.get("last_cell")
        ):
            return sha256_file(path)
    return _atomic_json(path, payload)


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _startticks(pid: int) -> int | str | None:
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        tail = stat.read_text().rsplit(")", 1)[1].split()
        return int(tail[19])
    # macOS has no /proc; retain a structured, stable process-start identity
    # for local contract tests and control-plane receipts.
    try:
        import subprocess
        value = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "lstart="], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


@dataclass(frozen=True)
class Qtip3ApiConfig:
    """Immutable acceptance settings for one QTIP V7 ladder tier."""

    bpw: float = 3.00
    tier: Literal["qtip1_v7", "qtip3_v7", "qtip4_v7"] = "qtip3_v7"
    codec_version: Literal["v6"] = "v6"
    provider: str = "qtip-native-v6@3.00"
    backend: Literal["cuda"] = "cuda"
    geometry: tuple[int, int, int] = (12, 16, 4)
    tlut_shape: tuple[int, int] = (512, 2)
    materialize_decoded: bool = False
    scale_factors: tuple[float, ...] = (1.0,)
    scale_semantics: Literal["rms_ratio"] = "rms_ratio"
    feedback_mode: Literal["off"] = "off"
    trellis_objective: Literal["sse"] = "sse"
    decode_repeats: int = 1
    # QTIP3 cells contain hundreds of thousands of independent compact
    # sequences. Keep enough of them in each CUDA dispatch to amortize the
    # public cell path's Python/launch overhead; the old implicit API default
    # of 2,048 produced hundreds of tiny launches per physical cell.
    solve_batch: int = 65_536
    decode_batch: int = 65_536
    # Explicit immutable safety reserve; CUDA enforces peak + reserve <= free.
    reserve_bytes: int = 256 << 20

    @classmethod
    def for_bpw(cls, bpw: float) -> "Qtip3ApiConfig":
        """Resolve the production Q1/Q3/Q4 V7 ladder without changing its path."""
        rate = float(bpw)
        if rate not in {1.0, 3.0, 4.0}:
            raise ValueError("QTIP V7 ladder supports only BPW 1.00, 3.00, and 4.00")
        tiers: dict[float, Literal["qtip1_v7", "qtip3_v7", "qtip4_v7"]] = {
            1.0: "qtip1_v7", 3.0: "qtip3_v7", 4.0: "qtip4_v7"
        }
        return cls(
            bpw=rate,
            tier=tiers[rate],
            provider=f"qtip-native-v6@{rate:.2f}",
            geometry=(int(rate * 4), 16, 4),
        )

    @classmethod
    def for_tier(cls, tier: str) -> "Qtip3ApiConfig":
        """Resolve a canonical V7 tier name to the proven producer geometry."""
        rates = {"qtip1_v7": 1.0, "qtip3_v7": 3.0, "qtip4_v7": 4.0}
        if tier not in rates:
            raise ValueError("QTIP V7 tier must be qtip1_v7, qtip3_v7, or qtip4_v7")
        return cls.for_bpw(rates[tier])

    def __post_init__(self) -> None:
        expected = {
            1.0: ("qtip1_v7", "qtip-native-v6@1.00", (4, 16, 4)),
            3.0: ("qtip3_v7", "qtip-native-v6@3.00", (12, 16, 4)),
            4.0: ("qtip4_v7", "qtip-native-v6@4.00", (16, 16, 4)),
        }.get(self.bpw)
        if expected is None or self.codec_version != "v6":
            raise ValueError("QTIP V7 ladder supports only codec v6 at BPW 1.00, 3.00, and 4.00")
        if (self.tier, self.provider, self.geometry) != expected:
            raise ValueError("QTIP V7 tier/provider/geometry does not match its declared BPW")
        if self.backend != "cuda":
            raise ValueError("QTIP V7 producer requires CUDA")
        if self.tlut_shape != (512, 2):
            raise ValueError("QTIP3 V7 producer requires float32 TLUT shape [512,2]")
        if self.materialize_decoded:
            raise ValueError("QTIP3 V7 producer requires materialize_decoded=false")
        if self.scale_factors != (1.0,) or self.scale_semantics != "rms_ratio":
            raise ValueError("QTIP3 V7 producer requires rms_ratio with scale_factors=[1.0]")
        if self.feedback_mode != "off" or self.trellis_objective != "sse":
            raise ValueError("QTIP3 V7 producer requires feedback off and sum_sse")
        if self.decode_repeats < 1:
            raise ValueError("decode_repeats must be positive")
        if self.solve_batch < 1 or self.decode_batch < 1:
            raise ValueError("solve_batch and decode_batch must be positive")
        if isinstance(self.reserve_bytes, bool) or not isinstance(self.reserve_bytes, int) or self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a non-negative integer")


@dataclass(frozen=True)
class Qtip3ApiPlan:
    """Immutable authority, source, and scope binding for one producer."""

    task_id: str
    board_run_id: int
    host: str
    allocation: str
    intended_basis_sha256: str
    driver_goals_path: Path
    driver_goals_sha256: str
    claim_path: Path
    shards_path: Path
    mission_root: Path
    model_index_path: Path
    tlut_path: Path
    layers: tuple[int, ...] = LAYERS
    cell_roster: tuple[tuple[int, int, str], ...] = ()
    # The canonical host claim is outside the mission root.  Binding its
    # exact RELEASED preimage prevents admission against a stale/foreign seat.
    expected_claim_sha256: str | None = None

    def __post_init__(self) -> None:
        _sha256(self.intended_basis_sha256, "intended_basis_sha256")
        _sha256(self.driver_goals_sha256, "driver_goals_sha256")
        if not self.task_id or not self.host or not self.allocation.startswith("HOST_ALLOCATION "):
            raise ValueError("task, host, and explicit HOST_ALLOCATION are required")
        if self.layers != LAYERS:
            raise ValueError(f"QTIP3 V7 missing scope is fixed to {LAYERS}")
        if self.cell_roster:
            if len(set(self.cell_roster)) != len(self.cell_roster):
                raise ValueError("cell_roster contains duplicates")
            for layer, expert, projection in self.cell_roster:
                if layer not in self.layers or expert not in EXPERTS or projection not in PROJECTIONS:
                    raise ValueError("cell_roster contains a cell outside the declared scope")
        if self.allocation.split()[1] != self.task_id or self.host not in self.allocation.split():
            raise ValueError("allocation must name this task and host")

    @property
    def expected_cells(self) -> int:
        return len(self.cell_roster) or len(self.layers) * len(EXPERTS) * len(PROJECTIONS)

    @property
    def expected_cell_scope(self) -> set[tuple[int, int, str]]:
        if self.cell_roster:
            return set(self.cell_roster)
        return {(layer, expert, projection) for layer in self.layers for expert in EXPERTS for projection in PROJECTIONS}


@dataclass(frozen=True)
class CellSpec:
    layer: int
    expert: int
    projection: Literal["fused13", "down"]
    source: Path
    control: Path
    output: Path

    def __post_init__(self) -> None:
        if self.layer not in LAYERS or self.expert not in EXPERTS or self.projection not in PROJECTIONS:
            raise ValueError("cell is outside the immutable QTIP3 V7 scope")

    @property
    def key(self) -> str:
        return f"L{self.layer:03d}/E{self.expert:03d}_{self.projection}"


def verify_basis(plan: Qtip3ApiPlan) -> dict[str, Any]:
    observed = sha256_file(plan.model_index_path)
    intended = _sha256(plan.intended_basis_sha256, "intended_basis_sha256")
    if observed != intended:
        raise RuntimeError(f"BASIS_GATE_REFUSED expected={intended} observed={observed}")
    return {"status": "PASS", "path": str(plan.model_index_path), "sha256": observed}


def verify_driver_authority(plan: Qtip3ApiPlan) -> dict[str, Any]:
    path = plan.driver_goals_path
    observed_sha = sha256_file(path)
    if observed_sha != plan.driver_goals_sha256:
        raise RuntimeError(f"DRIVER_GOALS_SHA_REFUSED expected={plan.driver_goals_sha256} observed={observed_sha}")
    text = path.read_text()
    line = plan.allocation
    if line not in text:
        raise RuntimeError(f"HOST_ALLOCATION_NOT_AUTHORIZED missing={line}")
    return {"status": "PASS", "path": str(path), "sha256": observed_sha, "allocation": line}


def _read_claim(plan: Qtip3ApiPlan) -> tuple[bytes, dict[str, Any], str]:
    raw = plan.claim_path.read_bytes()
    return raw, json.loads(raw), _sha_bytes(raw)


def _claim_lock(path: Path):
    lock = path.with_name(path.name + ".lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    return lock


def admit_host_and_shard(
    plan: Qtip3ApiPlan,
    *,
    gpu_probe: Callable[[], Sequence[Any]] = lambda: (),
    pid: int | None = None,
    config: Qtip3ApiConfig | None = None,
) -> dict[str, Any]:
    """Perform fail-closed basis/authority checks and exact local CAS admission.

    ``claim_path`` and ``shards_path`` must be task-owned destinations.  The
    predecessor NONCOMPLIANT_NON_API shard is never overwritten or adopted.
    """
    basis = verify_basis(plan)
    authority = verify_driver_authority(plan)
    if tuple(gpu_probe()):
        raise RuntimeError("GPU_NOT_EMPTY_REFUSED")
    plan.mission_root.mkdir(parents=True, exist_ok=True)
    shard_before = None
    if plan.shards_path.exists():
        shard_before = json.loads(plan.shards_path.read_text())
        shard_layers = tuple(shard_before.get("layers", shard_before.get("scope_layers", ())))
        if shard_before.get("task_id") != plan.task_id or shard_layers != plan.layers:
            raise RuntimeError("SHARD_COLLISION_REFUSED")
        if shard_before.get("status") not in {"CLAIMED", "RUNNING", "PASS", "RELEASED"}:
            raise RuntimeError("SHARD_NONCOMPLIANT_REFUSED")
        if plan.cell_roster and int(shard_before.get("missing_members", plan.expected_cells)) != plan.expected_cells:
            raise RuntimeError("SHARD_CELL_ROSTER_COUNT_REFUSED")
    pid = os.getpid() if pid is None else int(pid)
    ticks = _startticks(pid)
    if ticks is None:
        raise RuntimeError("PID_STARTTICKS_UNAVAILABLE")
    lock = _claim_lock(plan.claim_path)
    try:
        preimage, previous, preimage_sha = _read_claim(plan)
        if plan.expected_claim_sha256 is not None and preimage_sha != _sha256(
            plan.expected_claim_sha256, "expected_claim_sha256"
        ):
            raise RuntimeError(
                "HOST_CLAIM_PREIMAGE_REFUSED "
                f"expected={plan.expected_claim_sha256} observed={preimage_sha}"
            )
        # Historical claim writers sometimes left a stale state="CLAIMED"
        # after sealing status="RELEASED". Status plus null workload/controller
        # identities is the authoritative released preimage; the exact-CAS
        # postimage below repairs the stale alias without preempting work.
        if previous.get("status") != "RELEASED":
            raise RuntimeError(f"HOST_CLAIM_NOT_RELEASED_REFUSED preimage={preimage_sha}")
        if previous.get("controller_pid") is not None or previous.get("workload_pid") is not None:
            raise RuntimeError("HOST_CLAIM_LIVE_WORKLOAD_REFUSED")
        now = time.time()
        claim = dict(previous)
        claim.update({
            "schema": "banana-smasher-host-claim-v3",
            "status": "CLAIMED", "state": "CLAIMED", "task_id": plan.task_id,
            "owner_task_id": plan.task_id, "owner": plan.task_id, "owner_profile": "bs02",
            "host": plan.host, "allocation": plan.allocation, "board_run_id": plan.board_run_id,
            "run_id": plan.board_run_id, "intended_basis": plan.intended_basis_sha256,
            "source_model_index_sha256": plan.intended_basis_sha256,
            "driver_goals_sha256": plan.driver_goals_sha256,
            "claim_preimage_sha256": preimage_sha, "exact_cas_from_sha256": preimage_sha,
            "claimed_unix": now, "updated_unix": now, "heartbeat_unix": now,
            "expiry_unix": now + 7 * 86400, "lease_until_unix": now + 7 * 86400,
            "controller_pid": pid, "controller_startticks": ticks,
            "workload_pid": pid, "workload_startticks": ticks,
            "mission_root": str(plan.mission_root), "shards_path": str(plan.shards_path),
            "scope_layers": list(plan.layers), "expected_members": plan.expected_cells,
        })
        if plan.claim_path.read_bytes() != preimage:
            raise RuntimeError("HOST_CLAIM_CAS_DRIFT_REFUSED")
        claim_sha = _atomic_json(plan.claim_path, claim)
    finally:
        lock.close()
    shard = shard_before or {
        "schema": "banana-smasher-qtip3-v7-public-api-shards-v1",
        "status": "CLAIMED", "state": "CLAIMED", "task_id": plan.task_id,
        "board_run_id": plan.board_run_id, "owner_profile": "bs02", "host": plan.host,
        "intended_basis": plan.intended_basis_sha256, "layers": list(plan.layers),
        "expected_members": plan.expected_cells,
        "scope": {"layers": list(plan.layers), "projections": list(PROJECTIONS), "experts": [0, 255]},
        "claim_sha256": claim_sha,
    }
    # A released/dead same-task attempt may leave the durable shard document
    # behind. Re-admission adopts that exact scope but must bind it to the new
    # board run and claim postimage rather than retaining stale identities.
    shard.update({
        "status": "CLAIMED", "state": "CLAIMED", "task_id": plan.task_id,
        "board_run_id": plan.board_run_id, "owner_profile": "bs02", "host": plan.host,
        "intended_basis": plan.intended_basis_sha256, "layers": list(plan.layers),
        "expected_members": plan.expected_cells, "claim_sha256": claim_sha,
    })
    shard_sha = _atomic_json(plan.shards_path, shard)
    admission = {
        "schema": ADMISSION_SCHEMA, "status": "PASS", "task_id": plan.task_id,
        "board_run_id": plan.board_run_id, "host": plan.host, "allocation": plan.allocation,
        "basis": basis, "authority": authority, "claim_preimage_sha256": preimage_sha,
        "claim_sha256": claim_sha, "shards_sha256": shard_sha, "pid": pid,
        "startticks": ticks, "scope_layers": list(plan.layers), "cells": plan.expected_cells,
        "config": (config or Qtip3ApiConfig()).__dict__,
    }
    receipt = plan.mission_root / "receipts" / "ADMISSION.json"
    if receipt.exists():
        receipt = plan.mission_root / "receipts" / f"ADMISSION_{claim_sha[:12]}.json"
    admission["receipt_sha256"] = _atomic_json(receipt, admission)
    return admission


def _valid_cuda_receipt(result: dict[str, Any], config: Qtip3ApiConfig) -> None:
    if result.get("status") != "PASS" or result.get("backend") != "cuda":
        raise RuntimeError("CELL_API_NOT_PASS_CUDA")
    if result.get("codec_version") != config.codec_version or result.get("provider") != config.provider:
        raise RuntimeError("CELL_API_CODEC_PROVIDER_MISMATCH")
    geometry = result.get("geometry", {})
    # Qtip3ApiConfig.geometry is documented in public API order (B, L, V).
    if (geometry.get("B"), geometry.get("L"), geometry.get("V")) != config.geometry:
        raise RuntimeError("CELL_API_GEOMETRY_MISMATCH")
    installed = result.get("installed_cuda_decode", {})
    counters = installed.get("counters", {})
    if int(counters.get("fallback_calls", result.get("fallback_calls", -1))) != 0:
        raise RuntimeError("CELL_API_FALLBACK_REFUSED")
    if int(counters.get("cuda_decode_calls", 0)) <= 0:
        raise RuntimeError("CELL_API_CUDA_DECODE_NOT_POSITIVE")


def _cell_terminal(plan: Qtip3ApiPlan, cell: CellSpec, api_receipt: dict[str, Any], config: Qtip3ApiConfig) -> dict[str, Any]:
    _valid_cuda_receipt(api_receipt, config)
    decoded = cell.output / "decoded.npy"
    if not config.materialize_decoded:
        decoded.unlink(missing_ok=True)
    payload = {
        "schema": f"{SCHEMA}-cell",
        "status": "PASS", "task_id": plan.task_id, "basis_sha256": plan.intended_basis_sha256,
        "cell": cell.key, "layer": cell.layer, "expert": cell.expert, "projection": cell.projection,
        "api_receipt": api_receipt.get("receipt", str(cell.output / "CELL_RECEIPT.json")),
        "api_receipt_sha256": api_receipt.get("receipt_sha256"),
        "tier": config.tier, "backend": config.backend, "codec_version": config.codec_version,
        "provider": config.provider,
        "geometry": {"B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]},
        "bpw": config.bpw,
        "materialize_decoded": False, "scale_factors": [1.0],
        "scale_semantics": config.scale_semantics, "feedback_mode": "off", "objective": "sum_sse",
        "cuda_decode_calls": int(api_receipt["installed_cuda_decode"]["counters"]["cuda_decode_calls"]),
        "fallback_calls": int(api_receipt["installed_cuda_decode"]["counters"]["fallback_calls"]),
    }
    payload["receipt_sha256"] = _atomic_json(cell.output / "PUBLIC_CELL_RECEIPT.json", payload)
    return payload


def build_clean102_option_row(
    public_receipt_path: str | Path,
    *,
    prediction_receipt_path: str | Path,
    config: Qtip3ApiConfig,
    physical_bytes: int,
    prediction_by_class: dict[str, float],
    bank_sha256: str,
    teacher_sha256: str,
    scorer_sha256: str,
    model_id: str = "deepseek-ai/DeepSeek-V4-Flash-0731",
    model_revision: str = "0731",
) -> dict[str, Any]:
    """Build one SHA-bound CLEAN102 predicted-KLD option-ledger row."""
    path = Path(public_receipt_path).expanduser().resolve()
    raw = path.read_bytes()
    receipt = json.loads(raw)
    prediction_path = Path(prediction_receipt_path).expanduser().resolve()
    prediction_raw = prediction_path.read_bytes()
    prediction_receipt = json.loads(prediction_raw)
    expected_geometry = {
        "B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]
    }
    if (
        receipt.get("schema") != f"{SCHEMA}-cell"
        or receipt.get("status") != "PASS"
        or receipt.get("tier") != config.tier
        or receipt.get("provider") != config.provider
        or receipt.get("geometry") != expected_geometry
        or receipt.get("bpw") != config.bpw
    ):
        raise ValueError("PUBLIC_CELL receipt does not match the requested QTIP V7 tier")
    basis = _sha256(str(receipt.get("basis_sha256", "")), "basis_sha256")
    if model_id != "deepseek-ai/DeepSeek-V4-Flash-0731" or model_revision != "0731":
        raise ValueError("CLEAN102 QTIP V7 rows require DeepSeek-V4-Flash-0731 revision 0731")
    for value, label in (
        (bank_sha256, "bank_sha256"),
        (teacher_sha256, "teacher_sha256"),
        (scorer_sha256, "scorer_sha256"),
    ):
        _sha256(value, label)
    if isinstance(physical_bytes, bool) or not isinstance(physical_bytes, int) or physical_bytes <= 0:
        raise ValueError("physical_bytes must be a positive integer")
    codes = path.parent / "codes.npy"
    if not codes.is_file() or codes.stat().st_size != physical_bytes:
        raise ValueError("physical_bytes must equal the retained codes.npy byte count")
    if not prediction_by_class or any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for name, value in prediction_by_class.items()
    ):
        raise ValueError("prediction_by_class must contain finite non-negative KLD values")
    match = re.fullmatch(r"L(\d{3})/E(\d{3})_(down|fused13)", str(receipt.get("cell", "")))
    if match is None:
        raise ValueError("PUBLIC_CELL receipt has an invalid cell identity")
    layer, expert, projection = int(match.group(1)), int(match.group(2)), match.group(3)
    cell_id = f"L{layer:03d}:E{expert:03d}:{projection}"
    if (
        prediction_receipt.get("status") != "PASS"
        or prediction_receipt.get("basis_sha256") != basis
        or prediction_receipt.get("cell_id") != cell_id
    ):
        raise ValueError("CLEAN102 prediction receipt does not match PUBLIC_CELL basis/cell")
    descriptor = {"path": str(path), "sha256": _sha_bytes(raw), "bytes": len(raw)}
    prediction_descriptor = {
        "path": str(prediction_path), "sha256": _sha_bytes(prediction_raw),
        "bytes": len(prediction_raw),
    }
    return {
        "schema": "banana-smasher-provenance-option-row-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "basis_sha256": basis,
        "bank_sha256": bank_sha256,
        "teacher_sha256": teacher_sha256,
        "scorer_sha256": scorer_sha256,
        "cell_id": cell_id,
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "tier": config.tier,
        "physical_bytes": physical_bytes,
        "activation_ids": [],
        "prediction_by_class": {
            name: float(value) for name, value in sorted(prediction_by_class.items())
        },
        "prediction_producer": prediction_descriptor,
        "physical_producer": {**descriptor, "artifact_sha256": sha256_file(codes)},
    }


def _adopt_existing_api_receipt(
    plan: Qtip3ApiPlan, cell: CellSpec, config: Qtip3ApiConfig
) -> dict[str, Any]:
    """Fail-closed adoption of a physically PASSed public cell API receipt."""
    path = cell.output / "CELL_RECEIPT.json"
    result = json.loads(path.read_text())
    _valid_cuda_receipt(result, config)
    identities = (
        (result.get("basis_sha256"), plan.intended_basis_sha256, "basis"),
        (result.get("source", {}).get("sha256"), sha256_file(cell.source), "source"),
        (result.get("control", {}).get("sha256"), sha256_file(cell.control), "control"),
        (result.get("tlut", {}).get("sha256"), sha256_file(plan.tlut_path), "tlut"),
    )
    for observed, expected, label in identities:
        if observed != expected:
            raise RuntimeError(
                f"CELL_API_ADOPTION_{label.upper()}_MISMATCH expected={expected} observed={observed}"
            )
    return result


def _is_resumable(path: Path, plan: Qtip3ApiPlan, cell: CellSpec, config: Qtip3ApiConfig) -> bool:
    try:
        value = json.loads(path.read_text())
        return (
            value.get("status") == "PASS" and value.get("task_id") == plan.task_id
            and value.get("cell") == cell.key and value.get("basis_sha256") == plan.intended_basis_sha256
            and value.get("tier") == config.tier and value.get("provider") == config.provider
            and value.get("geometry") == {
                "B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]
            }
            and value.get("bpw") == config.bpw and value.get("fallback_calls") == 0
            and int(value.get("cuda_decode_calls", 0)) > 0
        )
    except (OSError, ValueError, TypeError):
        return False


def run_cells(
    plan: Qtip3ApiPlan,
    config: Qtip3ApiConfig,
    cells: Iterable[CellSpec],
    *,
    api: Callable[..., dict[str, Any]] = build_qtip_native_cell,
    prepare_cell: Callable[[CellSpec], None] | None = None,
    cleanup_cell: Callable[[CellSpec], None] | None = None,
) -> dict[str, Any]:
    """Run exactly the declared cells, resuming only valid PASS receipts."""
    rows = tuple(cells)
    expected = plan.expected_cell_scope
    actual = {(cell.layer, cell.expert, cell.projection) for cell in rows}
    if actual != expected or len(rows) != plan.expected_cells:
        raise ValueError(f"cell scope mismatch expected={len(expected)} actual={len(actual)}")
    admission_path = plan.mission_root / "receipts" / "ADMISSION.json"
    if not admission_path.is_file():
        raise RuntimeError("ADMISSION_RECEIPT_REQUIRED")
    pid = os.getpid()
    ticks = _startticks(pid)
    pid_receipt = plan.mission_root / "receipts" / "PID_STARTTICKS.json"
    if pid_receipt.exists():
        pid_receipt = plan.mission_root / "receipts" / f"PID_STARTTICKS_{pid}_{ticks}.json"
    _atomic_json(pid_receipt, {
        "schema": f"{SCHEMA}-pid", "status": "RUNNING", "task_id": plan.task_id,
        "pid": pid, "startticks": ticks, "created_unix": time.time(),
    })
    passed: list[dict[str, Any]] = []
    layer_counts: dict[int, int] = {layer: 0 for layer in plan.layers}
    accepted_layers: set[int] = set()
    for cell in sorted(rows, key=lambda value: (value.layer, value.expert, PROJECTIONS.index(value.projection))):
        cell.output.mkdir(parents=True, exist_ok=True)
        public_receipt = cell.output / "PUBLIC_CELL_RECEIPT.json"
        if _is_resumable(public_receipt, plan, cell, config):
            accepted = json.loads(public_receipt.read_text())
            accepted["receipt_sha256"] = sha256_file(public_receipt)
        else:
            api_receipt = cell.output / "CELL_RECEIPT.json"
            if api_receipt.is_file():
                # A wrapper-only failure after the public API sealed PASS must
                # adopt exact bytes instead of replaying expensive CUDA work.
                result = _adopt_existing_api_receipt(plan, cell, config)
            else:
                if prepare_cell is not None:
                    prepare_cell(cell)
                if cell.source.is_symlink() or cell.control.is_symlink():
                    raise RuntimeError(f"CELL_INPUT_SYMLINK_REFUSED {cell.key}")
                result = api(
                    cell.source, cell.control, plan.tlut_path, cell.output,
                    bpw=config.bpw, codec_version=config.codec_version, backend=config.backend,
                    intended_basis_sha256=plan.intended_basis_sha256,
                    observed_basis_sha256=plan.intended_basis_sha256,
                    scale_factors=config.scale_factors, ldlq_scale_semantics=config.scale_semantics,
                    feedback_mode=config.feedback_mode, trellis_objective=config.trellis_objective,
                    decode_repeats=config.decode_repeats, reserve_bytes=config.reserve_bytes,
                    solve_batch=config.solve_batch, decode_batch=config.decode_batch,
                )
            accepted = _cell_terminal(plan, cell, result, config)
            if cleanup_cell is not None:
                cleanup_cell(cell)
        passed.append(accepted)
        layer_counts[cell.layer] += 1
        if layer_counts[cell.layer] == len(EXPERTS) * len(PROJECTIONS):
            accepted_layers.add(cell.layer)
            layer_root = plan.mission_root / "receipts" / "layers" / f"L{cell.layer:03d}"
            layer_payload = {
                "schema": f"{SCHEMA}-layer-terminal", "status": "PASS",
                "task_id": plan.task_id, "board_run_id": plan.board_run_id,
                "basis_sha256": plan.intended_basis_sha256, "layer": cell.layer,
                "accepted_cells": layer_counts[cell.layer], "expected_cells": len(EXPERTS) * len(PROJECTIONS),
                "pid": pid, "startticks": ticks,
            }
            layer_sha = _atomic_json(layer_root / "LAYER_PRODUCT_TERMINAL.json", layer_payload)
            _atomic_json(layer_root / "ACK.json", {
                "schema": f"{SCHEMA}-layer-ack", "status": "PASS", "task_id": plan.task_id,
                "layer": cell.layer, "terminal_sha256": layer_sha,
            })
        progress = {
            "schema": PROGRESS_SCHEMA, "status": "RUNNING", "task_id": plan.task_id,
            "board_run_id": plan.board_run_id, "basis_sha256": plan.intended_basis_sha256,
            "accepted_cells": len(passed), "accepted_layers": len(accepted_layers),
            "cells_passed": len(passed),
            "cells_expected": plan.expected_cells, "last_cell": cell.key,
            "latest_receipt": str(public_receipt),
            "latest_receipt_sha256": accepted.get("receipt_sha256"),
            "pid_receipt": str(pid_receipt),
            "pid": pid, "startticks": ticks,
        }
        _write_progress_monotone(plan.mission_root / "PROGRESS.json", progress)
        _write_progress_monotone(plan.mission_root / "receipts" / "PROGRESS.json", progress)
    terminal_payload = {
        "schema": TERMINAL_SCHEMA, "status": "PASS", "task_id": plan.task_id,
        "board_run_id": plan.board_run_id, "basis_sha256": plan.intended_basis_sha256,
        "host": plan.host, "allocation": plan.allocation, "layers": list(plan.layers),
        "cells": len(passed), "expected_cells": plan.expected_cells,
        "cuda_positive": all(int(row["cuda_decode_calls"]) > 0 for row in passed),
        "fallback_calls": sum(int(row["fallback_calls"]) for row in passed),
        "materialize_decoded": False, "tier": config.tier, "bpw": config.bpw,
        "provider": config.provider,
        "geometry": {"B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]},
        "cell_receipts": [row["receipt_sha256"] for row in passed],
        "pid": pid, "startticks": ticks,
    }
    terminal = plan.mission_root / "receipts" / "PRODUCER_TERMINAL.json"
    terminal_payload["receipt_sha256"] = _atomic_json(terminal, terminal_payload)
    return terminal_payload


def run_cells_batched(
    plan: Qtip3ApiPlan,
    config: Qtip3ApiConfig,
    cells: Iterable[CellSpec],
    *,
    batch_api: Callable[..., Sequence[dict[str, Any]]],
    # Forty cells is the smallest production batch that amortizes cross-cell
    # dispatch while retaining the exact source/output ordering and bounded
    # resident footprint proven by the public-path probe.
    batch_size: int = 40,
    prepare_cell: Callable[[CellSpec], None] | None = None,
    cleanup_cell: Callable[[CellSpec], None] | None = None,
    batch_source_root: Path | None = None,
    max_new_batches: int | None = None,
) -> dict[str, Any]:
    """Resume exact cells while invoking the public CUDA API in bounded batches.

    Already sealed public receipts are adopted without replay. A wrapper-only
    failure is adopted from its exact cell receipt before any new batch starts.
    New source tensors are copied to unique durable paths so one public batch
    cannot alias the producer's shared materialization scratch file.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if max_new_batches is not None and (
        isinstance(max_new_batches, bool)
        or not isinstance(max_new_batches, int)
        or max_new_batches < 1
    ):
        raise ValueError("max_new_batches must be a positive integer or None")
    rows = tuple(cells)
    expected = plan.expected_cell_scope
    actual = {(cell.layer, cell.expert, cell.projection) for cell in rows}
    if actual != expected or len(rows) != plan.expected_cells:
        raise ValueError(f"cell scope mismatch expected={len(expected)} actual={len(actual)}")
    if not (plan.mission_root / "receipts" / "ADMISSION.json").is_file():
        raise RuntimeError("ADMISSION_RECEIPT_REQUIRED")
    pid = os.getpid()
    ticks = _startticks(pid)
    pid_receipt = plan.mission_root / "receipts" / "PID_STARTTICKS.json"
    if pid_receipt.exists():
        pid_receipt = plan.mission_root / "receipts" / f"PID_STARTTICKS_{pid}_{ticks}.json"
    _atomic_json(pid_receipt, {
        "schema": f"{SCHEMA}-pid", "status": "RUNNING", "task_id": plan.task_id,
        "pid": pid, "startticks": ticks, "created_unix": time.time(),
        "execution": "public-cross-cell-batch",
    })
    source_root = batch_source_root or plan.mission_root / "working_full_api" / "batch_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda value: (value.layer, value.expert, PROJECTIONS.index(value.projection)))
    passed: list[dict[str, Any]] = []
    layer_counts: dict[int, int] = {layer: 0 for layer in plan.layers}
    accepted_layers: set[int] = set()
    new_batches = 0
    new_cells = 0

    def accept(cell: CellSpec, accepted: dict[str, Any]) -> None:
        passed.append(accepted)
        layer_counts[cell.layer] += 1
        if layer_counts[cell.layer] == len(EXPERTS) * len(PROJECTIONS):
            accepted_layers.add(cell.layer)
            layer_root = plan.mission_root / "receipts" / "layers" / f"L{cell.layer:03d}"
            layer_payload = {
                "schema": f"{SCHEMA}-layer-terminal", "status": "PASS",
                "task_id": plan.task_id, "board_run_id": plan.board_run_id,
                "basis_sha256": plan.intended_basis_sha256, "layer": cell.layer,
                "accepted_cells": layer_counts[cell.layer],
                "expected_cells": len(EXPERTS) * len(PROJECTIONS),
                "pid": pid, "startticks": ticks, "execution": "public-cross-cell-batch",
            }
            layer_sha = _atomic_json(layer_root / "LAYER_PRODUCT_TERMINAL.json", layer_payload)
            _atomic_json(layer_root / "ACK.json", {
                "schema": f"{SCHEMA}-layer-ack", "status": "PASS", "task_id": plan.task_id,
                "layer": cell.layer, "terminal_sha256": layer_sha,
            })
        public_receipt = cell.output / "PUBLIC_CELL_RECEIPT.json"
        progress = {
            "schema": PROGRESS_SCHEMA, "status": "RUNNING", "task_id": plan.task_id,
            "board_run_id": plan.board_run_id, "basis_sha256": plan.intended_basis_sha256,
            "accepted_cells": len(passed), "accepted_layers": len(accepted_layers),
            "cells_passed": len(passed), "cells_expected": plan.expected_cells,
            "last_cell": cell.key, "latest_receipt": str(public_receipt),
            "latest_receipt_sha256": accepted.get("receipt_sha256"),
            "pid_receipt": str(pid_receipt), "pid": pid, "startticks": ticks,
            "execution": "public-cross-cell-batch", "batch_size": batch_size,
        }
        _write_progress_monotone(plan.mission_root / "PROGRESS.json", progress)
        _write_progress_monotone(plan.mission_root / "receipts" / "PROGRESS.json", progress)

    index = 0
    while index < len(ordered):
        cell = ordered[index]
        cell.output.mkdir(parents=True, exist_ok=True)
        public_receipt = cell.output / "PUBLIC_CELL_RECEIPT.json"
        if _is_resumable(public_receipt, plan, cell, config):
            accepted = json.loads(public_receipt.read_text())
            accepted["receipt_sha256"] = sha256_file(public_receipt)
            accept(cell, accepted)
            index += 1
            continue
        api_receipt = cell.output / "CELL_RECEIPT.json"
        if api_receipt.is_file():
            result = _adopt_existing_api_receipt(plan, cell, config)
            accepted = _cell_terminal(plan, cell, result, config)
            if cleanup_cell is not None:
                cleanup_cell(cell)
            accept(cell, accepted)
            index += 1
            continue

        pending: list[tuple[CellSpec, CellSpec, Path]] = []
        cursor = index
        while cursor < len(ordered) and len(pending) < batch_size:
            candidate = ordered[cursor]
            candidate.output.mkdir(parents=True, exist_ok=True)
            if (candidate.output / "PUBLIC_CELL_RECEIPT.json").is_file() or (candidate.output / "CELL_RECEIPT.json").is_file():
                break
            if prepare_cell is not None:
                prepare_cell(candidate)
            if candidate.source.is_symlink() or candidate.control.is_symlink():
                raise RuntimeError(f"CELL_INPUT_SYMLINK_REFUSED {candidate.key}")
            durable_source = source_root / f"{candidate.key.replace('/', '_')}.npy"
            temporary = durable_source.with_name(f".{durable_source.name}.tmp")
            shutil.copyfile(candidate.source, temporary)
            os.replace(temporary, durable_source)
            batch_cell = CellSpec(
                layer=candidate.layer, expert=candidate.expert, projection=candidate.projection,
                source=durable_source, control=candidate.control, output=candidate.output,
            )
            pending.append((candidate, batch_cell, durable_source))
            cursor += 1
        if not pending:
            continue
        results = list(batch_api(
            [{"source": batch_cell.source, "control": batch_cell.control, "output": batch_cell.output}
             for _cell, batch_cell, _source in pending],
            plan.tlut_path, bpw=config.bpw, codec_version=config.codec_version,
            backend=config.backend, intended_basis_sha256=plan.intended_basis_sha256,
            observed_basis_sha256=plan.intended_basis_sha256,
            solve_batch=config.solve_batch, decode_batch=config.decode_batch,
            decode_repeats=config.decode_repeats, scale_factors=config.scale_factors,
            ldlq_scale_semantics=config.scale_semantics, feedback_mode=config.feedback_mode,
            trellis_objective=config.trellis_objective, reserve_bytes=config.reserve_bytes,
        ))
        if len(results) != len(pending):
            raise RuntimeError("PUBLIC_BATCH_RESULT_COUNT_MISMATCH")
        for (original, _batch_cell, durable_source), result in zip(pending, results, strict=True):
            accepted = _cell_terminal(plan, original, result, config)
            if cleanup_cell is not None:
                cleanup_cell(original)
            durable_source.unlink(missing_ok=True)
            accept(original, accepted)
            new_cells += 1
        index += len(pending)
        new_batches += 1
        if max_new_batches is not None and new_batches >= max_new_batches:
            break

    terminal_payload = {
        "schema": TERMINAL_SCHEMA, "status": "PASS", "task_id": plan.task_id,
        "board_run_id": plan.board_run_id, "basis_sha256": plan.intended_basis_sha256,
        "host": plan.host, "allocation": plan.allocation, "layers": list(plan.layers),
        "cells": len(passed), "expected_cells": plan.expected_cells,
        "cuda_positive": all(int(row["cuda_decode_calls"]) > 0 for row in passed),
        "fallback_calls": sum(int(row["fallback_calls"]) for row in passed),
        "materialize_decoded": False, "tier": config.tier, "bpw": config.bpw,
        "provider": config.provider,
        "geometry": {"B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]},
        "cell_receipts": [row["receipt_sha256"] for row in passed],
        "pid": pid, "startticks": ticks, "execution": "public-cross-cell-batch",
        "batch_size": batch_size, "bounded_partial": max_new_batches is not None,
        "new_batches": new_batches, "new_cells": new_cells,
    }
    if plan.cell_roster:
        terminal_payload["cell_roster"] = [list(row) for row in plan.cell_roster]
    terminal = plan.mission_root / "receipts" / "PRODUCER_TERMINAL.json"
    terminal_payload["receipt_sha256"] = _atomic_json(terminal, terminal_payload)
    return terminal_payload


def release_bounded_host(plan: Qtip3ApiPlan, terminal_path: str | Path) -> dict[str, Any]:
    """Release this PID after a sealed, nonempty bounded production batch."""
    terminal = Path(terminal_path)
    value = json.loads(terminal.read_text())
    if (
        value.get("status") != "PASS"
        or value.get("task_id") != plan.task_id
        or not value.get("bounded_partial")
        or int(value.get("new_batches", 0)) != 1
        or int(value.get("new_cells", 0)) < 1
        or int(value.get("new_cells", 0)) > int(value.get("batch_size", 0))
    ):
        raise RuntimeError("BOUNDED_RELEASE_REQUIRES_ONE_NONEMPTY_BATCH")
    lock = _claim_lock(plan.claim_path)
    try:
        preimage, claim, preimage_sha = _read_claim(plan)
        if claim.get("task_id") != plan.task_id or claim.get("controller_pid") != os.getpid():
            raise RuntimeError("BOUNDED_RELEASE_IDENTITY_REFUSED")
        post = dict(claim)
        post.update({
            "status": "RELEASED", "state": "RELEASED", "controller_pid": None,
            "controller_startticks": None, "workload_pid": None, "workload_startticks": None,
            "released_unix": time.time(), "release_terminal_path": str(terminal),
            "release_terminal_sha256": sha256_file(terminal),
            "release_reason": "PASS_BOUNDED_PRODUCTION_BATCH",
        })
        if plan.claim_path.read_bytes() != preimage:
            raise RuntimeError("BOUNDED_RELEASE_CAS_DRIFT_REFUSED")
        post_sha = _atomic_json(plan.claim_path, post)
    finally:
        lock.close()
    result = {
        "schema": f"{RELEASE_SCHEMA}-bounded", "status": "PASS", "task_id": plan.task_id,
        "host": plan.host, "preimage_sha256": preimage_sha, "postimage_sha256": post_sha,
        "terminal_sha256": sha256_file(terminal), "new_cells": int(value["new_cells"]),
    }
    result["receipt_sha256"] = _atomic_json(
        plan.mission_root / "receipts" / "BOUNDED_RELEASE.json", result
    )
    return result


def release_unstarted_admission(plan: Qtip3ApiPlan) -> dict[str, Any]:
    """Exact-CAS recovery for a claim-owning admission that never launched.

    This is intentionally narrower than ``release_host``: it accepts only a
    same-task claim whose recorded controller/workload identity is no longer
    live, so a failed one-shot admission cannot strand Spark-1.
    """
    lock = _claim_lock(plan.claim_path)
    try:
        preimage, claim, preimage_sha = _read_claim(plan)
        if claim.get("task_id") != plan.task_id or claim.get("board_run_id") != plan.board_run_id:
            raise RuntimeError("ABORT_RELEASE_IDENTITY_REFUSED")
        if claim.get("status") != "CLAIMED":
            raise RuntimeError("ABORT_RELEASE_REQUIRES_CLAIMED")
        pids = [claim.get("controller_pid"), claim.get("workload_pid")]
        for value in pids:
            if value is not None and int(value) != os.getpid() and _startticks(int(value)) is not None:
                raise RuntimeError("ABORT_RELEASE_LIVE_WORKLOAD_REFUSED")
        post = dict(claim)
        post.update({"status": "RELEASED", "state": "RELEASED", "controller_pid": None,
                     "controller_startticks": None, "workload_pid": None,
                     "workload_startticks": None, "released_unix": time.time(),
                     "release_reason": "UNSTARTED_ADMISSION_RECOVERY",
                     "release_preimage_sha256": preimage_sha})
        if plan.claim_path.read_bytes() != preimage:
            raise RuntimeError("ABORT_RELEASE_CAS_DRIFT_REFUSED")
        post_sha = _atomic_json(plan.claim_path, post)
    finally:
        lock.close()
    result = {"schema": f"{RELEASE_SCHEMA}-unstarted-recovery", "status": "PASS",
              "task_id": plan.task_id, "preimage_sha256": preimage_sha,
              "postimage_sha256": post_sha}
    result["receipt_sha256"] = _atomic_json(
        plan.mission_root / "receipts" / (
            "UNSTARTED_ADMISSION_RELEASE.json"
            if not (plan.mission_root / "receipts" / "UNSTARTED_ADMISSION_RELEASE.json").exists()
            else f"UNSTARTED_ADMISSION_RELEASE_{post_sha[:12]}.json"
        ), result
    )
    return result


def release_smoke_host(plan: Qtip3ApiPlan, terminal_path: str | Path) -> dict[str, Any]:
    """Release this process's claim after a sealed same-basis smoke PASS."""
    terminal = Path(terminal_path)
    value = json.loads(terminal.read_text())
    if (
        value.get("status") != "PASS"
        or value.get("task_id") != plan.task_id
        or value.get("basis_sha256") != plan.intended_basis_sha256
        or int(value.get("cells", 0)) < 20
        or float(value.get("cells_per_minute", 0.0)) < 20.0
        or value.get("parity") != "bitwise"
    ):
        raise RuntimeError("SMOKE_RELEASE_REQUIRES_PARITY_THROUGHPUT_PASS")
    lock = _claim_lock(plan.claim_path)
    try:
        preimage, claim, preimage_sha = _read_claim(plan)
        if claim.get("task_id") != plan.task_id or claim.get("controller_pid") != os.getpid():
            raise RuntimeError("SMOKE_RELEASE_IDENTITY_REFUSED")
        post = dict(claim)
        post.update({
            "status": "RELEASED", "state": "RELEASED", "controller_pid": None,
            "controller_startticks": None, "workload_pid": None, "workload_startticks": None,
            "released_unix": time.time(), "release_terminal_path": str(terminal),
            "release_terminal_sha256": sha256_file(terminal), "release_reason": "SMOKE_PASS",
        })
        if plan.claim_path.read_bytes() != preimage:
            raise RuntimeError("SMOKE_RELEASE_CAS_DRIFT_REFUSED")
        post_sha = _atomic_json(plan.claim_path, post)
    finally:
        lock.close()
    result = {
        "schema": f"{RELEASE_SCHEMA}-smoke", "status": "PASS", "task_id": plan.task_id,
        "host": plan.host, "preimage_sha256": preimage_sha, "postimage_sha256": post_sha,
        "terminal_sha256": sha256_file(terminal),
    }
    result["receipt_sha256"] = _atomic_json(
        plan.mission_root / "receipts" / "SMOKE_RELEASE.json", result
    )
    return result


def release_host(plan: Qtip3ApiPlan, terminal_path: str | Path) -> dict[str, Any]:
    """Release only a claim owned by this task/PID after a sealed PASS terminal."""
    terminal = Path(terminal_path)
    value = json.loads(terminal.read_text())
    if value.get("status") != "PASS" or value.get("task_id") != plan.task_id or int(value.get("cells", -1)) != plan.expected_cells:
        raise RuntimeError("RELEASE_REQUIRES_COMPLETE_PASS_TERMINAL")
    lock = _claim_lock(plan.claim_path)
    try:
        preimage, claim, preimage_sha = _read_claim(plan)
        if claim.get("task_id") != plan.task_id or claim.get("controller_pid") != os.getpid():
            raise RuntimeError("RELEASE_IDENTITY_REFUSED")
        post = dict(claim)
        post.update({"status": "RELEASED", "state": "RELEASED", "controller_pid": None,
                     "controller_startticks": None, "workload_pid": None, "workload_startticks": None,
                     "released_unix": time.time(), "release_terminal_path": str(terminal),
                     "release_terminal_sha256": sha256_file(terminal), "release_reason": "PASS_PRODUCER"})
        if plan.claim_path.read_bytes() != preimage:
            raise RuntimeError("RELEASE_CAS_DRIFT_REFUSED")
        post_sha = _atomic_json(plan.claim_path, post)
    finally:
        lock.close()
    result = {"schema": RELEASE_SCHEMA, "status": "PASS", "task_id": plan.task_id,
              "host": plan.host, "preimage_sha256": preimage_sha, "postimage_sha256": post_sha,
              "terminal_sha256": sha256_file(terminal)}
    result["receipt_sha256"] = _atomic_json(plan.mission_root / "receipts" / "RELEASE.json", result)
    return result


__all__ = [
    "BASIS", "LAYERS", "PROJECTIONS", "EXPECTED_CELLS", "CellSpec", "Qtip3ApiConfig",
    "Qtip3ApiPlan", "admit_host_and_shard", "release_bounded_host", "release_host", "release_smoke_host", "release_unstarted_admission", "run_cells", "run_cells_batched", "sha256_file",
    "verify_basis", "verify_driver_authority",
]
