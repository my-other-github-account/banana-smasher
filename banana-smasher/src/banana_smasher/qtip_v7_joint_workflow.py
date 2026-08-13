"""One-line orchestration for the public QTIP V7 joint-repair workflow."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from .qtip_v7_repair import load_qtip_v7_artifact

_FREEZE_SCHEMA = "banana-smasher-qtip-v7-joint-freeze-v1"
_CHECKPOINT_FORMAT = "banana-smasher-qtip-v7-joint-checkpoint-v1"
_CHECKPOINT_RECEIPT_SCHEMA = "banana-smasher-qtip-v7-joint-checkpoint-receipt-v1"
_SHARD_SCHEMA = "banana-smasher-qtip-v7-balanced64-shard-v1"
_INVENTORY = {
    "layers": 43,
    "layer_luts": 43,
    "rmsnorm_masters": 235,
    "output_gains": 43,
}
_HOST_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$")
_REMOTE_ROOT_PATTERN = re.compile(r"^/(?:[A-Za-z0-9._-]+/?)+$")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")



def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} requires a lowercase SHA-256 identity")
    return value


def _normalize_host(value: str) -> str:
    host = value.strip().lower().rstrip(".")
    if not host or not _HOST_PATTERN.fullmatch(host):
        raise ValueError(f"invalid host identity {value!r}")
    return host


def _same_host(left: str, right: str) -> bool:
    """Compare sealed/observed host identities without ignoring route identity."""
    left = _normalize_host(left)
    right = _normalize_host(right)
    if left == right:
        return True
    try:
        ipaddress.ip_address(left)
        left_is_ip = True
    except ValueError:
        left_is_ip = False
    try:
        ipaddress.ip_address(right)
        right_is_ip = True
    except ValueError:
        right_is_ip = False
    if left_is_ip or right_is_ip:
        return False
    return left.split(".", 1)[0] == right.split(".", 1)[0]


def _host_aliases(host: str, aliases: Sequence[str] = ()) -> list[str]:
    identities = {_normalize_host(host), *(_normalize_host(value) for value in aliases)}
    for value in tuple(identities):
        try:
            identities.update(
                _normalize_host(str(row[4][0]))
                for row in socket.getaddrinfo(value, None)
                if row[4]
            )
        except socket.gaierror:
            pass
        try:
            canonical, names, addresses = socket.gethostbyaddr(value)
            identities.update(_normalize_host(item) for item in (canonical, *names, *addresses))
        except (socket.gaierror, socket.herror):
            pass
    return sorted(identities)


def _matches_any_host(value: str, identities: Sequence[str]) -> bool:
    return any(_same_host(value, identity) for identity in identities)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> dict[str, Any]:
    payload = _canonical(value)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {resolved}")
    return resolved, value


def _load_freeze(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved, value = _load_json(path)
    if resolved.stat().st_mode & 0o222:
        raise RuntimeError("QTIP V7 frozen-input receipt must be read-only")
    if value.get("schema") != _FREEZE_SCHEMA or value.get("status") != "PASS":
        raise ValueError("QTIP V7 joint repair requires a PASS frozen-input receipt")
    if value.get("inventory") != _INVENTORY:
        raise ValueError("QTIP V7 frozen inventory surface drift")
    manifest = Path(str(value["manifest"]["path"]))
    bank = Path(str(value["teacher_bank"]["path"]))
    for label, target, record in (
        ("manifest", manifest, value["manifest"]),
        ("teacher bank", bank, value["teacher_bank"]),
    ):
        if not target.is_file() or _sha256(target) != record.get("sha256"):
            raise RuntimeError(f"frozen QTIP V7 {label} identity drift")
    return resolved, value


def inspect_joint_inputs(
    *,
    manifest: str | Path,
    teacher_bank: str | Path,
    run_root: str | Path,
    trainer_host: str,
    trainer_aliases: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and freeze the exact all-43 V7 inventory and 64-window teacher bank."""
    manifest_path = Path(manifest).expanduser().resolve()
    source = load_qtip_v7_artifact(manifest_path)
    external_rows = source.document.get("external_layers", [])
    if len(source.external_paths) != len(external_rows) or any(
        not isinstance(row, dict) or not isinstance(row.get("members"), list)
        for row in external_rows
    ):
        raise ValueError(
            "QTIP V7 joint inspection requires physical readback and exact roster for every external layer"
        )
    layers = sorted(source.layer_luts)
    if layers != list(range(43)):
        raise ValueError(f"QTIP V7 joint repair requires exact layers 0..42, got {layers}")
    members_by_layer = {layer: 0 for layer in range(43)}
    physical_identities: dict[int, set[tuple[int, str]]] = {
        layer: set() for layer in range(43)
    }
    for row in source.document["members"]:
        layer = int(row["layer"])
        members_by_layer[layer] += 1
        physical_identities[layer].add((int(row["expert"]), str(row["projection"])))
    for row in source.document.get("external_layers", []):
        members_by_layer[int(row["layer"])] += int(row["member_count"])
    if set(members_by_layer.values()) != {768}:
        raise ValueError(
            "QTIP V7 joint repair requires exactly 768 members for each of 43 layers: "
            f"{members_by_layer}"
        )
    expected_identities = {
        (expert, projection)
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
    }
    for layer, identities in physical_identities.items():
        if identities and identities != expected_identities:
            raise ValueError(
                f"QTIP V7 layer {layer} requires exact experts 0..255 × w1/w2/w3"
            )
    trainable_surface = source.document.get("joint_trainable_surface")
    if not isinstance(trainable_surface, dict):
        raise ValueError("QTIP V7 manifest requires exact joint_trainable_surface keys and shapes")
    expected_keys = {
        "layer_luts": {f"L{i:03d}": [1024] for i in range(43)},
        "norms": {f"rmsnorm_{i:03d}": [2] for i in range(235)},
        "outputs": {f"output_gain_L{i:03d}": [] for i in range(43)},
    }
    if trainable_surface != expected_keys:
        raise ValueError("QTIP V7 joint_trainable_surface keys/shapes drift")
    bank_path, bank = _load_json(teacher_bank)
    windows = bank.get("windows")
    if not isinstance(windows, list) or len(windows) != 64 or len({json.dumps(row, sort_keys=True) for row in windows}) != 64:
        raise ValueError("QTIP V7 teacher bank requires exactly 64 unique ordered windows")
    teacher_identity = bank.get("teacher_sha256")
    if teacher_identity is None and isinstance(bank.get("identities"), dict):
        teacher_identity = bank["identities"].get("teacher", {}).get("sha256")
    teacher_identity = _require_sha256(teacher_identity, "QTIP V7 teacher SHA-256")
    logits_sha256 = _require_sha256(
        bank.get("teacher_logits_sha256"), "QTIP V7 teacher-logits SHA-256"
    )
    if logits_sha256 != hashlib.sha256(_canonical(windows)).hexdigest():
        raise ValueError("QTIP V7 teacher-logits SHA-256 identity drift")
    trainer_identities = _host_aliases(trainer_host, trainer_aliases)
    document = {
        "schema": _FREEZE_SCHEMA,
        "status": "PASS",
        "inventory": dict(_INVENTORY),
        "trainable_surface": trainable_surface,
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "members": len(source.member_paths) + source.external_member_count,
            "physical_accounting": "requires qtip-v7-wire verified layer receipts",
        },
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": _sha256(bank_path),
            "teacher_sha256": teacher_identity,
            "teacher_logits_sha256": logits_sha256,
            "windows": len(windows),
        },
        "objective": "teacher_kld",
        "trainer_host": _normalize_host(trainer_host),
        "trainer_identities": trainer_identities,
    }
    output = Path(run_root).expanduser().resolve() / "FROZEN_INPUTS.json"
    record = _write_json(output, document, exclusive=True)
    os.chmod(output, 0o444)
    return {**document, "freeze": record}


def _checkpoint_surface(
    checkpoint: str | Path,
    expected_surface: dict[str, dict[str, list[int]]],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    import torch

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"joint checkpoint format must be {_CHECKPOINT_FORMAT!r}")
    update = payload.get("update")
    kld = payload.get("teacher_kld")
    objective = payload.get("objective")
    freeze_sha256 = payload.get("freeze_sha256")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("joint checkpoint requires a nonnegative integer update")
    if isinstance(kld, bool) or not isinstance(kld, (int, float)) or not math.isfinite(float(kld)) or float(kld) < 0:
        raise ValueError("joint checkpoint requires a finite nonnegative teacher_kld")
    if objective != "teacher_kld":
        raise ValueError("joint checkpoint objective must be teacher_kld")
    if not isinstance(freeze_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256):
        raise ValueError("joint checkpoint requires the frozen-input SHA-256 identity")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("joint checkpoint requires state")
    names = ("layer_luts", "norms", "outputs")
    if any(not isinstance(state.get(name), dict) for name in names):
        raise ValueError("joint checkpoint state requires layer_luts/norms/outputs maps")
    counts = {
        "layer_luts": len(state["layer_luts"]),
        "rmsnorm_masters": len(state["norms"]),
        "output_gains": len(state["outputs"]),
    }
    expected = {name: _INVENTORY[name] for name in counts}
    if counts != expected:
        raise ValueError(f"joint checkpoint trainable surface drift: expected={expected} actual={counts}")
    for group in names:
        if set(state[group]) != set(expected_surface[group]):
            raise ValueError(f"joint checkpoint {group} keys drift from frozen surface")
        for name, tensor in state[group].items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"joint checkpoint {group} entries must be named tensors")
            if list(tensor.shape) != expected_surface[group][name]:
                raise ValueError(
                    f"joint checkpoint {group}/{name} shape drift: "
                    f"{list(tensor.shape)} != {expected_surface[group][name]}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"joint checkpoint contains nonfinite tensor {group}/{name}")
    return path, payload, counts


def _authenticated_kld(bank: dict[str, Any], predictions: object) -> tuple[float, int]:
    if not isinstance(predictions, list) or len(predictions) != 64:
        raise ValueError("joint checkpoint requires 64 authenticated prediction rows")
    losses = []
    for ordinal, (window, predicted) in enumerate(
        zip(bank["windows"], predictions, strict=True)
    ):
        teacher = window.get("teacher_logits") if isinstance(window, dict) else None
        if (
            not isinstance(teacher, list)
            or len(teacher) < 2
            or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in teacher)
            or not isinstance(predicted, list)
            or len(predicted) != len(teacher)
            or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in predicted)
        ):
            raise ValueError(f"invalid authenticated KLD logits at teacher window {ordinal}")
        teacher_array = np.asarray(teacher, dtype=np.float64)
        predicted_array = np.asarray(predicted, dtype=np.float64)
        teacher_array -= teacher_array.max()
        predicted_array -= predicted_array.max()
        teacher_probability = np.exp(teacher_array)
        teacher_probability /= teacher_probability.sum()
        predicted_probability = np.exp(predicted_array)
        predicted_probability /= predicted_probability.sum()
        losses.append(float(np.sum(
            teacher_probability
            * (np.log(teacher_probability) - np.log(predicted_probability))
        )))
    return math.fsum(losses) / 64, 64


def _validate_continuity(
    payload: dict[str, Any], checkpoint_path: Path, resume: dict[str, Any] | None = None
) -> None:
    import torch

    continuity = payload.get("continuity")
    if not isinstance(continuity, dict):
        raise ValueError("joint checkpoint requires optimizer/scheduler/RNG continuity")
    required = ("optimizer", "scheduler", "rng_state", "trainer_identity")
    if any(name not in continuity for name in required):
        raise ValueError("joint checkpoint requires optimizer/scheduler/RNG/trainer continuity")
    _require_sha256(continuity["trainer_identity"], "joint checkpoint trainer identity")
    optimizer = continuity["optimizer"]
    scheduler = continuity["scheduler"]
    if not isinstance(optimizer, dict) or optimizer.get("step") != payload["update"]:
        raise ValueError("joint checkpoint optimizer continuity drift")
    if not isinstance(scheduler, dict) or scheduler.get("update") != payload["update"]:
        raise ValueError("joint checkpoint scheduler continuity drift")
    rng_state = continuity["rng_state"]
    if (
        not isinstance(rng_state, torch.Tensor)
        or rng_state.dtype != torch.uint8
        or rng_state.ndim != 1
        or rng_state.numel() == 0
    ):
        raise ValueError("joint checkpoint RNG continuity drift")
    parent = continuity.get("parent")
    if resume is None:
        if parent is not None:
            if not isinstance(parent, dict):
                raise ValueError("joint checkpoint parent continuity drift")
            parent_path = Path(str(parent.get("path", ""))).expanduser().resolve()
            parent_sha = parent.get("sha256")
            parent_update = parent.get("update")
            if (
                not parent_path.is_file()
                or parent_path.stat().st_mode & 0o222
                or _sha256(parent_path) != parent_sha
                or isinstance(parent_update, bool)
                or not isinstance(parent_update, int)
                or parent_update >= payload["update"]
            ):
                raise ValueError("joint checkpoint parent continuity drift")
        return
    expected = {
        "path": str(Path(resume["checkpoint"]).resolve()),
        "sha256": resume["checkpoint_sha256"],
        "update": resume["update"],
    }
    if parent != expected:
        raise ValueError(f"joint checkpoint parent continuity drift: {checkpoint_path}")


def verify_joint_checkpoint(
    *, freeze: str | Path, checkpoint: str | Path, receipt: str | Path | None = None
) -> dict[str, Any]:
    """Rehash and validate one immutable joint checkpoint and optional PASS receipt."""
    freeze_path, frozen = _load_freeze(freeze)
    checkpoint_path, payload, counts = _checkpoint_surface(
        checkpoint, frozen["trainable_surface"]
    )
    if checkpoint_path.stat().st_mode & 0o222:
        raise RuntimeError("joint checkpoint must be immutable/read-only")
    freeze_sha256 = _sha256(freeze_path)
    if payload["freeze_sha256"] != freeze_sha256:
        raise RuntimeError("joint checkpoint frozen-input identity drift")
    _, bank = _load_json(frozen["teacher_bank"]["path"])
    measured_kld, authenticated_windows = _authenticated_kld(
        bank, payload.get("predictions")
    )
    if not math.isclose(
        measured_kld, float(payload["teacher_kld"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError("joint checkpoint self-reported teacher_kld does not match logits")
    _validate_continuity(payload, checkpoint_path)
    result = {
        "schema": _CHECKPOINT_RECEIPT_SCHEMA,
        "status": "PASS",
        "objective": "teacher_kld",
        "update": int(payload["update"]),
        "teacher_kld": float(payload["teacher_kld"]),
        "authenticated_kld_windows": authenticated_windows,
        "trainer": "external",
        "trainable_surface": counts,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "freeze": str(freeze_path),
        "freeze_sha256": freeze_sha256,
        "manifest_sha256": frozen["manifest"]["sha256"],
        "teacher_bank_sha256": frozen["teacher_bank"]["sha256"],
    }
    if receipt is not None:
        receipt_path, expected = _load_json(receipt)
        if receipt_path.stat().st_mode & 0o222:
            raise RuntimeError("joint checkpoint PASS receipt must be immutable/read-only")
        if expected != result:
            raise RuntimeError(f"joint checkpoint PASS receipt mismatch: {receipt_path}")
        result["receipt"] = str(receipt_path)
    return result


def train_joint(
    *,
    freeze: str | Path,
    checkpoint: str | Path,
    target_update: int,
    trainer: str | Path,
    resume_from: str | Path | None = None,
    inputs_ready: str | Path | None = None,
) -> dict[str, Any]:
    """Run the caller-supplied public trainer and seal its authenticated checkpoint."""
    if target_update < 0:
        raise ValueError("target update must be nonnegative")
    freeze_path, frozen = _load_freeze(freeze)
    resumed_from_update: int | None = None
    resume_path: Path | None = None
    prior: dict[str, Any] | None = None
    if resume_from is not None:
        prior = verify_joint_checkpoint(freeze=freeze_path, checkpoint=resume_from)
        resumed_from_update = int(prior["update"])
        if resumed_from_update >= target_update:
            raise ValueError("resume checkpoint must precede target update")
        resume_path = Path(resume_from).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_path = Path(trainer).expanduser().resolve()
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)
    trainer_sha256 = _sha256(trainer_path)
    environment = os.environ.copy()
    environment.update({
        "QTIP_V7_FREEZE": str(freeze_path),
        "QTIP_V7_MANIFEST": str(frozen["manifest"]["path"]),
        "QTIP_V7_TEACHER_BANK": str(frozen["teacher_bank"]["path"]),
        "QTIP_V7_CHECKPOINT": str(checkpoint_path),
        "QTIP_V7_TARGET_UPDATE": str(target_update),
        "QTIP_V7_RESUME_FROM": "" if resume_path is None else str(resume_path),
        "QTIP_V7_OBJECTIVE": "teacher_kld",
        "QTIP_V7_LAYER_LUTS": "43",
        "QTIP_V7_RMSNORM_MASTERS": "235",
        "QTIP_V7_OUTPUT_GAINS": "43",
        "QTIP_V7_TRAINER_SHA256": trainer_sha256,
    })
    if inputs_ready is not None:
        ready_path = Path(inputs_ready).expanduser().resolve()
        if not ready_path.is_file():
            raise FileNotFoundError(ready_path)
        environment["QTIP_V7_INPUTS_READY"] = str(ready_path)
    command = (
        [str(trainer_path)]
        if os.access(trainer_path, os.X_OK)
        else [os.environ.get("PYTHON", "python3"), str(trainer_path)]
    )
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(f"QTIP V7 trainer exited with status {completed.returncode}")
    if not checkpoint_path.is_file():
        raise RuntimeError("QTIP V7 trainer did not emit the requested checkpoint")
    os.chmod(checkpoint_path, 0o444)
    result = verify_joint_checkpoint(freeze=freeze_path, checkpoint=checkpoint_path)
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["continuity"]["trainer_identity"] != trainer_sha256:
        raise ValueError("joint checkpoint trainer identity does not match supplied trainer")
    _validate_continuity(
        payload,
        checkpoint_path,
        prior,
    )
    if result["update"] != target_update:
        raise RuntimeError(f"trainer checkpoint update drift: {result['update']} != {target_update}")
    receipt_path = checkpoint_path.with_name(f"{checkpoint_path.name}.PASS.json")
    receipt_record = _write_json(receipt_path, result, exclusive=True)
    os.chmod(checkpoint_path, 0o444)
    os.chmod(receipt_path, 0o444)
    return {
        **result,
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_record["sha256"],
        "resumed_from_update": resumed_from_update,
    }


def _parse_worker(value: str) -> tuple[str, str | None, Path | None, Path]:
    target, separator, command = value.partition("=")
    if not separator or not target or not command:
        raise ValueError("--worker must be LOCAL=COMMAND or HOST:REMOTE_ROOT=COMMAND")
    command_path = Path(command).expanduser().resolve()
    if not command_path.is_file():
        raise FileNotFoundError(command_path)
    if target.startswith("local"):
        return target, None, None, command_path
    host, separator, root = target.partition(":")
    if (
        not separator
        or not _REMOTE_ROOT_PATTERN.fullmatch(root)
        or ".." in Path(root).parts
    ):
        raise ValueError(
            "remote --worker requires a shell-safe absolute REMOTE_ROOT"
        )
    expected, at, route = host.partition("@")
    if not at:
        raise ValueError("remote --worker requires EXPECTED_HOSTNAME@HOST route identity")
    return _normalize_host(route), _normalize_host(expected), Path(root), command_path


def _cancel_workers(
    processes: Sequence[tuple[subprocess.Popen[bytes], dict[str, Any], Path, str | None]],
) -> None:
    for process, _, _, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, _, _, _ in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def launch_balanced64_shards(
    *,
    candidate: str | Path,
    freeze: str | Path,
    teacher_bank: str | Path,
    output: str | Path,
    workers: Sequence[str],
    remote_python: str = "python3",
) -> dict[str, Any]:
    """Stage and launch disjoint BALANCED64 shards in parallel on side workers."""
    if not workers or len(workers) > 64:
        raise ValueError("BALANCED64 shard launch requires 1..64 workers")
    if not remote_python or any(character.isspace() for character in remote_python):
        raise ValueError("remote Python command must be one shell-safe executable path")
    verified_candidate = verify_joint_checkpoint(freeze=freeze, checkpoint=candidate)
    candidate_path = Path(verified_candidate["checkpoint"])
    _, frozen = _load_freeze(freeze)
    trainer_host = _normalize_host(str(frozen.get("trainer_host", "")))
    trainer_identities = frozen.get("trainer_identities")
    if not isinstance(trainer_identities, list) or not trainer_identities:
        raise ValueError("frozen trainer identity set is missing")
    bank_path, bank = _load_json(teacher_bank)
    if not isinstance(bank.get("windows"), list) or len(bank["windows"]) != 64:
        raise ValueError("BALANCED64 shard launch requires a 64-window teacher bank")
    candidate_sha = _sha256(candidate_path)
    bank_sha = _sha256(bank_path)
    if bank_sha != frozen["teacher_bank"]["sha256"]:
        raise RuntimeError("BALANCED64 teacher bank differs from the frozen run identity")
    root = Path(output).expanduser().resolve()
    parsed = [_parse_worker(value) for value in workers]
    remote_hosts = [target for target, _, remote_root, _ in parsed if remote_root is not None]
    for route, expected_host, remote_root, _ in parsed:
        if remote_root is not None and (
            _matches_any_host(route, trainer_identities)
            or (expected_host is not None and _matches_any_host(expected_host, trainer_identities))
        ):
            raise ValueError(
                f"refusing BALANCED64 shard worker on live trainer host {trainer_host}"
            )
    if len(remote_hosts) != len(set(remote_hosts)):
        raise ValueError("BALANCED64 remote worker hosts must be distinct")
    if root.exists():
        raise FileExistsError(root)
    route_identities: dict[str, str] = {}
    for target, expected_host, remote_root, _ in parsed:
        if remote_root is None:
            continue
        assert expected_host is not None
        preflight = subprocess.run(
            ["ssh", target, "hostname"],
            check=True,
            text=True,
            capture_output=True,
        )
        observed = _normalize_host(preflight.stdout.splitlines()[0])
        if not _same_host(observed, expected_host):
            raise RuntimeError(
                f"route identity mismatch for {target}: expected={expected_host} observed={observed}"
            )
        if _matches_any_host(observed, trainer_identities):
            raise ValueError(
                f"refusing BALANCED64 shard worker on live trainer host {trainer_host}"
            )
        route_identities[target] = observed
    if len(route_identities.values()) != len(set(route_identities.values())):
        raise ValueError("BALANCED64 observed worker hosts must be distinct")
    root.mkdir(parents=True)
    ranges = []
    for index in range(len(parsed)):
        start = index * 64 // len(parsed)
        end = (index + 1) * 64 // len(parsed) - 1
        ranges.append((start, end))
    processes: list[tuple[subprocess.Popen[bytes], dict[str, Any], Path, str | None]] = []
    for (target, _, remote_root, worker), (start, end) in zip(parsed, ranges, strict=True):
        try:
            shard = root / f"o{start:02d}-{end:02d}"
            shard.mkdir()
            local_receipt = shard / "BALANCED64_SHARD_TERMINAL.json"
            environment = {
                "QTIP_V7_CANDIDATE": str(candidate_path),
                "QTIP_V7_CANDIDATE_SHA256": candidate_sha,
                "QTIP_V7_TEACHER_BANK": str(bank_path),
                "QTIP_V7_TEACHER_BANK_SHA256": bank_sha,
                "QTIP_V7_SHARD_START": str(start),
                "QTIP_V7_SHARD_END": str(end),
                "QTIP_V7_SHARD_RECEIPT": str(local_receipt),
            }
            remote_receipt: str | None = None
            if remote_root is None:
                command = (
                    [str(worker)]
                    if os.access(worker, os.X_OK)
                    else [sys.executable, str(worker)]
                )
                process = subprocess.Popen(command, env={**os.environ, **environment})
            else:
                remote = remote_root / f"o{start:02d}-{end:02d}"
                subprocess.run(["ssh", target, "mkdir", "-p", str(remote)], check=True)
                staged_worker = worker
                subprocess.run(["scp", str(candidate_path), str(bank_path), str(staged_worker), f"{target}:{remote}/"], check=True)
                remote_candidate = remote / candidate_path.name
                remote_bank = remote / bank_path.name
                remote_worker = remote / staged_worker.name
                remote_receipt = str(remote / "BALANCED64_SHARD_TERMINAL.json")
                readback = subprocess.run(
                    [
                        "ssh",
                        target,
                        "sha256sum",
                        str(remote_candidate),
                        str(remote_bank),
                        str(remote_worker),
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                observed_hashes = [line.split()[0] for line in readback.stdout.splitlines()]
                if observed_hashes != [candidate_sha, bank_sha, _sha256(staged_worker)]:
                    raise RuntimeError(f"remote staged SHA-256 readback drift on {target}")
                remote_env = {
                    **environment,
                    "QTIP_V7_CANDIDATE": str(remote_candidate),
                    "QTIP_V7_TEACHER_BANK": str(remote_bank),
                    "QTIP_V7_SHARD_RECEIPT": remote_receipt,
                }
                assignments = [f"{name}={shlex.quote(value)}" for name, value in remote_env.items()]
                remote_command = " ".join([
                    "env", *assignments, shlex.quote(remote_python), shlex.quote(str(remote_worker))
                ])
                process = subprocess.Popen(["ssh", target, remote_command])
        except Exception:
            _cancel_workers(processes)
            raise
        processes.append((process, {
            "target": target, "ordinal_start": start, "ordinal_end": end,
            "candidate_sha256": candidate_sha, "teacher_bank_sha256": bank_sha,
        }, local_receipt, remote_receipt))
    rows = []
    try:
        pending = list(processes)
        while pending:
            for item in tuple(pending):
                process, row, _, _ = item
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(item)
                if return_code:
                    raise RuntimeError(
                        f"BALANCED64 worker {row['target']} exited with status {return_code}"
                    )
            if pending:
                time.sleep(0.05)
        for _, row, local_receipt, remote_receipt in processes:
            if remote_receipt is not None:
                subprocess.run(["scp", f"{row['target']}:{remote_receipt}", str(local_receipt)], check=True)
            _, receipt = _load_json(local_receipt)
            expected = (row["ordinal_start"], row["ordinal_end"], candidate_sha)
            observed = (receipt.get("ordinal_start"), receipt.get("ordinal_end"), receipt.get("candidate_sha256"))
            receipt_rows = receipt.get("rows")
            observed_ordinals = (
                [item.get("ordinal") for item in receipt_rows]
                if isinstance(receipt_rows, list)
                and all(isinstance(item, dict) for item in receipt_rows)
                else None
            )
            expected_ordinals = list(range(row["ordinal_start"], row["ordinal_end"] + 1))
            if (
                receipt.get("schema") != _SHARD_SCHEMA
                or receipt.get("status") != "PASS"
                or observed != expected
                or receipt.get("teacher_bank_sha256") != bank_sha
                or observed_ordinals != expected_ordinals
            ):
                raise RuntimeError(f"BALANCED64 shard receipt drift: expected={expected} actual={observed}")
            rows.append({**row, "receipt": str(local_receipt), "receipt_sha256": _sha256(local_receipt)})
    except Exception:
        _cancel_workers(processes)
        raise
    result = {
        "schema": "banana-smasher-qtip-v7-shard-launch-v1",
        "status": "PASS",
        "trainer_host": trainer_host,
        "route_identities": route_identities,
        "shards": rows,
    }
    _write_json(root / "SHARD_LAUNCH_RECEIPT.json", result, exclusive=True)
    return result


def aggregate_balanced64(*, shards: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(shards).expanduser().resolve()
    receipts = sorted(root.glob("o??-??/BALANCED64_SHARD_TERMINAL.json"))
    if not receipts:
        raise ValueError("no BALANCED64 shard receipts found")
    by_ordinal: dict[int, dict[str, Any]] = {}
    candidate_sha: str | None = None
    teacher_bank_sha: str | None = None
    source_receipts = []
    for path in receipts:
        _, receipt = _load_json(path)
        if receipt.get("schema") != _SHARD_SCHEMA or receipt.get("status") != "PASS":
            raise ValueError(f"invalid BALANCED64 shard receipt: {path}")
        if candidate_sha is None:
            candidate_sha = receipt.get("candidate_sha256")
            if not isinstance(candidate_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha):
                raise ValueError("BALANCED64 shard does not bind one candidate")
        if receipt.get("candidate_sha256") != candidate_sha:
            raise ValueError("BALANCED64 shards do not bind one candidate")
        teacher_sha = receipt.get("teacher_bank_sha256")
        if not isinstance(teacher_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", teacher_sha):
            raise ValueError("BALANCED64 shard does not bind one teacher bank")
        if teacher_bank_sha is None:
            teacher_bank_sha = teacher_sha
        elif teacher_sha != teacher_bank_sha:
            raise ValueError("BALANCED64 shards do not bind one teacher bank")
        rows = receipt.get("rows")
        start = receipt.get("ordinal_start")
        end = receipt.get("ordinal_end")
        if (
            not isinstance(rows, list)
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start <= end < 64
            or len(rows) != end - start + 1
        ):
            raise ValueError(f"BALANCED64 shard range/row closure drift: {path}")
        observed_ordinals: list[int] = []
        for row in rows:
            ordinal = row.get("ordinal") if isinstance(row, dict) else None
            positions = row.get("positions") if isinstance(row, dict) else None
            support = row.get("support") if isinstance(row, dict) else None
            kld_sum = row.get("kld_sum_binary64") if isinstance(row, dict) else None
            top1 = row.get("top1_matches") if isinstance(row, dict) else None
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal in by_ordinal
                or positions != 1024
                or support != 8192
                or isinstance(kld_sum, bool)
                or not isinstance(kld_sum, (int, float))
                or not math.isfinite(float(kld_sum))
                or float(kld_sum) < 0
                or isinstance(top1, bool)
                or not isinstance(top1, int)
                or not 0 <= top1 <= positions
                or any(
                    row.get(field) != 0
                    for field in (
                        "fallback_calls",
                        "pass_through_bytes",
                        "hidden_fp32_control_bytes",
                    )
                )
            ):
                raise ValueError(f"invalid/duplicate BALANCED64 row in {path}")
            observed_ordinals.append(ordinal)
            by_ordinal[ordinal] = {
                "ordinal": ordinal,
                "positions": positions,
                "support": support,
                "kld_sum_binary64": float(kld_sum),
                "top1_matches": top1,
                "fallback_calls": 0,
                "pass_through_bytes": 0,
                "hidden_fp32_control_bytes": 0,
            }
        if observed_ordinals != list(range(start, end + 1)):
            raise ValueError(f"BALANCED64 shard range/row closure drift: {path}")
        source_receipts.append({"path": str(path), "sha256": _sha256(path)})
    if set(by_ordinal) != set(range(64)):
        raise ValueError(f"BALANCED64 aggregate requires exact ordinals 0..63, got {sorted(by_ordinal)}")
    ordered = [by_ordinal[index] for index in range(64)]
    result = {
        "schema": "banana-smasher-qtip-v7-balanced64-aggregate-v1",
        "status": "PASS",
        "candidate_sha256": candidate_sha,
        "teacher_bank_sha256": teacher_bank_sha,
        "windows": 64,
        "positions": 65_536,
        "support": 8192,
        "mean_kld": math.fsum(row["kld_sum_binary64"] for row in ordered) / 65_536,
        "top1_matches": sum(row["top1_matches"] for row in ordered),
        "rows": ordered,
        "source_receipts": source_receipts,
    }
    _write_json(Path(output), result, exclusive=True)
    return result


def _validate_aggregate(label: str, value: dict[str, Any]) -> None:
    rows = value.get("rows")
    if (
        value.get("schema") != "banana-smasher-qtip-v7-balanced64-aggregate-v1"
        or value.get("status") != "PASS"
        or value.get("windows") != 64
        or not isinstance(value.get("candidate_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["candidate_sha256"])
        or not isinstance(value.get("teacher_bank_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["teacher_bank_sha256"])
        or not isinstance(rows, list)
        or len(rows) != 64
    ):
        raise ValueError(f"{label} is not a complete BALANCED64 aggregate")
    normalized: list[tuple[int, float, int]] = []
    for expected, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} has invalid BALANCED64 rows")
        ordinal = row.get("ordinal")
        positions = row.get("positions")
        support = row.get("support")
        kld_sum = row.get("kld_sum_binary64")
        top1 = row.get("top1_matches")
        if (
            ordinal != expected
            or positions != 1024
            or support != 8192
            or isinstance(kld_sum, bool)
            or not isinstance(kld_sum, (int, float))
            or not math.isfinite(float(kld_sum))
            or float(kld_sum) < 0
            or isinstance(top1, bool)
            or not isinstance(top1, int)
            or not 0 <= top1 <= positions
            or any(
                row.get(field) != 0
                for field in (
                    "fallback_calls",
                    "pass_through_bytes",
                    "hidden_fp32_control_bytes",
                )
            )
        ):
            raise ValueError(f"{label} has invalid BALANCED64 rows")
        normalized.append((expected, float(kld_sum), top1))
    expected_mean = math.fsum(row[1] for row in normalized) / 65_536
    expected_top1 = sum(row[2] for row in normalized)
    if (
        value.get("positions") != 65_536
        or value.get("support") != 8192
        or value.get("mean_kld") != expected_mean
        or value.get("top1_matches") != expected_top1
    ):
        raise ValueError(f"{label} BALANCED64 summary does not match its rows")


def compare_aggregates(*, baseline: str | Path, candidate: str | Path, output: str | Path) -> dict[str, Any]:
    baseline_path, left = _load_json(baseline)
    candidate_path, right = _load_json(candidate)
    for label, value in (("baseline", left), ("candidate", right)):
        _validate_aggregate(label, value)
    if left["teacher_bank_sha256"] != right["teacher_bank_sha256"]:
        raise ValueError("BALANCED64 comparison requires one teacher bank identity")
    candidate_nonworse = float(right["mean_kld"]) <= float(left["mean_kld"]) and int(right["top1_matches"]) >= int(left["top1_matches"])
    champion = "candidate" if candidate_nonworse else "baseline"
    selected_path, selected = (candidate_path, right) if candidate_nonworse else (baseline_path, left)
    result = {
        "schema": "banana-smasher-qtip-v7-champion-selection-v1",
        "status": "PASS",
        "champion": champion,
        "selected_candidate_sha256": selected["candidate_sha256"],
        "selected_aggregate": str(selected_path),
        "selected_aggregate_sha256": _sha256(selected_path),
        "delta_mean_kld": float(right["mean_kld"]) - float(left["mean_kld"]),
        "delta_top1_matches": int(right["top1_matches"]) - int(left["top1_matches"]),
    }
    _write_json(Path(output), result, exclusive=True)
    return result


def materialize_joint(
    *,
    freeze: str | Path,
    manifest: str | Path,
    checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Materialize trained LUT/dense state and account exact stored bytes without rewriting packed wire."""
    from safetensors.numpy import save_file

    manifest_path = Path(manifest).expanduser().resolve()
    _, frozen = _load_freeze(freeze)
    if _sha256(manifest_path) != frozen["manifest"]["sha256"]:
        raise RuntimeError("joint materialization manifest differs from frozen run identity")
    verify_joint_checkpoint(freeze=freeze, checkpoint=checkpoint)
    source = load_qtip_v7_artifact(manifest_path)
    checkpoint_path, payload, counts = _checkpoint_surface(
        checkpoint, frozen["trainable_surface"]
    )
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)
    try:
        state = payload["state"]
        lut_values_by_layer = {
            int(name[1:]): tensor for name, tensor in state["layer_luts"].items()
        }
        document = json.loads(json.dumps(source.document))
        if source.member_paths:
            document["member_root"] = str(source.member_paths[0].parent)
        if source.external_paths:
            document["external_root"] = str(source.external_paths[0].parent)
        for row, source_path in zip(document["members"], source.member_paths, strict=True):
            row["path"] = source_path.name
        for row in sorted(document["layer_luts"], key=lambda item: int(item["layer"])):
            layer = int(row["layer"])
            tensor = lut_values_by_layer[layer]
            array = tensor.detach().cpu().float().reshape(-1).numpy()
            if array.shape != (1024,):
                raise ValueError("joint layer LUT must contain 1024 values")
            path = output_path / str(row["path"])
            np.ascontiguousarray(array, dtype="<f2").tofile(path)
            row["bytes"] = path.stat().st_size
            row["sha256"] = _sha256(path)
        document["update"] = int(payload["update"])
        _write_json(output_path / "QTIP_V7_MANIFEST.json", document)
        readback = load_qtip_v7_artifact(output_path / "QTIP_V7_MANIFEST.json")
        dense: dict[str, np.ndarray] = {}
        for prefix, values in (("norms", state["norms"]), ("outputs", state["outputs"])):
            for name, tensor in sorted(values.items()):
                dense[f"{prefix}/{name}"] = np.ascontiguousarray(tensor.detach().cpu().float().numpy())
        repair_path = output_path / "repair_state.safetensors"
        save_file(dense, repair_path, metadata={
            "format": _CHECKPOINT_FORMAT,
            "update": str(payload["update"]),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "objective": "teacher_kld",
        })

        result = {
            "schema": "banana-smasher-qtip-v7-joint-materialization-v1",
            "status": "PASS",
            "update": int(payload["update"]),
            "teacher_kld": float(payload["teacher_kld"]),
            "trainable_surface": counts,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "packed_identity": (
                readback.member_wire_sha256 == source.member_wire_sha256
                and readback.external_wire_sha256 == source.external_wire_sha256
            ),

            "repair_state_sha256": _sha256(repair_path),
            "repair_state_bytes": repair_path.stat().st_size,
            "physical_accounting": "requires qtip-v7-wire verified layer receipts",
        }
        if not result["packed_identity"]:
            raise RuntimeError("joint materialization changed fixed QTIP member identity")
        _write_json(output_path / "QTIP_V7_JOINT_MATERIALIZATION.json", result, exclusive=True)
        return result
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise


__all__ = [
    "aggregate_balanced64",
    "compare_aggregates",
    "inspect_joint_inputs",
    "launch_balanced64_shards",
    "materialize_joint",
    "train_joint",
    "verify_joint_checkpoint",
]
