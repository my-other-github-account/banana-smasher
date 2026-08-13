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
import subprocess
import tempfile
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
) -> dict[str, Any]:
    """Validate and freeze the exact all-43 V7 inventory and 64-window teacher bank."""
    manifest_path = Path(manifest).expanduser().resolve()
    source = load_qtip_v7_artifact(manifest_path)
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
    bank_path, bank = _load_json(teacher_bank)
    windows = bank.get("windows")
    if not isinstance(windows, list) or len(windows) != 64 or len({json.dumps(row, sort_keys=True) for row in windows}) != 64:
        raise ValueError("QTIP V7 teacher bank requires exactly 64 unique ordered windows")
    teacher_identity = bank.get("teacher_sha256")
    if teacher_identity is None and isinstance(bank.get("identities"), dict):
        teacher_identity = bank["identities"].get("teacher", {}).get("sha256")
    if not isinstance(teacher_identity, str) or len(teacher_identity) != 64:
        raise ValueError("QTIP V7 teacher bank requires an exact teacher SHA-256 identity")
    document = {
        "schema": _FREEZE_SCHEMA,
        "status": "PASS",
        "inventory": dict(_INVENTORY),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "complete_wire_bytes": source.complete_wire_bytes,
            "members": len(source.member_paths) + source.external_member_count,
        },
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": _sha256(bank_path),
            "teacher_sha256": teacher_identity,
            "windows": len(windows),
        },
        "objective": "teacher_kld",
        "trainer_host": _normalize_host(trainer_host),
    }
    output = Path(run_root).expanduser().resolve() / "FROZEN_INPUTS.json"
    record = _write_json(output, document, exclusive=True)
    os.chmod(output, 0o444)
    return {**document, "freeze": record}


def _checkpoint_surface(checkpoint: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
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
    lut_layers: set[int] = set()
    for name in state["layer_luts"]:
        if isinstance(name, str) and name.startswith("L") and name[1:].isdigit():
            layer = int(name[1:])
        else:
            raise ValueError(f"joint checkpoint has invalid layer LUT key {name!r}")
        if layer in lut_layers:
            raise ValueError(f"joint checkpoint has duplicate layer LUT {layer}")
        lut_layers.add(layer)
    if lut_layers != set(range(43)):
        raise ValueError("joint checkpoint layer LUTs must bind exact layers 0..42")
    for group in names:
        for name, tensor in state[group].items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"joint checkpoint {group} entries must be named tensors")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"joint checkpoint contains nonfinite tensor {group}/{name}")
    return path, payload, counts


def verify_joint_checkpoint(
    *, freeze: str | Path, checkpoint: str | Path, receipt: str | Path | None = None
) -> dict[str, Any]:
    """Rehash and validate one immutable joint checkpoint and optional PASS receipt."""
    freeze_path, frozen = _load_freeze(freeze)
    checkpoint_path, payload, counts = _checkpoint_surface(checkpoint)
    if checkpoint_path.stat().st_mode & 0o222:
        raise RuntimeError("joint checkpoint must be immutable/read-only")
    freeze_sha256 = _sha256(freeze_path)
    if payload["freeze_sha256"] != freeze_sha256:
        raise RuntimeError("joint checkpoint frozen-input identity drift")
    result = {
        "schema": _CHECKPOINT_RECEIPT_SCHEMA,
        "status": "PASS",
        "objective": "teacher_kld",
        "update": int(payload["update"]),
        "teacher_kld": float(payload["teacher_kld"]),
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
) -> dict[str, Any]:
    """Launch an external public trainer, then seal its all-surface teacher-KLD checkpoint."""
    if target_update <= 0:
        raise ValueError("target update must be positive")
    freeze_path, frozen = _load_freeze(freeze)
    resumed_from_update: int | None = None
    resume_path: Path | None = None
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
    })
    command = [str(trainer_path)] if os.access(trainer_path, os.X_OK) else [os.environ.get("PYTHON", "python3"), str(trainer_path)]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(f"QTIP V7 trainer exited with status {completed.returncode}")
    os.chmod(checkpoint_path, 0o444)
    result = verify_joint_checkpoint(freeze=freeze_path, checkpoint=checkpoint_path)
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


def launch_balanced64_shards(
    *,
    candidate: str | Path,
    freeze: str | Path,
    teacher_bank: str | Path,
    output: str | Path,
    workers: Sequence[str],
) -> dict[str, Any]:
    """Stage and launch disjoint BALANCED64 shards in parallel on side workers."""
    if not workers or len(workers) > 64:
        raise ValueError("BALANCED64 shard launch requires 1..64 workers")
    verified_candidate = verify_joint_checkpoint(freeze=freeze, checkpoint=candidate)
    candidate_path = Path(verified_candidate["checkpoint"])
    _, frozen = _load_freeze(freeze)
    trainer_host = _normalize_host(str(frozen.get("trainer_host", "")))
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
            _same_host(route, trainer_host)
            or (expected_host is not None and _same_host(expected_host, trainer_host))
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
        if _same_host(observed, trainer_host):
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
            command = [str(worker)] if os.access(worker, os.X_OK) else [os.environ.get("PYTHON", "python3"), str(worker)]
            process = subprocess.Popen(command, env={**os.environ, **environment})
        else:
            remote = remote_root / f"o{start:02d}-{end:02d}"
            subprocess.run(["ssh", target, "mkdir", "-p", str(remote)], check=True)
            subprocess.run(["scp", str(candidate_path), str(bank_path), str(worker), f"{target}:{remote}/"], check=True)
            remote_candidate = remote / candidate_path.name
            remote_bank = remote / bank_path.name
            remote_worker = remote / worker.name
            remote_receipt = str(remote / "BALANCED64_SHARD_TERMINAL.json")
            remote_env = {
                **environment,
                "QTIP_V7_CANDIDATE": str(remote_candidate),
                "QTIP_V7_TEACHER_BANK": str(remote_bank),
                "QTIP_V7_SHARD_RECEIPT": remote_receipt,
            }
            assignments = [f"{name}={shlex.quote(value)}" for name, value in remote_env.items()]
            remote_command = " ".join(["env", *assignments, shlex.quote(str(remote_worker))])
            process = subprocess.Popen(["ssh", target, remote_command])
        processes.append((process, {
            "target": target, "ordinal_start": start, "ordinal_end": end,
            "candidate_sha256": candidate_sha, "teacher_bank_sha256": bank_sha,
        }, local_receipt, remote_receipt))
    rows = []
    for process, row, local_receipt, remote_receipt in processes:
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"BALANCED64 worker {row['target']} exited with status {return_code}")
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
        if not isinstance(rows, list):
            raise ValueError(f"BALANCED64 shard rows missing: {path}")
        for row in rows:
            ordinal = row.get("ordinal") if isinstance(row, dict) else None
            kld = row.get("mean_kld") if isinstance(row, dict) else None
            top1 = row.get("top1_match") if isinstance(row, dict) else None
            if (
                isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal in by_ordinal
                or isinstance(kld, bool) or not isinstance(kld, (int, float)) or not math.isfinite(float(kld)) or float(kld) < 0
                or top1 not in (0, 1)
            ):
                raise ValueError(f"invalid/duplicate BALANCED64 row in {path}")
            by_ordinal[ordinal] = {"ordinal": ordinal, "mean_kld": float(kld), "top1_match": int(top1)}
        start = receipt.get("ordinal_start")
        end = receipt.get("ordinal_end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or [row["ordinal"] for row in rows] != list(range(start, end + 1))
        ):
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
        "mean_kld": math.fsum(row["mean_kld"] for row in ordered) / 64,
        "top1_matches": sum(row["top1_match"] for row in ordered),
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
        kld = row.get("mean_kld")
        top1 = row.get("top1_match")
        if (
            ordinal != expected
            or isinstance(kld, bool)
            or not isinstance(kld, (int, float))
            or not math.isfinite(float(kld))
            or float(kld) < 0
            or top1 not in (0, 1)
        ):
            raise ValueError(f"{label} has invalid BALANCED64 rows")
        normalized.append((ordinal, float(kld), int(top1)))
    expected_mean = math.fsum(row[1] for row in normalized) / 64
    expected_top1 = sum(row[2] for row in normalized)
    if value.get("mean_kld") != expected_mean or value.get("top1_matches") != expected_top1:
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
    checkpoint_path, payload, counts = _checkpoint_surface(checkpoint)
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
        qtip_wire_bytes = readback.complete_wire_bytes
        dense_bytes = repair_path.stat().st_size
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
            "wire_size_delta": qtip_wire_bytes - source.complete_wire_bytes,
            "qtip_wire_bytes": qtip_wire_bytes,
            "dense_repair_bytes": dense_bytes,
            "stored_wire_bytes": qtip_wire_bytes + dense_bytes,
            "repair_state_sha256": _sha256(repair_path),
        }
        if not result["packed_identity"] or result["wire_size_delta"] != 0:
            raise RuntimeError("joint materialization changed fixed QTIP wire identity/accounting")
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
