#!/usr/bin/env python3
"""Claim-bound detached launcher for a sealed route and bounded consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    right = text.rfind(")")
    return int(text[right + 2 :].split()[19])


def live(pid: int, ticks: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        right = text.rfind(")")
        fields = text[right + 2 :].split()
        return fields[0] != "Z" and int(fields[19]) == ticks
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def terminate_exact(process: subprocess.Popen[bytes] | None, pid: int | None, ticks: int | None) -> bool:
    if pid is None or ticks is None or not live(pid, ticks):
        return True
    if process is not None and process.poll() is None:
        process.terminate()
    else:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline and live(pid, ticks):
        time.sleep(0.25)
    if live(pid, ticks):
        os.kill(pid, signal.SIGKILL)
        deadline = time.time() + 15
        while time.time() < deadline and live(pid, ticks):
            time.sleep(0.25)
    return not live(pid, ticks)


def main(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    task_id = str(config["task_id"])
    run_id = int(config["run_id"])
    run_root = Path(config["run_root"])
    claim_path = Path(config["host_claim_path"])
    progress_path = run_root / "CONTROLLER_PROGRESS.json"
    terminal_path = run_root / "CONTROLLER_TERMINAL.json"
    run_root.mkdir(parents=True, exist_ok=True)
    controller_pid = os.getpid()
    controller_ticks = start_ticks(controller_pid)
    state: dict[str, Any] = {
        "schema": "banana-smasher-bounded-top1-controller-progress-v1",
        "status": "WAITING_CLAIM",
        "task_id": task_id,
        "run_id": run_id,
        "controller_pid": controller_pid,
        "controller_startticks": controller_ticks,
        "started_unix": time.time(),
        "updated_unix": time.time(),
    }

    def publish(status: str, **fields: Any) -> str:
        state.update(status=status, updated_unix=time.time(), **fields)
        return atomic_json(progress_path, state)

    route: subprocess.Popen[bytes] | None = None
    consumer: subprocess.Popen[bytes] | None = None
    route_pid: int | None = None
    route_ticks: int | None = None
    consumer_pid: int | None = None
    consumer_ticks: int | None = None
    try:
        publish("WAITING_CLAIM")
        deadline = time.time() + int(config.get("claim_wait_seconds", 180))
        while time.time() < deadline:
            claim = json.loads(claim_path.read_text())
            expected_holder_pid = int(config.get("claim_holder_pid", controller_pid))
            expected_holder_ticks = int(config.get("claim_holder_startticks", controller_ticks))
            if (
                claim.get("state") == "CLAIMED"
                and claim.get("task_id") == task_id
                and int(claim.get("run_id", -1)) == run_id
                and int(claim.get("holder_pid", -1)) == expected_holder_pid
                and int(claim.get("holder_start_ticks", -1)) == expected_holder_ticks
            ):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("CLAIM_BIND_TIMEOUT")
        claim_sha = sha256(claim_path)
        basis_path = Path(config["basis_path"])
        if sha256(basis_path) != config["basis_sha256"]:
            raise RuntimeError("BASIS_GATE_MISMATCH")
        route_binary = Path(config["route_binary"])
        route_source = Path(config["route_source"])
        model_path = Path(config["model_path"])
        candidate_manifest_path = Path(config["candidate_manifest_path"])
        if sha256(route_binary) != config["route_binary_sha256"]:
            raise RuntimeError("ROUTE_BINARY_DRIFT")
        if sha256(route_source) != config["route_source_sha256"]:
            raise RuntimeError("ROUTE_SOURCE_DRIFT")
        if sha256(candidate_manifest_path) != config["candidate_manifest_sha256"]:
            raise RuntimeError("CANDIDATE_MANIFEST_DRIFT")
        if not model_path.is_file():
            raise RuntimeError("MODEL_MEMBER_MISSING")
        client_path = Path(config["client_path"])
        if sha256(client_path) != config["client_sha256"]:
            raise RuntimeError("CLIENT_SOURCE_DRIFT")
        route_progress = Path(config["route_progress_path"])
        route_key = Path(config["route_key_path"])
        route_key.parent.mkdir(parents=True, exist_ok=True)
        if not route_key.exists():
            fd = os.open(route_key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as output:
                output.write(secrets.token_hex(32) + "\n")
                output.flush()
                os.fsync(output.fileno())
        route_command = [
            str(route_binary),
            str(model_path),
            str(config["route_port"]),
            str(route_progress),
            str(route_key),
            str(config["candidate_manifest_sha256"]),
        ]
        route_log_path = run_root / "route.log"
        route_log = route_log_path.open("ab", buffering=0)
        route = subprocess.Popen(
            route_command,
            cwd=run_root,
            stdin=subprocess.DEVNULL,
            stdout=route_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        route_pid = route.pid
        route_ticks = start_ticks(route_pid)
        publish(
            "ROUTE_LOADING",
            claim_sha256=claim_sha,
            route_pid=route_pid,
            route_startticks=route_ticks,
            route_command=route_command,
            route_log_path=str(route_log_path),
            route_binary_sha256=config["route_binary_sha256"],
            route_source_sha256=config["route_source_sha256"],
        )
        deadline = time.time() + int(config.get("route_ready_timeout_seconds", 1800))
        route_state: dict[str, Any] | None = None
        while time.time() < deadline:
            if route.poll() is not None:
                raise RuntimeError(f"ROUTE_EXITED_BEFORE_READY rc={route.returncode}")
            if route_progress.is_file():
                current_route_state = json.loads(route_progress.read_text())
                route_state = current_route_state
                if current_route_state.get("status") == "READY" and int(current_route_state.get("requests", -1)) == 0:
                    break
            publish(
                "ROUTE_LOADING",
                claim_sha256=claim_sha,
                route_pid=route_pid,
                route_startticks=route_ticks,
                route_command=route_command,
                route_log_path=str(route_log_path),
                route_progress=route_state if "route_state" in locals() else None,
            )
            time.sleep(5)
        else:
            raise RuntimeError("ROUTE_READY_TIMEOUT")
        client_config_path = Path(config["client_config_path"])
        client_config = json.loads(client_config_path.read_text())
        client_config.update(route_pid=route_pid, route_startticks=route_ticks)
        atomic_json(client_config_path, client_config)
        consumer_command = [
            str(config["python"]),
            "-u",
            str(client_path),
            "--config",
            str(client_config_path),
        ]
        consumer_log_path = run_root / "consumer.log"
        consumer_log = consumer_log_path.open("ab", buffering=0)
        consumer = subprocess.Popen(
            consumer_command,
            cwd=run_root,
            stdin=subprocess.DEVNULL,
            stdout=consumer_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        consumer_pid = consumer.pid
        consumer_ticks = start_ticks(consumer_pid)
        publish(
            "CONSUMER_RUNNING",
            claim_sha256=claim_sha,
            route_pid=route_pid,
            route_startticks=route_ticks,
            route_command=route_command,
            consumer_pid=consumer_pid,
            consumer_startticks=consumer_ticks,
            consumer_command=consumer_command,
            consumer_log_path=str(consumer_log_path),
            consumer_progress_path=str(config["consumer_progress_path"]),
        )
        while consumer.poll() is None:
            if route.poll() is not None:
                raise RuntimeError(f"ROUTE_DIED_DURING_CONSUMER rc={route.returncode}")
            consumer_progress = None
            consumer_progress_path = Path(config["consumer_progress_path"])
            if consumer_progress_path.is_file():
                consumer_progress = json.loads(consumer_progress_path.read_text())
            publish(
                "CONSUMER_RUNNING",
                claim_sha256=claim_sha,
                route_pid=route_pid,
                route_startticks=route_ticks,
                route_command=route_command,
                consumer_pid=consumer_pid,
                consumer_startticks=consumer_ticks,
                consumer_command=consumer_command,
                consumer_log_path=str(consumer_log_path),
                consumer_progress_path=str(consumer_progress_path),
                consumer_progress=consumer_progress,
            )
            time.sleep(10)
        consumer_rc = consumer.wait()
        consumer_progress_path = Path(config["consumer_progress_path"])
        consumer_progress = json.loads(consumer_progress_path.read_text()) if consumer_progress_path.is_file() else {}
        if consumer_rc != 0 or consumer_progress.get("status") != "PASS":
            raise RuntimeError(f"CONSUMER_FAILED rc={consumer_rc} status={consumer_progress.get('status')} error={consumer_progress.get('error')}")
        if not terminate_exact(route, route_pid, route_ticks):
            raise RuntimeError("ROUTE_STOP_DEAD_VERIFY_FAILED")
        final = {
            **state,
            "status": "PASS",
            "finished_unix": time.time(),
            "claim_sha256": claim_sha,
            "route_pid": route_pid,
            "route_startticks": route_ticks,
            "route_dead_verified": True,
            "consumer_pid": consumer_pid,
            "consumer_startticks": consumer_ticks,
            "consumer_dead_verified": not live(consumer_pid, consumer_ticks),
            "consumer_progress": consumer_progress,
            "next_gate": "fresh full-registry SHA then exact-CAS HOST_CLAIM and registry release",
        }
        atomic_json(terminal_path, final)
        publish("PASS_AWAITING_RELEASE", terminal_path=str(terminal_path), terminal_sha256=sha256(terminal_path))
    except BaseException as error:
        consumer_dead = terminate_exact(consumer, consumer_pid, consumer_ticks)
        route_dead = terminate_exact(route, route_pid, route_ticks)
        failure = {
            **state,
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "finished_unix": time.time(),
            "route_pid": route_pid,
            "route_startticks": route_ticks,
            "route_dead_verified": route_dead,
            "consumer_pid": consumer_pid,
            "consumer_startticks": consumer_ticks,
            "consumer_dead_verified": consumer_dead,
            "next_gate": "seal exact failure and fresh full-registry SHA before exact-CAS release",
        }
        atomic_json(terminal_path, failure)
        publish("FAILED_AWAITING_RELEASE", error=failure["error"], terminal_path=str(terminal_path), terminal_sha256=sha256(terminal_path))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    main(arguments.config)
