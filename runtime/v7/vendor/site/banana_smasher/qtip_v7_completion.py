"""Terminal-aware, never-kill completion loop for QTIP V7 evaluations."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "banana-smasher-qtip-v7-completion-request-v1"
STATE_SCHEMA = "banana-smasher-qtip-v7-completion-state-v1"
RESULT_SCHEMA = "banana-smasher-qtip-v7-completion-result-v1"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(base: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _start_token(pid: int) -> str | None:
    proc = Path(f"/proc/{pid}/stat")
    if proc.is_file():
        try:
            fields = proc.read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            return "linux:" + fields[19]
        except (FileNotFoundError, ProcessLookupError, IndexError):
            return None
    try:
        observed = subprocess.run(
            ["ps", "-o", "state=", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    state_and_token = observed.stdout.strip().split(maxsplit=1)
    if (
        observed.returncode != 0
        or len(state_and_token) != 2
        or state_and_token[0].startswith("Z")
    ):
        return None
    return "ps:" + state_and_token[1]


def _process_command(pid: int) -> str:
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.is_file():
        try:
            return proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            return ""
    try:
        observed = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return observed.stdout.strip() if observed.returncode == 0 else ""


def _live_process(active: Mapping[str, Any], command: Sequence[str]) -> bool:
    try:
        pid = int(active["pid"])
        token = str(active["start_token"])
        command_sha256 = str(active["command_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_command_sha256 = hashlib.sha256(_canonical(list(command))).hexdigest()
    return command_sha256 == expected_command_sha256 and _start_token(pid) == token


def _validate_request(request: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"completion request schema must be {REQUEST_SCHEMA!r}")
    command = request.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise ValueError("completion command must be a non-empty string array")
    contract = request.get("terminal_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("terminal_contract must be an object")
    ordered_windows = contract.get("ordered_windows")
    if (
        not isinstance(ordered_windows, list)
        or not ordered_windows
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ordered_windows)
        or len(set(ordered_windows)) != len(ordered_windows)
    ):
        raise ValueError("terminal ordered_windows must be unique integers")
    statuses = contract.get("allowed_statuses")
    if (
        not isinstance(statuses, list)
        or not statuses
        or any(not isinstance(value, str) or not value for value in statuses)
    ):
        raise ValueError("terminal allowed_statuses must be non-empty strings")
    run_root = _resolve(base_dir, request.get("run_root"), "run_root")
    cwd = _resolve(base_dir, request.get("cwd", str(run_root)), "cwd")
    terminal = _resolve(base_dir, request.get("terminal"), "terminal")
    retry_delay = request.get("retry_delay_seconds", 5.0)
    if (
        isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or float(retry_delay) < 0
    ):
        raise ValueError("retry_delay_seconds must be finite and non-negative")
    max_attempts = request.get("max_attempts")
    if max_attempts is not None and (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer or null")
    declared_environment = request.get("env", {})
    if not isinstance(declared_environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in declared_environment.items()
    ):
        raise ValueError("completion env must map strings to strings")
    return {
        **dict(request),
        "command": list(command),
        "terminal_contract": dict(contract),
        "run_root": run_root,
        "cwd": cwd,
        "terminal": terminal,
        "retry_delay_seconds": float(retry_delay),
        "max_attempts": max_attempts,
        "env": dict(declared_environment),
    }


def _validate_terminal(request: Mapping[str, Any]) -> dict[str, Any] | None:
    terminal_path = Path(request["terminal"])
    if not terminal_path.is_file():
        return None
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)
    if not isinstance(terminal, dict):
        raise ValueError("completion terminal must be an object")
    contract = request["terminal_contract"]
    status = terminal.get("status")
    if status not in contract["allowed_statuses"]:
        raise ValueError(f"terminal status is not admitted: {status!r}")
    metrics_path = Path(str(terminal.get("metrics_path", ""))).expanduser().resolve()
    if not metrics_path.is_file():
        raise ValueError("terminal metrics_path is unavailable")
    metrics_bytes = metrics_path.read_bytes()
    metrics_sha = hashlib.sha256(metrics_bytes).hexdigest()
    if terminal.get("metrics_sha256") != metrics_sha:
        raise ValueError("terminal metrics SHA-256 mismatch")
    metrics = json.loads(metrics_bytes)
    if not isinstance(metrics, dict):
        raise ValueError("completion metrics must be an object")
    rows = metrics.get("per_window")
    if not isinstance(rows, list):
        raise ValueError("completion metrics require per_window rows")
    observed_windows = [int(row.get("win", -1)) for row in rows]
    expected_windows = list(contract["ordered_windows"])
    expected_positions = int(contract["positions"])
    expected_support = int(contract["support"])
    checks = {
        "checkpoint_sha256": terminal.get("checkpoint_sha256") == contract["checkpoint_sha256"],
        "windows": int(metrics.get("windows", -1)) == len(expected_windows)
        and len(rows) == len(expected_windows),
        "ordered_windows": observed_windows == expected_windows,
        "positions": int(metrics.get("positions", -1)) == expected_positions
        and sum(int(row.get("positions", -1)) for row in rows) == expected_positions,
        "support": int(metrics.get("support", -1)) == expected_support
        and all(int(row.get("support", -1)) == expected_support for row in rows),
        "fallback_calls": int(terminal.get("fallback_calls", -1))
        == int(contract.get("fallback_calls", 0)),
        "pass_through_bytes": int(terminal.get("pass_through_bytes", -1))
        == int(contract.get("pass_through_bytes", 0)),
        "hidden_fp32_control_bytes": int(terminal.get("hidden_fp32_control_bytes", -1))
        == int(contract.get("hidden_fp32_control_bytes", 0)),
        "frozen_layer_count": int(terminal.get("frozen_layer_count", -1))
        == int(contract["frozen_layer_count"]),
        "complete_members": int(terminal.get("complete_members", -1))
        == int(contract["complete_members"]),
    }
    if not all(checks.values()):
        raise ValueError(f"completion terminal contract failed: {checks}")
    return {
        "schema": RESULT_SCHEMA,
        "status": "TERMINAL_VALIDATED",
        "scientific_status": status,
        "terminal": str(terminal_path),
        "terminal_sha256": hashlib.sha256(terminal_bytes).hexdigest(),
        "metrics": str(metrics_path),
        "metrics_sha256": metrics_sha,
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "windows": int(metrics["windows"]),
        "positions": int(metrics["positions"]),
        "support": int(metrics["support"]),
        "fallback_calls": int(terminal["fallback_calls"]),
        "kld_mean_binary64": float(metrics["kld_mean_binary64"]),
        "top1_matches": int(metrics["top1_matches"]),
        "top1_rate": float(metrics["top1_rate"]),
        "quality_gate": terminal.get("quality_gate"),
    }


def _wait_for_terminal_or_exit(
    request: Mapping[str, Any], active: Mapping[str, Any]
) -> dict[str, Any] | None:
    while _live_process(active, request["command"]):
        terminal = _validate_terminal(request)
        if terminal is not None:
            return terminal
        time.sleep(1.0)
    return _validate_terminal(request)


def run_qtip_v7_completion(request: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Drive one exact command until its declared terminal validates.

    There is no runtime timeout and no signal/kill path. A live exact PID is
    adopted from durable state. A naturally exited attempt is preserved and the
    identical command is retried. A valid RED quality split is still a completed
    scientific terminal and is returned honestly rather than retried.
    """
    if isinstance(request, Mapping):
        source = dict(request)
        base_dir = Path.cwd()
    else:
        request_path = Path(request).expanduser().resolve()
        source = _load_object(request_path)
        base_dir = request_path.parent
    normalized = _validate_request(source, base_dir=base_dir)
    run_root = Path(normalized["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    driver_root = run_root / "completion_driver"
    attempts_root = driver_root / "attempts"
    state_path = driver_root / "STATE.json"
    request_identity = {
        key: value
        for key, value in source.items()
        if key not in {"retry_delay_seconds", "max_attempts"}
    }
    request_sha = hashlib.sha256(_canonical(request_identity)).hexdigest()
    if state_path.is_file():
        state = _load_object(state_path)
        if state.get("schema") != STATE_SCHEMA or state.get("request_sha256") != request_sha:
            raise ValueError("completion state/request identity drift")
    else:
        state = {
            "schema": STATE_SCHEMA,
            "request_sha256": request_sha,
            "attempts_started": 0,
            "active_process": None,
        }
        _atomic_json(state_path, state)

    terminal = _validate_terminal(normalized)
    if terminal is not None:
        result = {**terminal, "attempts_started": int(state["attempts_started"])}
        state.update(status="TERMINAL_VALIDATED", active_process=None, result=result)
        _atomic_json(state_path, state)
        return result

    active = state.get("active_process")
    if isinstance(active, Mapping) and _live_process(active, normalized["command"]):
        terminal = _wait_for_terminal_or_exit(normalized, active)
        if terminal is not None:
            result = {**terminal, "attempts_started": int(state["attempts_started"])}
            state.update(status="TERMINAL_VALIDATED", active_process=None, result=result)
            _atomic_json(state_path, state)
            return result
        state.update(status="ATTEMPT_EXITED", active_process=None)
        _atomic_json(state_path, state)

    while True:
        attempt = int(state["attempts_started"]) + 1
        maximum = normalized["max_attempts"]
        if maximum is not None and attempt > int(maximum):
            raise RuntimeError(
                f"completion exhausted {maximum} natural attempts without a valid terminal"
            )
        attempt_root = attempts_root / f"attempt_{attempt:04d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        stdout_path = attempt_root / "stdout.log"
        stderr_path = attempt_root / "stderr.log"
        environment = os.environ.copy()
        environment.update(normalized["env"])
        environment["BANANA_SMASHER_COMPLETION_ATTEMPT"] = str(attempt)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                normalized["command"],
                cwd=normalized["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        token = _start_token(process.pid)
        if token is None:
            raise RuntimeError("launched process has no stable start token")
        active = {
            "pid": process.pid,
            "start_token": token,
            "attempt": attempt,
            "command_sha256": hashlib.sha256(_canonical(normalized["command"])).hexdigest(),
            "started_unix": time.time(),
        }
        state.update(status="RUNNING", attempts_started=attempt, active_process=active)
        _atomic_json(state_path, state)
        terminal = _wait_for_terminal_or_exit(normalized, active)
        return_code = process.poll()
        attempt_receipt = {
            "schema": "banana-smasher-qtip-v7-completion-attempt-v1",
            "attempt": attempt,
            "pid": process.pid,
            "start_token": token,
            "returncode": return_code,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "terminal_observed": terminal is not None,
            "completed_unix": time.time(),
        }
        _atomic_json(attempt_root / "ATTEMPT.json", attempt_receipt)
        if terminal is not None:
            result = {**terminal, "attempts_started": attempt}
            state.update(status="TERMINAL_VALIDATED", active_process=None, result=result)
            _atomic_json(state_path, state)
            return result
        state.update(status="ATTEMPT_EXITED", active_process=None)
        _atomic_json(state_path, state)
        time.sleep(normalized["retry_delay_seconds"])
