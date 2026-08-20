from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REQUEST_SCHEMA = "banana-smasher-update-request-v1"
_LEDGER_SCHEMA = "banana-smasher-segment-queue-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TERMINAL_STATES = ("PASS", "FAIL")
_REQUIRED_PHASES = (
    "decode",
    "staging_resident_layout",
    "kernel_forward",
    "backward",
    "optimizer",
    "checkpoint",
    "total",
)


class IdentityMismatch(RuntimeError):
    """A queued request does not match the resident worker identity."""


class DuplicateSegment(RuntimeError):
    """A segment identifier already exists in the durable exactly-once ledger."""


class SegmentStateConflict(RuntimeError):
    """A segment cannot make the requested durable state transition."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_json(path: Path, value: object, *, exclusive: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            # ``exists()`` followed by ``replace()`` is a TOCTOU race: two
            # enqueuers can both observe absence and the loser can overwrite
            # the winner. A same-directory hard link is an atomic create-if-
            # absent CAS and fails with EEXIST without touching the winner.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(data)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_sha(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def validate_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("update request must be a JSON object")
    request = dict(value)
    if request.get("schema") != _REQUEST_SCHEMA:
        raise ValueError(f"update request schema must be {_REQUEST_SCHEMA}")
    segment_id = request.get("segment_id", request.get("request_id"))
    request_id = request.get("request_id", segment_id)
    if (
        not isinstance(segment_id, str)
        or _REQUEST_ID.fullmatch(segment_id) is None
    ):
        raise ValueError("segment_id must be a filesystem-safe identifier")
    if request_id != segment_id:
        raise ValueError("request_id and segment_id must match when both are supplied")
    request["segment_id"] = segment_id
    request["request_id"] = segment_id
    checkpoint = request.get("input_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("input_checkpoint must be a non-empty path")
    for key in ("input_checkpoint_sha256", "config_sha256", "aot_sha256"):
        _validate_sha(key, request.get(key))
    payload = request.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    request["payload"] = payload
    return request


def request_identity(request: dict[str, Any]) -> str:
    normalized = validate_request(request)
    bound = {
        key: value
        for key, value in normalized.items()
        if key
        not in {
            "attempt_id",
            "ledger_state",
            "queued_unix",
            "request_id",
            "request_identity_sha256",
        }
    }
    return _sha256_bytes(_canonical_json(bound))


class _UpdateQueue:
    """Private compatibility queue retained for historical tests and fixtures.

    ``SEGMENT_QUEUE.json`` is the legacy control-plane artifact. Its
    segment identifiers are immutable tombstones: submitting an identifier a
    second time is refused in every state, including after completion and after
    the worker process exits.  Per-state receipt files are projections for
    operators; the ledger is authoritative across their crash windows.
    """

    def __init__(self, root: str | Path):
        selected = Path(root).resolve()
        if selected.name == "SEGMENT_QUEUE.json":
            self.root = selected.parent
            self.ledger_path = selected
        else:
            self.root = selected
            self.ledger_path = self.root / "SEGMENT_QUEUE.json"
        self.requests = self.root / "requests"
        self.receipts = self.root / "receipts"
        self.requests.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._ledger_lock_path = self.root / "SEGMENT_QUEUE.lock"

    @staticmethod
    def _empty_ledger() -> dict[str, Any]:
        now = time.time()
        return {
            "schema": _LEDGER_SCHEMA,
            "revision": 0,
            "created_unix": now,
            "updated_unix": now,
            "segments": {},
        }

    def _legacy_ledger(self) -> dict[str, Any]:
        """Import the pre-ledger request projections without replaying work."""

        ledger = self._empty_ledger()
        migrated = 0
        for path in sorted(self.requests.glob("*.json")):
            request = validate_request(_load_json(path))
            segment_id = request["segment_id"]
            identity = request_identity(request)
            external_state = "QUEUED"
            receipt: dict[str, Any] | None = None
            for candidate in ("PASS", "FAIL", "RUNNING", "QUEUED"):
                candidate_path = self._receipt_path(segment_id, candidate)
                if candidate_path.is_file():
                    external_state = candidate
                    receipt = _load_json(candidate_path)
                    break
            state = {
                "QUEUED": "QUEUED",
                "RUNNING": "INFLIGHT",
                "PASS": "COMPLETED",
                "FAIL": "FAILED",
            }[external_state]
            attempts: list[dict[str, Any]] = []
            attempt_id = None
            if state != "QUEUED":
                observed_attempt = receipt.get("attempt_id") if receipt is not None else None
                attempt_id = (
                    observed_attempt
                    if isinstance(observed_attempt, str) and observed_attempt
                    else _sha256_bytes(
                        f"legacy:{segment_id}:{identity}".encode()
                    )[:32]
                )
                attempts = [
                    {
                        "attempt_id": attempt_id,
                        "claimed_unix": (
                            receipt.get("updated_unix", request.get("queued_unix", 0.0))
                            if receipt is not None
                            else request.get("queued_unix", 0.0)
                        ),
                        "claimer_pid": receipt.get("worker_pid") if receipt is not None else None,
                        "migrated_from_projection": True,
                    }
                ]
            entry = {
                "segment_id": segment_id,
                "request_identity_sha256": identity,
                "request": request,
                "state": state,
                "attempts": attempts,
                "queued_unix": request.get("queued_unix", 0.0),
                "migrated_from_projection": str(path),
            }
            if attempt_id is not None:
                entry["attempt_id"] = attempt_id
            if state in {"COMPLETED", "FAILED"}:
                entry["terminal_state"] = external_state
                entry["terminal_fields"] = receipt or {}
                entry["sealed_unix"] = (
                    receipt.get("updated_unix", 0.0) if receipt is not None else 0.0
                )
            ledger["segments"][segment_id] = entry
            migrated += 1
        if migrated:
            ledger["migration"] = {
                "schema": "banana-smasher-segment-queue-migration-v1",
                "request_projections": migrated,
                "migrated_unix": time.time(),
            }
        return ledger

    def _read_ledger_unlocked(self) -> dict[str, Any]:
        ledger = (
            _load_json(self.ledger_path)
            if self.ledger_path.is_file()
            else self._legacy_ledger()
        )
        if ledger.get("schema") != _LEDGER_SCHEMA:
            raise RuntimeError(f"invalid segment queue schema: {self.ledger_path}")
        if not isinstance(ledger.get("segments"), dict):
            raise RuntimeError(f"segment queue has no segment map: {self.ledger_path}")
        return ledger

    def _locked_ledger(self, mutate: Callable[[dict[str, Any]], Any]) -> Any:
        """Serialize one read/compare/replace transition with an fsynced postimage."""

        with self._ledger_lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            ledger = self._read_ledger_unlocked()
            result = mutate(ledger)
            ledger["revision"] = int(ledger.get("revision", 0)) + 1
            ledger["updated_unix"] = time.time()
            _atomic_json(self.ledger_path, ledger)
            return result

    def ledger(self) -> dict[str, Any]:
        with self._ledger_lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            return self._read_ledger_unlocked()

    def _request_path(self, request_id: str) -> Path:
        return self.requests / f"{request_id}.json"

    def _receipt_path(self, request_id: str, state: str) -> Path:
        return self.receipts / f"{request_id}.{state}.json"

    def acquire_worker_lock(self):
        """Acquire the queue's process-scoped single-worker lock, fail-closed."""

        if not self.ledger_path.is_file():
            self._locked_ledger(lambda _ledger: None)
        path = self.root / "WORKER.lock"
        stream = path.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(
                f"persistent queue already has a live worker: {self.root}"
            ) from exc
        row = _canonical_json({"worker_pid": os.getpid(), "locked_unix": time.time()})
        stream.seek(0)
        stream.truncate()
        stream.write(row)
        stream.flush()
        os.fsync(stream.fileno())
        return stream

    def enqueue(self, value: object) -> dict[str, Any]:
        request = validate_request(value)
        request_id = request["segment_id"]
        identity = request_identity(request)
        request_row = {
            **request,
            "request_identity_sha256": identity,
            "queued_unix": time.time(),
        }

        def create(ledger: dict[str, Any]) -> None:
            segments = ledger["segments"]
            existing = segments.get(request_id)
            if existing is not None:
                state = existing.get("state", "UNKNOWN") if isinstance(existing, dict) else "INVALID"
                raise DuplicateSegment(
                    f"duplicate segment_id refused: {request_id} state={state}"
                )
            segments[request_id] = {
                "segment_id": request_id,
                "request_identity_sha256": identity,
                "request": request_row,
                "state": "QUEUED",
                "attempts": [],
                "queued_unix": request_row["queued_unix"],
            }

        self._locked_ledger(create)
        request_path = self._request_path(request_id)
        try:
            _atomic_json(request_path, request_row, exclusive=True)
        except FileExistsError as exc:
            existing = validate_request(_load_json(request_path))
            if request_identity(existing) != identity:
                raise RuntimeError(
                    f"request projection identity mismatch for segment {request_id}"
                ) from exc
        queued = {
            "schema": "banana-smasher-update-queue-receipt-v1",
            "state": "QUEUED",
            "status": "QUEUED",
            "segment_id": request_id,
            "request_id": request_id,
            "request_identity_sha256": identity,
            "input_checkpoint_sha256": request["input_checkpoint_sha256"],
            "config_sha256": request["config_sha256"],
            "aot_sha256": request["aot_sha256"],
            "queued_unix": request_row["queued_unix"],
        }
        try:
            queued["receipt_sha256"] = _atomic_json(
                self._receipt_path(request_id, "QUEUED"), queued, exclusive=True
            )
        except FileExistsError:
            pass
        return queued

    def get(self, request_id: str) -> dict[str, Any]:
        if _REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("invalid request_id")
        entry = self.ledger()["segments"].get(request_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("request"), dict):
            raise FileNotFoundError(self._request_path(request_id))
        return validate_request(entry["request"])

    def status(self, request_id: str) -> dict[str, Any]:
        if _REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("invalid request_id")
        entry = self.ledger()["segments"].get(request_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"no segment ledger entry for {request_id}")
        external_state = {
            "QUEUED": "QUEUED",
            "INFLIGHT": "RUNNING",
            "COMPLETED": "PASS",
            "FAILED": "FAIL",
        }.get(entry.get("state"))
        if external_state is None:
            raise RuntimeError(f"invalid ledger state for {request_id}: {entry.get('state')}")
        request = validate_request(entry["request"])
        canonical = {
            "schema": "banana-smasher-update-queue-receipt-v1",
            "state": external_state,
            "status": external_state,
            "segment_id": request_id,
            "request_id": request_id,
            "request_identity_sha256": request_identity(request),
            "input_checkpoint_sha256": request["input_checkpoint_sha256"],
            "config_sha256": request["config_sha256"],
            "aot_sha256": request["aot_sha256"],
        }
        path = self._receipt_path(request_id, external_state)
        if path.is_file():
            receipt = _load_json(path)
            mismatches = [
                key for key, value in canonical.items() if receipt.get(key) != value
            ]
            if mismatches:
                raise RuntimeError(
                    f"receipt identity mismatch for segment {request_id}: {mismatches}"
                )
            return receipt
        terminal_fields = entry.get("terminal_fields", {})
        if not isinstance(terminal_fields, dict):
            raise RuntimeError(f"invalid terminal fields for segment {request_id}")
        return {
            **terminal_fields,
            **canonical,
            "attempt_id": entry.get("attempt_id"),
            "ledger_state": entry["state"],
            "ledger_revision": self.ledger().get("revision"),
        }

    def pending(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        ledger = self.ledger()
        for entry in ledger["segments"].values():
            if not isinstance(entry, dict) or entry.get("state") not in {"QUEUED", "INFLIGHT"}:
                continue
            row = validate_request(entry.get("request"))
            row["ledger_state"] = entry["state"]
            row["attempt_id"] = entry.get("attempt_id")
            rows.append(row)
        rows.sort(key=lambda row: (float(row.get("queued_unix", 0.0)), row["request_id"]))
        return rows

    def claim_segment(self, request: dict[str, Any]) -> dict[str, Any]:
        """CAS QUEUED to INFLIGHT before any segment child/compute is launched.

        An INFLIGHT entry is returned only as a resume of the same immutable
        attempt; it never allocates a second attempt identifier.
        """

        request = validate_request(request)
        segment_id = request["segment_id"]
        identity = request_identity(request)

        def claim(ledger: dict[str, Any]) -> dict[str, Any]:
            entry = ledger["segments"].get(segment_id)
            if not isinstance(entry, dict):
                raise FileNotFoundError(f"no queued segment {segment_id}")
            if entry.get("request_identity_sha256") != identity:
                raise IdentityMismatch(f"segment identity mismatch for {segment_id}")
            state = entry.get("state")
            if state == "QUEUED":
                attempt_id = uuid.uuid4().hex
                attempt = {
                    "attempt_id": attempt_id,
                    "claimed_unix": time.time(),
                    "claimer_pid": os.getpid(),
                }
                entry["state"] = "INFLIGHT"
                entry["attempt_id"] = attempt_id
                entry["inflight_unix"] = attempt["claimed_unix"]
                entry["attempts"] = [attempt]
                return {**attempt, "segment_id": segment_id, "resumed": False}
            if state == "INFLIGHT":
                attempts = entry.get("attempts")
                if not isinstance(attempts, list) or len(attempts) != 1:
                    raise RuntimeError(f"segment {segment_id} has invalid exactly-once attempts")
                return {
                    **attempts[0],
                    "segment_id": segment_id,
                    "resumed": True,
                }
            raise SegmentStateConflict(
                f"segment {segment_id} cannot launch from durable state {state}"
            )

        return self._locked_ledger(claim)

    def write_state(self, request: dict[str, Any], state: str, **fields: object) -> dict[str, Any]:
        if state not in {"RUNNING", "PASS", "FAIL"}:
            raise ValueError(f"unsupported queue state {state}")
        reserved = {
            "schema",
            "state",
            "status",
            "segment_id",
            "request_id",
            "request_identity_sha256",
            "input_checkpoint_sha256",
            "config_sha256",
            "aot_sha256",
            "attempt_id",
            "updated_unix",
        }
        collisions = sorted(reserved.intersection(fields))
        if collisions:
            raise ValueError(f"reserved receipt fields cannot be overridden: {collisions}")
        request = validate_request(request)
        segment_id = request["segment_id"]
        if state == "RUNNING":
            attempt = self.claim_segment(request)
        else:
            target = "COMPLETED" if state == "PASS" else "FAILED"

            def seal(ledger: dict[str, Any]) -> dict[str, Any]:
                entry = ledger["segments"].get(segment_id)
                if not isinstance(entry, dict):
                    raise FileNotFoundError(f"no segment ledger entry for {segment_id}")
                if entry.get("request_identity_sha256") != request_identity(request):
                    raise IdentityMismatch(f"segment identity mismatch for {segment_id}")
                if entry.get("state") != "INFLIGHT":
                    raise SegmentStateConflict(
                        f"segment {segment_id} cannot seal {state} from {entry.get('state')}"
                    )
                entry["state"] = target
                entry["sealed_unix"] = time.time()
                entry["terminal_state"] = state
                entry["terminal_fields"] = dict(fields)
                return {
                    "attempt_id": entry.get("attempt_id"),
                    "sealed_unix": entry["sealed_unix"],
                }

            attempt = self._locked_ledger(seal)
        row = {
            "schema": "banana-smasher-update-queue-receipt-v1",
            "state": state,
            "status": state,
            "segment_id": segment_id,
            "request_id": segment_id,
            "request_identity_sha256": request_identity(request),
            "input_checkpoint_sha256": request["input_checkpoint_sha256"],
            "config_sha256": request["config_sha256"],
            "aot_sha256": request["aot_sha256"],
            "attempt_id": attempt.get("attempt_id"),
            "updated_unix": time.time(),
            **fields,
        }
        path = self._receipt_path(segment_id, state)
        try:
            row["receipt_sha256"] = _atomic_json(path, row, exclusive=True)
        except FileExistsError:
            existing = _load_json(path)
            if existing.get("request_identity_sha256") != row["request_identity_sha256"]:
                raise RuntimeError(f"receipt identity mismatch for segment {segment_id}")
            return existing
        return row

    def heartbeat(self, state: str, **fields: object) -> dict[str, Any]:
        if state not in {"INITIALIZING", "WAITING", "RUNNING", "STOPPED"}:
            raise ValueError(f"unsupported queue worker state {state}")
        row = {
            "schema": "banana-smasher-segment-queue-heartbeat-v1",
            "status": state,
            "state": state,
            "worker_pid": os.getpid(),
            "heartbeat_unix": time.time(),
            "segment_queue": str(self.ledger_path),
            **fields,
        }
        row["receipt_sha256"] = _atomic_json(self.root / "QUEUE_HEARTBEAT.json", row)
        return row


def _validate_cycle_result(
    result: object,
    *,
    allowed_statuses: tuple[str, ...] = ("PASS",),
) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("status") not in allowed_statuses:
        raise RuntimeError(
            f"persistent update cycle did not return an allowed status: {allowed_statuses}"
        )
    _validate_sha("output_checkpoint_sha256", result.get("output_checkpoint_sha256"))

    phases = result.get("phase_seconds")
    if not isinstance(phases, dict):
        raise RuntimeError("persistent cycle omitted phase_seconds")
    missing = [name for name in _REQUIRED_PHASES if not isinstance(phases.get(name), (int, float))]
    if missing:
        raise RuntimeError(f"persistent cycle omitted measured phases: {missing}")
    for key in ("rchar_delta", "memory_floor_bytes"):
        if not isinstance(result.get(key), int):
            raise RuntimeError(f"persistent cycle omitted {key}")
    if result.get("aot_engaged") is not True:
        raise RuntimeError("persistent cycle did not engage AOT")
    if result.get("fallback_used") is not False:
        raise RuntimeError("persistent cycle used or omitted fallback status")
    if not isinstance(result.get("loss"), (int, float)):
        raise RuntimeError("persistent cycle omitted loss")
    return dict(result)


def recover_committed_cycle(
    checkpoints: str | Path,
    *,
    update: int,
    request_id: str,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    """Return a crash-committed cycle from its immutable update sidecar.

    ``LATEST.pt`` is a hard-link alias and intentionally has no JSON sidecar.
    Recovery therefore binds the durable ``UPDATE_NNN.json`` marker to the
    still-RUNNING request and to the bytes of the promoted checkpoint.
    """

    if update < 0:
        raise ValueError("recovery update must be non-negative")
    if _REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("invalid recovery request_id")
    checkpoint_sha256 = _validate_sha("checkpoint SHA256", checkpoint_sha256)
    sidecar_path = Path(checkpoints).resolve() / f"UPDATE_{update:03d}.json"
    if not sidecar_path.is_file():
        return None
    sidecar = _load_json(sidecar_path)
    if sidecar.get("persistent_request_id") != request_id:
        return None
    result = sidecar.get("persistent_result")
    if not isinstance(result, dict):
        return None
    validated = _validate_cycle_result(
        result,
        allowed_statuses=("PASS", "FAIL_MAX_SEGMENT_SECONDS"),
    )
    if validated["output_checkpoint_sha256"] != checkpoint_sha256:
        raise RuntimeError(
            f"committed checkpoint identity mismatch for {request_id}: "
            f"{validated['output_checkpoint_sha256']} != {checkpoint_sha256}"
        )
    return validated


def _serve_queue_unlocked(
    queue: _UpdateQueue,
    *,
    expected_config_sha256: str,
    expected_aot_sha256: str,
    initialize: Callable[[], Any],
    cycle: Callable[[Any, dict[str, Any]], dict[str, Any]],
    recover: Callable[[Any, dict[str, Any]], dict[str, Any] | None] | None = None,
    stop_after: int | None = None,
    poll_seconds: float = 1.0,
    idle_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Initialize a resident worker once, then execute queued cycles in this PID."""

    expected_config_sha256 = _validate_sha("expected config SHA256", expected_config_sha256)
    expected_aot_sha256 = _validate_sha("expected AOT SHA256", expected_aot_sha256)
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    if idle_timeout_seconds is not None and idle_timeout_seconds < 0:
        raise ValueError("idle_timeout_seconds must be non-negative")

    init_started = time.monotonic()
    worker = initialize()
    init_seconds = time.monotonic() - init_started
    worker_pid = os.getpid()
    checkpoint_sha256 = (
        worker.get("checkpoint_sha256")
        if isinstance(worker, dict)
        else getattr(worker, "checkpoint_sha256")
    )
    checkpoint_sha256 = _validate_sha(
        "initialized checkpoint SHA256", checkpoint_sha256
    )
    init_receipt = {
        "schema": "banana-smasher-persistent-init-v1",
        "status": "PASS_INITIALIZED",
        "worker_pid": worker_pid,
        "init_seconds": init_seconds,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": expected_config_sha256,
        "aot_sha256": expected_aot_sha256,
        "initialized_unix": time.time(),
    }
    init_receipt["receipt_sha256"] = _atomic_json(queue.root / "INIT.json", init_receipt)
    queue.heartbeat(
        "WAITING",
        initialized=True,
        init_receipt=str(queue.root / "INIT.json"),
        config_sha256=expected_config_sha256,
        aot_sha256=expected_aot_sha256,
    )

    completed = 0
    failed = 0
    idle_started = time.monotonic()
    while stop_after is None or completed + failed < stop_after:
        pending = queue.pending()
        if not pending:
            queue.heartbeat(
                "WAITING",
                cycles_completed=completed,
                cycles_failed=failed,
            )
            if idle_timeout_seconds is not None and time.monotonic() - idle_started >= idle_timeout_seconds:
                return {
                    "status": "PASS_IDLE_TIMEOUT" if failed == 0 else "FAIL_IDLE_TIMEOUT",
                    "worker_pid": worker_pid,
                    "init_seconds": init_seconds,
                    "cycles_completed": completed,
                    "cycles_failed": failed,
                }
            time.sleep(poll_seconds)
            continue
        idle_started = time.monotonic()
        request = pending[0]
        request_id = request["request_id"]
        try:
            previous_state = queue.status(request_id).get("state")
            if previous_state == "RUNNING":
                recovered = recover(worker, request) if recover is not None else None
                if recovered is None:
                    raise RuntimeError(
                        f"inflight segment {request_id} has no committed recovery; "
                        "refusing exactly-once replay"
                    )
                result = _validate_cycle_result(
                    recovered,
                    allowed_statuses=("PASS", "FAIL_MAX_SEGMENT_SECONDS"),
                )
                current_sha = (
                    worker.get("checkpoint_sha256")
                    if isinstance(worker, dict)
                    else getattr(worker, "checkpoint_sha256")
                )
                if current_sha != result["output_checkpoint_sha256"]:
                    raise RuntimeError(
                        f"recovered checkpoint identity mismatch for {request_id}: "
                        f"{result['output_checkpoint_sha256']} != {current_sha}"
                    )
                recovered_status = result["status"]
                result_fields = {
                    key: value for key, value in result.items() if key != "status"
                }
                if recovered_status == "PASS":
                    queue.write_state(
                        request,
                        "PASS",
                        worker_pid=worker_pid,
                        init_seconds=init_seconds,
                        recovered_after_crash=True,
                        completed_unix=time.time(),
                        **result_fields,
                    )
                    completed += 1
                else:
                    queue.write_state(
                        request,
                        "FAIL",
                        worker_pid=worker_pid,
                        init_seconds=init_seconds,
                        recovered_after_crash=True,
                        error_type="SegmentWallExceeded",
                        error="recovered committed segment exceeded wall gate",
                        completed_unix=time.time(),
                        **result_fields,
                    )
                    failed += 1
                queue.heartbeat(
                    "WAITING",
                    last_segment_id=request_id,
                    recovered_status=recovered_status,
                    cycles_completed=completed,
                    cycles_failed=failed,
                )
                continue
            running = queue.write_state(
                request,
                "RUNNING",
                worker_pid=worker_pid,
                init_receipt=str(queue.root / "INIT.json"),
                started_unix=time.time(),
            )
            queue.heartbeat(
                "RUNNING",
                segment_id=request_id,
                attempt_id=running.get("attempt_id"),
            )
            if request["config_sha256"] != expected_config_sha256:
                raise IdentityMismatch(
                    f"config identity mismatch for {request_id}: "
                    f"{request['config_sha256']} != {expected_config_sha256}"
                )
            if request["aot_sha256"] != expected_aot_sha256:
                raise IdentityMismatch(
                    f"AOT identity mismatch for {request_id}: "
                    f"{request['aot_sha256']} != {expected_aot_sha256}"
                )
            current_sha = worker.get("checkpoint_sha256") if isinstance(worker, dict) else getattr(worker, "checkpoint_sha256")
            if current_sha != request["input_checkpoint_sha256"]:
                raise IdentityMismatch(
                    f"input checkpoint identity mismatch for {request_id}: "
                    f"{request['input_checkpoint_sha256']} != {current_sha}"
                )
            cycle_started = time.monotonic()
            result = _validate_cycle_result(cycle(worker, request))
            measured_total = time.monotonic() - cycle_started
            if abs(float(result["phase_seconds"]["total"]) - measured_total) > max(1.0, measured_total * 0.5):
                raise RuntimeError("cycle total phase is not consistent with measured worker wall")
            current_sha = (
                worker.get("checkpoint_sha256")
                if isinstance(worker, dict)
                else getattr(worker, "checkpoint_sha256")
            )
            if current_sha != result["output_checkpoint_sha256"]:
                raise RuntimeError(
                    f"resident checkpoint identity mismatch for {request_id}: "
                    f"{current_sha} != {result['output_checkpoint_sha256']}"
                )
            result_fields = {
                key: value for key, value in result.items() if key != "status"
            }
            queue.write_state(
                request,
                "PASS",
                worker_pid=worker_pid,
                running_receipt_sha256=running["receipt_sha256"],
                init_seconds=init_seconds,
                completed_unix=time.time(),
                **result_fields,
            )
            completed += 1
            queue.heartbeat("WAITING", last_segment_id=request_id)
        except Exception as exc:
            if queue.status(request_id).get("state") == "QUEUED":
                queue.write_state(request, "RUNNING", worker_pid=worker_pid)
            if queue.status(request_id).get("state") == "RUNNING":
                queue.write_state(
                    request,
                    "FAIL",
                    worker_pid=worker_pid,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    failed_unix=time.time(),
                    fallback_used=False,
                )
            failed += 1
            queue.heartbeat("WAITING", last_failed_segment_id=request_id)

    return {
        "status": "PASS_STOP_AFTER" if failed == 0 else "FAIL_STOP_AFTER",
        "worker_pid": worker_pid,
        "init_seconds": init_seconds,
        "cycles_completed": completed,
        "cycles_failed": failed,
    }


def _serve_queue(
    queue: _UpdateQueue,
    *,
    expected_config_sha256: str,
    expected_aot_sha256: str,
    initialize: Callable[[], Any],
    cycle: Callable[[Any, dict[str, Any]], dict[str, Any]],
    recover: Callable[[Any, dict[str, Any]], dict[str, Any] | None] | None = None,
    stop_after: int | None = None,
    poll_seconds: float = 1.0,
    idle_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    lock = queue.acquire_worker_lock()
    try:
        return _serve_queue_unlocked(
            queue,
            expected_config_sha256=expected_config_sha256,
            expected_aot_sha256=expected_aot_sha256,
            initialize=initialize,
            cycle=cycle,
            recover=recover,
            stop_after=stop_after,
            poll_seconds=poll_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
    finally:
        lock.close()