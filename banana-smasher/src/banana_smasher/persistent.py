from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class PersistentQueueError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PersistentUpdateQueue:
    """Durable explicit FIFO with a segment clock that starts only on transition."""

    def __init__(self, root: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root)
        self.clock = clock
        self.pending = self.root / "pending"
        self.active = self.root / "active"
        self.terminal = self.root / "terminal"
        for directory in (self.pending, self.active, self.terminal):
            directory.mkdir(parents=True, exist_ok=True)

    def waiting(self) -> dict[str, Any]:
        state = {
            "schema": "banana-smasher-persistent-state-v1",
            "state": "WAITING",
            "queued": len(list(self.pending.glob("*.json"))),
        }
        _atomic_json(self.root / "state.json", state)
        return state

    def enqueue(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping) or not request:
            raise ValueError("queue request must be a non-empty mapping")
        queued_at = float(self.clock())
        stable = json.dumps(dict(request), sort_keys=True, allow_nan=False).encode()
        job_id = f"{int(queued_at * 1_000_000):020d}-{hashlib.sha256(stable).hexdigest()[:12]}-{uuid.uuid4().hex[:8]}"
        record = {
            "schema": "banana-smasher-persistent-job-v1",
            "state": "QUEUED",
            "job_id": job_id,
            "queued_at": queued_at,
            "request": dict(request),
        }
        _atomic_json(self.pending / f"{job_id}.json", record)
        return record

    def claim_next(self, *, worker: str) -> dict[str, Any]:
        if not worker:
            raise ValueError("worker must be non-empty")
        candidates = sorted(self.pending.glob("*.json"))
        if not candidates:
            return self.waiting()
        source = candidates[0]
        destination = self.active / source.name
        try:
            os.replace(source, destination)
        except FileNotFoundError as exc:
            raise PersistentQueueError("queue head was concurrently claimed") from exc
        record = json.loads(destination.read_text())
        record.update({"state": "CLAIMED", "worker": worker})
        _atomic_json(destination, record)
        _atomic_json(self.root / "state.json", record)
        return record

    def segment_start(self, job_id: str, *, worker: str) -> dict[str, Any]:
        path = self.active / f"{job_id}.json"
        if not path.is_file():
            raise PersistentQueueError(f"active job does not exist: {job_id}")
        record = json.loads(path.read_text())
        if record.get("state") != "CLAIMED" or record.get("worker") != worker:
            raise PersistentQueueError("only the claiming worker may start a segment")
        record.update({"state": "SEGMENT_START", "segment_started_at": float(self.clock())})
        _atomic_json(path, record)
        _atomic_json(self.root / "state.json", record)
        return record

    def finish(self, job_id: str, *, worker: str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        source = self.active / f"{job_id}.json"
        if not source.is_file():
            raise PersistentQueueError(f"active job does not exist: {job_id}")
        record = json.loads(source.read_text())
        if record.get("state") != "SEGMENT_START" or record.get("worker") != worker:
            raise PersistentQueueError("job is not in SEGMENT_START for this worker")
        record.update({"state": "COMPLETE", "checkpoint": dict(checkpoint)})
        destination = self.terminal / source.name
        _atomic_json(destination, record)
        source.unlink()
        _atomic_json(self.root / "state.json", record)
        return record
