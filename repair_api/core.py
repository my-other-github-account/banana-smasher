"""Shared identity, claim, memory, and checkpoint preflight for the façade."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

CANONICAL_BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
_BASIS_RE = re.compile(r"^[0-9a-f]{64}$")
_GIB = 1 << 30


def canonical_checkpoint(value: int | str) -> int | str:
    """Normalize the campaign's canonical U0 spelling without changing milestones."""
    text = str(value).strip()
    match = re.fullmatch(r"[Uu]0+", text)
    return "UPDATE_000" if match else value


def _available_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _drop_caches() -> dict[str, Any]:
    """Run the proven cache-drop step when the kernel permits it, without prompting."""
    path = Path("/proc/sys/vm/drop_caches")
    if not path.exists():
        return {"status": "UNAVAILABLE", "reason": "no-/proc/sys/vm/drop_caches"}
    try:
        if hasattr(os, "sync"):
            os.sync()
        path.write_text("3\n")
    except OSError as exc:
        return {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}:{exc.errno}"}
    return {"status": "PASS", "mode": "sync+drop_caches=3"}


class SharedPreflight:
    """One preflight implementation used by every ResidentRepairAPI operation."""

    def __init__(self, artifact: Any):
        self.artifact = artifact
        self._memory: dict[str, Any] | None = None

    def _basis(self) -> str:
        identity = self.artifact.manifest.get("identity", {})
        basis = identity.get("basis_sha256") if isinstance(identity, Mapping) else None
        if basis is None:
            basis = self.artifact.manifest.get("basis_sha256")
        if not isinstance(basis, str) or _BASIS_RE.fullmatch(basis) is None:
            raise ValueError("basis_sha256 must be 64 lowercase hex characters")
        return basis

    def _memory_preflight(self, peak_gib: float) -> dict[str, Any]:
        if self._memory is None:
            before = _available_bytes()
            cache_drop = _drop_caches()
            after = _available_bytes()
            self._memory = {
                "status": "PASS",
                "available_before_bytes": before,
                "available_after_bytes": after,
                "cache_drop": cache_drop,
            }
        available = self._memory.get("available_after_bytes")
        required = int(float(peak_gib) * _GIB) + 4 * _GIB
        if available is not None and peak_gib > 0 and available < required:
            raise MemoryError(
                f"memory preflight failed: available={available} required_peak_plus_4GiB={required}"
            )
        return {**self._memory, "peak_estimate_gib": float(peak_gib), "reserve_gib": 4}

    @staticmethod
    def _claim(preflight: Mapping[str, Any] | None, basis: str) -> dict[str, Any]:
        if not preflight or preflight.get("claim_path") is None:
            return {"status": "NOT_REQUESTED"}
        path = Path(preflight["claim_path"]).expanduser()
        try:
            claim = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"claim preflight could not read {path}: {exc}") from exc
        expected_task = preflight.get("task_id")
        if claim.get("state") != "CLAIMED" or claim.get("task_id") != expected_task:
            raise RuntimeError("claim preflight task/state mismatch")
        if claim.get("intended_basis") != basis:
            raise RuntimeError("claim preflight basis mismatch")
        return {
            "status": "PASS",
            "path": str(path),
            "task_id": expected_task,
            "workload_pid": claim.get("workload_pid"),
        }

    def run(
        self,
        operation: str,
        checkpoint: str,
        windows: tuple[int, ...],
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = {} if preflight is None else dict(preflight)
        basis = self._basis()
        return {
            "status": "PASS",
            "operation": operation,
            "checkpoint": checkpoint,
            "windows": list(windows),
            "basis_sha256": basis,
            "claim": self._claim(options, basis),
            "memory": self._memory_preflight(float(options.get("peak_gib", 0.0))),
        }
