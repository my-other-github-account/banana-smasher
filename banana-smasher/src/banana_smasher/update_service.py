from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from .persistent import UpdateQueue, serve_queue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_adapter(name: str) -> ModuleType:
    if not name.startswith("banana_smasher."):
        raise ValueError("update adapter must be a package-owned banana_smasher module")
    module = importlib.import_module(name)
    for member in ("initialize", "cycle"):
        if not callable(getattr(module, member, None)):
            raise RuntimeError(f"update adapter {name!r} omits callable {member}()")
    recover = getattr(module, "recover", None)
    if recover is not None and not callable(recover):
        raise RuntimeError(f"update adapter {name!r} has a non-callable recover attribute")
    return module


def serve_persistent_updates(
    *,
    queue_root: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    aot: str | Path,
    adapter: str,
    expected_config_sha256: str,
    expected_aot_sha256: str,
    poll_seconds: float = 1.0,
    idle_timeout_seconds: float | None = None,
    stop_after_requests: int | None = None,
) -> dict[str, Any]:
    """Bind explicit artifacts to one resident, exactly-once update worker."""
    checkpoint_path = Path(checkpoint).resolve()
    config_path = Path(config).resolve()
    aot_path = Path(aot).resolve()
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
        ("AOT", aot_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} artifact is not a regular file: {path}")
    config_sha = _sha256(config_path)
    aot_sha = _sha256(aot_path)
    if config_sha != expected_config_sha256:
        raise RuntimeError(
            f"config identity mismatch: {config_sha} != {expected_config_sha256}"
        )
    if aot_sha != expected_aot_sha256:
        raise RuntimeError(f"AOT identity mismatch: {aot_sha} != {expected_aot_sha256}")

    implementation = _load_adapter(adapter)

    def initialize() -> Any:
        worker = implementation.initialize(
            checkpoint=checkpoint_path,
            config=config_path,
            aot=aot_path,
        )
        expected_checkpoint_sha = _sha256(checkpoint_path)
        observed = (
            worker.get("checkpoint_sha256")
            if isinstance(worker, dict)
            else getattr(worker, "checkpoint_sha256", None)
        )
        if observed != expected_checkpoint_sha:
            raise RuntimeError(
                "update adapter initialized the wrong checkpoint: "
                f"{observed} != {expected_checkpoint_sha}"
            )
        return worker

    recover = getattr(implementation, "recover", None)
    return serve_queue(
        UpdateQueue(queue_root),
        expected_config_sha256=config_sha,
        expected_aot_sha256=aot_sha,
        initialize=initialize,
        cycle=implementation.cycle,
        recover=recover,
        stop_after=stop_after_requests,
        poll_seconds=poll_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
    )
