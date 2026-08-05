from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TEACHER_SIDECAR_SCHEMA = "banana-smasher-anchor-teacher-sidecars-v1"
CANDIDATE_SIDECAR_SCHEMA = "banana-smasher-anchor-candidate-sidecars-v1"
CANDIDATE_WINDOW_SCHEMA = "banana-smasher-anchor-candidate-window-v1"
ANCHOR_POSITION_CUTOFF = 1024
_HEX = frozenset("0123456789abcdef")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Anchor tensor sidecars require PyTorch") from exc
    return torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _window_key(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_pt(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} sidecar is missing: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"{label} sidecar SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    try:
        value = _torch().load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load {label} sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} sidecar must contain a tensor dictionary")
    return value


def _relative_sidecar(manifest_path: Path, prefix: str, width: int, slot: int, window_id: object) -> Path:
    suffix = str(window_id) if isinstance(window_id, int) and not isinstance(window_id, bool) and window_id >= 0 else f"{slot:03d}"
    return Path(f"{manifest_path.stem}.sidecars") / f"{prefix}{width}_win{suffix}.pt"


def _tensor_record(value: Any) -> dict[str, object]:
    return {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}


def _validate_teacher_rows(idx: Any, logprob: Any) -> None:
    sorted_idx = idx.sort(dim=1).values
    if bool((sorted_idx[:, 1:] == sorted_idx[:, :-1]).any()):
        raise ValueError("teacher sidecar support rows require unique token IDs")
    if bool((logprob[:, 1:] > logprob[:, :-1]).any()):
        raise ValueError("teacher sidecar support rows require descending logprob order")


def _validate_window_ids(window_ids: object) -> list[object]:
    if not isinstance(window_ids, list) or not window_ids:
        raise ValueError("sidecar manifest window_ids must be a non-empty ordered list")
    keys = [_window_key(value) for value in window_ids]
    if len(set(keys)) != len(keys):
        raise ValueError("sidecar manifest window_ids must be unique")
    return window_ids


def _read_manifest(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} manifest must be an object")
    return value


def _payload_path(manifest_path: Path, relative_path: Path, *, label: str) -> Path:
    path = manifest_path.parent
    for part in relative_path.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{label} sidecar payload path must not contain symlinks")
    return path


def _teacher_entry(
    manifest_path: Path,
    entry: object,
    *,
    width: int,
    expected_window_id: object,
    load: bool,
) -> tuple[Any, Any] | None:
    torch = _torch()
    required = {"window_id", "path", "bytes", "sha256", "tensors"}
    if not isinstance(entry, Mapping) or set(entry) != required:
        raise ValueError("teacher sidecar window entry fields mismatch")
    if _window_key(entry["window_id"]) != _window_key(expected_window_id):
        raise ValueError("teacher sidecar window order/identity mismatch")
    relative = entry["path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("teacher sidecar path must be relative")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("teacher sidecar path escapes its manifest root")
    if not _is_sha256(entry["sha256"]):
        raise ValueError("teacher sidecar SHA-256 is invalid")
    path = _payload_path(manifest_path, relative_path, label="teacher")
    expected_bytes = entry["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise ValueError("teacher sidecar byte count is invalid")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("teacher sidecar byte count or SHA-256 mismatch")
    value = _load_pt(path, expected_sha256=entry["sha256"], label="teacher")
    if set(value) != {"idx", "logprob"}:
        raise ValueError("teacher sidecar tensors must be idx and logprob")
    idx, logprob = value["idx"], value["logprob"]
    if (
        not isinstance(idx, torch.Tensor)
        or not isinstance(logprob, torch.Tensor)
        or idx.dtype != torch.int32
        or logprob.dtype != torch.float16
        or idx.ndim != 2
        or idx.shape != logprob.shape
        or idx.shape[1] != width
        or idx.shape[0] < 1
        or bool((idx < 0).any())
        or not bool(torch.isfinite(logprob).all())
    ):
        raise ValueError("teacher sidecar tensor shape/dtype/value mismatch")
    _validate_teacher_rows(idx, logprob)
    expected_tensors = {
        "idx": _tensor_record(idx),
        "logprob": _tensor_record(logprob),
    }
    if entry["tensors"] != expected_tensors:
        raise ValueError("teacher sidecar tensor metadata mismatch")
    return (idx, logprob) if load else None


def write_teacher_support_manifest(
    manifest_path: str | Path,
    *,
    windows: Sequence[Mapping[str, Any]],
    bank_sha256: str,
    teacher_sha256: str,
) -> dict[str, Any]:
    """Write historical-compatible t<width> tensor sidecars and their compact manifest."""

    torch = _torch()
    if not _is_sha256(bank_sha256) or not _is_sha256(teacher_sha256):
        raise ValueError("teacher sidecar identities require lowercase SHA-256 values")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise ValueError("teacher sidecars require at least one window")
    manifest_path = Path(manifest_path).expanduser().resolve()
    width: int | None = None
    window_ids: list[object] = []
    entries: list[dict[str, Any]] = []
    for slot, window in enumerate(windows):
        if not isinstance(window, Mapping) or set(window) != {"window_id", "idx", "logprob"}:
            raise ValueError("teacher sidecar windows require window_id, idx, and logprob")
        window_id = window["window_id"]
        if _window_key(window_id) in {_window_key(value) for value in window_ids}:
            raise ValueError("teacher sidecar window ids must be unique")
        idx = window["idx"]
        logprob = window["logprob"]
        if (
            not isinstance(idx, torch.Tensor)
            or not isinstance(logprob, torch.Tensor)
            or idx.dtype != torch.int32
            or logprob.dtype != torch.float16
            or idx.ndim != 2
            or idx.shape != logprob.shape
            or idx.shape[0] < 1
            or idx.shape[1] < 1
            or bool((idx < 0).any())
            or not bool(torch.isfinite(logprob).all())
        ):
            raise ValueError("teacher sidecar tensors require int32 idx and fp16 logprob [T,S]")
        _validate_teacher_rows(idx, logprob)
        if width is None:
            width = int(idx.shape[1])
        elif idx.shape[1] != width:
            raise ValueError("teacher sidecar support row widths are inconsistent")
        relative = _relative_sidecar(manifest_path, "t", width, slot, window_id)
        path = manifest_path.parent / relative
        if path.exists():
            raise FileExistsError(path)
        _atomic_torch_save(path, {"idx": idx.cpu(), "logprob": logprob.cpu()})
        entries.append(
            {
                "window_id": window_id,
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "tensors": {
                    "idx": _tensor_record(idx),
                    "logprob": _tensor_record(logprob),
                },
            }
        )
        window_ids.append(window_id)
    assert width is not None
    manifest = {
        "schema": TEACHER_SIDECAR_SCHEMA,
        "support_width": width,
        "window_ids": window_ids,
        "identities": {
            "bank_sha256": bank_sha256,
            "teacher_sha256": teacher_sha256,
        },
        "windows": entries,
    }
    _atomic_bytes(manifest_path, _canonical(manifest))
    return manifest


def load_teacher_support_manifest(
    manifest_path: str | Path,
    *,
    expected_bank_sha256: str | None = None,
    expected_teacher_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(manifest_path, label="teacher sidecar")
    if set(manifest) != {"schema", "support_width", "window_ids", "identities", "windows"}:
        raise ValueError("teacher sidecar manifest fields mismatch")
    if manifest["schema"] != TEACHER_SIDECAR_SCHEMA:
        raise ValueError(f"teacher sidecar schema must be {TEACHER_SIDECAR_SCHEMA}")
    width = manifest["support_width"]
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("teacher sidecar support_width must be positive")
    window_ids = _validate_window_ids(manifest["window_ids"])
    identities = manifest["identities"]
    if not isinstance(identities, Mapping) or set(identities) != {"bank_sha256", "teacher_sha256"}:
        raise ValueError("teacher sidecar identities fields mismatch")
    if not all(_is_sha256(value) for value in identities.values()):
        raise ValueError("teacher sidecar identities require lowercase SHA-256 values")
    if expected_bank_sha256 is not None and identities["bank_sha256"] != expected_bank_sha256:
        raise ValueError("teacher sidecar bank_sha256 mismatch")
    if expected_teacher_sha256 is not None and identities["teacher_sha256"] != expected_teacher_sha256:
        raise ValueError("teacher sidecar teacher_sha256 mismatch")
    entries = manifest["windows"]
    if not isinstance(entries, list) or len(entries) != len(window_ids):
        raise ValueError("teacher sidecar window coverage mismatch")
    for window_id, entry in zip(window_ids, entries, strict=True):
        _teacher_entry(
            manifest_path,
            entry,
            width=width,
            expected_window_id=window_id,
            load=False,
        )
    return manifest


def load_teacher_window(
    manifest_path: str | Path,
    window_id: object,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if manifest is None:
        manifest = load_teacher_support_manifest(manifest_path)
    keys = [_window_key(value) for value in manifest["window_ids"]]
    key = _window_key(window_id)
    if key not in keys:
        raise ValueError(f"teacher sidecar has no window {window_id!r}")
    slot = keys.index(key)
    loaded = _teacher_entry(
        manifest_path,
        manifest["windows"][slot],
        width=manifest["support_width"],
        expected_window_id=window_id,
        load=True,
    )
    assert loaded is not None
    return loaded


def _candidate_entry(
    manifest_path: Path,
    entry: object,
    *,
    width: int,
    expected_window_id: object,
    expected_identities: Mapping[str, Any],
) -> tuple[Any, Any]:
    torch = _torch()
    required = {"window_id", "path", "bytes", "sha256", "tensors"}
    if not isinstance(entry, Mapping) or set(entry) != required:
        raise ValueError("candidate sidecar window entry fields mismatch")
    if _window_key(entry["window_id"]) != _window_key(expected_window_id):
        raise ValueError("candidate sidecar window order/identity mismatch")
    relative = entry["path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("candidate sidecar path must be relative")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("candidate sidecar path escapes its manifest root")
    path = _payload_path(manifest_path, relative_path, label="candidate")
    if not _is_sha256(entry["sha256"]):
        raise ValueError("candidate sidecar SHA-256 is invalid")
    expected_bytes = entry["bytes"]
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not path.is_file()
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError("candidate sidecar byte count mismatch")
    value = _load_pt(path, expected_sha256=entry["sha256"], label="candidate")
    if set(value) != {"q_lp_at_ref", "q_argmax", "_banana_smasher"}:
        raise ValueError(
            "candidate sidecar must contain q_lp_at_ref, q_argmax, and identity metadata"
        )
    metadata = value["_banana_smasher"]
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != {"schema", "window_id", "support_width", "identities"}
        or metadata["schema"] != CANDIDATE_WINDOW_SCHEMA
        or _window_key(metadata["window_id"]) != _window_key(expected_window_id)
        or metadata["support_width"] != width
        or metadata["identities"] != expected_identities
    ):
        raise ValueError("candidate sidecar embedded identity metadata mismatch")
    q_lp, q_argmax = value["q_lp_at_ref"], value["q_argmax"]
    if (
        not isinstance(q_lp, torch.Tensor)
        or not isinstance(q_argmax, torch.Tensor)
        or q_lp.dtype != torch.float16
        or q_argmax.dtype != torch.int32
        or q_lp.ndim != 2
        or q_lp.shape[1] != width
        or q_argmax.shape != (q_lp.shape[0],)
        or q_lp.shape[0] < 1
        or not bool(torch.isfinite(q_lp).all())
        or bool((q_argmax < 0).any())
    ):
        raise ValueError("candidate sidecar tensor shape/dtype/value mismatch")
    expected_tensors = {
        "q_argmax": _tensor_record(q_argmax),
        "q_lp_at_ref": _tensor_record(q_lp),
    }
    if entry["tensors"] != expected_tensors:
        raise ValueError("candidate sidecar tensor metadata mismatch")
    return q_lp, q_argmax


class CandidateSidecarWriter:
    """Atomically append historical-compatible q<width> windows without replay."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        teacher_manifest_path: str | Path,
        window_ids: Sequence[object],
        basis_sha256: str,
        bank_sha256: str,
        model_id: str,
        pack_sha256: str,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.teacher_manifest_path = Path(teacher_manifest_path).expanduser().resolve()
        self.teacher = load_teacher_support_manifest(
            self.teacher_manifest_path, expected_bank_sha256=bank_sha256
        )
        self.window_ids = list(window_ids)
        _validate_window_ids(self.window_ids)
        if [_window_key(value) for value in self.window_ids] != [
            _window_key(value) for value in self.teacher["window_ids"]
        ]:
            raise ValueError("candidate ordered window identities differ from teacher support")
        if not _is_sha256(basis_sha256) or not _is_sha256(pack_sha256):
            raise ValueError("candidate sidecar basis/pack identities require lowercase SHA-256")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("candidate sidecar model_id must be non-empty")
        teacher_manifest_sha = _sha256_file(self.teacher_manifest_path)
        self.identities = {
            "bank_sha256": bank_sha256,
            "basis_sha256": basis_sha256,
            "model_id": model_id,
            "pack_sha256": pack_sha256,
            "teacher_manifest_sha256": teacher_manifest_sha,
            "teacher_sha256": self.teacher["identities"]["teacher_sha256"],
        }
        if self.manifest_path.exists():
            manifest = load_candidate_manifest(self.manifest_path)
            expected = {
                "schema": CANDIDATE_SIDECAR_SCHEMA,
                "support_width": self.teacher["support_width"],
                "window_ids": self.window_ids,
                "identities": self.identities,
            }
            for field, value in expected.items():
                if manifest.get(field) != value:
                    raise ValueError(f"candidate sidecar resume {field} mismatch")
            self._entries = list(manifest["windows"])
        else:
            self._entries: list[dict[str, Any]] = []
            self._write_manifest()
        self._adopt_committed_windows()

    @property
    def completed_window_ids(self) -> list[object]:
        return [entry["window_id"] for entry in self._entries]

    def _write_manifest(self) -> None:
        _atomic_bytes(
            self.manifest_path,
            _canonical(
                {
                    "schema": CANDIDATE_SIDECAR_SCHEMA,
                    "support_width": self.teacher["support_width"],
                    "window_ids": self.window_ids,
                    "identities": self.identities,
                    "windows": self._entries,
                }
            ),
        )

    def _adopt_committed_windows(self) -> None:
        """Commit authenticated sidecars left by an interrupted manifest update."""

        while len(self._entries) < len(self.window_ids):
            slot = len(self._entries)
            window_id = self.window_ids[slot]
            relative = _relative_sidecar(
                self.manifest_path,
                "q",
                self.teacher["support_width"],
                slot,
                window_id,
            )
            path = self.manifest_path.parent / relative
            if not path.exists():
                return
            digest = _sha256_file(path)
            payload = _load_pt(path, expected_sha256=digest, label="candidate")
            torch = _torch()
            q_lp_at_ref = payload.get("q_lp_at_ref")
            q_argmax = payload.get("q_argmax")
            if not isinstance(q_lp_at_ref, torch.Tensor) or not isinstance(
                q_argmax, torch.Tensor
            ):
                raise ValueError("candidate orphan sidecar tensor fields are invalid")
            entry = {
                "window_id": window_id,
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "tensors": {
                    "q_argmax": _tensor_record(q_argmax),
                    "q_lp_at_ref": _tensor_record(q_lp_at_ref),
                },
            }
            q_lp, _ = _candidate_entry(
                self.manifest_path,
                entry,
                width=self.teacher["support_width"],
                expected_window_id=window_id,
                expected_identities=self.identities,
            )
            teacher_idx, _ = load_teacher_window(
                self.teacher_manifest_path, window_id, manifest=self.teacher
            )
            if (
                q_lp.shape[1] != teacher_idx.shape[1]
                or q_lp.shape[0] > teacher_idx.shape[0]
            ):
                raise ValueError(
                    "candidate orphan sidecar position/support shape differs from teacher"
                )
            self._entries.append(entry)
            self._write_manifest()

    def write_window(
        self,
        window_id: object,
        *,
        q_lp_at_ref: Any,
        q_argmax: Any,
    ) -> bool:
        key = _window_key(window_id)
        completed = [_window_key(value) for value in self.completed_window_ids]
        if key in completed:
            return False
        slot = len(self._entries)
        if slot >= len(self.window_ids) or key != _window_key(self.window_ids[slot]):
            raise ValueError("candidate sidecars must be written in ordered window sequence")
        torch = _torch()
        if (
            not isinstance(q_lp_at_ref, torch.Tensor)
            or not isinstance(q_argmax, torch.Tensor)
            or q_lp_at_ref.dtype != torch.float16
            or q_argmax.dtype != torch.int32
            or q_lp_at_ref.ndim != 2
            or q_lp_at_ref.shape[1] != self.teacher["support_width"]
            or q_lp_at_ref.shape[0] < 1
            or q_argmax.shape != (q_lp_at_ref.shape[0],)
            or not bool(torch.isfinite(q_lp_at_ref).all())
            or bool((q_argmax < 0).any())
        ):
            raise ValueError("candidate sidecars require fp16 q_lp_at_ref [T,S] and int32 q_argmax [T]")
        teacher_idx, _ = load_teacher_window(
            self.teacher_manifest_path, window_id, manifest=self.teacher
        )
        if (
            q_lp_at_ref.shape[1] != teacher_idx.shape[1]
            or q_lp_at_ref.shape[0] > teacher_idx.shape[0]
        ):
            raise ValueError("candidate sidecar position/support shape differs from teacher")
        relative = _relative_sidecar(
            self.manifest_path,
            "q",
            self.teacher["support_width"],
            slot,
            window_id,
        )
        path = self.manifest_path.parent / relative
        if path.exists():
            raise ValueError(f"candidate sidecar exists without a manifest receipt: {path}")
        q_lp = q_lp_at_ref.detach().cpu().contiguous()
        q_am = q_argmax.detach().cpu().contiguous()
        _atomic_torch_save(
            path,
            {
                "q_lp_at_ref": q_lp,
                "q_argmax": q_am,
                "_banana_smasher": {
                    "schema": CANDIDATE_WINDOW_SCHEMA,
                    "window_id": window_id,
                    "support_width": self.teacher["support_width"],
                    "identities": self.identities,
                },
            },
        )
        self._entries.append(
            {
                "window_id": window_id,
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "tensors": {
                    "q_argmax": _tensor_record(q_am),
                    "q_lp_at_ref": _tensor_record(q_lp),
                },
            }
        )
        self._write_manifest()
        return True


def load_candidate_manifest(
    manifest_path: str | Path,
    *,
    expected_basis_sha256: str | None = None,
    expected_bank_sha256: str | None = None,
    expected_model_id: str | None = None,
    expected_pack_sha256: str | None = None,
    expected_teacher_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(manifest_path, label="candidate sidecar")
    if set(manifest) != {"schema", "support_width", "window_ids", "identities", "windows"}:
        raise ValueError("candidate sidecar manifest fields mismatch")
    if manifest["schema"] != CANDIDATE_SIDECAR_SCHEMA:
        raise ValueError(f"candidate sidecar schema must be {CANDIDATE_SIDECAR_SCHEMA}")
    width = manifest["support_width"]
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("candidate sidecar support_width must be positive")
    window_ids = _validate_window_ids(manifest["window_ids"])
    identities = manifest["identities"]
    required_identities = {
        "bank_sha256",
        "basis_sha256",
        "model_id",
        "pack_sha256",
        "teacher_manifest_sha256",
        "teacher_sha256",
    }
    if not isinstance(identities, Mapping) or set(identities) != required_identities:
        raise ValueError("candidate sidecar identities fields mismatch")
    if not isinstance(identities["model_id"], str) or not identities["model_id"]:
        raise ValueError("candidate sidecar model_id is invalid")
    if not all(
        _is_sha256(identities[field]) for field in required_identities - {"model_id"}
    ):
        raise ValueError("candidate sidecar identities require lowercase SHA-256 values")
    expected_identities = {
        "basis_sha256": expected_basis_sha256,
        "bank_sha256": expected_bank_sha256,
        "model_id": expected_model_id,
        "pack_sha256": expected_pack_sha256,
        "teacher_manifest_sha256": expected_teacher_manifest_sha256,
    }
    for field, expected in expected_identities.items():
        if expected is not None and identities[field] != expected:
            raise ValueError(f"candidate sidecar {field} mismatch")
    entries = manifest["windows"]
    if not isinstance(entries, list) or len(entries) > len(window_ids):
        raise ValueError("candidate sidecar window coverage is invalid")
    for window_id, entry in zip(window_ids, entries, strict=False):
        _candidate_entry(
            manifest_path,
            entry,
            width=width,
            expected_window_id=window_id,
            expected_identities=identities,
        )
    return manifest


def score_anchor_sidecars(
    teacher_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
) -> dict[str, Any]:
    """Score support-renormalized KLD and full-vocabulary top-1 like kld_score.py."""

    torch = _torch()
    teacher_path = Path(teacher_manifest_path).expanduser().resolve()
    candidate_path = Path(candidate_manifest_path).expanduser().resolve()
    teacher = load_teacher_support_manifest(teacher_path)
    candidate = load_candidate_manifest(candidate_path)
    if candidate["identities"]["teacher_manifest_sha256"] != _sha256_file(teacher_path):
        raise ValueError("candidate teacher manifest identity mismatch")
    if candidate["identities"]["teacher_sha256"] != teacher["identities"]["teacher_sha256"]:
        raise ValueError("candidate teacher identity mismatch")
    if candidate["identities"]["bank_sha256"] != teacher["identities"]["bank_sha256"]:
        raise ValueError("candidate bank identity mismatch")
    if candidate["support_width"] != teacher["support_width"]:
        raise ValueError("candidate/teacher support width mismatch")
    if candidate["window_ids"] != teacher["window_ids"] or len(candidate["windows"]) != len(
        teacher["window_ids"]
    ):
        raise ValueError("candidate/teacher ordered window coverage mismatch")

    per_window: list[dict[str, Any]] = []
    kld_sum = math.fsum(())
    top1_equal = 0
    positions = 0
    for window_id, entry in zip(candidate["window_ids"], candidate["windows"], strict=True):
        idx, ref_lp = load_teacher_window(
            teacher_path, window_id, manifest=teacher
        )
        q_lp, q_argmax = _candidate_entry(
            candidate_path,
            entry,
            width=candidate["support_width"],
            expected_window_id=window_id,
            expected_identities=candidate["identities"],
        )
        if q_lp.shape[1] != ref_lp.shape[1]:
            raise ValueError("candidate/teacher sidecar support shape mismatch")
        count = min(
            int(ref_lp.shape[0]),
            int(q_lp.shape[0]),
            ANCHOR_POSITION_CUTOFF,
        )
        idx = idx[:count]
        ref = ref_lp[:count].float()
        cand = q_lp[:count].float()
        q_argmax = q_argmax[:count]
        lp_n = ref - ref.logsumexp(-1, keepdim=True)
        lq_n = cand - cand.logsumexp(-1, keepdim=True)
        kld = (lp_n.exp() * (lp_n - lq_n)).sum(-1)
        if not bool(torch.isfinite(kld).all()):
            raise ValueError("sidecar KLD is non-finite")
        agreements = q_argmax.long() == idx[:, 0].long()
        window_sum = float(kld.sum().item())
        window_equal = int(agreements.sum().item())
        per_window.append(
            {
                "window_id": window_id,
                "positions": count,
                "kld_sum": window_sum,
                "mean_kld": window_sum / count,
                "top1_matches": window_equal,
                "top1_agreement": window_equal / count,
            }
        )
        kld_sum += window_sum
        top1_equal += window_equal
        positions += count
    return {
        "schema": "banana-smasher-anchor-sidecar-score-v1",
        "status": "PASS",
        "support_width": teacher["support_width"],
        "position_cutoff": ANCHOR_POSITION_CUTOFF,
        "windows": len(per_window),
        "positions": positions,
        "kld_sum": kld_sum,
        "mean_kld": kld_sum / positions,
        "top1_matches": top1_equal,
        "top1_agreement": top1_equal / positions,
        "per_window": per_window,
        "identities": dict(candidate["identities"]),
    }
