from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np


class DurabilityError(ValueError):
    """Raised when a durable artifact violates its declared identity."""


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise DurabilityError(f"SCHEMA_TYPE_UNSUPPORTED: {expected}")


def _schema_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise DurabilityError(f"SCHEMA_REF_UNSUPPORTED: {reference}")
    current: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise DurabilityError(f"SCHEMA_REF_MISSING: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise DurabilityError(f"SCHEMA_REF_NOT_OBJECT: {reference}")
    return current


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema_value(value, _schema_ref(root, schema["$ref"]), root=root, path=path)
    if "oneOf" in schema:
        matches = 0
        for choice in schema["oneOf"]:
            try:
                _validate_schema_value(value, choice, root=root, path=path)
            except DurabilityError:
                continue
            matches += 1
        if matches != 1:
            raise DurabilityError(f"{path}: expected exactly one schema match, got {matches}")
    expected_type = schema.get("type")
    if expected_type is not None:
        choices = [expected_type] if isinstance(expected_type, str) else expected_type
        if not isinstance(choices, list) or not any(
            isinstance(choice, str) and _schema_type_matches(value, choice)
            for choice in choices
        ):
            raise DurabilityError(f"{path}: type mismatch")
    if "const" in schema and value != schema["const"]:
        raise DurabilityError(f"{path}: constant mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise DurabilityError(f"{path}: enum mismatch")
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise DurabilityError(f"{path}: schema required is invalid")
        missing = [name for name in required if name not in value]
        if missing:
            raise DurabilityError(f"{path}: missing required keys {missing}")
        minimum = schema.get("minProperties")
        if isinstance(minimum, int) and len(value) < minimum:
            raise DurabilityError(f"{path}: too few properties")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise DurabilityError(f"{path}: schema properties is invalid")
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = f"{path}.{name}"
            if name in properties:
                _validate_schema_value(item, properties[name], root=root, path=child)
            elif additional is False:
                raise DurabilityError(f"{path}: additional property {name!r}")
            elif isinstance(additional, dict):
                _validate_schema_value(item, additional, root=root, path=child)
    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise DurabilityError(f"{path}: too few items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise DurabilityError(f"{path}: too many items")
        if schema.get("uniqueItems") is True:
            encoded = [canonical_json_bytes(item) for item in value]
            if len(set(encoded)) != len(encoded):
                raise DurabilityError(f"{path}: duplicate items")
        prefix = schema.get("prefixItems", [])
        if isinstance(prefix, list):
            for index, child_schema in enumerate(prefix[: len(value)]):
                _validate_schema_value(
                    value[index], child_schema, root=root, path=f"{path}[{index}]"
                )
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, items, root=root, path=f"{path}[{index}]")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise DurabilityError(f"{path}: string too short")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise DurabilityError(f"{path}: pattern mismatch")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise DurabilityError(f"{path}: number must be finite")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise DurabilityError(f"{path}: below minimum")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise DurabilityError(f"{path}: above maximum")


def validate_portable_schema(value: Any, filename: str, *, label: str) -> None:
    packaged = Path(__file__).with_name("schema") / filename
    source = Path(__file__).parents[2] / "schema" / filename
    path = packaged if packaged.is_file() else source
    schema = load_json_object(path, label=f"{label}_SCHEMA_DOCUMENT")
    try:
        _validate_schema_value(value, schema, root=schema, path="$")
    except DurabilityError as exc:
        raise DurabilityError(f"{label}_SCHEMA_INVALID: {exc}") from exc


def ensure_output_subdirectory(
    root: str | Path, relative: str | Path, *, label: str
) -> Path:
    directory = Path(root)
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise DurabilityError(f"{label}_PATH_UNSAFE: {relative!s}")
    current = directory
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {relative!s}")
        if current.exists() and not current.is_dir():
            raise DurabilityError(f"{label}_DIRECTORY_INVALID: {relative!s}")
        current.mkdir(exist_ok=True)
        if current.is_symlink():
            raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {relative!s}")
    try:
        current.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise DurabilityError(f"{label}_PATH_ESCAPE: {relative!s}") from exc
    return current


def _output_destination(
    path: str | Path,
    *,
    root: str | Path,
    label: str,
    reject_leaf_symlink: bool,
) -> tuple[Path, Path]:
    directory = Path(root)
    if directory.is_symlink():
        raise DurabilityError(f"{label}_ROOT_SYMLINK_FORBIDDEN: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    destination = Path(os.path.abspath(Path(path)))
    try:
        relative = destination.relative_to(directory)
    except ValueError as exc:
        raise DurabilityError(f"{label}_PATH_ESCAPE: {destination}") from exc
    if not relative.parts:
        raise DurabilityError(f"{label}_PATH_UNSAFE: {destination}")
    parent = directory
    if relative.parent != Path("."):
        parent = ensure_output_subdirectory(
            directory, relative.parent, label=f"{label}_PARENT"
        )
    if reject_leaf_symlink and destination.is_symlink():
        raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {destination}")
    return destination, parent


@contextlib.contextmanager
def _atomic_stream(
    path: str | Path,
    *,
    root: str | Path,
    label: str,
    mode: str,
) -> Iterator[Any]:
    destination, parent = _output_destination(
        path, root=root, label=label, reject_leaf_symlink=True
    )
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd = os.open(parent, parent_flags)
    temporary_name: str | None = None
    try:
        for _ in range(128):
            candidate = f".{destination.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise DurabilityError(f"{label}_TEMPORARY_NAME_EXHAUSTED")
        with os.fdopen(descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_symlink():
            raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {destination}")
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: str | Path, payload: bytes, *, root: str | Path) -> Path:
    destination = Path(path)
    with _atomic_stream(
        destination, root=root, label="ATOMIC_BYTES", mode="wb"
    ) as stream:
        stream.write(payload)
    return destination


def atomic_json(path: str | Path, value: Any, *, root: str | Path) -> Path:
    return atomic_bytes(path, canonical_json_bytes(value), root=root)


def atomic_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    root: str | Path,
) -> Path:
    destination = Path(path)
    with _atomic_stream(
        destination, root=root, label="ATOMIC_NPZ", mode="w+b"
    ) as stream:
        np.savez(stream, **cast(dict[str, Any], dict(arrays)))
    return destination


def safe_unlink_output(
    path: str | Path, *, root: str | Path, label: str, missing_ok: bool = True
) -> None:
    destination, parent = _output_destination(
        path, root=root, label=label, reject_leaf_symlink=False
    )
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd = os.open(parent, parent_flags)
    try:
        try:
            os.unlink(destination.name, dir_fd=parent_fd)
        except FileNotFoundError:
            if not missing_ok:
                raise
        else:
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DurabilityError(f"{label}_INVALID: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise DurabilityError(f"{label}_NOT_OBJECT: {source}")
    return value


def safe_relative_path(root: str | Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise DurabilityError(f"{label}_PATH_UNSAFE: {relative!r}")
    directory = Path(root).resolve()
    path = directory / candidate
    current = directory
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {relative}")
    try:
        path.resolve().relative_to(directory)
    except ValueError as exc:
        raise DurabilityError(f"{label}_PATH_ESCAPE: {relative!r}") from exc
    if not path.is_file():
        raise DurabilityError(f"{label}_MISSING: {relative}")
    return path


def file_identity(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DurabilityError(f"FILE_INVALID: {source}")
    relative = source.name if root is None else source.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def tree_identity(
    root: str | Path,
    *,
    excluded_names: Sequence[str] = (),
    excluded_suffixes: Sequence[str] = (".tmp",),
) -> dict[str, Any]:
    directory = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DurabilityError(f"TREE_SYMLINK_FORBIDDEN: {path}")
        if not path.is_file() or path.name in excluded_names:
            continue
        relative = path.relative_to(directory).as_posix()
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if any(relative.endswith(suffix) for suffix in excluded_suffixes):
            continue
        rows.append(file_identity(path, root=directory))
    return {
        "sha256": canonical_sha256(rows),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }


@contextlib.contextmanager
def output_lock(root: str | Path) -> Iterator[None]:
    directory = Path(root)
    if directory.is_symlink():
        raise DurabilityError(f"OUTPUT_LOCK_ROOT_SYMLINK_FORBIDDEN: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    lock_path = directory / ".banana-smasher.lock"
    if lock_path.is_symlink():
        raise DurabilityError(f"OUTPUT_LOCK_SYMLINK_FORBIDDEN: {lock_path}")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd = os.open(directory, parent_flags)
    try:
        try:
            descriptor = os.open(
                lock_path.name,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if lock_path.is_symlink():
                raise DurabilityError(
                    f"OUTPUT_LOCK_SYMLINK_FORBIDDEN: {lock_path}"
                ) from exc
            raise DurabilityError(f"OUTPUT_LOCK_INVALID: {lock_path}: {exc}") from exc
        with os.fdopen(descriptor, "a+b") as stream:
            try:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DurabilityError(f"OUTPUT_LOCKED: {directory}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(parent_fd)
