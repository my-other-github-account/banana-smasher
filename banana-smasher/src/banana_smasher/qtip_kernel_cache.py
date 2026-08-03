from __future__ import annotations

import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable
import uuid

from .qtip_rings import (
    PERSISTENT_BACKENDS,
    TRELLIS_V2_BACKEND,
    QtipRing,
    resolve_qtip_ring,
)

_RECEIPT_NAME = "QTIP_KERNEL_RECEIPT.json"
_SCHEMA = "banana-smasher-qtip-kernel-build-plan-v1"
_PACKAGE_ROOT = Path(__file__).resolve().parent
_MEMINFO_PATH = Path("/proc/meminfo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _source_files() -> tuple[Path, ...]:
    trellis = _PACKAGE_ROOT / "trellis_v2"
    paths = [
        _PACKAGE_ROOT / "qtip_rings.json",
        _PACKAGE_ROOT / "qtip_rings.py",
        _PACKAGE_ROOT / "qtip_kernel_cache.py",
        _PACKAGE_ROOT / "qtip_viterbi.py",
    ]
    paths.extend(
        sorted(
            path
            for path in trellis.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh"}
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"QTIP kernel build source missing: {missing}")
    return tuple(paths)


def build_plan(
    bpw: object,
    *,
    architecture: str,
) -> dict[str, Any]:
    ring = resolve_qtip_ring(bpw)
    source_sha256 = {
        str(path.relative_to(_PACKAGE_ROOT)): _sha256(path)
        for path in _source_files()
    }
    recipes = [
        {
            "geometry": {
                key: value
                for key, value in zip(("L", "K", "V"), component.geometry)
            },
            "backend": component.backend,
            "compile_constants": {
                "states": 1 << component.geometry[0],
                "branch_bits": component.geometry[1] * component.geometry[2],
                "steps": 128,
            },
            **(
                {
                    "compile_variants": [
                        {"has_overlap": False},
                        {"has_overlap": True},
                    ],
                    "execution_contract": {
                        "host_launches_per_call": 1,
                        "transition_loop": "device-resident",
                        "exact_identity": "discrete-trellis",
                        "target_seconds_per_unit": 2.2,
                    },
                }
                if component.backend in PERSISTENT_BACKENDS
                else {}
            ),
        }
        for component in ring.components
    ]
    unsigned = {
        "schema": _SCHEMA,
        "bpw": ring.canonical_bpw,
        "tier": ring.tier,
        "architecture": architecture,
        "recipes": recipes,
        "source_sha256": source_sha256,
        "aot": dict(ring.aot),
        "codebook": dict(ring.codebook),
    }
    unsigned["cache_key_sha256"] = hashlib.sha256(_json_bytes(unsigned)).hexdigest()
    return unsigned


def _assert_inside(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"kernel artifact escapes cache directory: {path}") from exc


def _assert_not_symlink(path: Path, parent: Path) -> None:
    relative = path.relative_to(parent)
    candidate = parent
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise RuntimeError(f"compiled QTIP kernel artifact may not be a symlink: {candidate}")


def _assert_private_cache_path(path: Path) -> None:
    try:
        details = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"QTIP kernel cache path is unavailable: {path}") from exc
    if details.st_uid != os.getuid():
        raise RuntimeError(f"QTIP kernel cache path is not owned by this user: {path}")
    if details.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"QTIP kernel cache path is group/world writable: {path}")


def _load_sha_pinned_extension(
    module_name: str,
    extension: Path,
    expected_sha256: str,
) -> Any:
    """Load one regular native extension through its verified open descriptor."""
    if not extension.is_absolute() or extension.is_symlink():
        raise RuntimeError(f"invalid QTIP2 CUDA producer path: {extension}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(extension, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"QTIP2 CUDA producer is not a regular file: {extension}")
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 8 << 20), b""):
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"QTIP2 CUDA producer SHA mismatch: {actual_sha256} != {expected_sha256}"
            )
        fd_path = Path(f"/proc/self/fd/{descriptor}")
        if not fd_path.is_file():
            raise RuntimeError("QTIP2 fd-pinned CUDA loading requires procfs")
        loader = importlib.machinery.ExtensionFileLoader(module_name, str(fd_path))
        spec = importlib.util.spec_from_file_location(module_name, fd_path, loader=loader)
        if spec is None:
            raise ImportError(f"cannot load QTIP2 CUDA producer: {extension}")
        module = importlib.util.module_from_spec(spec)
        prior = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            loader.exec_module(module)
        except Exception:
            if prior is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = prior
            raise
        return module
    finally:
        os.close(descriptor)


def seal_kernel_cache(
    plan: dict[str, Any],
    cache_dir: Path,
    artifacts: Iterable[Path],
) -> Path:
    cache_dir = cache_dir.resolve()
    rows = []
    for raw_path in sorted({Path(path).resolve() for path in artifacts}):
        _assert_inside(raw_path, cache_dir)
        if not raw_path.is_file() or raw_path.name == _RECEIPT_NAME:
            raise ValueError(f"invalid compiled kernel artifact: {raw_path}")
        rows.append(
            {
                "path": str(raw_path.relative_to(cache_dir)),
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256(raw_path),
            }
        )
    if not rows:
        raise RuntimeError("AOT build produced no compiled kernel artifacts")
    receipt = {
        "schema": "banana-smasher-qtip-kernel-cache-v1",
        "status": "PASS",
        "plan": plan,
        "cache_key_sha256": plan["cache_key_sha256"],
        "artifacts": rows,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    for raw_path in sorted({Path(path).resolve() for path in artifacts}):
        os.chmod(raw_path, 0o400)
    output = cache_dir / _RECEIPT_NAME
    temporary = cache_dir / f".{_RECEIPT_NAME}.{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    return output


def verify_kernel_cache(plan: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    receipt_path = cache_dir / _RECEIPT_NAME
    _assert_private_cache_path(cache_dir)
    _assert_private_cache_path(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid QTIP kernel cache receipt: {receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "banana-smasher-qtip-kernel-cache-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("plan") != plan
        or receipt.get("cache_key_sha256") != plan.get("cache_key_sha256")
    ):
        raise RuntimeError(f"QTIP kernel cache plan mismatch: {receipt_path}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"QTIP kernel cache has no artifacts: {receipt_path}")
    for row in artifacts:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(f"invalid QTIP kernel cache artifact row: {receipt_path}")
        unresolved = cache_dir / row["path"]
        _assert_not_symlink(unresolved, cache_dir)
        artifact = unresolved.resolve()
        _assert_inside(artifact, cache_dir)
        if not artifact.is_file():
            raise RuntimeError(f"compiled QTIP kernel artifact missing: {artifact}")
        _assert_private_cache_path(artifact)
        actual = _sha256(artifact)
        if actual != row.get("sha256"):
            raise RuntimeError(
                f"compiled QTIP kernel artifact SHA mismatch: {artifact}: "
                f"{actual} != {row.get('sha256')}"
            )
        if artifact.stat().st_size != row.get("bytes"):
            raise RuntimeError(f"compiled QTIP kernel artifact size mismatch: {artifact}")
    return receipt


def _architecture(torch: Any) -> str:
    major, minor = torch.cuda.get_device_capability()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", torch.cuda.get_device_name()).strip("-")
    return f"sm_{major}{minor}-{name}"


def _trellis_v2_extension(artifact_root: Path) -> Path | None:
    artifact_root = artifact_root.resolve()
    extension_dir = artifact_root / "torch-extensions" / "trellis-v2"
    if not extension_dir.exists():
        return None
    if extension_dir.is_symlink():
        raise RuntimeError(f"QTIP2 CUDA producer directory may not be a symlink: {extension_dir}")
    resolved_dir = extension_dir.resolve()
    _assert_inside(resolved_dir, artifact_root)
    extensions = sorted(extension_dir.glob("trellis_v2_cuda_exact*.so"))
    if len(extensions) != 1:
        raise RuntimeError(
            f"QTIP2 kernel cache must contain exactly one CUDA producer: {extensions}"
        )
    extension = extensions[0]
    if extension.is_symlink():
        raise RuntimeError(f"QTIP2 CUDA producer may not be a symlink: {extension}")
    resolved = extension.resolve()
    _assert_inside(resolved, resolved_dir)
    if not resolved.is_file():
        raise RuntimeError(f"QTIP2 CUDA producer is not a regular file: {resolved}")
    return resolved


def _configure_compiler_cache(artifact_root: Path) -> None:
    os.environ["TRITON_CACHE_DIR"] = str(artifact_root / "triton")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(artifact_root / "torch-extensions")
    os.environ.pop("BANANA_SMASHER_TRELLIS_V2_EXTENSION", None)
    os.environ.pop("BANANA_SMASHER_TRELLIS_V2_EXTENSION_SHA256", None)
    receipt_path = artifact_root.parent / _RECEIPT_NAME
    if not receipt_path.is_file():
        return
    extension = _trellis_v2_extension(artifact_root)
    if extension is None:
        return
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid QTIP kernel receipt: {receipt_path}") from exc
    relative = str(extension.relative_to(artifact_root.parent.resolve()))
    rows = [
        row
        for row in receipt.get("artifacts", [])
        if isinstance(row, dict) and row.get("path") == relative
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"QTIP2 CUDA producer is not uniquely sealed in {receipt_path}: {relative}"
        )
    expected = rows[0].get("sha256")
    actual = _sha256(extension)
    if not isinstance(expected, str) or actual != expected:
        raise RuntimeError(
            f"QTIP2 CUDA producer SHA mismatch: {extension}: {actual} != {expected}"
        )
    os.environ["BANANA_SMASHER_TRELLIS_V2_EXTENSION"] = str(extension)
    os.environ["BANANA_SMASHER_TRELLIS_V2_EXTENSION_SHA256"] = actual


def _build_trellis_v2_extension(artifact_root: Path) -> Path:
    """Build and atomically publish the packaged non-Ninja QTIP2 extension."""
    _configure_compiler_cache(artifact_root)
    source_root = _PACKAGE_ROOT / "trellis_v2"
    extensions_root = artifact_root / "torch-extensions"
    extensions_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    build_lib = extensions_root / f".trellis-v2-stage-{token}"
    build_temp = extensions_root / f".trellis-v2-build-stage-{token}"
    build_lib.mkdir(parents=True)
    build_temp.mkdir(parents=True)
    command = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--build-lib",
        str(build_lib),
        "--build-temp",
        str(build_temp),
    ]
    try:
        subprocess.run(
            command,
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(build_lib, ignore_errors=True)
        shutil.rmtree(build_temp, ignore_errors=True)
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            "public QTIP2 CUDA producer build failed via the packaged non-Ninja "
            f"build_ext path: {detail}"
        ) from exc
    staged = sorted(build_lib.glob("trellis_v2_cuda_exact*.so"))
    if len(staged) != 1 or staged[0].is_symlink():
        shutil.rmtree(build_lib, ignore_errors=True)
        shutil.rmtree(build_temp, ignore_errors=True)
        raise RuntimeError(
            f"public QTIP2 CUDA producer build emitted invalid extensions: {staged}"
        )
    final_lib = extensions_root / "trellis-v2"
    final_temp = extensions_root / "trellis-v2-build"
    shutil.rmtree(final_lib, ignore_errors=True)
    shutil.rmtree(final_temp, ignore_errors=True)
    os.replace(build_lib, final_lib)
    os.replace(build_temp, final_temp)
    extension = _trellis_v2_extension(artifact_root)
    if extension is None:  # pragma: no cover - staged population was checked
        raise RuntimeError("public QTIP2 CUDA producer publication failed")
    digest = _sha256(extension)
    os.environ["BANANA_SMASHER_TRELLIS_V2_EXTENSION"] = str(extension)
    os.environ["BANANA_SMASHER_TRELLIS_V2_EXTENSION_SHA256"] = digest
    return extension


def _memory_preflight(torch: Any) -> dict[str, Any]:
    device_free, device_total = torch.cuda.mem_get_info()
    result = {
        "source": "cuda:mem_get_info",
        "available_bytes": device_free,
        "device_free_bytes": device_free,
        "device_total_bytes": device_total,
    }
    try:
        fields = {
            parts[0].removesuffix(":"): parts[1:]
            for line in _MEMINFO_PATH.read_text().splitlines()
            if len(parts := line.split()) >= 2
        }
        value = fields.get("MemAvailable")
        if value is None or len(value) != 2 or value[1] != "kB":
            raise ValueError("MemAvailable kB field missing")
        available = int(value[0]) * 1024
        if available < 0:
            raise ValueError("MemAvailable is negative")
    except (OSError, ValueError) as exc:
        result["host_memory_probe_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["host_available_bytes"] = available
    result["meminfo_path"] = str(_MEMINFO_PATH)
    return result


def _compile_ring(ring: QtipRing, artifact_root: Path, torch: Any) -> None:
    _configure_compiler_cache(artifact_root)
    if any(component.backend == TRELLIS_V2_BACKEND for component in ring.components):
        _build_trellis_v2_extension(artifact_root)
    qtip_viterbi = importlib.import_module("banana_smasher.qtip_viterbi")

    for component in ring.components:
        L, K, V = component.geometry
        lut_dtype = (
            torch.float32
            if component.backend == TRELLIS_V2_BACKEND
            else torch.float16
        )
        cb = SimpleNamespace(
            L=L,
            K=K,
            V=V,
            lut=torch.zeros((V, 1 << L), device="cuda", dtype=lut_dtype),
        )
        batch = 256 if component.backend == TRELLIS_V2_BACKEND else 1
        x = torch.zeros((256, batch), device="cuda", dtype=torch.float16)
        if component.backend == TRELLIS_V2_BACKEND:
            from .trellis_v2 import install_trellis_v2

            install_trellis_v2(cb)
            cb.quantize_seq(x)
            overlap = torch.zeros((batch,), device="cuda", dtype=torch.int32)
            cb.quantize_seq(x, overlap)
            del overlap
        elif component.backend in PERSISTENT_BACKENDS:
            qtip_viterbi.exact_prefix_viterbi(cb, x)
            overlap = torch.zeros(
                (batch,), device="cuda", dtype=torch.int32
            )
            qtip_viterbi.exact_prefix_viterbi(cb, x, overlap)
            del overlap
        else:
            raise RuntimeError(
                f"QTIP backend {component.backend!r} has no AOT compiler; run "
                f"`smash kernels build --tier qtip --bpw {ring.canonical_bpw}`"
            )
        torch.cuda.synchronize()
        del x, cb
        torch.cuda.empty_cache()


def _materialize_qtip_kernels(
    *,
    plan: dict[str, Any],
    ring: QtipRing,
    producer: str,
    cache_dir: Path,
    torch: Any,
) -> dict[str, Any]:
    receipt_path = cache_dir / _RECEIPT_NAME
    artifact_root = cache_dir / "artifacts"
    if receipt_path.is_file():
        receipt = verify_kernel_cache(plan, cache_dir)
        _configure_compiler_cache(artifact_root)
        return {**receipt, "cache_hit": True, "receipt": str(receipt_path)}

    max_prefixes = max(
        1 << (component.geometry[0] - component.geometry[1] * component.geometry[2])
        for component in ring.components
    )
    estimate = 128 * max_prefixes * 4 + 2 * max_prefixes * 4 + (8 << 20)
    if any(component.backend == TRELLIS_V2_BACKEND for component in ring.components):
        # Two graph variants (with/without overlap), each retaining exact input,
        # LUT, state, packed-backpointer and ping-pong cost buffers, plus the
        # bounded eight-entry transposed-LUT cache.
        graph_state = (
            256 * 256 * 2
            + 65_536 * 2 * 4
            + 256 * 4
            + 128 * 256 * 4
            + 64 * 256 * 4_096
            + 2 * 256 * 4_096 * 4
        )
        estimate += 2 * graph_state + 8 * 65_536 * 2 * 4
    memory_preflight = _memory_preflight(torch)
    reserve = 4 << 30
    if memory_preflight["available_bytes"] - reserve < estimate:
        raise RuntimeError(
            "AOT kernel build memory preflight failed: "
            f"available={memory_preflight['available_bytes']} "
            f"source={memory_preflight['source']} "
            f"device_free={memory_preflight['device_free_bytes']} "
            f"reserve={reserve} estimate={estimate}; producer `{producer}`"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    _compile_ring(ring, artifact_root, torch)
    artifacts = [path for path in artifact_root.rglob("*") if path.is_file()]
    seal_kernel_cache(plan, cache_dir, artifacts)
    receipt = verify_kernel_cache(plan, cache_dir)
    return {
        **receipt,
        "cache_hit": False,
        "receipt": str(receipt_path),
        "memory_preflight": memory_preflight,
    }


def build_qtip_kernels(
    bpw: object,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Ahead-of-solve compile every component for one packaged QTIP ring."""
    import torch

    ring = resolve_qtip_ring(bpw)
    producer = f"smash kernels build --tier qtip --bpw {ring.canonical_bpw}"
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA required for AOT kernel build; run `{producer}` on a CUDA host")
    architecture = _architecture(torch)
    plan = build_plan(ring.canonical_bpw, architecture=architecture)
    root = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else Path(
            os.environ.get(
                "BANANA_SMASHER_KERNEL_CACHE",
                "~/.cache/banana-smasher/qtip-kernels",
            )
        ).expanduser()
    ).resolve()
    cache_dir = root / architecture / plan["cache_key_sha256"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".build.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _materialize_qtip_kernels(
            plan=plan,
            ring=ring,
            producer=producer,
            cache_dir=cache_dir,
            torch=torch,
        )
