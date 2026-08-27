"""Model-neutral public BALANCED64 teacher capture and PRE orchestration."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from math import fsum
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
from typing import Any, Protocol

from .hf_moe import HF_UNIFORM_ARTIFACT_SCHEMA, admit_hf_source, open_hf_moe_uniform

POSITIONS_PER_WINDOW = 1024
POSITION_COUNT = 64 * POSITIONS_PER_WINDOW
SUPPORT = 8192


class Balanced64Runtime(Protocol):
    """Architecture/runtime plugin selected behind the public orchestration API."""

    runtime_id: str

    def capture_teacher(self, *, source, suite_lock, corpus, output, windows): ...

    def score_pre(self, *, artifact, teacher_capture, suite_lock, corpus): ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"{label} must be a JSON object")
    return loaded


def _suite_lock(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    lock = _mapping(value, "BALANCED64 suite lock")
    declared = lock.get("suite_lock_sha256")
    unhashed = dict(lock)
    unhashed.pop("suite_lock_sha256", None)
    canonical = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if declared != actual:
        raise ValueError(f"BALANCED64 suite-lock SHA mismatch: expected={declared} actual={actual}")
    if (
        lock.get("schema") != "banana-smasher.balanced64-suite-lock.v1"
        or lock.get("window_count") != 64
        or lock.get("positions_per_window") != POSITIONS_PER_WINDOW
        or lock.get("positions") != POSITION_COUNT
        or lock.get("support") != SUPPORT
    ):
        raise ValueError("BALANCED64 suite lock does not declare canonical 64x1024x8192 geometry")
    windows = lock.get("windows")
    if not isinstance(windows, list) or len(windows) != 64:
        raise ValueError("BALANCED64 suite lock must contain 64 ordered windows")
    ordinals = [row.get("ordinal") for row in windows if isinstance(row, Mapping)]
    ids = [row.get("window_id") for row in windows if isinstance(row, Mapping)]
    if ordinals != list(range(64)) or len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("BALANCED64 suite lock window ordinal/identity drift")
    return lock


def _runtime_id(runtime: Balanced64Runtime) -> str:
    value = getattr(runtime, "runtime_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("BALANCED64 runtime must declare a non-empty runtime_id")
    return value


def _resolve_runtime(
    runtime: Balanced64Runtime | None, *, subject: Mapping[str, Any], role: str
) -> Balanced64Runtime:
    if runtime is not None:
        return runtime
    discovered: list[Balanced64Runtime] = []
    for entry_point in importlib_metadata.entry_points().select(
        group="banana_smasher.balanced64_runtimes"
    ):
        candidate = entry_point.load()()
        supports = getattr(candidate, "supports", None)
        if callable(supports) and supports(subject=subject, role=role) is True:
            discovered.append(candidate)
    if len(discovered) != 1:
        raise ValueError(
            "BALANCED64 runtime selection must resolve exactly once: "
            f"role={role} matched={[getattr(item, 'runtime_id', None) for item in discovered]}"
        )
    return discovered[0]


def _zero_mechanisms(value: Any, *, timed: bool) -> dict[str, int]:
    required = ["fallback", "relay", "reconstruction", "streaming"]
    if timed:
        required = ["timed_payload_reads", "timed_model_reads", *required]
    if not isinstance(value, Mapping):
        raise ValueError("BALANCED64 runtime did not return mechanism counters")
    counters = {key: value.get(key) for key in required}
    if any(counter != 0 for counter in counters.values()):
        raise ValueError(f"BALANCED64 runtime mechanism gate is nonzero: {counters}")
    return {key: int(counter) for key, counter in counters.items()}


def capture_balanced64_teacher(
    model: str | Path,
    *,
    revision: str,
    suite_lock: Mapping[str, Any] | str | Path,
    corpus: str | Path,
    output: str | Path,
    receipt_path: str | Path,
    windows: list[int] | tuple[int, ...] | None = None,
    runtime: Balanced64Runtime | None = None,
) -> dict[str, Any]:
    """Capture a model's own teacher rows under one immutable BALANCED64 lock."""

    lock = _suite_lock(suite_lock)
    selected_ids = (
        [row["window_id"] for row in lock["windows"]]
        if windows is None
        else list(windows)
    )
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("teacher capture windows must be a non-empty unique sequence")
    by_id = {row["window_id"]: row for row in lock["windows"]}
    if set(selected_ids) - set(by_id):
        raise ValueError("teacher capture windows must belong to the frozen suite lock")
    selected_windows = [by_id[window_id] for window_id in selected_ids]
    complete = len(selected_windows) == 64 and selected_ids == [
        row["window_id"] for row in lock["windows"]
    ]
    corpus_path = Path(corpus).expanduser().resolve()
    if _sha256(corpus_path) != lock.get("source_windows_sha256"):
        raise ValueError("BALANCED64 corpus SHA does not match the suite lock")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"BALANCED64 teacher output already exists: {destination}")
    receipt_destination = Path(receipt_path).expanduser().resolve()
    source = admit_hf_source(
        model,
        revision=revision,
        receipt_path=receipt_destination.with_name("TEACHER_SOURCE_ADMISSION.json"),
    )
    if source["model_index_sha256"] != lock.get("teacher_source_model_index_sha256"):
        raise ValueError("teacher model index does not match the model-specific suite lock")
    runtime = _resolve_runtime(runtime, subject=source, role="teacher")
    runtime_result = runtime.capture_teacher(
        source=source,
        suite_lock=lock,
        corpus=corpus_path,
        output=destination,
        windows=selected_windows,
    )
    if not isinstance(runtime_result, Mapping):
        raise TypeError("BALANCED64 teacher runtime must return a mapping")
    rows = runtime_result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(selected_windows):
        raise ValueError(
            "BALANCED64 teacher capture returned the wrong selected row count"
        )
    expected_windows = selected_windows
    verified_rows: list[dict[str, Any]] = []
    for expected, row in zip(expected_windows, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("BALANCED64 teacher row must be a mapping")
        if any(row.get(key) != expected[key] for key in ("ordinal", "window_id", "source_class")):
            raise ValueError("BALANCED64 teacher row order/class identity drift")
        if row.get("positions") != POSITIONS_PER_WINDOW or row.get("support") != SUPPORT:
            raise ValueError("BALANCED64 teacher row geometry drift")
        relative = row.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("BALANCED64 teacher row path must be relative")
        path = (destination / relative).resolve()
        try:
            path.relative_to(destination)
        except ValueError as exc:
            raise ValueError("BALANCED64 teacher row escapes the output root") from exc
        if not path.is_file() or row.get("bytes") != path.stat().st_size or row.get("sha256") != _sha256(path):
            raise ValueError(f"BALANCED64 teacher row byte/hash mismatch: {relative}")
        verified_rows.append(dict(row))
    counters = _zero_mechanisms(runtime_result.get("runtime_counters"), timed=False)
    receipt = {
        "schema": "banana-smasher-balanced64-teacher-capture-v1",
        "status": "PASS" if complete else "PASS_DIAGNOSTIC",
        "artifact_admissible": complete,
        "api": {"method": "capture_balanced64_teacher", "version": 1},
        "runtime": {"id": _runtime_id(runtime)},
        "source": source,
        "suite_lock_sha256": lock["suite_lock_sha256"],
        "window_population_sha256": lock["window_population_sha256"],
        "corpus_sha256": _sha256(corpus_path),
        "teacher_bank": lock["teacher_bank"],
        "row_count": len(verified_rows),
        "positions": len(verified_rows) * POSITIONS_PER_WINDOW,
        "support": SUPPORT,
        "rows": verified_rows,
        "runtime_counters": counters,
    }
    _atomic_json(receipt_destination, receipt)
    return receipt


def score_balanced64_pre(
    artifact: Mapping[str, Any] | str | Path,
    *,
    teacher_capture: Mapping[str, Any] | str | Path,
    suite_lock: Mapping[str, Any] | str | Path,
    corpus: str | Path,
    receipt_path: str | Path,
    runtime: Balanced64Runtime | None = None,
) -> dict[str, Any]:
    """Produce the canonical resident BALANCED64 PRE terminal for one artifact."""

    lock = _suite_lock(suite_lock)
    corpus_path = Path(corpus).expanduser().resolve()
    if _sha256(corpus_path) != lock.get("source_windows_sha256"):
        raise ValueError("BALANCED64 corpus SHA does not match the suite lock")
    admitted = (
        dict(artifact)
        if isinstance(artifact, Mapping)
        else open_hf_moe_uniform(Path(artifact).expanduser().resolve())
    )
    if (
        admitted.get("schema") != HF_UNIFORM_ARTIFACT_SCHEMA
        or admitted.get("status") != "PASS"
        or admitted.get("reload_verified") is not True
    ):
        raise ValueError("score_pre requires a reloaded admitted HF MoE artifact")
    source = admitted.get("source")
    if not isinstance(source, Mapping) or source.get("model_index_sha256") != lock.get(
        "teacher_source_model_index_sha256"
    ):
        raise ValueError("candidate and teacher suite-lock model identities differ")
    teacher = _mapping(teacher_capture, "BALANCED64 teacher capture")
    if (
        teacher.get("schema") != "banana-smasher-balanced64-teacher-capture-v1"
        or teacher.get("status") != "PASS"
        or teacher.get("suite_lock_sha256") != lock["suite_lock_sha256"]
        or teacher.get("corpus_sha256") != _sha256(corpus_path)
        or teacher.get("row_count") != 64
    ):
        raise ValueError("score_pre teacher capture is not complete for this suite/corpus")
    runtime = _resolve_runtime(runtime, subject=admitted, role="candidate_pre")
    runtime_result = runtime.score_pre(
        artifact=admitted,
        teacher_capture=teacher,
        suite_lock=lock,
        corpus=corpus_path,
    )
    if not isinstance(runtime_result, Mapping):
        raise TypeError("BALANCED64 PRE runtime must return a mapping")
    if runtime_result.get("resident_ready") is not True:
        raise ValueError("BALANCED64 PRE runtime was not resident-ready before timing")
    rows = runtime_result.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("BALANCED64 PRE must return exactly 64 rows")
    all_values: list[float] = []
    top1_matches = 0
    sealed_rows: list[dict[str, Any]] = []
    for expected, row in zip(lock["windows"], rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("BALANCED64 PRE row must be a mapping")
        if any(row.get(key) != expected[key] for key in ("ordinal", "window_id", "source_class")):
            raise ValueError("BALANCED64 PRE row order/class identity drift")
        values = row.get("kld_values")
        if not isinstance(values, list) or len(values) != POSITIONS_PER_WINDOW:
            raise ValueError("BALANCED64 PRE row must contain 1024 binary64 KLD values")
        parsed: list[float] = []
        for text in values:
            if not isinstance(text, str):
                raise ValueError("BALANCED64 KLD values must use shortest binary64 decimal strings")
            value = float(text)
            if not math.isfinite(value) or value < 0 or repr(value) != text:
                raise ValueError(f"noncanonical/negative/non-finite BALANCED64 KLD value: {text}")
            parsed.append(value)
        matches = row.get("top1_matches")
        if isinstance(matches, bool) or not isinstance(matches, int) or not 0 <= matches <= POSITIONS_PER_WINDOW:
            raise ValueError("BALANCED64 PRE row Top-1 count is invalid")
        all_values.extend(parsed)
        top1_matches += matches
        sealed_rows.append(dict(row))
    if len(all_values) != POSITION_COUNT:
        raise ValueError("BALANCED64 PRE position closure failed")
    counters = _zero_mechanisms(runtime_result.get("runtime_counters"), timed=True)
    elapsed = runtime_result.get("timed_wall_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
        raise ValueError("BALANCED64 PRE runtime must report non-negative timed wall seconds")
    receipt = {
        "schema": "banana-smasher-balanced64-pre-v1",
        "status": "PASS",
        "api": {"method": "score_balanced64_pre", "version": 1},
        "runtime": {"id": _runtime_id(runtime)},
        "suite_lock_sha256": lock["suite_lock_sha256"],
        "window_population_sha256": lock["window_population_sha256"],
        "corpus_sha256": _sha256(corpus_path),
        "teacher_bank": lock["teacher_bank"],
        "teacher_model_index_sha256": lock["teacher_source_model_index_sha256"],
        "candidate_model_index_sha256": source["model_index_sha256"],
        "artifact_accounting": admitted.get("accounting"),
        "rows_sealed": 64,
        "positions": POSITION_COUNT,
        "support": SUPPORT,
        "direction": "KL(teacher||candidate)",
        "reduction": "binary64/math.fsum ordered window then position",
        "mean_kld": fsum(all_values) / POSITION_COUNT,
        "top1_matches": top1_matches,
        "top1_denominator": POSITION_COUNT,
        "resident_ready": True,
        "timed_wall_seconds": float(elapsed),
        "runtime_counters": counters,
        "rows": sealed_rows,
    }
    _atomic_json(Path(receipt_path).expanduser().resolve(), receipt)
    return receipt
