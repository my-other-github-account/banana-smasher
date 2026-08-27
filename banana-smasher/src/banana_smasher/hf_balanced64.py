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


class Balanced64Tokenizer(Protocol):
    """Minimal tokenizer seam used by the public token-ledger builder."""

    tokenizer_id: str
    tokenizer_sha256: str

    def encode(self, text: str) -> Any: ...

    def decode(self, token_ids: list[int]) -> str: ...


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class _TokenizerJsonAdapter:
    def __init__(self, model: Path) -> None:
        tokenizer_path = model / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise ValueError(f"model has no tokenizer.json: {tokenizer_path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "normal banana-smasher installation requires tokenizers for token-ledger construction"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer_sha256 = _sha256(tokenizer_path)
        self.tokenizer_id = f"tokenizer-json-sha256:{self.tokenizer_sha256}"

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)


def _historical_rows(value: Any) -> list[Mapping[str, Any]]:
    rows = value if isinstance(value, list) else value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("historical BALANCED64 token ledger must contain a non-empty row list")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("historical BALANCED64 token-ledger rows must be objects")
    return rows


def _historical_window_id(row: Mapping[str, Any]) -> int:
    fields = [field for field in ("window_id", "id_gold") if field in row]
    if not fields:
        raise ValueError("missing historical window identity (expected window_id or id_gold)")
    if len(fields) != 1:
        raise ValueError(f"ambiguous historical window identity: fields={fields}")
    value = row[fields[0]]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid historical window identity: field={fields[0]} value={value!r}")
    return value


def _historical_item_id(row: Mapping[str, Any], *, window_id: int) -> str:
    for field in ("item_id", "id_ds4", "name"):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"invalid historical item identity: window_id={window_id} field={field}")
        rendered = str(value)
        if rendered:
            return rendered
    raise ValueError(f"missing historical item identity: window_id={window_id}")


def recover_balanced64_source_text(
    historical_token_ledger: str | Path,
    *,
    suite_lock: Mapping[str, Any] | str | Path,
    output: str | Path,
    receipt_path: str | Path,
    source_tokenizer_model: str | Path | None = None,
    tokenizer: Balanced64Tokenizer | None = None,
) -> dict[str, Any]:
    """Recover source text only when its authenticated tokenizer round-trips exactly."""

    lock = _suite_lock(suite_lock)
    historical_path = Path(historical_token_ledger).expanduser().resolve()
    historical_sha256 = _sha256(historical_path)
    if historical_sha256 != lock.get("source_windows_sha256"):
        raise ValueError(
            "historical BALANCED64 token-ledger SHA does not match the suite lock: "
            f"expected={lock.get('source_windows_sha256')} actual={historical_sha256}"
        )
    try:
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"historical BALANCED64 token ledger is not readable JSON: {historical_path}"
        ) from exc
    rows = _historical_rows(historical)
    destination = Path(output).expanduser().resolve()
    receipt_destination = Path(receipt_path).expanduser().resolve()
    for label, path in (("source-text manifest", destination), ("recovery receipt", receipt_destination)):
        if path.exists():
            raise FileExistsError(f"BALANCED64 {label} already exists: {path}")

    if tokenizer is not None and source_tokenizer_model is not None:
        raise ValueError("supply either source_tokenizer_model or tokenizer, not both")
    if tokenizer is None:
        if source_tokenizer_model is None:
            raise ValueError("source_tokenizer_model is required when tokenizer is not supplied")
        tokenizer = _TokenizerJsonAdapter(Path(source_tokenizer_model).expanduser().resolve())
    tokenizer_id = getattr(tokenizer, "tokenizer_id", None)
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("BALANCED64 source tokenizer must declare a non-empty tokenizer_id")
    tokenizer_sha256 = getattr(tokenizer, "tokenizer_sha256", None)
    if (
        not isinstance(tokenizer_sha256, str)
        or len(tokenizer_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_sha256)
    ):
        raise ValueError("BALANCED64 source tokenizer must declare its lowercase SHA-256")
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise ValueError("BALANCED64 source tokenizer must provide decode(token_ids)")

    by_window: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        window_id = _historical_window_id(row)
        if window_id in by_window:
            raise ValueError(f"duplicate historical BALANCED64 window_id: {window_id}")
        by_window[window_id] = row

    items: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for window in lock["windows"]:
        window_id = window["window_id"]
        row = by_window.get(window_id)
        if row is None:
            raise ValueError(f"historical token ledger missing frozen window_id={window_id}")
        raw_token_ids = row.get("token_ids")
        if not isinstance(raw_token_ids, list) or not raw_token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in raw_token_ids
        ):
            raise ValueError(f"invalid historical token_ids: window_id={window_id}")
        real_len = row.get("real_len", len(raw_token_ids))
        if (
            isinstance(real_len, bool)
            or not isinstance(real_len, int)
            or real_len < POSITIONS_PER_WINDOW + 1
            or real_len > len(raw_token_ids)
        ):
            raise ValueError(f"invalid historical real_len: window_id={window_id}")
        token_ids = raw_token_ids[:real_len]
        text = decoder(token_ids)
        if not isinstance(text, str):
            raise ValueError(f"source tokenizer decode did not return text: window_id={window_id}")
        encoded = tokenizer.encode(text)
        recovered_ids = list(encoded.ids) if hasattr(encoded, "ids") else list(encoded)
        if recovered_ids != token_ids:
            raise ValueError(f"tokenizer round-trip mismatch: window_id={window_id}")
        item = {
            "item_id": _historical_item_id(row, window_id=window_id),
            "window_id": window_id,
            "source_class": window["source_class"],
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        items.append(item)
        descriptors.append(
            {
                "item_id": item["item_id"],
                "window_id": window_id,
                "source_class": item["source_class"],
                "text_sha256": item["text_sha256"],
            }
        )

    manifest = {
        "schema": "banana-smasher.balanced64-source-text.v1",
        "source_provenance_sha256": lock["source_provenance_sha256"],
        "item_roster_sha256": _canonical_sha256(descriptors),
        "historical_token_ledger": {
            "sha256": historical_sha256,
            "source_tokenizer_id": tokenizer_id,
            "source_tokenizer_sha256": tokenizer_sha256,
        },
        "items": items,
    }
    _atomic_json(destination, manifest)
    receipt = {
        "schema": "banana-smasher-balanced64-source-text-recovery-receipt-v1",
        "status": "PASS",
        "api": {"method": "recover_balanced64_source_text", "version": 1},
        "suite_lock_sha256": lock["suite_lock_sha256"],
        "source_provenance_sha256": lock["source_provenance_sha256"],
        "historical_token_ledger_path": str(historical_path),
        "historical_token_ledger_sha256": historical_sha256,
        "source_tokenizer": {"id": tokenizer_id, "sha256": tokenizer_sha256},
        "row_count": len(items),
        "roundtrip_verified_rows": len(items),
        "item_roster_sha256": manifest["item_roster_sha256"],
        "manifest_path": str(destination),
        "manifest_bytes": destination.stat().st_size,
        "manifest_sha256": _sha256(destination),
    }
    _atomic_json(receipt_destination, receipt)
    return receipt


def build_balanced64_token_ledger(
    model: str | Path,
    *,
    revision: str,
    suite_lock: Mapping[str, Any] | str | Path,
    source_manifest: Mapping[str, Any] | str | Path,
    output: str | Path,
    bound_suite_lock: str | Path,
    receipt_path: str | Path,
    tokenizer: Balanced64Tokenizer | None = None,
) -> dict[str, Any]:
    """Tokenize authenticated source text and bind its model-specific suite lock."""

    lock = _suite_lock(suite_lock)
    destination = Path(output).expanduser().resolve()
    bound_lock_destination = Path(bound_suite_lock).expanduser().resolve()
    for label, path in (
        ("token ledger", destination),
        ("bound suite lock", bound_lock_destination),
    ):
        if path.exists():
            raise FileExistsError(f"BALANCED64 {label} already exists: {path}")
    manifest = _mapping(source_manifest, "BALANCED64 raw-source manifest")
    if manifest.get("schema") != "banana-smasher.balanced64-source-text.v1":
        raise ValueError("BALANCED64 source manifest is not authenticated raw source text")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("BALANCED64 source manifest must contain an item list")
    by_window: dict[int, Mapping[str, Any]] = {}
    descriptors: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("BALANCED64 raw-source item must be an object")
        if "token_ids" in item:
            raise ValueError("BALANCED64 raw-source items must not supply historical token_ids")
        window_id = item.get("window_id")
        item_id = item.get("item_id")
        source_class = item.get("source_class")
        text = item.get("text")
        text_sha = item.get("text_sha256")
        if (
            isinstance(window_id, bool)
            or not isinstance(window_id, int)
            or not isinstance(item_id, str)
            or not item_id
            or not isinstance(source_class, str)
            or not source_class
            or not isinstance(text, str)
            or not isinstance(text_sha, str)
        ):
            raise ValueError("BALANCED64 raw-source item identity/text fields are invalid")
        actual_text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_sha != actual_text_sha:
            raise ValueError(f"raw source text hash mismatch: window_id={window_id}")
        if window_id in by_window:
            raise ValueError(f"duplicate BALANCED64 raw-source window_id: {window_id}")
        by_window[window_id] = item
        descriptors.append(
            {
                "item_id": item_id,
                "window_id": window_id,
                "source_class": source_class,
                "text_sha256": text_sha,
            }
        )
    roster_sha256 = _canonical_sha256(descriptors)
    provenance = manifest.get("source_provenance_sha256")
    if (
        manifest.get("item_roster_sha256") != roster_sha256
        or not isinstance(provenance, str)
        or lock.get("source_provenance_sha256") != provenance
    ):
        raise ValueError("BALANCED64 raw-source provenance does not match the suite lock")

    receipt_destination = Path(receipt_path).expanduser().resolve()
    source = admit_hf_source(
        model,
        revision=revision,
        receipt_path=receipt_destination.with_name("TOKEN_LEDGER_SOURCE_ADMISSION.json"),
    )
    if source["model_index_sha256"] != lock.get("teacher_source_model_index_sha256"):
        raise ValueError("token-ledger model index does not match the model-specific suite lock")
    selected_tokenizer = tokenizer or _TokenizerJsonAdapter(Path(model).expanduser().resolve())
    tokenizer_id = getattr(selected_tokenizer, "tokenizer_id", None)
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("BALANCED64 tokenizer must declare a non-empty tokenizer_id")

    rows: list[dict[str, Any]] = []
    for window in lock["windows"]:
        item = by_window.get(window["window_id"])
        if item is None:
            raise ValueError(f"raw source missing frozen window_id={window['window_id']}")
        if item["source_class"] != window["source_class"]:
            raise ValueError(f"raw-source class drift: window_id={window['window_id']}")
        encoded = selected_tokenizer.encode(item["text"])
        token_ids = list(encoded.ids) if hasattr(encoded, "ids") else list(encoded)
        if len(token_ids) < POSITIONS_PER_WINDOW + 1 or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise ValueError(
                f"model tokenization cannot supply 1024 next-token positions: window_id={window['window_id']}"
            )
        rows.append(
            {
                **window,
                "item_id": item["item_id"],
                "text_sha256": item["text_sha256"],
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
        )

    ledger = {
        "schema": "banana-smasher.balanced64-token-ledger.v1",
        "window_population_sha256": lock["window_population_sha256"],
        "source_provenance_sha256": provenance,
        "item_roster_sha256": roster_sha256,
        "model_index_sha256": source["model_index_sha256"],
        "revision": revision,
        "tokenizer": {"id": tokenizer_id},
        "positions_per_window": POSITIONS_PER_WINDOW,
        "rows": rows,
    }
    _atomic_json(destination, ledger)
    ledger_sha256 = _sha256(destination)
    bound_lock = dict(lock)
    bound_lock.pop("suite_lock_sha256", None)
    bound_lock["historical_source_windows_sha256"] = lock["source_windows_sha256"]
    bound_lock["source_windows_sha256"] = ledger_sha256
    bound_lock["token_ledger"] = {
        "schema": ledger["schema"],
        "sha256": ledger_sha256,
        "model_index_sha256": source["model_index_sha256"],
        "tokenizer_id": tokenizer_id,
        "row_count": len(rows),
    }
    bound_lock["suite_lock_sha256"] = _canonical_sha256(bound_lock)
    _atomic_json(bound_lock_destination, bound_lock)
    receipt = {
        "schema": "banana-smasher-balanced64-token-ledger-receipt-v1",
        "status": "PASS",
        "api": {"method": "build_balanced64_token_ledger", "version": 1},
        "input_suite_lock_sha256": lock["suite_lock_sha256"],
        "bound_suite_lock_path": str(bound_lock_destination),
        "bound_suite_lock_sha256": bound_lock["suite_lock_sha256"],
        "window_population_sha256": lock["window_population_sha256"],
        "source_provenance_sha256": provenance,
        "item_roster_sha256": roster_sha256,
        "model_index_sha256": source["model_index_sha256"],
        "revision": revision,
        "tokenizer": {"id": tokenizer_id},
        "row_count": len(rows),
        "positions": len(rows) * POSITIONS_PER_WINDOW,
        "ledger_path": str(destination),
        "ledger_bytes": destination.stat().st_size,
        "ledger_sha256": ledger_sha256,
    }
    _atomic_json(receipt_destination, receipt)
    return receipt


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
