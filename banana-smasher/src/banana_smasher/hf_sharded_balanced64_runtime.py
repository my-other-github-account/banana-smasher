"""Auto-discovered sharded Hugging Face BALANCED64 runtime.

The runtime selects inputs from admitted config/index/artifact semantics.  Execution
is delegated to a package-level layer-streamed executor factory so architecture
adapters do not leak into the public orchestration API.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np

from .hf_sharded_balanced64_executor import _hybrid_semantics


_SOURCE_SCHEMA = "banana-smasher-hf-source-admission-v1"
_ARTIFACT_SCHEMA = "banana-smasher-hf-moe-uniform-artifact-v1"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT = re.compile(r"(?:^|\.)experts(?:\.\d+)?\..+")
_INPUT_POLICY_SCHEMA = "model-specific-token-ledger-v1:no-retokenization"


class ShardedHFBalanced64Runtime:
    """Model-neutral BALANCED64 runtime for immutable sharded HF inputs."""

    runtime_id = "hf-sharded-balanced64-v1"

    #: Hardware/capability contract of the package-owned layer-streamed executor.  It
    #: materializes each admitted layer into CUDA device memory and runs a real forward
    #: pass, so it is not executable on a CPU-only or Apple-MPS host.
    PACKAGE_EXECUTOR_HARDWARE_CONTRACT = {
        "schema": "banana-smasher.balanced64-hardware-contract.v1",
        "required_accelerator": "cuda",
        "minimum_ranks": 1,
        "reason": (
            "the package-owned layer-streamed executor materializes each admitted layer "
            "into CUDA device memory and runs a real forward pass over the BALANCED64 "
            "windows"
        ),
        "not_admissible": [
            "cpu-only hosts",
            "Apple MPS hosts — torch.backends.mps is not a CUDA device",
        ],
        "check": "torch.cuda.is_available() and torch.cuda.device_count() >= 1",
    }

    @property
    def hardware_contract(self) -> dict[str, Any] | None:
        """Contract of the executor this instance will actually use.

        The requirement is a property of the executor, not of the orchestration seam:
        an instance constructed with a caller-supplied ``executor_factory`` runs that
        caller's executor and is therefore not gated by the package executor's CUDA
        contract.  The public API enforces whatever this returns.
        """

        if self._executor_factory is not None:
            return None
        return dict(self.PACKAGE_EXECUTOR_HARDWARE_CONTRACT)

    def __init__(self, executor_factory: Callable[..., Any] | None = None) -> None:
        self._executor_factory = executor_factory

    @staticmethod
    def _source_semantics(source: object) -> tuple[dict[str, Any], list[str]] | None:
        if not isinstance(source, Mapping) or source.get("schema") != _SOURCE_SCHEMA:
            return None
        if source.get("status") != "PASS":
            return None
        root_value = source.get("model_root")
        if not isinstance(root_value, str):
            return None
        root = Path(root_value).expanduser().resolve()
        try:
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            index = json.loads(
                (root / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(config, dict) or not isinstance(weight_map, dict) or not weight_map:
            return None
        architectures = config.get("architectures")
        text_config = config.get("text_config")
        semantic = text_config if isinstance(text_config, Mapping) else config
        layers = semantic.get("num_hidden_layers")
        vocab = semantic.get("vocab_size")
        routed_experts = semantic.get("n_routed_experts")
        nextn_layers = semantic.get("num_nextn_predict_layers", 0)
        names = sorted(weight_map)
        observed = {
            int(match.group(1))
            for name in names
            if "vision" not in name.lower() and (match := _LAYER.search(name)) is not None
        }
        if (
            not isinstance(architectures, list)
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", architectures[0]) is None
            or not _hybrid_semantics(config)
            or isinstance(layers, bool)
            or not isinstance(layers, int)
            or layers < 1
            or isinstance(vocab, bool)
            or not isinstance(vocab, int)
            or vocab < 8192
            or isinstance(routed_experts, bool)
            or not isinstance(routed_experts, int)
            or routed_experts < 1
            or isinstance(nextn_layers, bool)
            or not isinstance(nextn_layers, int)
            or nextn_layers < 0
            or observed != set(range(layers + nextn_layers))
            or not any("embed" in name for name in names)
            or not any("head" in name for name in names)
            or not any(_EXPERT.search(name) and "vision" not in name.lower() for name in names)
            or source.get("config_sha256") != ShardedHFBalanced64Runtime._sha256(root / "config.json")
            or source.get("model_index_sha256")
            != ShardedHFBalanced64Runtime._sha256(root / "model.safetensors.index.json")
        ):
            return None
        return config, names

    def supports(self, *, subject: Mapping[str, Any], role: str) -> bool:
        if role == "teacher":
            return self._source_semantics(subject) is not None
        if role != "candidate_pre" or subject.get("schema") != _ARTIFACT_SCHEMA:
            return False
        if subject.get("status") != "PASS" or subject.get("reload_verified") is not True:
            return False
        if subject.get("intent") != {
            "tier": "q2",
            "scope": "routed_only",
            "native_rest": True,
        }:
            return False
        artifact_root = subject.get("artifact_root")
        if not isinstance(artifact_root, str) or not artifact_root:
            return False
        if self._source_semantics(subject.get("source")) is None:
            return False
        routed = subject.get("routed_tensors")
        native = subject.get("native_tensors")
        if not isinstance(routed, list) or not routed or not isinstance(native, list) or not native:
            return False
        routed_names = [row.get("name") for row in routed if isinstance(row, Mapping)]
        native_names = [row.get("name") for row in native if isinstance(row, Mapping)]
        return (
            len(routed_names) == len(routed)
            and len(native_names) == len(native)
            and all(
                isinstance(name, str)
                and _EXPERT.search(name)
                and "vision" not in name.lower()
                for name in routed_names
            )
            and not set(routed_names).intersection(native_names)
        )

    def _verify_artifact_members(self, artifact: Mapping[str, Any]) -> None:
        root = Path(artifact["artifact_root"]).expanduser().resolve()
        geometry = {
            "L": 16,
            "K": 2,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
        }
        source = artifact.get("source")
        source_root = (
            Path(source["model_root"]).expanduser().resolve()
            if isinstance(source, Mapping) and isinstance(source.get("model_root"), str)
            else None
        )
        try:
            index = json.loads(
                (source_root / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            expected_names = set(index["weight_map"])
        except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("candidate artifact source inventory is unreadable") from exc
        routed_names = [row.get("name") for row in artifact["routed_tensors"]]
        native_names = [row.get("name") for row in artifact["native_tensors"]]
        accounting = artifact.get("accounting")
        if (
            len(set(routed_names)) != len(routed_names)
            or len(set(native_names)) != len(native_names)
            or set(routed_names).intersection(native_names)
            or set(routed_names).union(native_names) != expected_names
            or artifact.get("coverage") != {"duplicates": [], "gaps": []}
            or not isinstance(accounting, Mapping)
            or accounting.get("routed_tensor_count") != len(routed_names)
            or accounting.get("planned_routed_tensor_count") != len(routed_names)
            or accounting.get("native_tensor_count") != len(native_names)
            or accounting.get("planned_native_tensor_count") != len(native_names)
        ):
            raise ValueError("candidate artifact routed/native inventory is not exact")

        def bound_path(value: object) -> Path:
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise ValueError("candidate artifact member path is not relative")
            path = (root / value).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("candidate artifact member escapes artifact_root") from exc
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"candidate artifact member is missing/non-regular: {value}")
            return path

        for row in artifact["routed_tensors"]:
            wire = row.get("wire") if isinstance(row, Mapping) else None
            if not isinstance(wire, Mapping) or wire.get("geometry") != geometry:
                raise ValueError("candidate artifact routed member has non-Q2 wire geometry")
            for key in ("trellis", "scales"):
                member = wire.get(key)
                if not isinstance(member, Mapping):
                    raise ValueError(f"candidate artifact member binding is missing: {key}")
                path = bound_path(member.get("path"))
                if (
                    path.stat().st_size != member.get("bytes")
                    or self._sha256(path) != member.get("sha256")
                ):
                    raise ValueError(f"candidate artifact member byte/hash mismatch: {key}")
        for row in artifact["native_tensors"]:
            if (
                not isinstance(row, Mapping)
                or row.get("representation") != "exact-source-data-bytes"
                or row.get("source_sha256") != row.get("artifact_sha256")
            ):
                raise ValueError("candidate native artifact member is not exact source bytes")
            path = bound_path(row.get("path"))
            if self._sha256(path) != row.get("artifact_sha256"):
                raise ValueError("candidate artifact member byte/hash mismatch: native")

    @staticmethod
    def _read_corpus(
        corpus: str | Path,
        suite_lock: Mapping[str, Any],
        *,
        model_index_sha256: str,
    ) -> tuple[list[dict[str, Any]], str]:
        windows = suite_lock.get("windows")
        token_binding = suite_lock.get("token_ledger")
        if not isinstance(windows, list) or len(windows) != 64:
            raise ValueError("BALANCED64 suite must contain 64 windows")
        if not isinstance(token_binding, Mapping):
            raise ValueError("BALANCED64 suite is not bound to a model token ledger")
        path = Path(corpus).expanduser().resolve()
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"BALANCED64 corpus contains an invalid model token ledger: {path}") from exc
        ledger_sha256 = ShardedHFBalanced64Runtime._sha256(path)
        tokenizer = ledger.get("tokenizer") if isinstance(ledger, Mapping) else None
        tokenizer_id = tokenizer.get("id") if isinstance(tokenizer, Mapping) else None
        rows = ledger.get("rows") if isinstance(ledger, Mapping) else None
        if (
            ledger.get("schema") != "banana-smasher.balanced64-token-ledger.v1"
            or ledger.get("positions_per_window") != 1024
            or ledger.get("model_index_sha256") != model_index_sha256
            or ledger.get("window_population_sha256")
            != suite_lock.get("window_population_sha256")
            or ledger.get("source_provenance_sha256")
            != suite_lock.get("source_provenance_sha256")
            or not isinstance(tokenizer_id, str)
            or not tokenizer_id
            or not isinstance(rows, list)
            or len(rows) != 64
            or token_binding.get("schema") != ledger.get("schema")
            or token_binding.get("sha256") != ledger_sha256
            or token_binding.get("model_index_sha256") != model_index_sha256
            or token_binding.get("tokenizer_id") != tokenizer_id
            or token_binding.get("row_count") != 64
            or suite_lock.get("source_windows_sha256") != ledger_sha256
        ):
            raise ValueError("BALANCED64 corpus token ledger does not match its bound suite/model")
        selected: list[dict[str, Any]] = []
        for expected, row in zip(windows, rows, strict=True):
            if not isinstance(expected, Mapping) or not isinstance(row, Mapping):
                raise ValueError("BALANCED64 corpus token-ledger row is invalid")
            if any(
                row.get(field) != expected.get(field)
                for field in ("ordinal", "window_id", "source_class")
            ):
                raise ValueError("BALANCED64 corpus token-ledger row order/identity is invalid")
            tokens = row.get("token_ids")
            if (
                not isinstance(tokens, list)
                or len(tokens) < 1025
                or row.get("token_count") != len(tokens)
                or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in tokens
                )
            ):
                raise ValueError("BALANCED64 corpus integer-token stimulus is invalid")
            selected.append(dict(row))
        input_policy = (
            f"{_INPUT_POLICY_SCHEMA}:ledger-sha256={ledger_sha256}:tokenizer-id={tokenizer_id}"
        )
        return selected, input_policy

    def _session(self, *, subject, role, suite_lock, corpus_rows):
        if self._executor_factory is not None:
            return self._executor_factory(
                subject=subject,
                role=role,
                suite_lock=suite_lock,
                corpus_rows=corpus_rows,
            )
        matches = []
        for point in importlib_metadata.entry_points().select(
            group="banana_smasher.sharded_hf_balanced64_executors"
        ):
            factory = point.load()
            supports = getattr(factory, "supports", None)
            if callable(supports) and supports(subject=subject, role=role) is True:
                matches.append(factory)
        if len(matches) != 1:
            raise RuntimeError(
                "sharded HF BALANCED64 executor selection must resolve exactly once: "
                f"role={role} matched={[getattr(item, '__name__', None) for item in matches]}"
            )
        return matches[0](
            subject=subject,
            role=role,
            suite_lock=suite_lock,
            corpus_rows=corpus_rows,
        )

    @staticmethod
    def _teacher_arrays(value: object) -> dict[str, np.ndarray]:
        if not isinstance(value, Mapping):
            raise ValueError("teacher executor result must be a mapping")
        arrays = {
            name: np.asarray(value.get(name))
            for name in (
                "support_token_ids",
                "support_logits",
                "position_map",
                "top1_token_ids",
            )
        }
        ids = arrays["support_token_ids"]
        logits = arrays["support_logits"]
        position_map = arrays["position_map"]
        top1 = arrays["top1_token_ids"]
        if (
            ids.ndim != 2
            or ids.shape[1] != 8192
            or ids.shape != logits.shape
            or ids.shape[0] < 1
            or position_map.shape != (1024,)
            or top1.shape != (1024,)
            or ids.dtype.kind not in "iu"
            or position_map.dtype.kind not in "iu"
            or top1.dtype.kind not in "iu"
            or np.any(position_map < 0)
            or np.any(position_map >= ids.shape[0])
            or not np.isfinite(logits).all()
        ):
            raise ValueError("teacher executor row violates 1024x8192 compact support geometry")
        for row in ids:
            if len(set(int(token) for token in row)) != 8192:
                raise ValueError("teacher support token ids must be unique and ordered per position")
        return arrays

    @staticmethod
    def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _counters(session: object, *, timed: bool) -> dict[str, int]:
        method = getattr(session, "counters", None)
        value = method() if callable(method) else None
        required = {
            "setup_model_reads",
            "setup_payload_reads",
            "working_set_loads",
            "fallback",
            "relay",
            "reconstruction",
            "streaming",
        }
        if timed:
            required.update(("timed_payload_reads", "timed_model_reads"))
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValueError("sharded HF executor did not report complete runtime counters")
        counters: dict[str, int] = {}
        for name in sorted(required):
            counter = value[name]
            if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
                raise ValueError(f"sharded HF executor counter is invalid: {name}")
            counters[name] = counter
        for name in ("fallback", "relay", "reconstruction", "streaming"):
            if counters[name] != 0:
                raise ValueError(f"sharded HF executor forbidden mechanism is nonzero: {name}")
        if timed and (counters["timed_payload_reads"] or counters["timed_model_reads"]):
            raise ValueError("sharded HF executor performed reads inside the timed region")
        return counters

    def capture_teacher(self, *, source, suite_lock, corpus, output, windows=None):
        if not self.supports(subject=source, role="teacher"):
            raise ValueError("sharded HF teacher source is not supported")
        locked_windows = suite_lock.get("windows") if isinstance(suite_lock, Mapping) else None
        if windows is None:
            windows = locked_windows
        if not isinstance(locked_windows, list) or not isinstance(windows, list) or not windows:
            raise ValueError("sharded HF teacher runtime requires non-empty locked windows")
        locked_by_id = {
            row.get("window_id"): row for row in locked_windows if isinstance(row, Mapping)
        }
        selected_ids = [
            row.get("window_id") if isinstance(row, Mapping) else None for row in windows
        ]
        if (
            len(locked_by_id) != 64
            or len(set(selected_ids)) != len(selected_ids)
            or any(locked_by_id.get(window_id) != row for window_id, row in zip(selected_ids, windows))
        ):
            raise ValueError("sharded HF teacher windows must be an ordered subset of the suite lock")
        all_corpus_rows, input_policy = self._read_corpus(
            corpus,
            suite_lock,
            model_index_sha256=source["model_index_sha256"],
        )
        corpus_by_id = {row["window_id"]: row for row in all_corpus_rows}
        corpus_rows = [corpus_by_id[window_id] for window_id in selected_ids]
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"BALANCED64 teacher output already exists: {destination}")
        destination.mkdir(parents=True)
        session = self._session(
            subject=source,
            role="teacher",
            suite_lock=suite_lock,
            corpus_rows=corpus_rows,
        )
        teacher_rows = getattr(session, "teacher_rows", None)
        teacher_window = getattr(session, "teacher_window", None)
        if callable(teacher_rows):
            raw_rows = teacher_rows()
            if not isinstance(raw_rows, list) or len(raw_rows) != len(windows):
                raise ValueError("sharded HF bulk teacher executor returned the wrong row count")
        elif callable(teacher_window):
            raw_rows = [
                teacher_window(window=expected, token_ids=corpus_row["token_ids"])
                for expected, corpus_row in zip(windows, corpus_rows, strict=True)
            ]
        else:
            raise ValueError("sharded HF teacher executor lacks bulk/window execution")
        rows = []
        for expected, raw in zip(windows, raw_rows, strict=True):
            arrays = self._teacher_arrays(raw)
            relative = f"rows/teacher-{expected['ordinal']:02d}.npz"
            path = destination / relative
            self._atomic_npz(path, arrays)
            rows.append(
                {
                    **expected,
                    "positions": 1024,
                    "support": 8192,
                    "row_schema": "banana-smasher-balanced64-teacher-row-v1",
                    "input_policy": input_policy,
                    "path": relative,
                    "output_root": str(destination),
                    "bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
            )
        return {"rows": rows, "runtime_counters": self._counters(session, timed=False)}

    @staticmethod
    def _candidate_arrays(value: object) -> dict[str, np.ndarray]:
        if not isinstance(value, Mapping):
            raise ValueError("candidate executor result must be a mapping")
        arrays = {
            name: np.asarray(value.get(name))
            for name in ("support_logits", "position_map", "top1_token_ids")
        }
        logits = arrays["support_logits"]
        position_map = arrays["position_map"]
        top1 = arrays["top1_token_ids"]
        if (
            logits.ndim != 2
            or logits.shape[0] < 1
            or logits.shape[1] != 8192
            or position_map.shape != (1024,)
            or top1.shape != (1024,)
            or position_map.dtype.kind not in "iu"
            or top1.dtype.kind not in "iu"
            or np.any(position_map < 0)
            or np.any(position_map >= logits.shape[0])
            or not np.isfinite(logits).all()
        ):
            raise ValueError("candidate executor row violates 1024x8192 support geometry")
        return arrays

    def _load_teacher_row(
        self,
        row: object,
        expected: Mapping[str, Any],
        *,
        input_policy: str,
    ) -> dict[str, np.ndarray]:
        if not isinstance(row, Mapping) or any(
            row.get(name) != expected.get(name)
            for name in ("ordinal", "window_id", "source_class")
        ):
            raise ValueError("teacher capture row order/identity is invalid")
        root_value = row.get("output_root")
        relative = row.get("path")
        if (
            row.get("row_schema") != "banana-smasher-balanced64-teacher-row-v1"
            or row.get("input_policy") != input_policy
            or row.get("positions") != 1024
            or row.get("support") != 8192
            or not isinstance(root_value, str)
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
        ):
            raise ValueError("teacher capture row binding/geometry is invalid")
        root = Path(root_value).expanduser().resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("teacher capture row path escapes output root") from exc
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or self._sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"teacher capture row byte/hash mismatch: {relative}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                return self._teacher_arrays({name: payload[name] for name in payload.files})
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"teacher capture row payload is corrupt: {relative}") from exc

    @staticmethod
    def _kld_values(
        teacher: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
    ) -> list[float]:
        teacher_logits = teacher["support_logits"].astype(np.float64, copy=False)
        candidate_logits = candidate["support_logits"].astype(np.float64, copy=False)
        teacher_map = teacher["position_map"].astype(np.int64, copy=False)
        candidate_map = candidate["position_map"].astype(np.int64, copy=False)
        values = np.empty(1024, dtype=np.float64)
        cache: dict[tuple[int, int], float] = {}
        for position, pair in enumerate(zip(teacher_map, candidate_map, strict=True)):
            key = int(pair[0]), int(pair[1])
            if key not in cache:
                teacher_row = teacher_logits[key[0]]
                candidate_row = candidate_logits[key[1]]
                teacher_normalizer = float(
                    np.max(teacher_row)
                    + np.log(np.exp(teacher_row - np.max(teacher_row)).sum())
                )
                candidate_normalizer = float(
                    np.max(candidate_row)
                    + np.log(np.exp(candidate_row - np.max(candidate_row)).sum())
                )
                teacher_log_probability = teacher_row - teacher_normalizer
                candidate_log_probability = candidate_row - candidate_normalizer
                probability = np.exp(teacher_log_probability)
                kld = float(
                    np.sum(
                        probability
                        * (teacher_log_probability - candidate_log_probability),
                        dtype=np.float64,
                    )
                )
                if kld < 0 and abs(kld) <= 1e-12:
                    kld = 0.0
                if not math.isfinite(kld) or kld < 0:
                    raise ValueError("candidate KLD is negative or non-finite")
                cache[key] = kld
            values[position] = cache[key]
        return values.tolist()

    def score_pre(self, *, artifact, teacher_capture, suite_lock, corpus):
        if not isinstance(artifact, Mapping) or not self.supports(
            subject=artifact, role="candidate_pre"
        ):
            raise ValueError("sharded HF candidate artifact is not supported")
        artifact_root = Path(artifact["artifact_root"]).expanduser().resolve()
        if not artifact_root.is_dir():
            raise ValueError("sharded HF candidate artifact_root is not a directory")
        self._verify_artifact_members(artifact)
        windows = suite_lock.get("windows") if isinstance(suite_lock, Mapping) else None
        corpus_rows, input_policy = self._read_corpus(
            corpus,
            suite_lock,
            model_index_sha256=artifact["source"]["model_index_sha256"],
        )
        teacher_rows = (
            teacher_capture.get("rows") if isinstance(teacher_capture, Mapping) else None
        )
        if not isinstance(teacher_rows, list) or len(teacher_rows) != 64:
            raise ValueError("teacher capture must contain exactly 64 rows")
        teachers = [
            self._load_teacher_row(row, expected, input_policy=input_policy)
            for row, expected in zip(teacher_rows, windows, strict=True)
        ]
        session = self._session(
            subject=artifact,
            role="candidate_pre",
            suite_lock=suite_lock,
            corpus_rows=corpus_rows,
        )
        candidate_rows = getattr(session, "candidate_rows", None)
        candidate_window = getattr(session, "candidate_window", None)
        if callable(candidate_rows):
            raw_candidates = candidate_rows(teachers)
            if not isinstance(raw_candidates, list) or len(raw_candidates) != 64:
                raise ValueError("sharded HF bulk candidate executor must return 64 rows")
        elif callable(candidate_window):
            raw_candidates = [
                candidate_window(
                    window=expected,
                    token_ids=corpus_row["token_ids"],
                    support_token_ids=teacher["support_token_ids"],
                    position_map=teacher["position_map"],
                )
                for expected, corpus_row, teacher in zip(
                    windows, corpus_rows, teachers, strict=True
                )
            ]
        else:
            raise ValueError("sharded HF candidate executor lacks bulk/window execution")
        candidates = [self._candidate_arrays(value) for value in raw_candidates]
        finish_setup = getattr(session, "finish_setup", None)
        if not callable(finish_setup):
            raise ValueError("sharded HF candidate executor lacks finish_setup boundary")
        finish_setup()
        resident_ready = getattr(session, "resident_ready", None)
        if not callable(resident_ready) or resident_ready() is not True:
            raise ValueError("sharded HF candidate executor is not resident-ready")
        started = time.perf_counter()
        rows = []
        for expected, teacher, candidate in zip(
            windows, teachers, candidates, strict=True
        ):
            values = self._kld_values(teacher, candidate)
            matches = int(
                np.count_nonzero(
                    teacher["top1_token_ids"] == candidate["top1_token_ids"]
                )
            )
            rows.append(
                {
                    **expected,
                    "positions": 1024,
                    "input_policy": input_policy,
                    "kld_values": [repr(value) for value in values],
                    "top1_matches": matches,
                }
            )
        elapsed = time.perf_counter() - started
        counters = self._counters(session, timed=True)
        return {
            "rows": rows,
            "resident_ready": True,
            "timed_wall_seconds": elapsed,
            "runtime_counters": counters,
        }
