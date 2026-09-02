"""Artifact-root-scoped canonical Balanced64 scoring.

The scorer has one implementation. A repair artifact supplies only relative
locations and checkpoint names; the checkpoint does not select a scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import fsum
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

ARTIFACT_SCHEMA = "repair-artifact-v1"
BALANCED64_SPEC = "balanced64-v1"
POSITIONS_PER_WINDOW = 1024
SUPPORT = 8192

Loader = Callable[[Path], Mapping[str, Any]]


class ArtifactError(ValueError):
    """A self-contained repair artifact or score input is invalid."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relative_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ArtifactError(f"{field} must be a non-empty relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"{field} escapes artifact root: {value}") from exc
    return path


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on runtime hosts
        raise ArtifactError("torch is required to score .pt candidate artifacts") from exc
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{path} must contain a mapping")
    return value


def _array(value: Any):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("numpy is required for Balanced64 scoring") from exc
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _required(mapping: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in mapping:
        raise ArtifactError(f"{path}: missing {key}")
    return mapping[key]


@dataclass(frozen=True)
class ScoreResult:
    checkpoint: str
    windows: tuple[int, ...]
    positions: int
    support: int
    kld: float
    top1: int
    top1_rate: float
    artifact_root: str
    spec: str
    candidate_dir: str
    execution_mode: str = "file_loader"
    resident_load_seconds: float | None = None
    timed_wall_seconds: float | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    runtime_counters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        timed_score_file_reads = self.runtime_counters.get(
            "timed_score_file_reads",
            self.runtime_counters.get("file_reads_during_timed_score"),
        )
        return {
            "status": "PASS",
            "checkpoint": self.checkpoint,
            "windows": list(self.windows),
            "positions": self.positions,
            "support": self.support,
            "kld_mean": self.kld,
            "top1": self.top1,
            "top1_rate": self.top1_rate,
            "artifact_root": self.artifact_root,
            "spec": self.spec,
            "candidate_dir": self.candidate_dir,
            "execution_mode": self.execution_mode,
            "resident_load_seconds": self.resident_load_seconds,
            "timed_wall_seconds": self.timed_wall_seconds,
            "scoring_wall_seconds": self.timed_wall_seconds,
            "timed_score_file_reads": timed_score_file_reads,
            "direction": "KL(teacher||candidate)",
            "reduction": "float64 support renormalization + math.fsum(window/position order)",
            "identity": dict(self.identity),
            "runtime_counters": dict(self.runtime_counters),
        }


@dataclass(frozen=True)
class ResidentBalanced64:
    """A fully loaded Balanced64 anchor with no score-time file I/O."""

    checkpoint: str
    windows: tuple[int, ...]
    rows: tuple[tuple[int, Any, Any, Any, Any], ...]
    artifact_root: str
    candidate_dir: str
    resident_load_seconds: float

    def score(self) -> ScoreResult:
        """Reduce the resident rows; the timed section performs no file reads."""
        started = time.perf_counter()
        all_terms: list[float] = []
        top1 = 0
        positions = 0
        import numpy as np
        for _window, ref_norm, cand_norm, ref_argmax, q_argmax in self.rows:
            terms = np.sum(
                np.exp(ref_norm) * (ref_norm - cand_norm),
                axis=1,
                dtype=np.float64,
            )
            all_terms.extend(float(value) for value in np.asarray(terms).reshape(-1))
            top1 += int((np.asarray(q_argmax) == np.asarray(ref_argmax)).sum())
            positions += int(np.asarray(terms).shape[0])
        elapsed = time.perf_counter() - started
        if positions != len(self.windows) * POSITIONS_PER_WINDOW:
            raise ArtifactError(f"resident score has {positions} positions, expected {len(self.windows) * POSITIONS_PER_WINDOW}")
        return ScoreResult(
            checkpoint=self.checkpoint,
            windows=self.windows,
            positions=positions,
            support=SUPPORT,
            kld=fsum(all_terms) / positions,
            top1=top1,
            top1_rate=top1 / positions,
            artifact_root=self.artifact_root,
            spec=BALANCED64_SPEC,
            candidate_dir=self.candidate_dir,
            execution_mode="resident_in_memory",
            resident_load_seconds=self.resident_load_seconds,
            timed_wall_seconds=elapsed,
        )


class RepairArtifact:
    """A self-contained repair artifact with one standardized score function."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = root.resolve()
        self.manifest = dict(manifest)

    @classmethod
    def open(cls, root: str | Path) -> "RepairArtifact":
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise ArtifactError(f"artifact root does not exist: {root_path}")
        manifest_path = root_path / "ARTIFACT.json"
        if not manifest_path.is_file():
            raise ArtifactError(f"missing artifact manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            raise ArtifactError(f"invalid artifact manifest: {manifest_path}: {exc}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != ARTIFACT_SCHEMA:
            raise ArtifactError(f"expected {ARTIFACT_SCHEMA} in {manifest_path}")
        score = manifest.get("score")
        if not isinstance(score, Mapping) or score.get("spec") != BALANCED64_SPEC:
            raise ArtifactError("artifact must declare the standardized Balanced64 score spec")
        if score.get("positions_per_window") != POSITIONS_PER_WINDOW or score.get("support") != SUPPORT:
            raise ArtifactError("artifact Balanced64 dimensions are not the fixed standard")
        windows = score.get("window_ids")
        if not isinstance(windows, list) or len(windows) != 64 or len(set(windows)) != 64:
            raise ArtifactError("artifact must declare 64 unique ordered Balanced64 window IDs")
        for field in ("teacher_dir", "candidate_dir_template"):
            _relative_path(root_path, score.get(field), f"score.{field}")
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, Mapping) or not checkpoints:
            raise ArtifactError("artifact must declare checkpoints")
        for key, value in checkpoints.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise ArtifactError("checkpoints must map names to metadata objects")
            _relative_path(root_path, value.get("path"), f"checkpoints.{key}.path")
        return cls(root_path, manifest)

    @property
    def windows(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.manifest["score"]["window_ids"])

    def checkpoint_key(self, checkpoint: int | str) -> str:
        raw = str(checkpoint)
        candidates = [raw]
        if raw.isdigit():
            candidates += [f"UPDATE_{int(raw):03d}", f"U{int(raw):03d}"]
        elif re.fullmatch(r"U\d+", raw, flags=re.IGNORECASE):
            candidates += [f"UPDATE_{int(raw[1:]):03d}"]
        elif raw.upper().startswith("UPDATE_"):
            candidates += [raw.split("_", 1)[1].lstrip("0") or "0"]
        checkpoints = self.manifest["checkpoints"]
        for candidate in candidates:
            if candidate in checkpoints:
                return candidate
        raise ArtifactError(f"checkpoint {checkpoint!r} is not declared in ARTIFACT.json")

    def checkpoint_path(self, checkpoint: int | str) -> Path:
        key = self.checkpoint_key(checkpoint)
        value = self.manifest["checkpoints"][key]
        path = _relative_path(self.root, value["path"], f"checkpoints.{key}.path")
        if not path.is_file():
            raise ArtifactError(f"declared checkpoint is missing: {path}")
        expected = value.get("sha256")
        if expected and _sha256(path) != expected:
            raise ArtifactError(f"checkpoint SHA mismatch: {key}")
        return path

    def score(
        self,
        checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        loader: Loader | None = None,
    ) -> ScoreResult:
        """Score one declared checkpoint; windows are the only runtime choice."""
        key = self.checkpoint_key(checkpoint)
        score_spec = self.manifest["score"]
        selected = tuple(self.windows if windows is None else (int(x) for x in windows))
        if not selected or len(set(selected)) != len(selected):
            raise ArtifactError("windows must be a non-empty unique sequence")
        unknown = sorted(set(selected) - set(self.windows))
        if unknown:
            raise ArtifactError(f"windows are not declared by this artifact: {unknown}")
        loader = loader or _load_torch
        teacher_dir = _relative_path(self.root, score_spec["teacher_dir"], "score.teacher_dir")
        candidate_dir = _relative_path(
            self.root,
            str(score_spec["candidate_dir_template"]).format(checkpoint=key),
            "score.candidate_dir_template",
        )
        if not candidate_dir.is_dir():
            raise ArtifactError(f"candidate artifacts missing for {key}: {candidate_dir}")
        all_terms: list[float] = []
        top1 = 0
        positions = 0
        for window in selected:
            teacher_path = teacher_dir / f"t8192_win{window}.pt"
            candidate_path = candidate_dir / f"q8192_win{window}.pt"
            if not teacher_path.is_file() or not candidate_path.is_file():
                raise ArtifactError(f"checkpoint {key} is missing window {window} candidate/teacher rows")
            teacher = loader(teacher_path)
            candidate = loader(candidate_path)
            idx = _array(_required(teacher, "idx", teacher_path)).astype("int64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            ref_lp = _array(_required(teacher, "logprob", teacher_path)).astype("float64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            q_lp = _array(_required(candidate, "q_lp_at_ref", candidate_path)).astype("float64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            q_argmax = _array(_required(candidate, "q_argmax", candidate_path)).astype("int64")[:POSITIONS_PER_WINDOW]
            if idx.ndim != 2 or idx.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} teacher idx has shape {idx.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if ref_lp.ndim != 2 or ref_lp.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} teacher logprob has shape {ref_lp.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if q_lp.ndim != 2 or q_lp.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} candidate q_lp_at_ref has shape {q_lp.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if q_argmax.ndim not in (1, 2) or q_argmax.shape[0] != POSITIONS_PER_WINDOW:
                raise ArtifactError(f"window {window} candidate q_argmax has shape {q_argmax.shape}; expected {POSITIONS_PER_WINDOW} rows")
            count = POSITIONS_PER_WINDOW
            ref_lp = ref_lp[:count]
            q_lp = q_lp[:count]
            idx = idx[:count]
            q_argmax = q_argmax[:count]
            import numpy as np
            ref_max = np.max(ref_lp, axis=1, keepdims=True)
            cand_max = np.max(q_lp, axis=1, keepdims=True)
            ref_norm = ref_lp - (ref_max + np.log(np.exp(ref_lp - ref_max).sum(axis=1, keepdims=True)))
            cand_norm = q_lp - (cand_max + np.log(np.exp(q_lp - cand_max).sum(axis=1, keepdims=True)))
            terms = np.sum(np.exp(ref_norm) * (ref_norm - cand_norm), axis=1, dtype=np.float64)
            all_terms.extend([float(x) for x in terms.tolist()])
            top1 += int((q_argmax == idx[:, 0]).sum())
            positions += count
        return ScoreResult(
            checkpoint=key,
            windows=selected,
            positions=positions,
            support=SUPPORT,
            kld=fsum(all_terms) / positions,
            top1=top1,
            top1_rate=top1 / positions,
            artifact_root=str(self.root),
            spec=BALANCED64_SPEC,
            candidate_dir=str(candidate_dir),
        )

    def load_resident(
        self,
        checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        loader: Loader | None = None,
    ) -> ResidentBalanced64:
        """Load one anchor completely before timing its in-memory reduction."""
        started = time.perf_counter()
        key = self.checkpoint_key(checkpoint)
        self.checkpoint_path(key)
        score_spec = self.manifest["score"]
        selected = tuple(self.windows if windows is None else (int(x) for x in windows))
        if not selected or len(set(selected)) != len(selected):
            raise ArtifactError("windows must be a non-empty unique sequence")
        unknown = sorted(set(selected) - set(self.windows))
        if unknown:
            raise ArtifactError(f"windows are not declared by this artifact: {unknown}")
        loader = loader or _load_torch
        teacher_dir = _relative_path(self.root, score_spec["teacher_dir"], "score.teacher_dir")
        candidate_dir = _relative_path(
            self.root,
            str(score_spec["candidate_dir_template"]).format(checkpoint=key),
            "score.candidate_dir_template",
        )
        if not candidate_dir.is_dir():
            raise ArtifactError(f"candidate artifacts missing for {key}: {candidate_dir}")
        import numpy as np
        rows: list[tuple[int, Any, Any, Any, Any]] = []
        for window in selected:
            teacher_path = teacher_dir / f"t8192_win{window}.pt"
            candidate_path = candidate_dir / f"q8192_win{window}.pt"
            if not teacher_path.is_file() or not candidate_path.is_file():
                raise ArtifactError(f"checkpoint {key} is missing window {window} candidate/teacher rows")
            teacher = loader(teacher_path)
            candidate = loader(candidate_path)
            idx = _array(_required(teacher, "idx", teacher_path)).astype("int64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            ref_lp = _array(_required(teacher, "logprob", teacher_path)).astype("float64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            q_lp = _array(_required(candidate, "q_lp_at_ref", candidate_path)).astype("float64")[:POSITIONS_PER_WINDOW, :SUPPORT]
            q_argmax = _array(_required(candidate, "q_argmax", candidate_path)).astype("int64")[:POSITIONS_PER_WINDOW]
            if idx.ndim != 2 or idx.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} teacher idx has shape {idx.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if ref_lp.ndim != 2 or ref_lp.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} teacher logprob has shape {ref_lp.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if q_lp.ndim != 2 or q_lp.shape != (POSITIONS_PER_WINDOW, SUPPORT):
                raise ArtifactError(f"window {window} candidate q_lp_at_ref has shape {q_lp.shape}; expected {(POSITIONS_PER_WINDOW, SUPPORT)}")
            if q_argmax.ndim not in (1, 2) or q_argmax.shape[0] != POSITIONS_PER_WINDOW:
                raise ArtifactError(f"window {window} candidate q_argmax has shape {q_argmax.shape}; expected {POSITIONS_PER_WINDOW} rows")
            count = POSITIONS_PER_WINDOW
            ref_lp = ref_lp[:count]
            q_lp = q_lp[:count]
            idx = idx[:count]
            q_argmax = q_argmax[:count]
            ref_max = np.max(ref_lp, axis=1, keepdims=True)
            cand_max = np.max(q_lp, axis=1, keepdims=True)
            ref_norm = ref_lp - (ref_max + np.log(np.exp(ref_lp - ref_max).sum(axis=1, keepdims=True)))
            cand_norm = q_lp - (cand_max + np.log(np.exp(q_lp - cand_max).sum(axis=1, keepdims=True)))
            rows.append((window, ref_norm, cand_norm, idx[:, 0].copy(), q_argmax.copy()))
        return ResidentBalanced64(
            checkpoint=key,
            windows=selected,
            rows=tuple(rows),
            artifact_root=str(self.root),
            candidate_dir=str(candidate_dir),
            resident_load_seconds=time.perf_counter() - started,
        )

    def score_in_memory(
        self,
        checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        loader: Loader | None = None,
    ) -> ScoreResult:
        """Load an anchor once, then score it without any score-time file I/O."""
        return self.load_resident(checkpoint, windows=windows, loader=loader).score()

    def generate_candidates(
        self,
        checkpoint: int | str,
        *,
        builder_template: str | Path,
        ref_dir: str | Path,
        corpus: str | Path,
        meta_dir: str | Path,
        python_executable: str | Path = sys.executable,
        mode: str = "w2",
        remote: str | None = None,
        local_dir: str | Path | None = None,
        windows: Iterable[int] | None = None,
        chunk: int = 8,
        mb: int = 1,
    ) -> dict[str, Any]:
        """Generate official candidate rows through the public API.

        The supplied builder is treated as a sealed implementation template. Its
        checkpoint constants are rewritten only from the declared checkpoint
        metadata; the builder still owns the model forward and emits the official
        ``q_lp_at_ref``/``q_argmax`` payloads. Final KLD is always computed by
        :meth:`score`, never by the builder.
        """
        key = self.checkpoint_key(checkpoint)
        checkpoint_meta = self.manifest["checkpoints"][key]
        checkpoint_path = self.checkpoint_path(key)
        checkpoint_sha = checkpoint_meta.get("sha256")
        identity_sha = checkpoint_meta.get("identity_sha256")
        next_update = checkpoint_meta.get("next_update")
        if not all(isinstance(value, str) and value for value in (checkpoint_sha, identity_sha)):
            raise ArtifactError(f"checkpoint {key} must declare sha256 and identity_sha256")
        if not isinstance(next_update, int) or next_update < 0:
            raise ArtifactError(f"checkpoint {key} must declare non-negative next_update")
        template_path = Path(builder_template).expanduser().resolve()
        if not template_path.is_file():
            raise ArtifactError(f"builder template is missing: {template_path}")
        template = template_path.read_text()

        def replace_line(pattern: str, value: str, label: str) -> None:
            nonlocal template
            compiled = re.compile(pattern, re.MULTILINE)
            if compiled.search(template) is None:
                raise ArtifactError(f"builder template missing {label}")
            template = compiled.sub(lambda _match: value, template, count=1)

        replace_line(r"^CHECKPOINT\s*=.*$", f"CHECKPOINT = {json.dumps(str(checkpoint_path))}", "CHECKPOINT")
        replace_line(r"^CHECKPOINT_SHA\s*=.*$", f"CHECKPOINT_SHA = {json.dumps(checkpoint_sha)}", "CHECKPOINT_SHA")
        replace_line(r"^CANDIDATE_IDENTITY\s*=.*$", f"CANDIDATE_IDENTITY = {json.dumps(identity_sha)}", "CANDIDATE_IDENTITY")
        replace_line(
            r'int\(value\.get\("next_update", -1\)\) != \d+',
            f'int(value.get("next_update", -1)) != {next_update}',
            "next_update gate",
        )

        candidate_dir = _relative_path(
            self.root,
            str(self.manifest["score"]["candidate_dir_template"]).format(checkpoint=key),
            "score.candidate_dir_template",
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)
        builder_dir = self.root / "builders"
        builder_dir.mkdir(parents=True, exist_ok=True)
        derived_builder = builder_dir / f"builder_{key}.py"
        if derived_builder.exists():
            derived_builder.chmod(0o700)
        derived_builder.write_text(template)
        derived_builder.chmod(0o500)
        selected = tuple(self.windows if windows is None else (int(x) for x in windows))
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(self.windows):
            raise ArtifactError("generation windows must be a unique declared subset")
        command = [
            str(python_executable),
            str(derived_builder),
            "--mode", mode,
            "--ref-dir", str(Path(ref_dir).expanduser()),
            "--corpus", str(Path(corpus).expanduser()),
            "--meta-dir", str(Path(meta_dir).expanduser()),
            "--out", str(candidate_dir),
            "--windows", ",".join(str(value) for value in selected),
            "--chunk", str(int(chunk)),
            "--mb", str(int(mb)),
            "--tag", key,
        ]
        if remote is not None:
            command += ["--remote", remote]
        if local_dir is not None:
            command += ["--local-dir", str(Path(local_dir).expanduser())]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        receipt = {
            "schema": "repair-api-candidate-generation-v1",
            "status": "PASS" if completed.returncode == 0 else "FAILED",
            "checkpoint": key,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": identity_sha,
            "next_update": next_update,
            "builder_template": str(template_path),
            "derived_builder": str(derived_builder),
            "windows": list(selected),
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        (candidate_dir / "REPAIR_API_GENERATION.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if completed.returncode != 0:
            raise ArtifactError(f"candidate generation failed for {key}: {completed.stderr[-1200:]}")
        missing = [window for window in selected if not (candidate_dir / f"q8192_win{window}.pt").is_file()]
        if missing:
            raise ArtifactError(f"candidate generation completed without windows: {missing}")
        return receipt

    def trend(self, checkpoints: Iterable[int | str], *, windows: Iterable[int] | None = None) -> list[dict[str, Any]]:
        """Return the same standardized score for each checkpoint in order."""
        return [self.score(step, windows=windows).as_dict() for step in checkpoints]
