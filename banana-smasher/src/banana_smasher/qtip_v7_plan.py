"""API-owned input staging and lifecycle for QTIP V7 joint repair plans."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from .qtip_v7_joint_workflow import _load_json, _sha256, _write_json

PLAN_SCHEMA = "banana-smasher-qtip-v7-repair-plan-v1"
INPUTS_READY_SCHEMA = "banana-smasher-qtip-v7-inputs-ready-v1"
PRE_SCHEMA = "banana-smasher-qtip-v7-pre-v1"
TRAIN_SCHEMA = "banana-smasher-qtip-v7-train-v1"
_REQUIRED_RUN_INPUTS = {
    "teacher_targets",
    "teacher_bank",
    "manifest",
    "corpus",
    "model",
    "runtime",
    "admission",
    "inventory",
    "roster",
    "planes",
    "trainer",
}


def _relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be run-root-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be run-root-relative")
    return root / relative


def _plan_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"QTIP V7 repair plan schema must be {PLAN_SCHEMA!r}")
    rows = document.get("inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("QTIP V7 repair plan requires declared inputs")
    names: set[str] = set()
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else None
        expected = row.get("expected") if isinstance(row, dict) else None
        count = expected.get("count") if isinstance(expected, dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"duplicate/invalid QTIP V7 repair plan input {name!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"input {name!r} requires a positive expected count")
        names.add(name)
    return rows


def _declared_files(row: dict[str, Any], root: Path) -> tuple[Path, list[dict[str, Any]]]:
    name = str(row["name"])
    source = Path(str(row.get("source", ""))).expanduser().resolve()
    destination = _relative(root, row.get("destination"), f"input {name!r} destination")
    try:
        destination.relative_to(root / "inputs")
    except ValueError as exc:
        raise ValueError(f"input {name!r} destination must be beneath inputs/") from exc
    expected = row["expected"]
    count = int(expected["count"])
    if source.is_dir():
        files = expected.get("files")
        if not isinstance(files, list) or len(files) != count:
            raise ValueError(f"input {name!r} requires one expected row per file")
    else:
        if count != 1:
            raise ValueError(f"file input {name!r} expected count must be 1")
        files = [{
            "path": "",
            "bytes": expected.get("bytes"),
            "sha256": expected.get("sha256"),
        }]
    declared: list[dict[str, Any]] = []
    for item in files:
        relative_value = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative_value, str):
            raise ValueError(f"input {name!r} has an invalid expected file row")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"input {name!r} expected path must be relative")
        source_file = source / relative if source.is_dir() else source
        target = destination / relative if source.is_dir() else destination
        display = target.relative_to(root / "inputs")
        if not source_file.is_file():
            raise FileNotFoundError(str(display))
        expected_bytes = item.get("bytes")
        expected_sha = item.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ValueError(f"input {name!r} identity is incomplete for {relative}")
        if source_file.stat().st_size != expected_bytes:
            raise RuntimeError(f"input size mismatch: {display}")
        if _sha256(source_file) != expected_sha:
            raise RuntimeError(f"input SHA-256 mismatch: {display}")
        declared.append({
            "source": str(source_file),
            "path": str(target),
            "relative_path": str(display),
            "bytes": expected_bytes,
            "sha256": expected_sha,
        })
    if source.is_dir():
        observed = sum(1 for path in source.rglob("*") if path.is_file() or path.is_symlink())
        if observed != count:
            raise ValueError(f"input {name!r} expected {count} files but source contains {observed}")
    return destination, declared


def _materialize(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(strict=True), target)
    except OSError:
        shutil.copyfile(source.resolve(strict=True), target)


def _verify_ready(path: Path, ready: dict[str, Any]) -> dict[str, Any]:
    if path.stat().st_mode & 0o222:
        raise RuntimeError("QTIP V7 INPUTS_READY receipt must be read-only")
    if ready.get("schema") != INPUTS_READY_SCHEMA or ready.get("status") != "PASS":
        raise ValueError("QTIP V7 repair requires a PASS INPUTS_READY receipt")
    inputs = ready.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("QTIP V7 INPUTS_READY receipt has no inputs")
    for name, row in inputs.items():
        files = row.get("files") if isinstance(row, dict) else None
        if not isinstance(files, list) or row.get("accepted") != row.get("total"):
            raise RuntimeError(f"QTIP V7 staged input is incomplete: {name}")
        for file_row in files:
            target = Path(str(file_row.get("path", "")))
            if (
                target.is_symlink()
                or not target.is_file()
                or target.stat().st_size != file_row.get("bytes")
                or _sha256(target) != file_row.get("sha256")
            ):
                raise RuntimeError(
                    f"QTIP V7 staged input identity drift: {file_row.get('relative_path')}"
                )
    return ready


def prepare_joint_inputs(*, plan: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Stage every declared immediate input and publish INPUTS_READY last."""
    plan_path, document = _load_json(plan)
    rows = _plan_rows(document)
    root = Path(run_root).expanduser().resolve()
    receipt = root / "receipts" / "INPUTS_READY.json"
    plan_sha256 = _sha256(plan_path)
    if receipt.exists():
        _, ready = _load_json(receipt)
        if ready.get("plan_sha256") != plan_sha256:
            raise RuntimeError("QTIP V7 INPUTS_READY binds a different repair plan")
        return _verify_ready(receipt, ready)

    staged: dict[str, dict[str, Any]] = {}
    for row in rows:
        destination, files = _declared_files(row, root)
        staged[str(row["name"])] = {
            "destination": str(destination),
            "accepted": len(files),
            "total": len(files),
            "files": files,
        }
    for row in staged.values():
        for file_row in row["files"]:
            source = Path(file_row["source"])
            target = Path(file_row["path"])
            if target.exists() or target.is_symlink():
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.stat().st_size != file_row["bytes"]
                    or _sha256(target) != file_row["sha256"]
                ):
                    raise RuntimeError(f"staged input collision: {file_row['relative_path']}")
            else:
                _materialize(source, target)
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"staged input is not regular: {file_row['relative_path']}")
            file_row.pop("source")
    total = sum(row["total"] for row in staged.values())
    ready = {
        "schema": INPUTS_READY_SCHEMA,
        "status": "PASS",
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "run_root": str(root),
        "accepted": total,
        "total": total,
        "inputs": staged,
    }
    _write_json(receipt, ready, exclusive=True)
    os.chmod(receipt, 0o444)
    return _verify_ready(receipt, ready)


def _binding(ready: dict[str, Any], value: object, label: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("input"), str):
        raise ValueError(f"QTIP V7 repair workflow requires {label} input binding")
    row = ready["inputs"].get(value["input"])
    if not isinstance(row, dict):
        raise ValueError(f"QTIP V7 repair workflow references unknown input {value['input']!r}")
    relative = value.get("path", "")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"QTIP V7 repair workflow {label} path must be relative")
    target = Path(row["destination"]) / relative if relative else Path(row["destination"])
    if target.is_symlink() or not target.is_file():
        raise FileNotFoundError(f"staged {label}: {target}")
    return target


def run_joint_plan(*, plan: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Prepare, freeze (PRE), and train through the existing public stages."""
    from . import qtip_v7_joint_workflow as workflow

    plan_path, document = _load_json(plan)
    root = Path(run_root).expanduser().resolve()
    ready = prepare_joint_inputs(plan=plan_path, run_root=root)
    missing = sorted(_REQUIRED_RUN_INPUTS - set(ready["inputs"]))
    if missing:
        raise ValueError(f"QTIP V7 repair plan missing required immediate inputs: {missing}")
    bindings = document.get("workflow")
    if not isinstance(bindings, dict):
        raise ValueError("QTIP V7 repair plan requires workflow bindings")
    manifest = _binding(ready, bindings.get("manifest"), "manifest")
    teacher_bank = _binding(ready, bindings.get("teacher_bank"), "teacher bank")
    trainer = _binding(ready, bindings.get("trainer"), "trainer")
    receipts = root / "receipts"
    pre_receipt = receipts / "PRE.json"
    freeze = root / "FROZEN_INPUTS.json"
    if pre_receipt.exists():
        _, pre = _load_json(pre_receipt)
        if pre.get("inputs_ready_sha256") != _sha256(receipts / "INPUTS_READY.json"):
            raise RuntimeError("sealed PRE does not bind current INPUTS_READY")
        if not freeze.is_file():
            raise RuntimeError("sealed PRE is missing FROZEN_INPUTS.json")
    else:
        inspected = workflow.inspect_joint_inputs(
            manifest=manifest,
            teacher_bank=teacher_bank,
            run_root=root,
            trainer_host=str(document.get("trainer_host", "")),
            trainer_aliases=document.get("trainer_aliases", []),
        )
        pre = {
            "schema": PRE_SCHEMA,
            "status": "PASS",
            "accepted": 1,
            "total": 1,
            "inputs_ready_sha256": _sha256(receipts / "INPUTS_READY.json"),
            "freeze": inspected["freeze"],
        }
        _write_json(pre_receipt, pre, exclusive=True)
        os.chmod(pre_receipt, 0o444)
    target_update = document.get("target_update")
    if isinstance(target_update, bool) or not isinstance(target_update, int) or target_update < 0:
        raise ValueError("QTIP V7 repair plan target_update must be nonnegative")
    checkpoint = _relative(root, document.get("checkpoint"), "checkpoint")
    train_receipt = receipts / "TRAIN.json"
    if train_receipt.exists():
        _, trained = _load_json(train_receipt)
        if trained.get("target_update") != target_update:
            raise RuntimeError("sealed TRAIN target update drift")
        return {"status": "PASS", "inputs": ready, "pre": pre, "train": trained}
    resume_value = document.get("resume_from")
    resume = None if resume_value is None else _relative(root, resume_value, "resume_from")
    result = workflow.train_joint(
        freeze=freeze,
        checkpoint=checkpoint,
        target_update=target_update,
        trainer=trainer,
        resume_from=resume,
        inputs_ready=receipts / "INPUTS_READY.json",
    )
    trained = {
        "schema": TRAIN_SCHEMA,
        "status": "PASS",
        "accepted": 1,
        "total": 1,
        "target_update": target_update,
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "receipt": result["receipt"],
    }
    _write_json(train_receipt, trained, exclusive=True)
    os.chmod(train_receipt, 0o444)
    return {"status": "PASS", "inputs": ready, "pre": pre, "train": trained}


def joint_plan_status(*, run_root: str | Path) -> dict[str, Any]:
    """Report INPUTS/PRE/TRAIN accepted totals and first incomplete stage."""
    root = Path(run_root).expanduser().resolve()
    stages: dict[str, dict[str, Any]] = {}
    specifications = (
        ("INPUTS", root / "receipts" / "INPUTS_READY.json"),
        ("PRE", root / "receipts" / "PRE.json"),
        ("TRAIN", root / "receipts" / "TRAIN.json"),
    )
    first_incomplete: str | None = None
    for name, path in specifications:
        accepted = total = 0
        status = "PENDING"
        if path.is_file():
            _, row = _load_json(path)
            accepted = int(row.get("accepted", 0))
            total = int(row.get("total", 0))
            status = "PASS" if row.get("status") == "PASS" and accepted == total else "INCOMPLETE"
        if first_incomplete is None and status != "PASS":
            first_incomplete = name
        stages[name] = {"status": status, "accepted": accepted, "total": total, "receipt": str(path)}
    return {
        "schema": "banana-smasher-qtip-v7-plan-status-v1",
        "status": "PASS",
        "stages": stages,
        "first_incomplete_stage": first_incomplete,
    }


__all__ = ["joint_plan_status", "prepare_joint_inputs", "run_joint_plan"]
