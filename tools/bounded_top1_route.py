#!/usr/bin/env python3
"""Run a bounded, one-request-per-window top-1 evaluation against a sealed route.

All campaign-specific identities and paths are supplied by a JSON config. The client
never computes or publishes KLD; it only compares the candidate full-vocabulary
argmax returned by the route with the teacher's first support token.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    right = text.rfind(")")
    return int(text[right + 2 :].split()[19])


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {
        "task_id", "run_id", "basis_sha256", "basis_path", "candidate_manifest_sha256",
        "host_claim_path", "route_pid", "route_startticks", "route_progress_path", "route_key_path",
        "route_host", "route_port", "corpus_path", "corpus_sha256", "class_map_path",
        "class_map_sha256", "teacher_root", "teacher_tree_sha256", "output_root",
        "cutoff", "support", "expected_class_counts", "expected_canary_matches",
        "expected_final_matches",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"CONFIG_FIELDS_MISSING:{missing}")
    return config


def main(config_path: Path) -> None:
    config = load_config(config_path)
    task_id = str(config["task_id"])
    run_id = int(config["run_id"])
    basis = str(config["basis_sha256"])
    candidate = str(config["candidate_manifest_sha256"])
    cutoff = int(config["cutoff"])
    support = int(config["support"])
    route_pid = int(config["route_pid"])
    route_startticks = int(config["route_startticks"])
    output_root = Path(config["output_root"])
    rows_root = output_root / "rows"
    checkpoints_root = output_root / "checkpoints"
    progress_path = output_root / "PROGRESS.json"
    terminal_path = output_root / "TERMINAL.json"
    result_path = output_root / "BALANCED64_TOP1.json"
    rows_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    state: dict[str, Any] = {
        "schema": "banana-smasher-bounded-top1-progress-v1",
        "status": "PREFLIGHT",
        "task_id": task_id,
        "run_id": run_id,
        "controller_pid": os.getpid(),
        "controller_startticks": start_ticks(os.getpid()),
        "route_pid": route_pid,
        "route_startticks": route_startticks,
        "basis_sha256": basis,
        "candidate_manifest_sha256": candidate,
        "rows_complete": 0,
        "positions_complete": 0,
        "requests_this_consumer": 0,
        "started_unix": started,
        "updated_unix": started,
    }

    def publish(status: str, **fields: Any) -> str:
        state.update(status=status, updated_unix=time.time(), **fields)
        return atomic_json(progress_path, state)

    try:
        claim = json.loads(Path(config["host_claim_path"]).read_text())
        if claim.get("state") != "CLAIMED" or claim.get("task_id") != task_id or int(claim.get("run_id", -1)) != run_id:
            raise RuntimeError("HOST_CLAIM_GATE_FAILED")
        if start_ticks(route_pid) != route_startticks:
            raise RuntimeError("ROUTE_IDENTITY_MISMATCH")
        if sha256(Path(config["basis_path"])) != basis:
            raise RuntimeError("BASIS_GATE_MISMATCH")
        corpus_path = Path(config["corpus_path"])
        class_map_path = Path(config["class_map_path"])
        if sha256(corpus_path) != config["corpus_sha256"]:
            raise RuntimeError("CORPUS_SHA_MISMATCH")
        if sha256(class_map_path) != config["class_map_sha256"]:
            raise RuntimeError("CLASS_MAP_SHA_MISMATCH")
        class_map = json.loads(class_map_path.read_text())
        rows = class_map["windows"]
        expected_counts = {str(k): int(v) for k, v in config["expected_class_counts"].items()}
        if len(rows) != 64 or Counter(row["class"] for row in rows) != Counter(expected_counts):
            raise RuntimeError("CLASS_MAP_QUOTA_MISMATCH")
        corpus = json.loads(corpus_path.read_text())
        teacher_root = Path(config["teacher_root"])
        key = Path(config["route_key_path"]).read_text().strip()
        if not 32 <= len(key) <= 4096 or any(ch in key for ch in "\r\n"):
            raise RuntimeError("ROUTE_KEY_INVALID")
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        peak_estimate = int(config.get("peak_estimate_bytes", 3 * 1024**3))
        reserve = int(config.get("memory_reserve_bytes", 4 * 1024**3))
        if peak_estimate > available - reserve:
            raise RuntimeError(
                f"MEMORY_PREFLIGHT_FAILED estimate={peak_estimate} available={available} reserve={reserve}"
            )
        route_initial = json.loads(Path(config["route_progress_path"]).read_text())
        if route_initial.get("status") != "READY" or int(route_initial.get("requests", -1)) != 0:
            raise RuntimeError(f"ROUTE_INITIAL_STATE_MISMATCH:{route_initial}")
        initial_route_requests = int(route_initial["requests"])
        requests_sent = 0
        publish(
            "RUNNING",
            memory_available_bytes=available,
            peak_estimate_bytes=peak_estimate,
            memory_reserve_bytes=reserve,
            topology={"transport": "one_loopback_bulk_request_per_window", "requests_total": 64},
        )

        completed: list[dict[str, Any]] = []
        for row in rows:
            ordinal = int(row["ordinal"])
            window_id = int(row["win"])
            class_name = str(row["class"])
            checkpoint_path = checkpoints_root / f"ROW_{ordinal:03d}_W{window_id:03d}.json"
            artifact_path = rows_root / f"ROW_{ordinal:03d}_W{window_id:03d}.npz"
            if checkpoint_path.is_file() and artifact_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_text())
                if checkpoint.get("artifact_sha256") != sha256(artifact_path):
                    raise RuntimeError(f"SEALED_ROW_DRIFT ordinal={ordinal}")
                if ordinal == 0 and int(checkpoint.get("top1_matches", -1)) != int(config["expected_canary_matches"]):
                    raise RuntimeError(
                        f"SEALED_CANARY_DRIFT observed={checkpoint.get('top1_matches')} expected={config['expected_canary_matches']}"
                    )
                completed.append(checkpoint)
                continue

            if start_ticks(route_pid) != route_startticks:
                raise RuntimeError("ROUTE_IDENTITY_LOST")
            claim_now = json.loads(Path(config["host_claim_path"]).read_text())
            if claim_now.get("state") != "CLAIMED" or claim_now.get("task_id") != task_id or int(claim_now.get("run_id", -1)) != run_id:
                raise RuntimeError("HOST_CLAIM_LOST")
            teacher_path = teacher_root / row["teacher_file"]
            teacher_sha = sha256(teacher_path)
            teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
            indices = teacher["idx"][:cutoff].detach().cpu().numpy()
            if indices.shape != (cutoff, support):
                raise RuntimeError(f"TEACHER_SHAPE_MISMATCH ordinal={ordinal} observed={indices.shape}")
            teacher_argmax = indices[:, 0].astype(np.int32, copy=False)
            corpus_row = corpus[window_id]
            tokens = corpus_row["token_ids"]
            if corpus_row.get("id_ds4") != window_id or len(tokens) < cutoff:
                raise RuntimeError(f"CORPUS_ROW_MISMATCH window={window_id}")
            request_id = f"bounded-top1-run{run_id}-{ordinal:03d}-win{window_id:03d}"
            payload = {
                "schema": "bs-teacher-support-logprob-request-v1",
                "basis_index_sha256": basis,
                "cutoff": cutoff,
                "input_token_ids": tokens,
                "request_id": request_id,
                "teacher_support_token_ids": indices.tolist(),
                "window_id": window_id,
            }
            body = json.dumps(payload, separators=(",", ":")).encode()
            request_sha = hashlib.sha256(body).hexdigest()
            publish(
                "REQUESTING",
                rows_complete=len(completed),
                positions_complete=len(completed) * cutoff,
                requests_this_consumer=requests_sent,
                active_ordinal=ordinal,
                active_window_id=window_id,
                active_class=class_name,
                request_sha256=request_sha,
                request_bytes=len(body),
            )
            connection = http.client.HTTPConnection(
                str(config["route_host"]), int(config["route_port"]), timeout=int(config.get("request_timeout_seconds", 600))
            )
            connection.request(
                "POST",
                "/v1/eval/teacher-support-logprobs",
                body=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read(int(config.get("max_response_bytes", 512 * 1024 * 1024)) + 1)
            connection.close()
            if response.status != 200:
                raise RuntimeError(f"ROUTE_HTTP_FAILED ordinal={ordinal} status={response.status} body={raw[:2000]!r}")
            response_sha = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw)
            del raw, body, payload
            expected_response_keys = {
                "schema", "basis_index_sha256", "candidate_argmax", "candidate_logprob",
                "positions", "request_id", "support", "window_id",
            }
            if (
                set(value) != expected_response_keys
                or value["schema"] != "bs-teacher-support-logprob-response-v1"
                or value["basis_index_sha256"] != basis
                or value["request_id"] != request_id
                or int(value["window_id"]) != window_id
                or int(value["positions"]) != cutoff
                or int(value["support"]) != support
            ):
                raise RuntimeError(f"ROUTE_RESPONSE_CONTRACT_MISMATCH ordinal={ordinal}")
            candidate_argmax = np.asarray(value["candidate_argmax"], dtype=np.int32)
            candidate_logprob = value["candidate_logprob"]
            if candidate_argmax.shape != (cutoff,) or len(candidate_logprob) != cutoff or any(len(item) != support for item in candidate_logprob):
                raise RuntimeError(f"ROUTE_RESPONSE_SHAPE_MISMATCH ordinal={ordinal}")
            del candidate_logprob, value
            equal = (candidate_argmax == teacher_argmax).astype(np.uint8)
            matches = int(equal.sum())
            if ordinal == 0 and matches != int(config["expected_canary_matches"]):
                raise RuntimeError(
                    f"CANARY_DRIFT observed={matches} expected={config['expected_canary_matches']} window={window_id}"
                )
            temporary = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, candidate_argmax=candidate_argmax, teacher_argmax=teacher_argmax, top1_equal=equal)
            os.replace(temporary, artifact_path)
            with artifact_path.open("rb") as artifact_file:
                os.fsync(artifact_file.fileno())
            if sha256(teacher_path) != teacher_sha:
                raise RuntimeError(f"TEACHER_CHANGED ordinal={ordinal}")
            route_after = json.loads(Path(config["route_progress_path"]).read_text())
            requests_sent += 1
            expected_route_requests = initial_route_requests + requests_sent
            if route_after.get("status") != "READY" or int(route_after.get("requests", -1)) != expected_route_requests:
                raise RuntimeError(
                    f"ROUTE_REQUEST_COUNT_MISMATCH ordinal={ordinal} expected={expected_route_requests} observed={route_after}"
                )
            checkpoint = {
                "schema": "banana-smasher-bounded-top1-row-v1",
                "status": "PASS",
                "task_id": task_id,
                "run_id": run_id,
                "ordinal": ordinal,
                "window_id": window_id,
                "class": class_name,
                "positions": cutoff,
                "teacher_sha256": teacher_sha,
                "artifact": str(artifact_path),
                "artifact_sha256": sha256(artifact_path),
                "request_sha256": request_sha,
                "response_sha256": response_sha,
                "top1_matches": matches,
                "top1_parity": matches / cutoff,
            }
            checkpoint_sha = atomic_json(checkpoint_path, checkpoint)
            completed.append({**checkpoint, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha})
            publish(
                "RUNNING",
                rows_complete=len(completed),
                positions_complete=len(completed) * cutoff,
                requests_this_consumer=requests_sent,
                sealed_rows_reused=len(completed) - requests_sent,
                last_row=completed[-1],
            )

        per_class: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"matches": 0, "positions": 0, "windows": 0}
        )
        matches = 0
        members = []
        for row in rows:
            ordinal = int(row["ordinal"])
            window_id = int(row["win"])
            class_name = str(row["class"])
            checkpoint_path = checkpoints_root / f"ROW_{ordinal:03d}_W{window_id:03d}.json"
            artifact_path = rows_root / f"ROW_{ordinal:03d}_W{window_id:03d}.npz"
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("artifact_sha256") != sha256(artifact_path):
                raise RuntimeError(f"FINAL_ARTIFACT_DRIFT ordinal={ordinal}")
            with np.load(artifact_path, allow_pickle=False) as arrays:
                equal = np.asarray(arrays["top1_equal"], dtype=np.uint8)
            row_matches = int(equal.sum())
            matches += row_matches
            per_class[class_name]["matches"] += row_matches
            per_class[class_name]["positions"] += cutoff
            per_class[class_name]["windows"] += 1
            members.append(
                {
                    "ordinal": ordinal,
                    "window_id": window_id,
                    "class": class_name,
                    "artifact": str(artifact_path),
                    "artifact_sha256": sha256(artifact_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
        if matches != int(config["expected_final_matches"]):
            raise RuntimeError(f"FINAL_VECTOR_DRIFT observed={matches} expected={config['expected_final_matches']}")
        by_class = {
            name: {**value, "top1_parity": value["matches"] / value["positions"]}
            for name, value in sorted(per_class.items())
        }
        manifest_path = output_root / "ARTIFACT_MANIFEST.json"
        manifest_sha = atomic_json(
            manifest_path,
            {
                "schema": "banana-smasher-bounded-top1-artifact-manifest-v1",
                "status": "PASS",
                "task_id": task_id,
                "run_id": run_id,
                "members": members,
            },
        )
        result = {
            "schema": "banana-smasher-bounded-top1-v1",
            "status": "PASS",
            "scientific_status": "PASS_EXACT_BASIS_BOUNDED_TOP1_ONLY",
            "task_id": task_id,
            "run_id": run_id,
            "basis_sha256": basis,
            "candidate_manifest_sha256": candidate,
            "teacher_tree_sha256": config["teacher_tree_sha256"],
            "corpus_sha256": config["corpus_sha256"],
            "class_map_sha256": config["class_map_sha256"],
            "windows": len(rows),
            "positions": len(rows) * cutoff,
            "matches": matches,
            "top1_parity": matches / (len(rows) * cutoff),
            "by_class": by_class,
            "topology": {
                "transport": "one_loopback_bulk_request_per_window",
                "logical_requests_total": len(rows),
                "route_requests_this_execution": requests_sent,
                "sealed_rows_reused": len(rows) - requests_sent,
            },
            "kld_computed": False,
            "kld_published": False,
            "artifact_manifest": str(manifest_path),
            "artifact_manifest_sha256": manifest_sha,
            "elapsed_seconds": time.time() - started,
            "created_unix": time.time(),
        }
        result_sha = atomic_json(result_path, result)
        terminal_sha = atomic_json(terminal_path, {**result, "result_path": str(result_path), "result_sha256": result_sha})
        publish(
            "PASS",
            rows_complete=len(rows),
            positions_complete=len(rows) * cutoff,
            requests_this_consumer=requests_sent,
            sealed_rows_reused=len(rows) - requests_sent,
            matches=matches,
            top1_parity=result["top1_parity"],
            result_path=str(result_path),
            result_sha256=result_sha,
            terminal_path=str(terminal_path),
            terminal_sha256=terminal_sha,
            artifact_manifest_sha256=manifest_sha,
        )
    except BaseException as error:
        failure = {
            "schema": "banana-smasher-bounded-top1-failure-v1",
            "status": "FAILED",
            "task_id": task_id,
            "run_id": run_id,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "created_unix": time.time(),
        }
        failure_sha = atomic_json(terminal_path, failure)
        publish("FAILED", error=failure["error"], terminal_path=str(terminal_path), terminal_sha256=failure_sha)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    main(arguments.config)
