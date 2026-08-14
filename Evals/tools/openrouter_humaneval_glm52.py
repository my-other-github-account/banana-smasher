#!/usr/bin/env python3
"""Reproduce the GLM-5.2 OpenRouter HumanEval generation rail.

Dry-run validation is the default. Pass --run to generate the 164 paid tasks serially.
The dry run may populate EvalPlus's public dataset cache, but it never calls OpenRouter.
The output canonical.jsonl is scored with the repository's pinned EvalPlus tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MODEL = "z-ai/glm-5.2"
CANONICAL_SLUG = "z-ai/glm-5.2-20260616"
PROVIDER = "Z.AI"
PROVIDER_QUANTIZATION = "fp8"
PROVIDER_TAG = "z-ai/fp8"
URL = "https://openrouter.ai/api/v1/chat/completions"
ENDPOINTS_URL = f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints"
SCHEMA = "banana-smasher.humaneval-openrouter-glm52.v1"
IDENTITY_SCHEMA = "banana-smasher.humaneval-openrouter-glm52-run-identity.v1"
HANDOFF_SCHEMA = "banana-smasher.humaneval-openrouter-glm52-generation-handoff.v1"
INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
TASK_COUNT = 164
MAX_COMPLETION_TOKENS = 16_384
EVALPLUS_VERSION = "0.4.0.dev44"
EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
HUMANEVALPLUS_RELEASE = "v0.1.10"
HUMANEVALPLUS_HASH = "fe585eb4df8c88d844eeb463ea4d0302"
MEASURED_PROMPT_LEDGER_SHA256 = "6b4f5f30169054a7505f6704c246984f53671c6f4f168c9ee05fdfbd8444b598"
RESUMABLE_PREDECESSOR_SCRIPT_SHA256 = {
    "d5d7afdc510a96d3e478dcfeac28d2d7a0d25c5c574b890b79bcb1a1e42a27ab"
}
FROZEN_TASK_MESSAGES_SHA256 = "e1a420e057d9e44f4dcde770334d67d33e4c44cf70daa20edbf69f163ac4bd96"
REQUEST_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.95,
    "n": 1,
    "max_tokens": MAX_COMPLETION_TOKENS,
    "reasoning": {"enabled": True, "effort": "medium"},
    "include_reasoning": True,
    "provider": {
        "only": [PROVIDER],
        "allow_fallbacks": False,
        "require_parameters": True,
    },
}
TRANSIENT_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_PRINT_LOCK = threading.Lock()


class RunnerError(RuntimeError):
    pass


def compact(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def record_digest(record: dict[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("checkpoint_sha256", None)
    return sha(compact(unsigned))


def evalplus_commit(distribution: importlib.metadata.Distribution) -> str | None:
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        value = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    vcs = value.get("vcs_info")
    return vcs.get("commit_id") if isinstance(vcs, dict) else None


def require_evalplus() -> tuple[Any, Callable[..., str]]:
    try:
        distribution = importlib.metadata.distribution("evalplus")
        from evalplus.data import get_human_eval_plus
        from evalplus.sanitize import sanitize
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        raise RunnerError(
            "EvalPlus is unavailable; install Evals/requirements-humaneval.txt in an isolated environment"
        ) from exc
    if distribution.version != EVALPLUS_VERSION:
        raise RunnerError(
            f"EvalPlus version must be {EVALPLUS_VERSION}, got {distribution.version}"
        )
    installed_commit = evalplus_commit(distribution)
    if installed_commit != EVALPLUS_COMMIT:
        raise RunnerError(
            f"EvalPlus commit must be {EVALPLUS_COMMIT}, got {installed_commit!r}"
        )
    return get_human_eval_plus(version="default"), sanitize


def build_prompts(dataset: Any) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for number in range(TASK_COUNT):
        task_id = f"HumanEval/{number}"
        task = dataset[task_id]
        message = INSTRUCTION + f"\n```python\n{task['prompt'].strip()}\n```"
        messages = [{"role": "user", "content": message}]
        prompts.append(
            {
                "task_id": task_id,
                "messages": messages,
                "messages_sha256": sha(compact(messages)),
            }
        )
    bound = b"".join(
        compact({"task_id": row["task_id"], "messages": row["messages"]})
        for row in prompts
    )
    actual = sha(bound)
    if actual != FROZEN_TASK_MESSAGES_SHA256:
        raise RunnerError(
            "installed EvalPlus prompt messages do not match the frozen measured rail: "
            f"{actual} != {FROZEN_TASK_MESSAGES_SHA256}"
        )
    return prompts


def write_prompts(path: Path, prompts: list[dict[str, Any]]) -> str:
    payload = b"".join(compact(row) for row in prompts)
    fsync_replace(path, payload)
    return sha(payload)


def read_token(token_file: Path | None) -> str:
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if token_file is not None:
        if token:
            raise RunnerError("use either OPENROUTER_API_KEY or --token-file, not both")
        try:
            token = token_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RunnerError(f"cannot read token file: {exc}") from exc
    if not token:
        raise RunnerError("set OPENROUTER_API_KEY or pass --token-file before using --run")
    return token


def fetch_json(url: str, token: str | None = None, timeout: int = 60) -> tuple[bytes, dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw, json.loads(raw)


def select_endpoint(endpoints: dict[str, Any]) -> dict[str, Any]:
    candidates = endpoints.get("data", {}).get("endpoints", [])
    endpoint = next(
        (item for item in candidates if item.get("provider_name") == PROVIDER), None
    )
    if endpoint is None:
        raise RunnerError(f"provider {PROVIDER} is not present for {MODEL}")
    expected = {
        "quantization": PROVIDER_QUANTIZATION,
        "tag": PROVIDER_TAG,
    }
    for field, wanted in expected.items():
        if endpoint.get(field) != wanted:
            raise RunnerError(
                f"selected endpoint {field} changed: {endpoint.get(field)!r} != {wanted!r}"
            )
    name = str(endpoint.get("name") or "")
    if CANONICAL_SLUG not in name:
        raise RunnerError(f"selected endpoint name does not contain {CANONICAL_SLUG}: {name!r}")
    maximum = endpoint.get("max_completion_tokens")
    if not isinstance(maximum, int) or maximum < MAX_COMPLETION_TOKENS:
        raise RunnerError(f"selected endpoint completion cap is insufficient: {maximum!r}")
    required = {
        "reasoning",
        "reasoning_effort",
        "include_reasoning",
        "max_tokens",
        "temperature",
        "top_p",
    }
    supported = set(endpoint.get("supported_parameters") or [])
    missing = sorted(required - supported)
    if missing:
        raise RunnerError(f"selected endpoint lacks required parameters: {missing}")
    return endpoint


def prepare_identity(
    root: Path,
    prompts_path: Path,
    script_path: Path,
    token: str,
) -> tuple[dict[str, Any], str]:
    identity_path = root / "RUN_IDENTITY.json"
    snapshot_path = root / "OPENROUTER_ENDPOINTS_SNAPSHOT.json"
    required = {
        "schema": IDENTITY_SCHEMA,
        "model": MODEL,
        "canonical_slug": CANONICAL_SLUG,
        "provider": PROVIDER,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "provider_tag": PROVIDER_TAG,
        "frozen_task_messages_sha256": FROZEN_TASK_MESSAGES_SHA256,
        "prompts_sha256": sha(prompts_path.read_bytes()),
        "script_sha256": sha(script_path.read_bytes()),
        "request_params": REQUEST_PARAMS,
    }
    raw, endpoints = fetch_json(ENDPOINTS_URL, token=token)
    endpoint = select_endpoint(endpoints)
    if identity_path.exists():
        identity_bytes = identity_path.read_bytes()
        identity = json.loads(identity_bytes)
        for field, wanted in required.items():
            if (
                field == "script_sha256"
                and identity.get(field) in RESUMABLE_PREDECESSOR_SCRIPT_SHA256
            ):
                continue
            if identity.get(field) != wanted:
                raise RunnerError(
                    f"existing run identity mismatch at {field}: {identity.get(field)!r} != {wanted!r}"
                )
        return identity, sha(identity_bytes)

    fsync_replace(snapshot_path, raw + (b"\n" if not raw.endswith(b"\n") else b""))
    identity = {
        **required,
        "status": "PASS",
        "measured_source_prompt_ledger_sha256": MEASURED_PROMPT_LEDGER_SHA256,
        "openrouter_model_name": endpoints.get("data", {}).get("name"),
        "provider_endpoint_name": endpoint.get("name"),
        "provider_context_length": endpoint.get("context_length"),
        "provider_max_completion_tokens": endpoint.get("max_completion_tokens"),
        "provider_supported_parameters": endpoint.get("supported_parameters"),
        "provider_pricing": endpoint.get("pricing"),
        "endpoints_snapshot_sha256": sha(snapshot_path.read_bytes()),
        "created_unix": time.time(),
    }
    data = compact(identity)
    fsync_replace(identity_path, data)
    return identity, sha(data)


def request_payload(prompt: dict[str, Any]) -> dict[str, Any]:
    return {"model": MODEL, "messages": prompt["messages"], **REQUEST_PARAMS}


def response_fields(
    raw: bytes,
    entrypoint: str,
    sanitizer: Callable[..., str],
) -> dict[str, Any]:
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"malformed provider JSON: {exc}") from exc
    choices = response.get("choices") or []
    if not choices:
        raise RunnerError("provider response contains no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content")
    if response.get("model") != MODEL:
        raise RunnerError(f"returned model {response.get('model')!r}")
    if response.get("provider") != PROVIDER:
        raise RunnerError(f"returned provider {response.get('provider')!r}")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RunnerError("reasoning-enabled response has empty reasoning")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise RunnerError(f"message content type is {type(content).__name__}")
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or not 0 <= completion_tokens <= MAX_COMPLETION_TOKENS:
        raise RunnerError(f"invalid completion_tokens={completion_tokens!r}")
    return {
        "model_returned": response.get("model"),
        "provider_returned": response.get("provider"),
        "response_id": response.get("id"),
        "reasoning": reasoning,
        "reasoning_chars": len(reasoning),
        "message_content": content,
        "message_content_chars": len(content),
        "semantic_empty_or_null": not bool(content.strip()),
        "solution": sanitizer(content, entrypoint=entrypoint),
        "entry_point": entrypoint,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
    }


def checkpoint_from_response(
    prompt: dict[str, Any],
    identity_sha: str,
    entrypoint: str,
    raw: bytes,
    status: int,
    elapsed_seconds: float,
    attempt: int,
    *,
    sanitizer: Callable[..., str],
) -> dict[str, Any]:
    if status != 200:
        raise RunnerError(f"invalid successful HTTP status {status!r}")
    if not isinstance(attempt, int) or not 1 <= attempt <= 8:
        raise RunnerError(f"invalid attempt {attempt!r}")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise RunnerError(f"invalid elapsed seconds {elapsed_seconds!r}")
    derived = response_fields(raw, entrypoint, sanitizer)
    body = json.dumps(
        request_payload(prompt), ensure_ascii=False, separators=(",", ":")
    ).encode()
    record = {
        "schema": SCHEMA,
        "task_id": prompt["task_id"],
        "task_number": int(prompt["task_id"].split("/")[-1]),
        "model_requested": MODEL,
        "canonical_slug": CANONICAL_SLUG,
        "provider_requested": PROVIDER,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "provider_tag": PROVIDER_TAG,
        "run_identity_sha256": identity_sha,
        "messages": prompt["messages"],
        "messages_sha256": prompt["messages_sha256"],
        "request_params": REQUEST_PARAMS,
        "request_body_sha256": sha(body),
        "raw_response_json": raw.decode(),
        "raw_response_sha256": sha(raw),
        **derived,
    }
    record["checkpoint_sha256"] = record_digest(record)
    return record


def request_one(
    prompt: dict[str, Any],
    token: str,
    identity_sha: str,
    entrypoint: str,
    *,
    url: str = URL,
    sanitizer: Callable[..., str],
) -> dict[str, Any]:
    body = json.dumps(
        request_payload(prompt), ensure_ascii=False, separators=(",", ":")
    ).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/my-other-github-account/banana-smasher",
        "X-Title": "Banana Smasher GLM-5.2 HumanEval",
    }
    last_error: Exception | None = None
    for attempt in range(1, 9):
        started = time.time()
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                raw = response.read()
                status = response.status
            return checkpoint_from_response(
                prompt,
                identity_sha,
                entrypoint,
                raw,
                status,
                time.time() - started,
                attempt,
                sanitizer=sanitizer,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            last_error = RunnerError(f"HTTP {exc.code}: {detail[:1000]}")
            if exc.code not in TRANSIENT_HTTP_CODES:
                raise last_error
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < 8:
            time.sleep(min(30.0, 1.5 * (2 ** (attempt - 1))) + random.random())
    raise RunnerError(f"exhausted retries for {prompt['task_id']}: {last_error}")


def validate_checkpoint(
    path: Path,
    prompt: dict[str, Any],
    identity_sha: str,
    entrypoint: str,
    sanitizer: Callable[..., str],
) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot load checkpoint {path}: {exc}") from exc
    raw = record.get("raw_response_json")
    if not isinstance(raw, str):
        raise RunnerError(f"{path.name} raw response is not a string")
    expected = checkpoint_from_response(
        prompt,
        identity_sha,
        entrypoint,
        raw.encode(),
        200,
        0.0,
        1,
        sanitizer=sanitizer,
    )
    if record == expected:
        return record
    legacy_transport_fields = {
        "http_status",
        "elapsed_seconds",
        "attempt",
        "committed_unix",
    }
    if set(record) == set(expected) | legacy_transport_fields:
        for field, wanted in expected.items():
            if field != "checkpoint_sha256" and record.get(field) != wanted:
                raise RunnerError(
                    f"{path.name} legacy raw response binding mismatch at {field}"
                )
        if record.get("http_status") != 200:
            raise RunnerError(f"{path.name} legacy HTTP status is not 200")
        attempt = record.get("attempt")
        elapsed = record.get("elapsed_seconds")
        committed = record.get("committed_unix")
        if not isinstance(attempt, int) or not 1 <= attempt <= 8:
            raise RunnerError(f"{path.name} invalid legacy attempt")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise RunnerError(f"{path.name} invalid legacy elapsed seconds")
        if not isinstance(committed, (int, float)) or not math.isfinite(committed):
            raise RunnerError(f"{path.name} invalid legacy commit time")
        if record_digest(record) != record.get("checkpoint_sha256"):
            raise RunnerError(f"{path.name} legacy checkpoint digest mismatch")
        fsync_replace(path, compact(expected))
        return expected
    if record != expected:
        differing = sorted(
            field
            for field in set(record) | set(expected)
            if record.get(field) != expected.get(field)
        )
        raise RunnerError(
            f"{path.name} is not an exact reconstruction from its raw response: {differing}"
        )
    return expected


def merge(
    root: Path,
    prompts: list[dict[str, Any]],
    identity_sha: str,
    dataset: Any,
    sanitizer: Callable[..., str],
) -> dict[str, Any]:
    records = [
        validate_checkpoint(
            root / "checkpoints" / f"task-{number:03d}.json",
            prompts[number],
            identity_sha,
            dataset[prompts[number]["task_id"]]["entry_point"],
            sanitizer,
        )
        for number in range(TASK_COUNT)
    ]
    canonical = b"".join(
        compact({"task_id": row["task_id"], "solution": row["solution"]})
        for row in records
    )
    fsync_replace(root / "canonical.jsonl", canonical)
    audit = b"".join(
        compact(
            {
                "task_id": row["task_id"],
                "messages_sha256": row["messages_sha256"],
                "request_body_sha256": row["request_body_sha256"],
                "response_id": row["response_id"],
                "raw_response_sha256": row["raw_response_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "model_returned": row["model_returned"],
                "provider_returned": row["provider_returned"],
                "finish_reason": row["finish_reason"],
                "semantic_empty_or_null": row["semantic_empty_or_null"],
                "usage": row["usage"],
                "reasoning_chars": row["reasoning_chars"],
                "message_content_chars": row["message_content_chars"],
            }
        )
        for row in records
    )
    fsync_replace(root / "request-audit.jsonl", audit)
    finish_reasons: dict[str, int] = {}
    for row in records:
        reason = str(row.get("finish_reason"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "status": "PASS",
        "rows": TASK_COUNT,
        "unique_task_ids": TASK_COUNT,
        "model": MODEL,
        "canonical_slug": CANONICAL_SLUG,
        "provider": PROVIDER,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "provider_tag": PROVIDER_TAG,
        "reasoning": "enabled, medium; accepted rows require nonempty reasoning",
        "max_completion_tokens_total": MAX_COMPLETION_TOKENS,
        "temperature": 0.0,
        "top_p": 0.95,
        "empty_or_null": sum(bool(row["semantic_empty_or_null"]) for row in records),
        "finish_reasons": finish_reasons,
        "total_completion_tokens": sum(
            int((row.get("usage") or {}).get("completion_tokens") or 0) for row in records
        ),
        "total_reasoning_tokens": sum(
            int(
                ((row.get("usage") or {}).get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                or 0
            )
            for row in records
        ),
        "total_openrouter_cost_usd": sum(
            float((row.get("usage") or {}).get("cost") or 0.0) for row in records
        ),
        "canonical_sha256": sha(canonical),
        "request_audit_sha256": sha(audit),
        "run_identity_sha256": identity_sha,
    }
    fsync_replace(root / "GENERATION_HANDOFF.json", compact(handoff))
    return handoff


def dry_run(root: Path, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    prompts_path = root / "PROMPT_MESSAGES.jsonl"
    prompt_file_sha = write_prompts(prompts_path, prompts)
    result = {
        "status": "DRY_RUN_PASS",
        "openrouter_requests_sent": 0,
        "tasks": len(prompts),
        "model": MODEL,
        "canonical_slug": CANONICAL_SLUG,
        "provider": PROVIDER,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "provider_tag": PROVIDER_TAG,
        "request_params": REQUEST_PARAMS,
        "frozen_task_messages_sha256": FROZEN_TASK_MESSAGES_SHA256,
        "generated_prompt_file_sha256": prompt_file_sha,
        "measured_source_prompt_ledger_sha256": MEASURED_PROMPT_LEDGER_SHA256,
        "next": "set OPENROUTER_API_KEY and re-run with --run to generate 164 paid tasks serially",
    }
    fsync_replace(root / "DRY_RUN.json", compact(result))
    return result


def run(root: Path, prompts: list[dict[str, Any]], dataset: Any, sanitizer: Callable[..., str], token: str) -> dict[str, Any]:
    prompts_path = root / "PROMPT_MESSAGES.jsonl"
    write_prompts(prompts_path, prompts)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    identity, identity_sha = prepare_identity(root, prompts_path, Path(__file__), token)

    for number, prompt in enumerate(prompts):
        path = root / "checkpoints" / f"task-{number:03d}.json"
        if path.exists():
            try:
                validate_checkpoint(
                    path,
                    prompt,
                    identity_sha,
                    dataset[prompt["task_id"]]["entry_point"],
                    sanitizer,
                )
                with _PRINT_LOCK:
                    print(f"SKIP_VALID {number:03d} accepted={number + 1}/{TASK_COUNT}", flush=True)
                continue
            except RunnerError as exc:
                quarantine = path.with_name(path.name + f".invalid.{int(time.time())}")
                os.replace(path, quarantine)
                print(f"QUARANTINE {number:03d}: {exc}", flush=True)
        record = request_one(
            prompt,
            token,
            identity_sha,
            dataset[prompt["task_id"]]["entry_point"],
            sanitizer=sanitizer,
        )
        fsync_replace(path, compact(record))
        validate_checkpoint(
            path,
            prompt,
            identity_sha,
            dataset[prompt["task_id"]]["entry_point"],
            sanitizer,
        )
        usage = record["usage"]
        print(
            f"COMMIT {number:03d} accepted={number + 1}/{TASK_COUNT} "
            f"completion={usage.get('completion_tokens')} "
            f"reasoning={((usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))} "
            f"finish={record['finish_reason']} empty={record['semantic_empty_or_null']} "
            f"cost={usage.get('cost')}",
            flush=True,
        )
    return merge(root, prompts, identity_sha, dataset, sanitizer)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "GLM-5.2 OpenRouter HumanEval producer. Default sends no paid OpenRouter requests; "
            "pass --run to generate all 164 tasks serially."
        )
    )
    result.add_argument("--root", type=Path, default=Path("work/humaneval/glm-5.2-openrouter"))
    result.add_argument("--token-file", type=Path)
    result.add_argument(
        "--run",
        action="store_true",
        help="generate 164 paid tasks serially, retrying only transient transport failures; otherwise validate prompts without calling OpenRouter",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        dataset, sanitizer = require_evalplus()
        prompts = build_prompts(dataset)
        args.root.mkdir(parents=True, exist_ok=True)
        if args.run:
            result = run(args.root, prompts, dataset, sanitizer, read_token(args.token_file))
        else:
            result = dry_run(args.root, prompts)
        print(json.dumps(result, indent=2, sort_keys=True))
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
