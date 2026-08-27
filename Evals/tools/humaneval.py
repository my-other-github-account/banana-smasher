from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SUITE_SCHEMA = "banana-smasher.humaneval-suite-lock.v1"
SUITE_NAME = "HUMANEVAL_0731_V1"
DEFAULT_SUITE_LOCK = Path(__file__).resolve().parents[1] / "configs" / "humaneval-0731-v1.json"
_TASK_ID = re.compile(r"^HumanEval/(0|[1-9][0-9]*)$")


class HumanEvalError(ValueError):
    """Raised when a HumanEval run violates the frozen contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanEvalError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HumanEvalError(f"{label} must be an array")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise HumanEvalError(f"{label} must be {expected!r}, got {actual!r}")


def load_suite_lock(path: str | Path = DEFAULT_SUITE_LOCK) -> dict[str, Any]:
    lock_path = Path(path)
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanEvalError(f"cannot load suite lock {lock_path}: {exc}") from exc
    lock = dict(_mapping(value, "suite lock"))

    _expect(lock.get("schema"), SUITE_SCHEMA, "suite lock schema")
    _expect(lock.get("name"), SUITE_NAME, "suite lock name")
    _expect(lock.get("model_family"), "DeepSeek-V4-Flash-0731", "model family")

    harness = _mapping(lock.get("harness"), "harness")
    _expect(harness.get("name"), "EvalPlus", "harness name")
    _expect(harness.get("version"), "0.4.0.dev44", "EvalPlus version")
    _expect(
        harness.get("git_commit"),
        "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "EvalPlus commit",
    )

    dataset = _mapping(lock.get("dataset"), "dataset")
    _expect(dataset.get("name"), "HumanEvalPlus", "dataset name")
    _expect(dataset.get("release"), "v0.1.10", "dataset release")
    _expect(dataset.get("hash"), "fe585eb4df8c88d844eeb463ea4d0302", "dataset hash")
    _expect(dataset.get("task_count"), 164, "task count")

    generation = _mapping(lock.get("generation"), "generation")
    frozen_generation = {
        "backend": "openai",
        "decode": "greedy",
        "samples_per_task": 1,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_completion_tokens": 4096,
        "prompt_tokens_counted": False,
        "scored_response_field": "message.content",
        "reasoning_response_field": "not-scored",
        "semantic_null_policy": "write-empty-and-fail",
        "client_concurrency": 1,
        "speculative_decoding": False,
    }
    for key, expected in frozen_generation.items():
        _expect(generation.get(key), expected, f"generation.{key}")

    ranges = [list(item) for item in _sequence(lock.get("shard_ranges"), "shard_ranges")]
    _expect(ranges, [[0, 41], [41, 82], [82, 123], [123, 164]], "shard_ranges")

    cap_shim = _mapping(lock.get("cap_shim"), "cap_shim")
    _expect(
        cap_shim.get("implementation"),
        "Evals.tools.humaneval.install_openai_cap_shim",
        "cap_shim.implementation",
    )
    _expect(
        cap_shim.get("historical_reference_sha256"),
        "459a84bb6c594b4e03ed74992c30eb4ddbe05bb04ba127a659f540d4fea99282",
        "cap_shim.historical_reference_sha256",
    )

    evaluation = _mapping(lock.get("evaluation"), "evaluation")
    _expect(evaluation.get("parallel_workers"), 8, "evaluation.parallel_workers")
    _expect(evaluation.get("minimum_time_limit_seconds"), 4.0, "minimum time limit")
    _expect(evaluation.get("ground_truth_time_limit_factor"), 4.0, "time factor")
    return lock


def _call_argument(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    position: int,
    name: str,
    default: Any = None,
) -> Any:
    return args[position] if len(args) > position else kwargs.get(name, default)


def _promote_reasoning_final_answer(response: Any) -> Any:
    """Expose a thinking-on response at EvalPlus' message.content boundary."""
    for choice in list(getattr(response, "choices", []) or []):
        message = getattr(choice, "message", None)
        if message is None or getattr(message, "content", None) is not None:
            continue
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning = getattr(message, "reasoning", None)
        final_answer = reasoning_content if isinstance(reasoning_content, str) else reasoning
        if isinstance(final_answer, str) and final_answer.strip():
            message.content = final_answer
    return response


def install_openai_final_answer_shim(request_module: Any) -> None:
    """Normalize thinking-on responses before downstream response auditing."""
    if getattr(request_module, "_banana_smasher_final_answer_shim", False):
        return
    original_make_request = request_module.make_request

    def make_request_with_final_answer(*args: Any, **kwargs: Any) -> Any:
        return _promote_reasoning_final_answer(original_make_request(*args, **kwargs))

    request_module.make_request = make_request_with_final_answer
    request_module._banana_smasher_final_answer_shim = True


def install_openai_request_audit(
    request_module: Any,
    cap: int,
    audit_path: str | Path,
) -> None:
    """Audit the last EvalPlus provider boundary before the OpenAI SDK request."""
    if getattr(request_module, "_banana_smasher_request_audit", False):
        return
    original_make_request = request_module.make_request
    active_audit_path = Path(audit_path)
    active_audit_path.parent.mkdir(parents=True, exist_ok=True)

    def make_request_with_audit(*args: Any, **kwargs: Any) -> Any:
        message = _call_argument(args, kwargs, 1, "message")
        model = _call_argument(args, kwargs, 2, "model")
        max_tokens = _call_argument(args, kwargs, 3, "max_tokens", 512)
        temperature = _call_argument(args, kwargs, 4, "temperature", 1)
        n = _call_argument(args, kwargs, 5, "n", 1)
        if not isinstance(message, str):
            raise HumanEvalError("provider message must be a string")
        _expect(max_tokens, cap, "provider max_completion_tokens")
        _expect(temperature, 0.0, "provider temperature")
        _expect(n, 1, "provider n")
        if not isinstance(model, str):
            raise HumanEvalError("provider model must be a string")
        messages = [{"role": "user", "content": message}]
        serialized_messages = json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "max_completion_tokens": max_tokens,
            "message_roles": ["user"],
            "messages_sha256": hashlib.sha256(serialized_messages).hexdigest(),
            "model": model,
            "n": n,
            "temperature": temperature,
            "top_p": 0.95,
        }
        with active_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return original_make_request(*args, **kwargs)

    request_module.make_request = make_request_with_audit
    request_module._banana_smasher_request_audit = True


def install_openai_cap_shim(
    codegen_module: Any,
    cap: int,
    *,
    request_audit: str | Path | None = None,
) -> None:
    """Bind the cap dropped by EvalPlus 26d6d00 and turn null content into an honest empty sample."""
    if getattr(codegen_module, "_banana_smasher_humaneval_shim", False):
        return
    original_make_model = codegen_module.make_model
    audit_path = Path(request_audit) if request_audit is not None else None
    if audit_path is not None:
        request_module = importlib.import_module("evalplus.gen.util.openai_request")
        install_openai_final_answer_shim(request_module)

    def make_model_with_true_cap(*args: Any, **kwargs: Any) -> Any:
        model = original_make_model(*args, **kwargs)
        if kwargs.get("backend") == "openai":
            model.max_new_tokens = cap
            original_batch = model._codegen_api_batch

            def codegen_batch_preserving_null_as_failure(*batch_args: Any, **batch_kwargs: Any) -> list[str]:
                outputs = original_batch(*batch_args, **batch_kwargs)
                return [item if isinstance(item, str) else "" for item in outputs]

            model._codegen_api_batch = codegen_batch_preserving_null_as_failure
            if audit_path is not None:
                batch_function = getattr(original_batch, "__func__", original_batch)
                batch_globals = getattr(batch_function, "__globals__", {})
                request_module = batch_globals.get("openai_request")
                if request_module is None or not hasattr(request_module, "make_request"):
                    raise HumanEvalError("cannot locate EvalPlus OpenAI request boundary")
                install_openai_request_audit(request_module, cap, audit_path)
            print(f"EFFECTIVE_DECODER_MAX_NEW_TOKENS={model.max_new_tokens}", flush=True)
        return model

    codegen_module.make_model = make_model_with_true_cap
    codegen_module._banana_smasher_humaneval_shim = True


def _evalplus_commit(distribution: importlib.metadata.Distribution) -> str | None:
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        value = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    vcs = value.get("vcs_info")
    return vcs.get("commit_id") if isinstance(vcs, Mapping) else None


def require_evalplus(lock: Mapping[str, Any]) -> Any:
    harness = _mapping(lock.get("harness"), "harness")
    try:
        distribution = importlib.metadata.distribution("evalplus")
        installed_version = distribution.version
        import evalplus.codegen as codegen_module
        import evalplus.config as evalplus_config
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        raise HumanEvalError(
            "EvalPlus is not installed; install Evals/requirements-humaneval.txt in an isolated environment"
        ) from exc

    _expect(installed_version, harness["version"], "installed EvalPlus version")
    _expect(_evalplus_commit(distribution), harness["git_commit"], "installed EvalPlus commit")
    _expect(evalplus_config.DEFAULT_MIN_TIME_LIMIT, 4.0, "EvalPlus minimum time limit")
    _expect(
        evalplus_config.DEFAULT_GT_TIME_LIMIT_FACTOR,
        4.0,
        "EvalPlus ground-truth time-limit factor",
    )
    return codegen_module


def audit_provider_requests(
    path: str | Path,
    *,
    cap: int,
    expected_unique_prompts: int,
    model: str,
) -> dict[str, Any]:
    audit_path = Path(path)
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HumanEvalError(f"cannot read provider request audit {audit_path}: {exc}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            record = _mapping(json.loads(line), f"{audit_path}:{line_number}")
        except json.JSONDecodeError as exc:
            raise HumanEvalError(f"{audit_path}:{line_number}: invalid JSON: {exc}") from exc
        _expect(record.get("max_completion_tokens"), cap, "provider max_completion_tokens")
        _expect(record.get("temperature"), 0.0, "provider temperature")
        _expect(record.get("top_p"), 0.95, "provider top_p")
        _expect(record.get("n"), 1, "provider n")
        _expect(record.get("message_roles"), ["user"], "provider message roles")
        _expect(record.get("model"), model, "provider model")
        message_hash = record.get("messages_sha256")
        if not isinstance(message_hash, str) or re.fullmatch(r"[0-9a-f]{64}", message_hash) is None:
            raise HumanEvalError(f"{audit_path}:{line_number}: invalid messages_sha256")
        records.append(record)
    unique_prompts = {str(record["messages_sha256"]) for record in records}
    _expect(len(unique_prompts), expected_unique_prompts, "unique audited provider prompts")
    return {
        "attempts": len(records),
        "max_completion_tokens": cap,
        "path": str(audit_path),
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "unique_prompts": len(unique_prompts),
    }


def _read_samples(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise HumanEvalError(f"cannot read sample file {path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HumanEvalError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row = dict(_mapping(row, f"{path}:{line_number}"))
            if set(row) != {"task_id", "solution"}:
                raise HumanEvalError(
                    f"{path}:{line_number}: expected exactly task_id and solution"
                )
            task_id = row.get("task_id")
            solution = row.get("solution")
            if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
                raise HumanEvalError(f"{path}:{line_number}: invalid task_id")
            if not isinstance(solution, str):
                raise HumanEvalError(f"{path}:{line_number}: solution must be a string")
            rows.append({"task_id": task_id, "solution": solution})
    return rows


def _task_number(task_id: str) -> int:
    match = _TASK_ID.fullmatch(task_id)
    if match is None:
        raise HumanEvalError(f"invalid task_id: {task_id}")
    return int(match.group(1))


def _audit_rows(rows: Sequence[Mapping[str, Any]], expected_ids: set[int]) -> dict[str, Any]:
    seen: set[int] = set()
    empty = 0
    syntax_ok = 0
    fence_free = 0
    for row in rows:
        task_id = str(row["task_id"])
        number = _task_number(task_id)
        if number in seen:
            raise HumanEvalError(f"duplicate task_id: {task_id}")
        seen.add(number)
        solution = str(row["solution"])
        if not solution.strip():
            empty += 1
        try:
            ast.parse(solution)
        except SyntaxError:
            pass
        else:
            syntax_ok += 1
        if "```" not in solution:
            fence_free += 1

    missing = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if missing or unexpected:
        raise HumanEvalError(f"task population mismatch: missing={missing} unexpected={unexpected}")
    return {
        "rows": len(rows),
        "unique_task_ids": len(seen),
        "empty_solutions": empty,
        "syntax_ok": syntax_ok,
        "fence_free": fence_free,
        "first_task_id": f"HumanEval/{min(seen)}" if seen else None,
        "last_task_id": f"HumanEval/{max(seen)}" if seen else None,
    }


def audit_samples(path: str | Path, *, low: int = 0, high: int = 164) -> dict[str, Any]:
    sample_path = Path(path)
    rows = _read_samples([sample_path])
    audit = _audit_rows(rows, set(range(low, high)))
    audit["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    audit["path"] = str(sample_path)
    return audit


def merge_samples(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    task_count: int = 164,
) -> dict[str, Any]:
    rows = _read_samples(paths)
    _audit_rows(rows, set(range(task_count)))
    ordered = sorted(rows, key=lambda row: _task_number(str(row["task_id"])))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in ordered
    )
    output_path.write_text(payload, encoding="utf-8")
    audit = audit_samples(output_path, low=0, high=task_count)
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def generate_shard(args: argparse.Namespace, lock: Mapping[str, Any]) -> dict[str, Any]:
    generation = _mapping(lock["generation"], "generation")
    low, high = args.id_range
    frozen_ranges = {tuple(item) for item in lock["shard_ranges"]}
    if (low, high) not in frozen_ranges:
        raise HumanEvalError(f"id range {(low, high)} is not a frozen shard range")
    if "OPENAI_API_KEY" not in os.environ:
        raise HumanEvalError("OPENAI_API_KEY must be set; any nonempty value works for keyless local endpoints")

    codegen_module = require_evalplus(lock)
    cap = int(generation["max_completion_tokens"])
    request_audit = args.root / "request-audit.jsonl"
    install_openai_cap_shim(codegen_module, cap, request_audit=request_audit)
    effective = {
        "backend": "openai",
        "base_url": args.base_url,
        "dataset": "humaneval",
        "greedy": True,
        "id_range": [low, high],
        "max_new_tokens": cap,
        "model": args.model,
    }
    print("EFFECTIVE_CONFIG=" + json.dumps(effective, sort_keys=True), flush=True)
    target = codegen_module.run_codegen(root=str(args.root), **effective)
    audit = audit_samples(target, low=low, high=high)
    audit["provider_request_audit"] = audit_provider_requests(
        request_audit,
        cap=cap,
        expected_unique_prompts=high - low,
        model=args.model,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def score_samples(args: argparse.Namespace, lock: Mapping[str, Any]) -> None:
    task_count = int(_mapping(lock["dataset"], "dataset")["task_count"])
    audit_samples(args.samples, low=0, high=task_count)
    require_evalplus(lock)
    from evalplus.evaluate import evaluate

    evaluate(
        dataset="humaneval",
        samples=str(args.samples),
        parallel=int(_mapping(lock["evaluation"], "evaluation")["parallel_workers"]),
        i_just_wanna_run=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen HumanEval 0731 generation and scoring tools")
    parser.add_argument("--suite-lock", type=Path, default=DEFAULT_SUITE_LOCK)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-config", help="print and validate the frozen suite lock")

    generate = subparsers.add_parser("generate", help="generate one frozen disjoint shard")
    generate.add_argument("--model", required=True)
    generate.add_argument("--base-url", required=True)
    generate.add_argument("--root", type=Path, required=True)
    generate.add_argument("--id-range", nargs=2, type=int, required=True, metavar=("LOW", "HIGH"))

    audit = subparsers.add_parser("audit", help="validate one complete 164-task JSONL")
    audit.add_argument("samples", type=Path)

    merge = subparsers.add_parser("merge", help="merge disjoint JSONL shards")
    merge.add_argument("shards", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser("score", help="score a complete JSONL inside the pinned sandbox")
    score.add_argument("samples", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        lock = load_suite_lock(args.suite_lock)
        if args.command == "show-config":
            print(json.dumps(lock, indent=2, sort_keys=True))
        elif args.command == "generate":
            generate_shard(args, lock)
        elif args.command == "audit":
            task_count = int(_mapping(lock["dataset"], "dataset")["task_count"])
            print(json.dumps(audit_samples(args.samples, high=task_count), indent=2, sort_keys=True))
        elif args.command == "merge":
            task_count = int(_mapping(lock["dataset"], "dataset")["task_count"])
            print(
                json.dumps(
                    merge_samples(args.shards, args.output, task_count=task_count),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "score":
            score_samples(args, lock)
        else:
            parser.error(f"unsupported command: {args.command}")
    except HumanEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
