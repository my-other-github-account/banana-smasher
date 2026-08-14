from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from Evals.tools import openrouter_humaneval_glm52 as runner


class _Handler(BaseHTTPRequestHandler):
    request_body: dict[str, object] | None = None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).request_body = json.loads(self.rfile.read(length))
        response = {
            "id": "fixture-response",
            "model": runner.MODEL,
            "provider": runner.PROVIDER,
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"reasoning": "reasoned until the cap", "content": None},
                }
            ],
            "usage": {
                "completion_tokens": runner.MAX_COMPLETION_TOKENS,
                "completion_tokens_details": {
                    "reasoning_tokens": runner.MAX_COMPLETION_TOKENS
                },
                "cost": 0.01,
            },
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class GLM52OpenRouterRunnerTests(unittest.TestCase):
    def test_previous_public_script_identity_remains_resumable(self) -> None:
        endpoint = {
            "provider_name": runner.PROVIDER,
            "quantization": runner.PROVIDER_QUANTIZATION,
            "tag": runner.PROVIDER_TAG,
            "name": runner.CANONICAL_SLUG,
            "max_completion_tokens": runner.MAX_COMPLETION_TOKENS,
            "supported_parameters": [
                "reasoning",
                "reasoning_effort",
                "include_reasoning",
                "max_tokens",
                "temperature",
                "top_p",
            ],
        }
        endpoints = {"data": {"endpoints": [endpoint]}}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts_path = root / "PROMPT_MESSAGES.jsonl"
            script_path = root / "runner.py"
            prompts_path.write_bytes(b"prompts\n")
            script_path.write_bytes(b"new public script\n")
            identity = {
                "schema": runner.IDENTITY_SCHEMA,
                "model": runner.MODEL,
                "canonical_slug": runner.CANONICAL_SLUG,
                "provider": runner.PROVIDER,
                "provider_quantization": runner.PROVIDER_QUANTIZATION,
                "provider_tag": runner.PROVIDER_TAG,
                "frozen_task_messages_sha256": runner.FROZEN_TASK_MESSAGES_SHA256,
                "prompts_sha256": runner.sha(prompts_path.read_bytes()),
                "script_sha256": "d5d7afdc510a96d3e478dcfeac28d2d7a0d25c5c574b890b79bcb1a1e42a27ab",
                "request_params": runner.REQUEST_PARAMS,
                "status": "PASS",
            }
            identity_bytes = runner.compact(identity)
            (root / "RUN_IDENTITY.json").write_bytes(identity_bytes)
            with mock.patch.object(
                runner,
                "fetch_json",
                return_value=(runner.compact(endpoints), endpoints),
            ):
                resumed, identity_sha = runner.prepare_identity(
                    root,
                    prompts_path,
                    script_path,
                    "x",
                )
            self.assertEqual(resumed, identity)
            self.assertEqual(identity_sha, runner.sha(identity_bytes))
            self.assertEqual((root / "RUN_IDENTITY.json").read_bytes(), identity_bytes)

    def test_default_cli_path_never_enters_paid_runner(self) -> None:
        prompts = [{"task_id": f"HumanEval/{number}"} for number in range(164)]
        receipt = {"status": "DRY_RUN_PASS", "openrouter_requests_sent": 0}
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(runner, "require_evalplus", return_value=({}, object())),
                mock.patch.object(runner, "build_prompts", return_value=prompts),
                mock.patch.object(runner, "dry_run", return_value=receipt) as dry,
                mock.patch.object(
                    runner,
                    "run",
                    side_effect=AssertionError("paid runner entered without --run"),
                ) as paid,
            ):
                self.assertEqual(runner.main(["--root", temp_dir]), 0)
        dry.assert_called_once()
        paid.assert_not_called()

    def test_token_file_is_read_but_never_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_path = Path(temp_dir) / "token"
            token_path.write_text("secret-value\n", encoding="utf-8")
            self.assertEqual(runner.read_token(token_path), "secret-value")

    def test_request_contract_and_empty_visible_answer_are_preserved(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        messages = [{"role": "user", "content": "fixture prompt"}]
        prompt = {
            "task_id": "HumanEval/145",
            "messages": messages,
            "messages_sha256": runner.sha(runner.compact(messages)),
        }
        record = runner.request_one(
            prompt,
            token="x",
            identity_sha="a" * 64,
            entrypoint="fixture",
            url=f"http://127.0.0.1:{server.server_port}/chat/completions",
            sanitizer=lambda content, *, entrypoint: content,
        )

        body = _Handler.request_body
        self.assertIsNotNone(body)
        assert body is not None
        self.assertEqual(body["model"], "z-ai/glm-5.2")
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["n"], 1)
        self.assertEqual(body["max_tokens"], 16_384)
        self.assertEqual(
            body["reasoning"], {"enabled": True, "effort": "medium"}
        )
        self.assertTrue(body["include_reasoning"])
        self.assertEqual(
            body["provider"],
            {
                "only": ["Z.AI"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        self.assertEqual(record["finish_reason"], "length")
        self.assertEqual(record["message_content"], "")
        self.assertEqual(record["solution"], "")
        self.assertTrue(record["semantic_empty_or_null"])
        self.assertEqual(
            record["usage"]["completion_tokens"], runner.MAX_COMPLETION_TOKENS
        )
        self.assertEqual(record["checkpoint_sha256"], runner.record_digest(record))
        self.assertNotIn("authorization", json.dumps(record).lower())

    def test_malformed_success_response_is_not_retried(self) -> None:
        state: dict[str, int] = {"count": 0}

        class MalformedHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                state["count"] += 1
                payload = b"not-json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), MalformedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prompt = {
            "task_id": "HumanEval/0",
            "messages": [{"role": "user", "content": "fixture"}],
        }
        prompt["messages_sha256"] = runner.sha(runner.compact(prompt["messages"]))
        try:
            with self.assertRaisesRegex(runner.RunnerError, "malformed provider JSON"):
                runner.request_one(
                    prompt,
                    token="x",
                    identity_sha="a" * 64,
                    entrypoint="fixture",
                    url=f"http://127.0.0.1:{server.server_port}/chat/completions",
                    sanitizer=lambda content, *, entrypoint: content,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(state["count"], 1)

    def test_existing_valid_checkpoint_is_strictly_resumable(self) -> None:
        prompt = {
            "task_id": "HumanEval/0",
            "messages": [{"role": "user", "content": "fixture"}],
        }
        prompt["messages_sha256"] = runner.sha(runner.compact(prompt["messages"]))
        identity_sha = "b" * 64
        response = {
            "id": "fixture-response",
            "model": runner.MODEL,
            "provider": runner.PROVIDER,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"reasoning": "reasoning", "content": "def f():\n    pass"},
                }
            ],
            "usage": {
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0,
            },
        }
        record = runner.checkpoint_from_response(
            prompt,
            identity_sha,
            "f",
            json.dumps(response, separators=(",", ":")).encode(),
            200,
            1.0,
            1,
            sanitizer=lambda content, *, entrypoint: content,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task-000.json"
            path.write_bytes(runner.compact(record))
            self.assertEqual(
                runner.validate_checkpoint(
                    path,
                    prompt,
                    identity_sha,
                    "f",
                    lambda content, *, entrypoint: content,
                )["task_id"],
                "HumanEval/0",
            )

    def test_legacy_checkpoint_is_migrated_without_regeneration(self) -> None:
        prompt = {
            "task_id": "HumanEval/0",
            "messages": [{"role": "user", "content": "fixture"}],
        }
        prompt["messages_sha256"] = runner.sha(runner.compact(prompt["messages"]))
        identity_sha = "e" * 64
        response = {
            "id": "fixture-response",
            "model": runner.MODEL,
            "provider": runner.PROVIDER,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"reasoning": "reasoning", "content": "def f():\n    pass"},
                }
            ],
            "usage": {
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0,
            },
        }
        def sanitizer(content: str, *, entrypoint: str) -> str:
            return content
        expected = runner.checkpoint_from_response(
            prompt,
            identity_sha,
            "f",
            json.dumps(response, separators=(",", ":")).encode(),
            200,
            1.0,
            1,
            sanitizer=sanitizer,
        )
        legacy = json.loads(json.dumps(expected))
        legacy.update(
            {
                "http_status": 200,
                "elapsed_seconds": 1.0,
                "attempt": 1,
                "committed_unix": 1_700_000_000.0,
            }
        )
        legacy["checkpoint_sha256"] = runner.record_digest(legacy)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task-000.json"
            path.write_bytes(runner.compact(legacy))
            validated = runner.validate_checkpoint(
                path,
                prompt,
                identity_sha,
                "f",
                sanitizer,
            )
            self.assertEqual(validated, expected)
            self.assertEqual(json.loads(path.read_text()), expected)

    def test_checkpoint_validation_rejects_tampered_prompt_binding(self) -> None:
        prompt = {
            "task_id": "HumanEval/0",
            "messages": [{"role": "user", "content": "frozen prompt"}],
        }
        prompt["messages_sha256"] = runner.sha(runner.compact(prompt["messages"]))
        identity_sha = "c" * 64
        response = {
            "id": "fixture-response",
            "model": runner.MODEL,
            "provider": runner.PROVIDER,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"reasoning": "reasoning", "content": "def f():\n    pass"},
                }
            ],
            "usage": {
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0,
            },
        }
        record = runner.checkpoint_from_response(
            prompt,
            identity_sha,
            "f",
            json.dumps(response, separators=(",", ":")).encode(),
            200,
            1.0,
            1,
            sanitizer=lambda content, *, entrypoint: content,
        )
        record = json.loads(json.dumps(record))
        record["messages"][0]["content"] = "tampered prompt"
        record["checkpoint_sha256"] = runner.record_digest(record)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task-000.json"
            path.write_bytes(runner.compact(record))
            with self.assertRaisesRegex(runner.RunnerError, "exact reconstruction"):
                runner.validate_checkpoint(
                    path,
                    prompt,
                    identity_sha,
                    "f",
                    lambda content, *, entrypoint: content,
                )

    def test_checkpoint_validation_rejects_response_field_tampering(self) -> None:
        prompt = {
            "task_id": "HumanEval/0",
            "messages": [{"role": "user", "content": "frozen prompt"}],
        }
        prompt["messages_sha256"] = runner.sha(runner.compact(prompt["messages"]))
        identity_sha = "d" * 64
        response = {
            "id": "fixture-response",
            "model": runner.MODEL,
            "provider": runner.PROVIDER,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "reasoning": "reasoning",
                        "content": "def f():\n    pass",
                    },
                }
            ],
            "usage": {
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0,
            },
        }
        def sanitizer(content: str, *, entrypoint: str) -> str:
            return f"{entrypoint}:{content}"
        original = runner.checkpoint_from_response(
            prompt,
            identity_sha,
            "f",
            json.dumps(response, separators=(",", ":")).encode(),
            200,
            1.0,
            1,
            sanitizer=sanitizer,
        )
        mutations = {
            "solution": "tampered solution",
            "message_content": "tampered content",
            "message_content_chars": 999,
            "reasoning": "tampered reasoning",
            "reasoning_chars": 999,
            "finish_reason": "length",
            "semantic_empty_or_null": True,
            "usage": {
                "completion_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0,
            },
            "entry_point": "other",
            "response_id": "tampered-response",
            "http_status": 201,
            "attempt": 9,
            "elapsed_seconds": -1.0,
            "committed_unix": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task-000.json"
            for field, value in mutations.items():
                with self.subTest(field=field):
                    record = json.loads(json.dumps(original))
                    record[field] = value
                    record["checkpoint_sha256"] = runner.record_digest(record)
                    path.write_bytes(runner.compact(record))
                    with self.assertRaises(runner.RunnerError):
                        runner.validate_checkpoint(
                            path,
                            prompt,
                            identity_sha,
                            "f",
                            sanitizer,
                        )


if __name__ == "__main__":
    unittest.main()
