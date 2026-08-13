from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"not-json")

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
                runner.validate_checkpoint(path, prompt, identity_sha)["task_id"],
                "HumanEval/0",
            )

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
            with self.assertRaisesRegex(runner.RunnerError, "messages binding mismatch"):
                runner.validate_checkpoint(path, prompt, identity_sha)


if __name__ == "__main__":
    unittest.main()
