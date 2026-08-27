from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from Evals.tools import humaneval


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "Evals" / "configs" / "humaneval-0731-v1.json"


class HumanEvalToolingTests(unittest.TestCase):
    def test_suite_lock_freezes_true_4096_completion_cap(self) -> None:
        lock = humaneval.load_suite_lock(LOCK)

        self.assertEqual(lock["name"], "HUMANEVAL_0731_V1")
        self.assertEqual(lock["generation"]["max_completion_tokens"], 4096)
        self.assertFalse(lock["generation"]["prompt_tokens_counted"])
        self.assertEqual(lock["generation"]["samples_per_task"], 1)
        self.assertEqual(lock["generation"]["temperature"], 0.0)

    def test_cap_shim_overrides_silent_768_and_preserves_null_as_failure(self) -> None:
        class FakeDecoder:
            def __init__(self) -> None:
                self.max_new_tokens = 768

            def _codegen_api_batch(self, prompt: str, batch_size: int) -> list[str | None]:
                return [None]

        module = types.SimpleNamespace(make_model=lambda *args, **kwargs: FakeDecoder())

        humaneval.install_openai_cap_shim(module, 4096)
        decoder = module.make_model(backend="openai", max_new_tokens=4096)

        self.assertEqual(decoder.max_new_tokens, 4096)
        self.assertEqual(decoder._codegen_api_batch("prompt", 1), [""])

    def test_request_audit_records_actual_provider_payload_without_prompt_text(self) -> None:
        request_module = types.SimpleNamespace(
            make_request=lambda *args, **kwargs: "ok"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "requests.jsonl"
            humaneval.install_openai_request_audit(request_module, 4096, audit_path)
            result = request_module.make_request(
                object(),
                message="secret prompt text",
                model="mock",
                max_tokens=4096,
                temperature=0.0,
                n=1,
            )

            self.assertEqual(result, "ok")
            record = json.loads(audit_path.read_text())
            self.assertEqual(record["max_completion_tokens"], 4096)
            self.assertEqual(record["message_roles"], ["user"])
            self.assertNotIn("secret prompt text", audit_path.read_text())
            self.assertEqual(len(record["messages_sha256"]), 64)

    def test_final_answer_shim_promotes_reasoning_field_when_content_is_null(self) -> None:
        message = types.SimpleNamespace(content=None, reasoning="def answer():\n    return 42")
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )
        request_module = types.SimpleNamespace(
            make_request=lambda *args, **kwargs: response
        )

        humaneval.install_openai_final_answer_shim(request_module)
        result = request_module.make_request(
            object(),
            message="prompt",
            model="mock",
            max_tokens=4096,
            temperature=0.0,
            n=1,
        )

        self.assertIs(result, response)
        self.assertEqual(message.content, "def answer():\n    return 42")

    def test_final_answer_shim_preserves_existing_final_content(self) -> None:
        message = types.SimpleNamespace(content="final", reasoning="thinking")
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )
        request_module = types.SimpleNamespace(
            make_request=lambda *args, **kwargs: response
        )

        humaneval.install_openai_final_answer_shim(request_module)
        request_module.make_request(
            object(),
            message="prompt",
            model="mock",
            max_tokens=4096,
            temperature=0.0,
            n=1,
        )

        self.assertEqual(message.content, "final")

    def test_merge_writes_exactly_one_sorted_sample_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards: list[Path] = []
            for shard_index, (low, high) in enumerate(
                ((0, 41), (41, 82), (82, 123), (123, 164))
            ):
                path = root / f"shard-{shard_index}.jsonl"
                rows = [
                    {"task_id": f"HumanEval/{index}", "solution": f"def f():\n    return {index}"}
                    for index in range(low, high)
                ]
                path.write_text(
                    "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8",
                )
                shards.append(path)

            output = root / "merged.jsonl"
            audit = humaneval.merge_samples(shards, output, task_count=164)

            self.assertEqual(audit["rows"], 164)
            self.assertEqual(audit["unique_task_ids"], 164)
            self.assertEqual(audit["empty_solutions"], 0)
            self.assertEqual(audit["syntax_ok"], 164)
            merged = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(merged[0]["task_id"], "HumanEval/0")
            self.assertEqual(merged[-1]["task_id"], "HumanEval/163")

    def test_merge_rejects_duplicate_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text('{"task_id":"HumanEval/0","solution":"pass"}\n')
            second.write_text('{"task_id":"HumanEval/0","solution":"pass"}\n')

            with self.assertRaisesRegex(humaneval.HumanEvalError, "duplicate task_id"):
                humaneval.merge_samples([first, second], root / "merged.jsonl", task_count=1)


if __name__ == "__main__":
    unittest.main()
