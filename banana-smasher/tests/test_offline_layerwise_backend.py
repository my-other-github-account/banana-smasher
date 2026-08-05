from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import banana_smasher
from banana_smasher.fixed_d4 import (
    produce_fixed_d4_layerwise_logits as public_layerwise,
)
from banana_smasher.hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime
from banana_smasher.offline_layerwise import produce_fixed_d4_layerwise_logits


BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


ADAPTER_SOURCE = '''
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np

class Runtime:
    API_VERSION = 1

    def __init__(self, *, model_root, parameters):
        self.root = Path(model_root)
        self.events = self.root / "events.jsonl"
        self._resident = 0
        self._peak = 0
        self._bytes = 0

    def _event(self, kind, **fields):
        with self.events.open("a") as handle:
            handle.write(json.dumps({"kind": kind, **fields}, sort_keys=True) + "\\n")

    @contextmanager
    def initial_stage(self):
        self._event("initial-enter")
        def embed(token_ids, *, window_id):
            self._bytes += len(token_ids) * 8
            return np.asarray(token_ids, dtype=np.float32)[:, None]
        yield embed
        self._event("initial-exit")

    @contextmanager
    def layer_stage(self, layer):
        if self._resident:
            raise RuntimeError("overlapping layers")
        self._resident = 100 + layer
        self._peak = max(self._peak, self._resident)
        self._bytes += 10
        self._event("layer-enter", layer=layer)
        def forward(activation, *, window_id):
            self._event("forward", layer=layer, window_id=window_id)
            return activation + (layer + 1)
        try:
            yield forward
        finally:
            self._resident = 0
            self._event("layer-exit", layer=layer)

    @contextmanager
    def terminal_stage(self):
        if self._resident:
            raise RuntimeError("layer still resident")
        self._resident = 50
        self._peak = max(self._peak, self._resident)
        self._event("terminal-enter")
        def score(activation, support_token_ids, *, window_id):
            self._event("score", window_id=window_id)
            logits = [[-float(pair[0]), -float(pair[1])] for pair in support_token_ids]
            top1 = [max(pair) for pair in support_token_ids]
            return {"logits": logits, "top1_token_ids": top1}
        try:
            yield score
        finally:
            self._resident = 0
            self._event("terminal-exit")

    def export_activation(self, activation):
        return np.asarray(activation)

    def import_activation(self, activation):
        return np.asarray(activation)

    def synchronize(self):
        self._event("synchronize")

    def resident_bytes(self):
        return self._resident

    def peak_resident_bytes(self):
        return self._peak

    def bytes_read(self):
        return self._bytes
'''


class OfflineLayerwiseBackendTests(unittest.TestCase):
    def test_deepseek_v4_d4_adapter_implements_runtime_api_v1(self) -> None:
        required = {
            "initial_stage",
            "layer_stage",
            "terminal_stage",
            "export_activation",
            "import_activation",
            "synchronize",
            "resident_bytes",
            "peak_resident_bytes",
            "bytes_read",
        }

        self.assertEqual(DeepseekV4D4Runtime.API_VERSION, 1)
        self.assertTrue(
            all(callable(getattr(DeepseekV4D4Runtime, name, None)) for name in required)
        )

    def test_deepseek_adapter_synchronizes_before_releasing_cuda_cache(self) -> None:
        events: list[str] = []
        runtime = DeepseekV4D4Runtime.__new__(DeepseekV4D4Runtime)
        runtime.torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache"))
        )
        runtime.synchronize = lambda: events.append("synchronize")
        runtime._resident_now = lambda: events.append("resident") or 0
        with patch(
            "banana_smasher.hf_deepseek_v4_d4_adapter.gc.collect",
            side_effect=lambda: events.append("collect"),
        ):
            runtime._release()

        self.assertEqual(
            events,
            ["collect", "synchronize", "empty-cache", "resident"],
        )

    def test_layerwise_backend_is_public_python_api(self) -> None:
        self.assertIs(
            banana_smasher.produce_fixed_d4_layerwise_logits,
            public_layerwise,
        )

    def test_public_fixed_d4_route_uses_builtin_layerwise_without_resident_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            config_path = root / "producer.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": "banana-smasher-candidate-producer-v1",
                        "producer": "fixed-d4-offline-layerwise",
                        "parameters": {},
                    }
                )
            )
            bank = root / "bank.jsonl"
            bank.write_text("{}\n")
            output = root / "candidate.jsonl"
            expected = {"status": "PASS", "execution_mode": "offline-layerwise"}
            with patch(
                "banana_smasher.offline_layerwise.produce_fixed_d4_layerwise_logits",
                return_value=expected,
            ) as layerwise:
                actual = public_layerwise(
                    model,
                    config_path,
                    bank,
                    output,
                    basis_sha256=BASIS,
                )
            self.assertEqual(actual, expected)
            layerwise.assert_called_once()

    def test_streams_one_layer_over_all_windows_then_scores_terminally(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            adapter_path = root / "adapter.py"
            adapter_path.write_text(ADAPTER_SOURCE)
            bank_path = root / "bank.jsonl"
            support_path = root / "support.jsonl"
            output_path = root / "candidate.jsonl"
            windows = list(range(64))
            positions = 4
            bank_path.write_text(
                "".join(
                    json.dumps({"window_id": window, "token_ids": [0, 1, 2, 3]}) + "\n"
                    for window in windows
                )
            )
            support_payload = "".join(
                json.dumps(
                    {
                        "window_id": window,
                        "support_token_ids": [[1, 2], [2, 3], [3, 4], [4, 5]],
                    }
                )
                + "\n"
                for window in windows
            ).encode()
            support_path.write_bytes(support_payload)
            config = {
                "schema": "banana-smasher-candidate-producer-v1",
                "producer": "fixed-d4-offline-layerwise",
                "parameters": {
                    "input_field": "token_ids",
                    "positions": positions,
                    "layers": [0, 1],
                    "teacher_support": {
                        "path": str(support_path),
                        "sha256": hashlib.sha256(support_payload).hexdigest(),
                        "field": "support_token_ids",
                    },
                    "execution_mode": "offline-layerwise",
                    "runtime_adapter": {
                        "path": str(adapter_path),
                        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                        "class": "Runtime",
                        "api_version": 1,
                    },
                    "physical_limits": {
                        "input_scope": "local-only",
                        "expected_read_bytes": 100,
                        "max_read_bytes": 4096,
                        "first_output_deadline_seconds": 300,
                        "max_elapsed_seconds": 3600,
                        "max_resident_bytes": 1000,
                    },
                },
            }
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model",
                return_value={"layers": [0, 1, 2]},
            ) as verify_model:
                receipt = produce_fixed_d4_layerwise_logits(
                    model,
                    config,
                    bank_path,
                    output_path,
                    basis_sha256=BASIS,
                )
            verify_model.assert_called_once_with(
                model.resolve(),
                basis_sha256=BASIS,
                verified_pack_receipt=None,
            )

            events = [json.loads(line) for line in (model / "events.jsonl").read_text().splitlines()]
            forwards = [event for event in events if event["kind"] == "forward"]
            self.assertEqual(
                [(event["layer"], event["window_id"]) for event in forwards],
                [(layer, window) for layer in (0, 1) for window in windows],
            )
            self.assertLess(events.index({"kind": "layer-exit", "layer": 1}), events.index({"kind": "terminal-enter"}))
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual([row["window_id"] for row in rows], windows)
            self.assertTrue(all(len(row["logits"]) == positions for row in rows))
            self.assertTrue(all(len(row["top1_token_ids"]) == positions for row in rows))
            self.assertEqual(receipt["execution_mode"], "offline-layerwise")
            self.assertEqual(receipt["layers_completed"], 2)
            self.assertEqual(receipt["configured_layer_count"], 2)
            self.assertEqual(receipt["manifest_layer_count"], 3)
            self.assertEqual(receipt["window_layer_forwards"], 128)
            self.assertEqual(receipt["output_rows"], 64)
            self.assertLessEqual(receipt["peak_resident_bytes"], 1000)
            self.assertNotIn("vllm", sys.modules)
            progress = json.loads(Path(receipt["progress_path"]).read_text())
            self.assertEqual(progress["stage"], "complete")
            self.assertEqual(progress["configured_layer_count"], 2)
            self.assertEqual(progress["manifest_layer_count"], 3)
            self.assertEqual(progress["output_rows"], 64)

            event_count = len(events)
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model",
                return_value={"layers": [0, 1, 2]},
            ):
                completed = produce_fixed_d4_layerwise_logits(
                    model,
                    config,
                    bank_path,
                    output_path,
                    basis_sha256=BASIS,
                )
            events_after_rerun = (model / "events.jsonl").read_text().splitlines()
            self.assertEqual(len(events_after_rerun), event_count)
            self.assertEqual(completed["window_layer_forwards"], 0)
            self.assertEqual(completed["resumed_output_rows"], 64)


if __name__ == "__main__":
    unittest.main()
