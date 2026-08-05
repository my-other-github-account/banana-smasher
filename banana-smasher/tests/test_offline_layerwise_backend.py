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
import torch
from banana_smasher.anchor_sidecars import (
    load_candidate_manifest,
    write_teacher_support_manifest,
)
from banana_smasher.fixed_d4 import (
    produce_fixed_d4_layerwise_logits as public_layerwise,
    rescore_fixed_d4_layerwise_terminal as public_terminal_rescore,
)
from banana_smasher.hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime
from banana_smasher.offline_layerwise import (
    produce_fixed_d4_layerwise_logits,
    rescore_fixed_d4_layerwise_terminal,
)


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
            support = np.asarray(support_token_ids)
            q_lp = -support.astype(np.float16)
            q_argmax = support.max(axis=1).astype(np.int32)
            return {"q_lp_at_ref": q_lp, "q_argmax": q_argmax}
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
        self.assertIs(
            banana_smasher.rescore_fixed_d4_layerwise_terminal,
            public_terminal_rescore,
        )

    def test_public_terminal_rescore_compatibly_forwards_window_id_field(self) -> None:
        expected = {"status": "PASS"}
        with patch(
            "banana_smasher.offline_layerwise.rescore_fixed_d4_layerwise_terminal",
            return_value=expected,
        ) as terminal:
            actual = public_terminal_rescore(
                "model",
                {},
                "bank.jsonl",
                "STATE.json",
                "teacher.json",
                "candidate.json",
                basis_sha256=BASIS,
                window_id_field="id_ds4",
            )

        self.assertEqual(actual, expected)
        self.assertEqual(terminal.call_args.kwargs["window_id_field"], "id_ds4")

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
                    "checkpoint_retention": "all",
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
                        "max_read_bytes": 1 << 20,
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

            run_root = Path(receipt["state_path"]).parent
            state = json.loads(Path(receipt["state_path"]).read_text())
            self.assertEqual(
                sorted(state["checkpoints"]),
                ["initial", "layer_0", "layer_1"],
            )
            self.assertEqual(receipt["checkpoint_stages_retained"], 3)
            self.assertEqual(receipt["checkpoint_files_retained"], 64 * 3)
            self.assertEqual(receipt["checkpoint_retention"], "all")
            for stage in ("initial", "layer_0", "layer_1"):
                self.assertEqual(len(list((run_root / stage).glob("window_*.npy"))), 64)

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

            frontier_config = json.loads(json.dumps(config))
            frontier_config["parameters"].pop("checkpoint_retention")
            frontier_output = root / "frontier-candidate.jsonl"
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model",
                return_value={"layers": [0, 1, 2]},
            ):
                frontier = produce_fixed_d4_layerwise_logits(
                    model,
                    frontier_config,
                    bank_path,
                    frontier_output,
                    basis_sha256=BASIS,
                )
            frontier_root = Path(frontier["state_path"]).parent
            frontier_state = json.loads(Path(frontier["state_path"]).read_text())
            self.assertEqual(frontier["checkpoint_retention"], "frontier")
            self.assertEqual(frontier["checkpoint_stages_retained"], 1)
            self.assertEqual(frontier["checkpoint_files_retained"], 64)
            self.assertEqual(sorted(frontier_state["checkpoints"]), ["layer_1"])
            self.assertFalse((frontier_root / "initial").exists())
            self.assertFalse((frontier_root / "layer_0").exists())
            self.assertEqual(
                len(list((frontier_root / "layer_1").glob("window_*.npy"))),
                64,
            )

    def test_width_8192_sidecars_emit_historical_candidate_shape_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            pack_manifest = model / "BANANA_PACK_MANIFEST.json"
            pack_manifest.write_text('{"model_id":"fixture-model"}\n')
            adapter_path = root / "adapter.py"
            adapter_path.write_text(ADAPTER_SOURCE)
            bank_path = root / "bank.jsonl"
            windows = list(range(64))
            bank_path.write_text(
                "".join(
                    json.dumps({"window_id": window, "token_ids": [0, 1]}) + "\n"
                    for window in windows
                )
            )
            bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
            support_width = 8192
            idx = torch.arange(support_width, dtype=torch.int32).repeat(3, 1)
            logprob = torch.log_softmax(
                torch.linspace(1.0, -1.0, support_width), dim=0
            ).to(torch.float16).repeat(3, 1)
            teacher_manifest = root / "teacher.json"
            write_teacher_support_manifest(
                teacher_manifest,
                windows=[
                    {"window_id": window, "idx": idx, "logprob": logprob}
                    for window in windows
                ],
                bank_sha256=bank_sha,
                teacher_sha256="a" * 64,
            )
            config = {
                "schema": "banana-smasher-candidate-producer-v1",
                "producer": "fixed-d4-offline-layerwise",
                "parameters": {
                    "input_field": "token_ids",
                    "positions": 2,
                    "layers": [0],
                    "checkpoint_retention": "all",
                    "teacher_support": {
                        "manifest": str(teacher_manifest),
                        "sha256": hashlib.sha256(teacher_manifest.read_bytes()).hexdigest(),
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
                        "max_read_bytes": 1 << 30,
                        "first_output_deadline_seconds": 300,
                        "max_elapsed_seconds": 3600,
                        "max_resident_bytes": 1000,
                    },
                },
            }
            output = root / "candidate.json"
            verified = {"layers": [0], "model_id": "fixture-model"}
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                first = produce_fixed_d4_layerwise_logits(
                    model, config, bank_path, output, basis_sha256=BASIS
                )

            teacher_link = root / "teacher-link.json"
            teacher_link.symlink_to(teacher_manifest)
            linked_config = json.loads(json.dumps(config))
            linked_config["parameters"]["teacher_support"]["manifest"] = str(
                teacher_link
            )
            linked_config_path = root / "teacher-link-producer.json"
            linked_config_path.write_text(
                json.dumps(linked_config, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(
                    ValueError, "teacher sidecar manifest must not be a symlink"
                ):
                    public_layerwise(
                        model,
                        linked_config_path,
                        bank_path,
                        root / "teacher-link-candidate.json",
                        basis_sha256=BASIS,
                    )

            output_link = root / "candidate-link.json"
            output_link.symlink_to(output)
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(
                    ValueError, "candidate sidecar manifest must not be a symlink"
                ):
                    public_layerwise(
                        model,
                        config,
                        bank_path,
                        output_link,
                        basis_sha256=BASIS,
                    )
            events_before = (model / "events.jsonl").read_bytes()
            manifest = load_candidate_manifest(output)
            first_sidecar = output.parent / manifest["windows"][0]["path"]
            candidate = torch.load(first_sidecar, weights_only=True)

            assert first["support_width"] == support_width
            assert manifest["support_width"] == support_width
            assert manifest["windows"][0]["tensors"]["q_lp_at_ref"]["shape"] == [
                2,
                support_width,
            ]
            assert candidate["q_lp_at_ref"].shape == (2, support_width)
            assert candidate["q_lp_at_ref"].dtype == torch.float16
            assert candidate["q_argmax"].shape == (2,)
            assert candidate["q_argmax"].dtype == torch.int32

            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                resumed = produce_fixed_d4_layerwise_logits(
                    model, config, bank_path, output, basis_sha256=BASIS
                )
            assert resumed["resumed_output_rows"] == 64
            assert (model / "events.jsonl").read_bytes() == events_before

            terminal_adapter_path = root / "terminal-adapter.py"
            terminal_adapter_path.write_text(ADAPTER_SOURCE + "\n# terminal-only revision\n")
            terminal_runtime_adapter = {
                "path": str(terminal_adapter_path),
                "sha256": hashlib.sha256(
                    terminal_adapter_path.read_bytes()
                ).hexdigest(),
                "class": "Runtime",
                "api_version": 1,
            }
            new_teacher_manifest = root / "new-teacher.json"
            write_teacher_support_manifest(
                new_teacher_manifest,
                windows=[
                    {"window_id": window, "idx": idx, "logprob": logprob}
                    for window in windows
                ],
                bank_sha256=bank_sha,
                teacher_sha256="b" * 64,
            )
            state_path = Path(first["state_path"])
            state_before = state_path.read_bytes()
            final_stage = state_path.parent / "layer_0"
            activations_before = {
                path.name: path.read_bytes() for path in final_stage.glob("*.npy")
            }
            forward_events_before = sum(
                json.loads(line)["kind"] == "forward"
                for line in (model / "events.jsonl").read_text().splitlines()
            )
            rescored_output = root / "rescored-candidate.json"

            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(
                    ValueError, "teacher sidecar manifest must not be a symlink"
                ):
                    public_terminal_rescore(
                        model,
                        linked_config_path,
                        bank_path,
                        state_path,
                        new_teacher_manifest,
                        root / "source-teacher-link-rescore.json",
                        basis_sha256=BASIS,
                        terminal_runtime_adapter=terminal_runtime_adapter,
                    )

            state = json.loads(state_before)
            source_support_sha = config["parameters"]["teacher_support"]["sha256"]
            config_sha = hashlib.sha256(
                (
                    json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
            ).hexdigest()
            legacy_binding_payload = {
                "basis_sha256": BASIS,
                "model_root": str(model.resolve()),
                "producer_config_sha256": config_sha,
                "bank_sha256": bank_sha,
                "teacher_support_sha256": source_support_sha,
                "runtime_adapter_sha256": config["parameters"]["runtime_adapter"][
                    "sha256"
                ],
                "layers": [0],
                "positions": 2,
            }
            legacy_binding = hashlib.sha256(
                json.dumps(
                    legacy_binding_payload, sort_keys=True, separators=(",", ":")
                ).encode()
                + b"\n"
            ).hexdigest()
            state["binding_sha256"] = "f" * 64
            state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(ValueError, "completed state identity mismatch"):
                    rescore_fixed_d4_layerwise_terminal(
                        model,
                        config,
                        bank_path,
                        state_path,
                        new_teacher_manifest,
                        rescored_output,
                        basis_sha256=BASIS,
                        terminal_runtime_adapter=terminal_runtime_adapter,
                    )
            state_path.write_bytes(state_before)

            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                rescored = rescore_fixed_d4_layerwise_terminal(
                    model,
                    config,
                    bank_path,
                    state_path,
                    new_teacher_manifest,
                    rescored_output,
                    basis_sha256=BASIS,
                    terminal_runtime_adapter=terminal_runtime_adapter,
                )

            new_teacher_link = root / "new-teacher-link.json"
            new_teacher_link.symlink_to(new_teacher_manifest)
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(
                    ValueError, "teacher sidecar manifest must not be a symlink"
                ):
                    public_terminal_rescore(
                        model,
                        config,
                        bank_path,
                        state_path,
                        new_teacher_link,
                        root / "teacher-link-rescore.json",
                        basis_sha256=BASIS,
                        terminal_runtime_adapter=terminal_runtime_adapter,
                    )

            rescored_output_link = root / "rescored-candidate-link.json"
            rescored_output_link.symlink_to(rescored_output)
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                with self.assertRaisesRegex(
                    ValueError, "candidate sidecar manifest must not be a symlink"
                ):
                    public_terminal_rescore(
                        model,
                        config,
                        bank_path,
                        state_path,
                        new_teacher_manifest,
                        rescored_output_link,
                        basis_sha256=BASIS,
                        terminal_runtime_adapter=terminal_runtime_adapter,
                    )

            forward_events_after = sum(
                json.loads(line)["kind"] == "forward"
                for line in (model / "events.jsonl").read_text().splitlines()
            )
            rescored_manifest = load_candidate_manifest(rescored_output)
            assert rescored["terminal_only"] is True
            assert rescored["window_layer_forwards"] == 0
            assert rescored["transformer_layer_forwards"] == 0
            assert rescored["classification"] == "backend-smoke-only"
            assert rescored["output_rows"] == 64
            assert rescored["source_runtime_adapter_sha256"] == config[
                "parameters"
            ]["runtime_adapter"]["sha256"]
            assert rescored["source_binding_schema"] == "support-width-v2"
            assert rescored["runtime_adapter_sha256"] == terminal_runtime_adapter[
                "sha256"
            ]
            assert rescored["windows"] == 64
            assert rescored["positions"] == 128
            assert isinstance(rescored["kld_sum"], float)
            assert isinstance(rescored["top1_matches"], int)
            assert Path(rescored["score"]["path"]).is_file()
            assert rescored["quality_rail"] == {
                "support_width": 8192,
                "position_cutoff": 1024,
                "kld_semantics": "support-renormalized",
                "top1_semantics": "full-vocabulary-argmax",
                "teacher_support_sidecar_manifest": rescored[
                    "teacher_support_sidecar_manifest"
                ],
                "candidate_output_sidecar_manifest": rescored[
                    "candidate_output_sidecar_manifest"
                ],
                "score": rescored["score"],
            }
            assert Path(
                rescored["teacher_support_sidecar_manifest"]["path"]
            ).is_file()
            assert Path(
                rescored["candidate_output_sidecar_manifest"]["path"]
            ).is_file()
            assert forward_events_after == forward_events_before
            assert state_path.read_bytes() == state_before
            assert {
                path.name: path.read_bytes() for path in final_stage.glob("*.npy")
            } == activations_before
            assert len(rescored_manifest["windows"]) == 64

            state = json.loads(state_before)
            state["binding_sha256"] = legacy_binding
            state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
            legacy_state = state_path.read_bytes()
            with patch(
                "banana_smasher.fixed_d4.verify_fixed_d4_model", return_value=verified
            ):
                legacy_rescore = rescore_fixed_d4_layerwise_terminal(
                    model,
                    config,
                    bank_path,
                    state_path,
                    new_teacher_manifest,
                    root / "legacy-rescored-candidate.json",
                    basis_sha256=BASIS,
                    terminal_runtime_adapter=terminal_runtime_adapter,
                )
            assert legacy_rescore["source_binding_schema"] == "legacy-producer-v1"
            assert legacy_rescore["transformer_layer_forwards"] == 0
            assert state_path.read_bytes() == legacy_state

if __name__ == "__main__":
    unittest.main()
