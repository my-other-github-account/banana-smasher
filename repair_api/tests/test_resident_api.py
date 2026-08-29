from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from repair_api import ArtifactError, ResidentRepairAPI, adapt_checkpointed_envelope

WINDOWS = [28, 56, 68, 71, 76, 99, 107, 122, 124, 130, 141, 156, 160, 171, 180, 183, 185, 186, 196, 210, 212, 213, 218, 228, 232, 235, 249, 270, 272, 273, 283, 288, 290, 295, 297, 306, 307, 309, 311, 328, 331, 357, 362, 365, 368, 374, 376, 380, 384, 385, 391, 396, 413, 429, 430, 437, 442, 447, 454, 462, 464, 475, 489, 499]


class ResidentApiTests(unittest.TestCase):
    def test_checkpointed_envelope_admits_canonical_persisted_state_without_mutation(self) -> None:
        state = {"luts": object(), "norms": object(), "outputs": object()}
        payload = {
            "schema": "resident-continuation-checkpoint-v1",
            "next_update": 24,
            "state": state,
            "identity": {"checkpoint_loaded": True, "next_update": 24},
        }

        admitted = adapt_checkpointed_envelope(payload)

        self.assertEqual(admitted["format"], "banana-smasher-qtip2-v7-joint-checkpoint-v1")
        self.assertIs(admitted["state"], state)
        self.assertNotIn("format", payload)

    def make_artifact(self, root: Path, *, partial: bool = False, mismatch: bool = False, teacher_rows: int = 1024) -> None:
        (root / "checkpoints").mkdir()
        (root / "score" / "teacher").mkdir(parents=True)
        (root / "score" / "candidates" / "UPDATE_016").mkdir(parents=True)
        (root / "score" / "candidates" / "UPDATE_064").mkdir(parents=True)
        teacher_path = root / "checkpoints" / "teacher.pt"
        teacher_path.write_bytes(b"teacher")
        entries = {}
        for key, update, parent in (("UPDATE_000_MIDPOINT", 0, ""), ("UPDATE_016", 16, "parent-0"), ("UPDATE_064", 64, "parent-16")):
            checkpoint = root / "checkpoints" / f"{key}.pt"
            checkpoint.write_bytes(key.encode())
            (root / "score" / "candidates" / key).mkdir(parents=True, exist_ok=True)
            entries[key] = {
                "path": str(checkpoint.relative_to(root)),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "identity_sha256": ("16" if key.endswith("016") else "64") * 32,
                "parent_sha256": ("bad" if mismatch and key.endswith("064") else parent),
                "next_update": update,
            }
        entries["UPDATE_016"]["parent_sha256"] = entries["UPDATE_000_MIDPOINT"]["sha256"]
        entries["UPDATE_064"]["parent_sha256"] = (
            "bad" if mismatch else entries["UPDATE_016"]["sha256"]
        )
        teacher = {
            "idx": np.zeros((teacher_rows, 8192), dtype=np.int32),
            "logprob": np.zeros((teacher_rows, 8192), dtype=np.float32),
        }
        candidate_rows = 1023 if partial else 1024
        candidate = {
            "q_lp_at_ref": np.zeros((candidate_rows, 8192), dtype=np.float32),
            "q_argmax": np.zeros(candidate_rows, dtype=np.int32),
        }
        np.savez(root / "teacher.npz", **teacher)
        np.savez(root / "candidate.npz", **candidate)
        (root / "score" / "teacher" / "t8192_win28.pt").touch()
        for key in entries:
            (root / "score" / "candidates" / key / "q8192_win28.pt").touch()
        (root / "ARTIFACT.json").write_text(json.dumps({
            "schema": "repair-artifact-v1",
            "artifact_id": "resident-api-test",
            "identity": {
                "basis_sha256": "b" * 64,
                "builder_eval_corpus_sha256": "c" * 64,
                "train_score_corpus_sha256": "d" * 64,
                "teacher_inventory": ["teacher-v1"],
            },
            "checkpoints": entries,
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "score/teacher",
                "candidate_dir_template": "score/candidates/{checkpoint}",
                "window_ids": WINDOWS,
                "positions_per_window": 1024,
                "support": 8192,
            },
        }))

    @staticmethod
    def loader(path: Path):
        if path.parent.name == "teacher":
            with np.load(path.parent.parent.parent / "teacher.npz") as value:
                return {name: value[name] for name in value.files}
        with np.load(path.parent.parent.parent.parent / "candidate.npz") as value:
            return {name: value[name] for name in value.files}

    def test_resident_facade_caches_rows_and_emits_identity_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            calls = []

            def counting_loader(path: Path):
                calls.append(path)
                return self.loader(path)

            api = ResidentRepairAPI.open(root, loader=counting_loader)
            first = api.score("UPDATE_016", windows=[28])
            second = api.score("UPDATE_016", windows=[28])
            self.assertEqual(len(calls), 2)
            self.assertEqual(first.as_dict()["execution_mode"], "resident_in_memory")
            self.assertEqual(first.as_dict()["identity"]["basis_sha256"], "b" * 64)
            self.assertEqual(first.as_dict()["runtime_counters"]["file_reads_during_timed_score"], 0)
            self.assertEqual(first.as_dict()["kld_mean"], second.as_dict()["kld_mean"])

    def test_resident_accepts_real_root_teacher_padding_beyond_scored_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root, teacher_rows=2048)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            result = api.score("UPDATE_016", windows=[28])
            self.assertEqual(result.positions, 1024)

    def test_resume_compare_requires_checkpoint_pair_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root, mismatch=True)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaises(ArtifactError):
                api.resume_compare("UPDATE_016", "UPDATE_064", windows=[28])

    def test_resume_compare_returns_bound_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            result = api.resume_compare("UPDATE_064", "UPDATE_016", windows=[28])
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["pair_binding"], "checkpoint-parent-and-shared-scientific-identity")
            self.assertEqual(result["resume"]["positions"], 1024)
            self.assertEqual(result["scratch"]["positions"], 1024)

    def test_continue_to_selects_u64_milestone_after_u16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            result = api.continue_to("UPDATE_016", "U64", windows=[28])
            self.assertEqual(result["start_checkpoint"], "UPDATE_016")
            self.assertEqual(result["target_checkpoint"], "UPDATE_064")
            self.assertEqual(result["milestones"], ["UPDATE_064"])
            self.assertEqual(result["score"]["positions"], 1024)

    def test_checkpoint_sha_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            checkpoint = root / "checkpoints" / "UPDATE_016.pt"
            checkpoint.write_bytes(b"tampered")
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaises(ArtifactError):
                api.score("UPDATE_016", windows=[28])

    def test_missing_scientific_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            del manifest["identity"]["train_score_corpus_sha256"]
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaises(ArtifactError):
                api.score("UPDATE_016", windows=[28])

    def test_teacher_inventory_is_derived_for_legacy_real_root_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            del manifest["identity"]["teacher_inventory"]
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, loader=self.loader)
            result = api.score("UPDATE_016", windows=[28])
            inventory = result.as_dict()["identity"]["teacher_inventory"]
            self.assertEqual(inventory["schema"], "teacher-file-inventory-v1")
            self.assertEqual(inventory["file_count"], 1)
            self.assertEqual(len(inventory["sha256"]), 64)

    def test_partial_resident_artifact_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root, partial=True)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaises(ArtifactError):
                api.score("UPDATE_016", windows=[28])

    def test_true_clean_u0_replay_runs_exact_updates_and_emits_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            class Model:
                def __init__(self):
                    self.value = 0
                def state_dict(self):
                    return {"value": self.value}
            class Optimizer:
                def __init__(self, model):
                    self.model = model
                    self.steps = 0
                def state_dict(self):
                    return {"steps": self.steps}
            class Scheduler:
                def __init__(self, optimizer):
                    self.optimizer = optimizer
                    self.epochs = 0
                def state_dict(self):
                    return {"epochs": self.epochs}
            expected = {
                "model": {"value": 16},
                "optimizer": {"steps": 16},
                "scheduler": {"epochs": 16},
            }
            import torch
            target_path = root / "checkpoints" / "UPDATE_016.pt"
            torch.save({"state": expected}, target_path)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            manifest["checkpoints"]["UPDATE_016"]["sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, loader=self.loader)
            receipt = root / "receipts" / "CLEAN_U0_REPLAY.json"
            result = api.construct_from_clean_u0(
                "UPDATE_000_MIDPOINT", "UPDATE_016", receipt_path=receipt,
                replay={
                    "model_factory": Model,
                    "optimizer_factory": Optimizer,
                    "scheduler_factory": Scheduler,
                    "update_fn": lambda model, optimizer, scheduler, update: (
                        setattr(model, "value", model.value + 1),
                        setattr(optimizer, "steps", optimizer.steps + 1),
                        setattr(scheduler, "epochs", scheduler.epochs + 1),
                    ),
                    "geometry": {"layers": 1, "hidden": 4},
                    "basis_sha256": "b" * 64,
                    "corpus_sha256": "c" * 64,
                    "seed": 1701,
                },
            )
            self.assertEqual(result["update_count"], 16)
            self.assertEqual(result["updates"], {"requested": 16, "executed": 16})
            self.assertFalse(result["checkpoint_loaded"])
            self.assertEqual(result["state_sha256"], result["target_state_sha256"])
            self.assertEqual(json.loads(receipt.read_text())["status"], "PASS")

    def test_resume_equivalence_is_u0_to_u4_with_one_u2_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            import torch

            u0 = root / "checkpoints" / "UPDATE_000_MIDPOINT.pt"
            torch.save({
                "state": {"value": torch.tensor([0.0])},
                "optimizer_state": {"state": {}, "param_groups": []},
                "scheduler_state": {},
            }, u0)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            manifest["checkpoints"]["UPDATE_000_MIDPOINT"]["sha256"] = hashlib.sha256(u0.read_bytes()).hexdigest()
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, loader=self.loader)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.value = torch.nn.Parameter(torch.tensor([0.0]))
                    self.checkpoint_loaded = False
                    self.resident_ready = True

            def update(model, optimizer, scheduler, update):
                model.value.grad = torch.tensor([float(update)])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                return {
                    "loss": float(update),
                    "optimizer_steps": 1,
                    "scheduler_steps": 1,
                }

            receipt = root / "receipts" / "RESUME_EQUIVALENCE.json"
            midpoint = root / "checkpoints" / "RESUME_MIDPOINT_U2.pt"
            result = api.resume_equivalence(
                "UPDATE_000_MIDPOINT",
                replay={
                    "model_factory": Model,
                    "optimizer_factory": lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
                    "scheduler_factory": lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                    "update_fn": update,
                    "geometry": {"layers": 43},
                    "basis_sha256": "b" * 64,
                    "corpus_sha256": "c" * 64,
                    "seed": 1701,
                },
                total_updates=4,
                midpoint_update=2,
                midpoint_checkpoint_path=midpoint,
                receipt_path=receipt,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["total_updates"], 4)
            self.assertEqual(result["midpoint_update"], 2)
            self.assertTrue(result["midpoint"]["reload_verified"])
            self.assertTrue(midpoint.is_file())
            self.assertEqual(hashlib.sha256(midpoint.read_bytes()).hexdigest(), result["midpoint"]["sha256"])
            self.assertEqual(result["midpoint"]["pre_save_fingerprint"], result["midpoint"]["post_reload_fingerprint"])
            self.assertTrue(result["midpoint"]["resident_byte_comparison"]["equal"])
            self.assertEqual(result["midpoint"]["resident_byte_comparison"]["first_mismatch"], None)
            self.assertEqual(result["runtime_counters"]["checkpoint_boundary_serializations"], 1)
            self.assertEqual(result["runtime_counters"]["checkpoint_boundary_reloads"], 1)
            self.assertTrue(result["terminal"]["bitwise_equal"])
            self.assertIsNone(result["first_divergence_update"])
            self.assertEqual([row["update"] for row in result["arms"]["resume"]], [1, 2, 3, 4])
            self.assertEqual(json.loads(receipt.read_text())["schema"], "resident-api-resume-equivalence-v1")

    def test_resume_equivalence_rejects_vacuous_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaisesRegex(ArtifactError, "at least two post-reload"):
                api.resume_equivalence(
                    "UPDATE_000_MIDPOINT", replay={}, total_updates=4,
                    midpoint_update=3, receipt_path=root / "bad.json",
                )

    def test_clean_u0_replay_rejects_raw_command_and_missing_true_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, loader=self.loader)
            with self.assertRaisesRegex(ArtifactError, "raw command"):
                api.construct_from_clean_u0(
                    "UPDATE_000_MIDPOINT", "UPDATE_016",
                    replay={"command": ["python", "fast_two_node_v7_continuous.py"]},
                )
            with self.assertRaisesRegex(ArtifactError, "model_factory"):
                api.construct_from_clean_u0(
                    "UPDATE_000_MIDPOINT", "UPDATE_016",
                    replay={"geometry": {"layers": 1}, "basis_sha256": "b" * 64,
                            "corpus_sha256": "c" * 64, "seed": 1701},
                )

    def test_two_spark_continuation_requires_explicit_resident_two_rank_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            # The continuation API must create durable U20/U32/U48/U64
            # artifacts; remove the legacy byte fixtures for this test.
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            manifest["checkpoints"].pop("UPDATE_064")
            (root / "checkpoints" / "UPDATE_064.pt").unlink()
            for child in (root / "score" / "candidates" / "UPDATE_064").iterdir():
                child.unlink()
            (root / "score" / "candidates" / "UPDATE_064").rmdir()
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            import torch
            checkpoint_path = root / "checkpoints" / "UPDATE_016.pt"
            torch.save({"state": {"value": torch.tensor([16.0])}}, checkpoint_path)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            manifest["checkpoints"]["UPDATE_016"]["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, loader=self.loader)
            loaded_state_sha = api._state_fingerprint({"state": {"value": 16}})
            base = {
                "authorized_api": True,
                "world_size": 2,
                "rank": 0,
                "layer_split": {0: [0, 20], 1: [21, 42]},
                "shared_optimizer_scheduler_lineage": "modern-green-u16-lineage",
                "local_only": True,
                "basis_sha256": "b" * 64,
                "checkpoint_sha256": api.artifact.manifest["checkpoints"]["UPDATE_016"]["sha256"],
                "resident_model": {"value": 16},
                "resident_planes": {"layers": [0, 1]},
                "resident_data": {"seed": 1701},
                "resident_api_state": {"cache": "warm"},
                "resident_state": {"value": 16},
                "resident_state_sha256": loaded_state_sha,
            }
            def advance(state, target, config):
                delta = target - state["value"]
                return {"value": target}, {
                    "resident_optimizer_step": True,
                    "optimizer_steps": delta,
                    "scheduler_steps": delta,
                }
            receipt = root / "receipts" / "CONTINUATION.json"
            with self.assertRaisesRegex(ArtifactError, "fixture continuation is forbidden"):
                api.continue_two_spark(
                    "UPDATE_016", [20, 32, 48, 64], config=base,
                    advance_fn=advance, receipt_path=receipt,
                )
    def test_real_two_spark_requires_official_resident_inputs_not_state_surrogate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            import torch
            checkpoint_path = root / "checkpoints" / "UPDATE_016.pt"
            torch.save({"state": {"luts": {"layer0": torch.tensor([1.0, -1.0])}, "step": 16}}, checkpoint_path)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            manifest["checkpoints"]["UPDATE_016"]["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            manifest["checkpoints"]["UPDATE_016"]["next_update"] = 16
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root)
            cfg = {
                "authorized_api": True, "world_size": 2, "rank": 0,
                "layer_split": {0: [0, 20], 1: [21, 42]},
                "shared_optimizer_scheduler_lineage": "modern-green-u16-lineage",
                "local_only": True,
                "basis_sha256": "b" * 64,
                "checkpoint_sha256": manifest["checkpoints"]["UPDATE_016"]["sha256"],
                "artifact_root": str(root),
            }
            with self.assertRaisesRegex(ArtifactError, "official resident student inputs"):
                api.continue_two_spark_real("UPDATE_016", [20], config=cfg,
                                            receipt_path=root / "receipts" / "real.json")

    def test_real_forward_is_not_constant_target_mse_or_state_hash_bump(self) -> None:
        api_text = (Path(__file__).resolve().parents[1] / "api.py").read_text()
        engine_text = (Path(__file__).resolve().parents[1] / "modern_green_resident.py").read_text()
        self.assertNotIn("0.1234567", api_text + engine_text)
        self.assertNotIn("parameter.float() - target", api_text + engine_text)
        self.assertIn("ShardStudent", api_text + engine_text)
        self.assertIn("_pipeline_pass", engine_text)

    def test_legacy_fixture_continuation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root)
            with self.assertRaisesRegex(ArtifactError, "fixture.*forbidden"):
                api.continue_two_spark("UPDATE_016", [20], config={},
                                       advance_fn=lambda *_: ({"x": 1}, {}),
                                       receipt_path=root / "legacy.json")

    def test_materialize_candidates_requires_real_two_rank_lineage_and_scores_all_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "score" / "teacher").mkdir(parents=True)
            windows = list(range(64))
            lineage = "modern-green-u16-to-u64-shared-adam-cosine-resident-lineage"
            import torch
            checkpoints = {}
            previous_sha = None
            for update in (16, 20, 32, 48, 64):
                path = root / "checkpoints" / f"UPDATE_{update:03d}.pt"
                torch.save({"state": {"value": update}}, path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                checkpoints[f"UPDATE_{update:03d}"] = {
                    "path": str(path.relative_to(root)), "sha256": sha,
                    "identity_sha256": (f"{update:02x}" * 32)[:64],
                    "parent_sha256": previous_sha, "next_update": update,
                    "optimizer_scheduler_lineage": lineage,
                }
                previous_sha = sha
            for window in windows:
                (root / "score" / "teacher" / f"t8192_win{window}.pt").touch()
            manifest = {
                "schema": "repair-artifact-v1", "artifact_id": "materialize-test",
                "identity": {
                    "basis_sha256": "b" * 64,
                    "builder_eval_corpus_sha256": "c" * 64,
                    "train_score_corpus_sha256": "d" * 64,
                    "teacher_inventory": ["teacher-v1"],
                },
                "checkpoints": checkpoints,
                "score": {
                    "spec": "balanced64-v1", "teacher_dir": "score/teacher",
                    "candidate_dir_template": "score/candidates/{checkpoint}",
                    "window_ids": windows, "positions_per_window": 1024, "support": 8192,
                },
            }
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            builder = root / "builder.py"
            builder.write_text(
                "import argparse, pathlib, re\n"
                "CHECKPOINT = 'old'\nCHECKPOINT_SHA = 'old'\nCANDIDATE_IDENTITY = 'old'\n"
                "def main():\n"
                " p=argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--windows'); p.add_argument('--tag'); a,_=p.parse_known_args()\n"
                " value={'next_update': int(re.search(r'UPDATE_(\\d+)', CHECKPOINT).group(1))}\n"
                " if int(value.get(\"next_update\", -1)) != 0: raise RuntimeError('bad next_update')\n"
                " out=pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)\n"
                " [ (out / f'q8192_win{w}.pt').touch() for w in a.windows.split(',') ]\n"
                "if __name__ == '__main__': main()\n"
            )
            zeros_lp = np.zeros((1024, 8192), dtype=np.float32)
            zeros_idx = np.zeros((1024, 8192), dtype=np.int32)
            zeros_argmax = np.zeros(1024, dtype=np.int32)
            def loader(path: Path):
                if path.parent.name == "teacher":
                    return {"idx": zeros_idx, "logprob": zeros_lp}
                return {"q_lp_at_ref": zeros_lp, "q_argmax": zeros_argmax}
            api = ResidentRepairAPI.open(root, loader=loader)
            provenance_paths = []
            for rank in (0, 1):
                rows = []
                parent = checkpoints["UPDATE_016"]["sha256"]
                for update in (20, 32, 48, 64):
                    current = checkpoints[f"UPDATE_{update:03d}"]["sha256"]
                    rows.append({"target_update": update, "parent_checkpoint_sha256": parent,
                                 "checkpoint_sha256": current, "state_sha256": f"state-{update}",
                                 "checkpoint_loaded": True, "immutable": True,
                                 "optimizer_steps": update - (16 if update == 20 else {32: 20, 48: 32, 64: 48}[update]),
                                 "scheduler_steps": update - (16 if update == 20 else {32: 20, 48: 32, 64: 48}[update])})
                    parent = current
                receipt = {"status": "PASS", "world_size": 2, "rank": rank,
                           "checkpoint_loaded": True, "start_checkpoint_sha256": checkpoints["UPDATE_016"]["sha256"],
                           "loaded_checkpoint_sha256": checkpoints["UPDATE_016"]["sha256"],
                           "shared_optimizer_scheduler_lineage": lineage, "milestones": rows}
                path = root / f"rank{rank}.json"; path.write_text(json.dumps(receipt)); provenance_paths.append(path)
            result = api.materialize_candidates(
                ["UPDATE_020", "UPDATE_032", "UPDATE_048", "UPDATE_064"],
                builder_template=builder, ref_dir=root / "score" / "teacher",
                corpus=root / "corpus.json", meta_dir=root, continuation_receipts=provenance_paths,
                receipt_dir=root / "receipts", windows=windows,
            )
            self.assertEqual(result["status"], "PASS_4_OF_4")
            self.assertEqual(result["milestone_count"], 4)
            self.assertTrue(result["terminal"])
            self.assertEqual(result["milestones"][0]["score"]["positions"], 65536)
            self.assertEqual(result["milestones"][0]["file_reads_during_timed_score"], 0)
            self.assertTrue((root / "receipts" / "U16_U64_CANDIDATE_BALANCED64_AGGREGATE.json").is_file())

    def _make_durable_continuation_case(self, root: Path):
        self.make_artifact(root)
        manifest = json.loads((root / "ARTIFACT.json").read_text())
        manifest["checkpoints"].pop("UPDATE_064")
        (root / "checkpoints" / "UPDATE_064.pt").unlink()
        for child in (root / "score" / "candidates" / "UPDATE_064").iterdir():
            child.unlink()
        (root / "score" / "candidates" / "UPDATE_064").rmdir()
        (root / "ARTIFACT.json").write_text(json.dumps(manifest))
        import torch
        checkpoint_path = root / "checkpoints" / "UPDATE_016.pt"
        torch.save({"state": {"value": torch.tensor([16.0])}}, checkpoint_path)
        manifest = json.loads((root / "ARTIFACT.json").read_text())
        manifest["checkpoints"]["UPDATE_016"]["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        (root / "ARTIFACT.json").write_text(json.dumps(manifest))
        api = ResidentRepairAPI.open(root)
        config = {
            "authorized_api": True, "world_size": 2, "rank": 0,
            "layer_split": {0: [0, 20], 1: [21, 42]},
            "shared_optimizer_scheduler_lineage": "modern-green-u16-lineage",
            "local_only": True, "artifact_root": str(root),
            "basis_sha256": "b" * 64,
            "checkpoint_sha256": manifest["checkpoints"]["UPDATE_016"]["sha256"],
        }
        return api, config

    def _persist_durable_fixture(self, api: ResidentRepairAPI, config: dict, *, ranks=(0,)):
        import torch
        all_rows = {}
        previous_sha = api.artifact.manifest["checkpoints"]["UPDATE_016"]["sha256"]
        previous_identity = api.artifact.manifest["checkpoints"]["UPDATE_016"].get("identity_sha256")
        for rank in ranks:
            rank_config = {**config, "rank": rank}
            rows = []
            previous_sha = api.artifact.manifest["checkpoints"]["UPDATE_016"]["sha256"]
            previous_identity = api.artifact.manifest["checkpoints"]["UPDATE_016"].get("identity_sha256")
            for update in (20, 32, 48, 64):
                step_delta = update - (16 if update == 20 else {32: 20, 48: 32, 64: 48}[update])
                state = {"value": torch.tensor([float(update)])}
                persisted = api._persist_continuation_checkpoint(
                    update, state,
                    {
                        "resident_optimizer_step": True,
                        "optimizer_steps": step_delta,
                        "scheduler_steps": step_delta,
                        "optimizer_state": {"state": {}, "param_groups": []},
                        "scheduler_state": {"last_epoch": update},
                    },
                    parent_sha=previous_sha,
                    parent_identity_sha=previous_identity,
                    lineage=rank_config["shared_optimizer_scheduler_lineage"],
                    config=rank_config,
                )
                rows.append({"target_update": update, "next_update": update, "checkpoint_loaded": True, "immutable": True, "world_size": 2, "rank": rank, **dict(persisted)})
                previous_sha = persisted["checkpoint_sha256"]
                previous_identity = persisted["checkpoint_identity_sha256"]
            all_rows[rank] = rows
            receipt = {
                "schema": "resident-two-spark-real-continuation-v2",
                "status": "PASS",
                "world_size": 2,
                "rank": rank,
                "checkpoint_loaded": True,
                "start_checkpoint_sha256": api.artifact.manifest["checkpoints"]["UPDATE_016"]["sha256"],
                "loaded_checkpoint_sha256": api.artifact.manifest["checkpoints"]["UPDATE_016"]["sha256"],
                "shared_optimizer_scheduler_lineage": rank_config["shared_optimizer_scheduler_lineage"],
                "milestones": rows,
            }
            (api.artifact.root / f"rank{rank}.json").write_text(json.dumps(receipt))
        return all_rows

    def test_durable_milestones_have_paths_manifest_shas_and_readback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api, config = self._make_durable_continuation_case(root)
            persisted_rows = self._persist_durable_fixture(api, config)[0]
            result = {"milestones": persisted_rows}
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            import torch
            previous = manifest["checkpoints"]["UPDATE_016"]["sha256"]
            for row in result["milestones"]:
                update = row["target_update"]
                entry = manifest["checkpoints"][f"UPDATE_{update:03d}"]
                path = root / entry["path"]
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                self.assertEqual(row["checkpoint_path"], entry["path"])
                self.assertEqual(row["checkpoint_sha256"], entry["sha256"])
                self.assertEqual(row["checkpoint_identity_sha256"], entry["identity_sha256"])
                self.assertEqual(entry["parent_sha256"], previous)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(payload["identity"]["identity_sha256"], entry["identity_sha256"])
                self.assertIn("state", payload["optimizer_state"])
                previous = entry["sha256"]
            reopened = ResidentRepairAPI.open(root)
            self.assertEqual(reopened.artifact.checkpoint_path("UPDATE_064").resolve(), (root / "checkpoints" / "UPDATE_064.pt").resolve())

    def test_second_rank_reuses_files_and_merges_rank_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api, config = self._make_durable_continuation_case(root)
            self._persist_durable_fixture(api, config, ranks=(0, 1))
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            for update in (20, 32, 48, 64):
                entry = manifest["checkpoints"][f"UPDATE_{update:03d}"]
                self.assertEqual(entry["rank_provenance"], [0, 1])
                self.assertEqual(entry["world_size"], 2)
                self.assertEqual(entry["rank"], 1)

    def test_materialize_rejects_receipt_only_state_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api, config = self._make_durable_continuation_case(root)
            self._persist_durable_fixture(api, config, ranks=(0, 1))
            (root / "checkpoints" / "UPDATE_020.pt").unlink()
            with self.assertRaisesRegex(ArtifactError, "declared checkpoint is missing"):
                api.materialize_candidates(
                    ["UPDATE_020", "UPDATE_032", "UPDATE_048", "UPDATE_064"],
                    builder_template=root / "missing_builder.py", ref_dir=root,
                    corpus=root / "corpus.json", meta_dir=root,
                    continuation_receipts=[root / "rank0.json", root / "rank1.json"],
                    receipt_dir=root / "receipts", windows=api.windows,
                )

    def test_continuation_rejects_artifact_root_not_consumed_by_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api, config = self._make_durable_continuation_case(root)
            with self.assertRaisesRegex(ArtifactError, "opened materialize_candidates root"):
                api._persist_continuation_checkpoint(
                    20, {"value": 20},
                    {"optimizer_steps": 4, "scheduler_steps": 4},
                    parent_sha=config["checkpoint_sha256"], parent_identity_sha="16" * 32,
                    lineage=config["shared_optimizer_scheduler_lineage"],
                    config={**config, "artifact_root": str(root / "other")},
                )


if __name__ == "__main__":
    unittest.main()
