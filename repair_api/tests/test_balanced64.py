from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from repair_api import ArtifactError, RepairArtifact


WINDOWS = [28, 56, 68, 71, 76, 99, 107, 122, 124, 130, 141, 156, 160, 171, 180, 183, 185, 186, 196, 210, 212, 213, 218, 228, 232, 235, 249, 270, 272, 273, 283, 288, 290, 295, 297, 306, 307, 309, 311, 328, 331, 357, 362, 365, 368, 374, 376, 380, 384, 385, 391, 396, 413, 429, 430, 437, 442, 447, 454, 462, 464, 475, 489, 499]


class Balanced64ApiTests(unittest.TestCase):
    def make_artifact(self, root: Path) -> None:
        (root / "checkpoints").mkdir()
        (root / "score" / "teacher").mkdir(parents=True)
        (root / "score" / "candidates" / "0").mkdir(parents=True)
        (root / "checkpoints" / "UPDATE_000.pt").write_bytes(b"checkpoint")
        (root / "score" / "teacher" / "t8192_win28.pt").touch()
        (root / "score" / "candidates" / "0" / "q8192_win28.pt").touch()
        (root / "ARTIFACT.json").write_text(json.dumps({
            "schema": "repair-artifact-v1",
            "artifact_id": "test",
            "checkpoints": {"0": {"path": "checkpoints/UPDATE_000.pt"}},
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "score/teacher",
                "candidate_dir_template": "score/candidates/{checkpoint}",
                "window_ids": WINDOWS,
                "positions_per_window": 1024,
                "support": 8192,
            },
        }))

    def test_checkpoint_selection_and_fixed_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            artifact = RepairArtifact.open(root)
            zeros = np.zeros((1024, 8192), dtype=np.float32)
            candidate = {"q_lp_at_ref": zeros, "q_argmax": np.zeros(1024, dtype=np.int32)}
            teacher = {"idx": np.zeros((1024, 8192), dtype=np.int32), "logprob": zeros}
            loaded = {"t8192_win28.pt": teacher, "q8192_win28.pt": candidate}
            result = artifact.score(0, windows=[28], loader=lambda path: loaded[path.name])
            self.assertEqual(result.positions, 1024)
            self.assertEqual(result.support, 8192)
            self.assertEqual(result.kld, 0.0)
            self.assertEqual(result.top1, 1024)
            resident = artifact.load_resident(0, windows=[28], loader=lambda path: loaded[path.name])
            resident_result = resident.score()
            self.assertEqual(resident_result.execution_mode, "resident_in_memory")
            self.assertEqual(resident_result.positions, 1024)
            self.assertEqual(resident_result.kld, 0.0)
            timed = resident_result.timed_wall_seconds
            self.assertIsNotNone(timed)
            assert timed is not None
            self.assertLess(timed, 1200.0)

    def test_public_generation_binds_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "score" / "teacher").mkdir(parents=True)
            checkpoint = root / "checkpoints" / "UPDATE_016.pt"
            checkpoint.write_bytes(b"checkpoint-u16")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (root / "ARTIFACT.json").write_text(json.dumps({
                "schema": "repair-artifact-v1",
                "checkpoints": {
                    "UPDATE_016": {
                        "path": "checkpoints/UPDATE_016.pt",
                        "sha256": checkpoint_sha,
                        "identity_sha256": "3b" * 32,
                        "next_update": 16,
                    }
                },
                "score": {
                    "spec": "balanced64-v1",
                    "teacher_dir": "score/teacher",
                    "candidate_dir_template": "score/candidates/{checkpoint}",
                    "window_ids": WINDOWS,
                    "positions_per_window": 1024,
                    "support": 8192,
                },
            }))
            template = root / "builder.py"
            template.write_text(
                "import argparse, os\n"
                'CHECKPOINT = "old"\n'
                'CHECKPOINT_SHA = "old"\n'
                'CANDIDATE_IDENTITY = "old"\n'
                'def load():\n'
                '    value={"next_update": 2}\n'
                '    if int(value.get("next_update", -1)) != 2: raise RuntimeError("bad")\n'
                'def main():\n'
                '    p=argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--windows"); a,_=p.parse_known_args()\n'
                '    os.makedirs(a.out, exist_ok=True)\n'
                '    [open(os.path.join(a.out, f"q8192_win{w}.pt"), "wb").write(b"candidate") for w in a.windows.split(",")]\n'
                'if __name__ == "__main__": main()\n'
            )
            result = RepairArtifact.open(root).generate_candidates(
                "UPDATE_016",
                builder_template=template,
                ref_dir=root / "score" / "teacher",
                corpus=root / "eval.json",
                meta_dir=root,
                windows=[28, 56],
                chunk=2,
                mb=1,
            )
            derived = (root / "builders" / "builder_UPDATE_016.py").read_text()
            self.assertEqual(result["status"], "PASS")
            self.assertIn(f"CHECKPOINT_SHA = {checkpoint_sha!r}".replace("'", '"'), derived)
            self.assertIn('CANDIDATE_IDENTITY = "' + ("3b" * 32) + '"', derived)
            self.assertIn("!= 16", derived)
            self.assertTrue((root / "score" / "candidates" / "UPDATE_016" / "q8192_win28.pt").is_file())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ARTIFACT.json").write_text(json.dumps({
                "schema": "repair-artifact-v1",
                "checkpoints": {"0": {"path": "../outside.pt"}},
                "score": {
                    "spec": "balanced64-v1",
                    "teacher_dir": "teacher",
                    "candidate_dir_template": "candidates/{checkpoint}",
                    "window_ids": WINDOWS,
                    "positions_per_window": 1024,
                    "support": 8192,
                },
            }))
            with self.assertRaises(ArtifactError):
                RepairArtifact.open(root)


if __name__ == "__main__":
    unittest.main()
