from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "repair_api" / "sealed_rail_runner.py"


class SealedRailRunnerTest(unittest.TestCase):
    def test_pins_source_and_forwards_task_run_and_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "proven.py"
            output = root / "output.json"
            receipt = root / "deploy.json"
            source.write_text(
                "import json,sys\n"
                "TASK='old'\n"
                "RUN=1\n"
                "def main():\n"
                " json.dump({'task':TASK,'run':RUN,'argv':sys.argv[1:]},open(sys.argv[2],'w'))\n"
                " return 0\n"
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--source", str(source),
                    "--source-sha256", source_sha,
                    "--task-id", "t_d9925e51",
                    "--run-id", "4623",
                    "--git-sha", "a" * 40,
                    "--deployment-receipt", str(receipt),
                    "--",
                    "--output", str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), {
                "task": "t_d9925e51",
                "run": 4623,
                "argv": ["--output", str(output)],
            })
            deploy = json.loads(receipt.read_text())
            self.assertEqual(deploy["status"], "PINNED")
            self.assertEqual(deploy["source_sha256"], source_sha)
            self.assertEqual(deploy["canonical_git_sha"], "a" * 40)

    def test_refuses_source_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "proven.py"
            source.write_text("def main(): return 0\n")
            result = subprocess.run(
                [
                    sys.executable, str(RUNNER),
                    "--source", str(source),
                    "--source-sha256", "0" * 64,
                    "--task-id", "t_d9925e51",
                    "--run-id", "4623",
                    "--git-sha", "a" * 40,
                    "--deployment-receipt", str(root / "deploy.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sealed rail source SHA drift", result.stderr)


if __name__ == "__main__":
    unittest.main()
