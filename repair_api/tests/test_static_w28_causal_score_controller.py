from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _controller():
    path = Path(__file__).parents[1] / "assets" / "static_w28_causal_score_controller.py"
    spec = spec_from_file_location("static_w28_causal_score_controller", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u57_target_identity_and_score_argv_are_parameterized():
    c = _controller()
    c.TASK = "t_e9be5b1b"
    c.RUN_ID = 7319
    c.PIN = "deadbeef"
    c.CHECKPOINT = "UPDATE_057"
    c.CHECKPOINT_SHA = "a563f2f6683e5041b8d6f64ae360754a50fdbfdbed79ccd86a204b6a58ff2d21"
    c.PARENT_SHA = "ed98a9daf6ccfb9dd7e17a6ca1ff9aaabb581dbec7563e970776f5906a7a55d9"
    c.ATTEMPT = 25
    c.SLUG = "u57w28.attempt25"
    receipt = Path("/tmp/SCORE.candidate_d.u57w28.attempt25.rank1.json")

    identity = c.target_identity()
    argv = c.score_argv(1, receipt)

    assert identity == {
        "task_id": "t_e9be5b1b",
        "board_run_id": 7319,
        "canonical_git_pin": "deadbeef",
        "basis_sha256": c.BASIS,
        "checkpoint": "UPDATE_057",
        "checkpoint_sha256": c.CHECKPOINT_SHA,
        "parent_sha256": c.PARENT_SHA,
        "attempt": 25,
        "slug": "u57w28.attempt25",
    }
    assert argv[argv.index("--node-rank") + 1] == "1"
    assert argv[argv.index("--checkpoint") + 1] == "UPDATE_057"
    assert argv[argv.index("--receipt") + 1] == str(receipt)
    assert "UPDATE_058" not in argv
