import json
from pathlib import Path
import inspect

from repair_api import api as api_module
from repair_api import continuous_four_updates_official as official_continuous
from repair_api.api import ResidentRepairAPI
from repair_api.cli import build_parser
from repair_api.modern_green_resident import ModernGreenResidentEngine


def test_training_resident_loader_batches_projection_before_cuda_transfer():
    source = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "fast_v7_expert_train_batched.py"
    ).read_text()
    assert "def _load_projection_payloads(" in source
    init_source = source[source.index("class FullyResidentGroupedV7Experts"):]
    assert "packed_cpu, su_cpu, sv_cpu" in init_source
    assert init_source.count("packed_cpu.to(device=device)") == 1
    assert init_source.count("su_cpu.to(device=device)") == 1
    assert init_source.count("sv_cpu.to(device=device)") == 1
    assert ".copy_(packed_cpu)" not in init_source


class _Artifact:
    windows = tuple(range(64))
    root = Path("/artifact")
    manifest = {
        "identity": {
            "basis_sha256": "b" * 64,
            "builder_eval_corpus_sha256": "c" * 64,
            "train_score_corpus_sha256": "c" * 64,
            "teacher_inventory": "t" * 64,
        },
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": "u" * 64,
                "identity_sha256": "i" * 64,
                "next_update": 0,
            }
        },
    }

    def checkpoint_key(self, value):
        assert value in (0, "UPDATE_000")
        return "UPDATE_000"

    def checkpoint_path(self, key):
        assert key == "UPDATE_000"
        return self.root / "checkpoints" / "UPDATE_000.pt"


class _Stateful:
    checkpoint_loaded = False

    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state.clear()
        self.state.update(state)


class _Model(_Stateful):
    def resident_ready(self):
        return True


class _Optimizer(_Stateful):
    pass


class _Scheduler(_Stateful):
    pass


def _api():
    value = ResidentRepairAPI.__new__(ResidentRepairAPI)
    value.artifact = _Artifact()
    return value


def test_public_continuous_four_updates_is_one_resident_zero_reload_trajectory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_module,
        "_load_torch",
        lambda _path: {
            "state": {"value": 0},
            "optimizer_state": {"step": 0},
            "scheduler_state": {"step": 0},
        },
    )
    created = {"model": 0, "optimizer": 0, "scheduler": 0}
    updates = []
    scores = []
    fingerprints = []
    releases = []

    def model_factory(payload):
        assert payload["state"] == {"value": 0}
        created["model"] += 1
        return _Model({})

    def optimizer_factory(model):
        assert isinstance(model, _Model)
        created["optimizer"] += 1
        return _Optimizer({})

    def scheduler_factory(optimizer):
        assert isinstance(optimizer, _Optimizer)
        created["scheduler"] += 1
        return _Scheduler({})

    def update_fn(model, optimizer, scheduler, update):
        updates.append(update)
        model.state["value"] = update
        optimizer.state["step"] = update
        scheduler.state["step"] = update
        return {
            "optimizer_steps": 1,
            "scheduler_steps": 1,
            "checkpoint_loaded": False,
            "loss": float(update),
            "timings": {"wall_seconds": float(update)},
        }

    def resident_score_fn(model, update, windows):
        assert model.state["value"] == update
        assert tuple(windows) == tuple(range(64))
        scores.append(update)
        return {
            "kld_mean": float(update) / 10.0,
            "top1": 65536 - update,
            "positions": 64 * 1024,
            "support": 8192,
            "windows": list(range(64)),
            "execution_mode": "resident_in_memory",
            "runtime_counters": {
                "timed_model_payload_reads": 0,
                "timed_score_file_reads": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "cpu_relay_bytes": 0,
                "resident_ready": [{"rank": 0}, {"rank": 1}],
            },
        }

    def state_fingerprint_fn(model, optimizer, scheduler, update):
        fingerprints.append(update)
        return {
            "model": f"model-{update}",
            "optimizer": f"optimizer-{update}",
            "scheduler": f"scheduler-{update}",
            "scope": "global_two_rank",
        }

    def release_fn(model, optimizer, scheduler):
        releases.append((id(model), id(optimizer), id(scheduler)))

    receipt = tmp_path / "CONTINUOUS.json"
    result = _api().continuous_four_updates(
        "UPDATE_000",
        replay={
            "model_factory": model_factory,
            "optimizer_factory": optimizer_factory,
            "scheduler_factory": scheduler_factory,
            "update_fn": update_fn,
            "resident_score_fn": resident_score_fn,
            "state_fingerprint_fn": state_fingerprint_fn,
            "release_fn": release_fn,
            "geometry": {"layers": 43, "ranks": 2, "windows_per_update": 4},
            "basis_sha256": "b" * 64,
            "corpus_sha256": "c" * 64,
            "seed": 1701,
        },
        receipt_path=receipt,
    )

    assert created == {"model": 1, "optimizer": 1, "scheduler": 1}
    assert updates == [1, 2, 3, 4]
    assert scores == [0, 2, 4]
    assert fingerprints == [0, 2, 4]
    assert len(releases) == 1
    assert result["status"] == "PASS"
    assert result["public_api"]["method"] == "ResidentRepairAPI.continuous_four_updates"
    assert result["runtime_counters"] == {
        "input_checkpoint_loads": 1,
        "checkpoint_saves": 0,
        "checkpoint_reloads": 0,
        "update_callbacks": 4,
        "resident_scores": 3,
        "release_calls": 1,
    }
    assert [row["update"] for row in result["milestones"]] == [0, 2, 4]
    assert [row["optimizer_steps"] for row in result["milestones"]] == [0, 2, 4]
    assert [row["scheduler_steps"] for row in result["milestones"]] == [0, 2, 4]
    assert all(row["model_fingerprint"] for row in result["milestones"])
    assert all(row["optimizer_fingerprint"] for row in result["milestones"])
    assert all(row["scheduler_fingerprint"] for row in result["milestones"])
    assert json.loads(receipt.read_text()) == result


def test_continuous_four_updates_cli_is_public_and_has_no_update_count_override():
    parser = build_parser()
    args = parser.parse_args(
        [
            "continuous-four-updates",
            "--artifact-root", "/artifact",
            "--config", "/config.json",
            "--receipt", "/continuous.json",
        ]
    )
    assert args.verb == "continuous-four-updates"
    assert not hasattr(args, "total_updates")
    assert not hasattr(args, "midpoint_update")


def test_official_continuous_adapter_uses_public_api_and_resident_scorer_only():
    source = inspect.getsource(official_continuous.run_official_continuous_four_updates)
    assert ".continuous_four_updates(" in source
    assert ".score_resident(" in source
    assert "progress_callback" in source
    assert "torch.save" not in source
    assert "BytesIO" not in source
    assert "checkpoint_dir" not in source
    assert source.count(".release()") == 1


def test_resident_engine_forwards_cold_load_progress_callback():
    forwarded = []
    engine = object.__new__(ModernGreenResidentEngine)
    engine.status = {}
    engine.config = {"progress_callback": lambda **fields: forwarded.append(fields)}
    engine._status(phase="model_load", layer=7)
    assert engine.status == {"phase": "model_load", "layer": 7}
    assert forwarded == [{"phase": "model_load", "layer": 7}]
