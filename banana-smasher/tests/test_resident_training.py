from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.grouped_k2 import block_hadamard_128, direct_decode_matrix
from banana_smasher.official_k2_resident import OfficialK2PackedResidentAdapter
from banana_smasher.resident_training import (
    ParameterDescriptor,
    ResidentModelAdapter,
    ResidentTrainingPlan,
    ResidentTrainingSession,
    _checkpoint_info as checkpoint_info,
    _ResidentTrainer as ResidentTrainer,
    require_local_compute_path,
    select_parameter_groups,
)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "model_source": "fixtures:create_model",
        "model_adapter": "fixtures:ToyResidentAdapter",
        "model_root": str(tmp_path / "model"),
        "payload_root": str(tmp_path / "payload"),
        "run_root": str(tmp_path / "run"),
        "topology": {
            "world_size": 2,
            "rank": 0,
            "layer_split": [[0, 20], [21, 42]],
        },
        "windows": [20, 21, 22, 23],
        "tokens_per_window": 128,
        "microbatch": 2,
        "gradient_accumulation": 2,
        "updates": 3,
        "optimizer": "adam",
        "parameter_groups": [
            {
                "name": "repair",
                "lr": 0.001,
                "warmup_updates": 4,
                "families": ["luts", "norms", "output_gains", "scales", "biases"],
                "include": ["model.layers.*", "model.norm"],
                "exclude": ["*.dormant_*"],
            },
            {
                "name": "head",
                "lr": 0.0001,
                "families": ["biases"],
                "include": ["lm_head.*"],
            },
        ],
    }


def test_plan_roundtrip_and_explicit_parameter_selectors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = ResidentTrainingPlan.from_dict(config)

    assert plan.topology.layers_for_rank == (0, 20)
    assert plan.windows == (20, 21, 22, 23)
    assert (
        ResidentTrainingPlan.from_dict(json.loads(plan.to_json())).to_dict()
        == plan.to_dict()
    )

    parameters = [
        ParameterDescriptor("model.layers.0.qtip.lut", "luts"),
        ParameterDescriptor("model.layers.0.input_norm.weight", "norms"),
        ParameterDescriptor("model.layers.0.output_log_gain", "output_gains"),
        ParameterDescriptor("model.layers.0.quant_scale", "scales"),
        ParameterDescriptor("model.layers.0.proj.bias", "biases"),
        ParameterDescriptor("model.layers.0.dormant_norm", "norms"),
        ParameterDescriptor("lm_head.bias", "biases"),
        ParameterDescriptor("lm_head.weight", "weights"),
    ]
    selected = select_parameter_groups(parameters, plan.parameter_groups)

    assert [item.name for item in selected["repair"]] == [
        "model.layers.0.input_norm.weight",
        "model.layers.0.output_log_gain",
        "model.layers.0.proj.bias",
        "model.layers.0.qtip.lut",
        "model.layers.0.quant_scale",
    ]
    assert [item.name for item in selected["head"]] == ["lm_head.bias"]


def test_parameter_group_overlap_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["parameter_groups"] = [
        {"name": "one", "lr": 0.1, "include": ["model.*"]},
        {"name": "two", "lr": 0.1, "include": ["*.weight"]},
    ]
    plan = ResidentTrainingPlan.from_dict(config)

    with pytest.raises(ValueError, match="selected by multiple parameter groups"):
        select_parameter_groups(
            [ParameterDescriptor("model.norm.weight", "norms")],
            plan.parameter_groups,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_rail", "offline"),
        ("input_checkpoint", "/tmp/staged.safetensors"),
        ("execution_mode", "subprocess"),
        ("training_mode", "replay"),
        ("staged_files", True),
        ("reload_per_step", True),
    ],
)
def test_legacy_training_modes_are_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _config(tmp_path)
    config[field] = value

    with pytest.raises(ValueError, match="resident-in-memory|not public"):
        ResidentTrainingPlan.from_dict(config)


def test_parameter_group_options_cannot_override_optimizer_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["parameter_groups"] = [
        {"name": "norms", "lr": 0.1, "options": {"params": []}}
    ]

    with pytest.raises(ValueError, match="reserved optimizer keys"):
        ResidentTrainingPlan.from_dict(config)


def test_remote_compute_inputs_are_rejected_before_staging(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        require_local_compute_path("s3://models/official-k2", "model root")

    config = _config(tmp_path)
    config["model_root"] = "sshfs://host/model"
    with pytest.raises(ValueError, match="local filesystem path"):
        ResidentTrainingPlan.from_dict(config)


def test_resume_checkpoint_must_be_local(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for name in ("model", "payload"):
        (tmp_path / name).mkdir()
    trainer = ResidentTrainer(
        ResidentTrainingPlan.from_dict(config), adapter=ToyResidentAdapter()
    )
    trainer.initialize()

    with pytest.raises(ValueError, match="local filesystem path"):
        trainer.load_checkpoint("sshfs://host/checkpoint.safetensors")


def test_adapter_stages_resolved_local_paths_not_replaceable_symlinks(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for name in ("real-model", "real-payload"):
        (tmp_path / name).mkdir()
    (tmp_path / "model").symlink_to(tmp_path / "real-model")
    (tmp_path / "payload").symlink_to(tmp_path / "real-payload")

    class PlanCapturingAdapter(ToyResidentAdapter):
        staged_plan = None

        def stage(self, plan):
            self.staged_plan = plan
            return super().stage(plan)

    adapter = PlanCapturingAdapter()
    ResidentTrainer(
        ResidentTrainingPlan.from_dict(config), adapter=adapter
    ).initialize()

    assert adapter.staged_plan is not None
    assert adapter.staged_plan.model_root == (tmp_path / "real-model").resolve()
    assert adapter.staged_plan.payload_root == (tmp_path / "real-payload").resolve()


class ToyResidentAdapter(ResidentModelAdapter):
    adapter_name = "toy-resident"
    adapter_version = "1"

    def __init__(self) -> None:
        self.stage_calls = 0
        self.selected: dict[str, tuple[str, ...]] = {}
        self.microbatches: list[tuple[tuple[int, ...], float]] = []
        self.optimizer_lrs: list[dict[str, float]] = []
        self.loaded_optimizer_state = None
        self.values = {
            "model.layers.0.qtip.lut": 1.0,
            "model.layers.0.input_norm.weight": 2.0,
            "model.layers.0.output_log_gain": 0.0,
            "model.layers.0.quant_scale": 3.0,
            "model.layers.0.proj.bias": 0.0,
            "model.layers.0.dormant_norm": 4.0,
            "lm_head.bias": 0.0,
            "lm_head.weight": 5.0,
        }

    def stage(self, plan: ResidentTrainingPlan) -> dict[str, object]:
        self.stage_calls += 1
        return {"resident_bytes": 128, "payload_disk_reads": 2}

    def parameters(self) -> list[ParameterDescriptor]:
        family_by_suffix = {
            "lut": "luts",
            "weight": "norms",
            "output_log_gain": "output_gains",
            "quant_scale": "scales",
            "bias": "biases",
            "dormant_norm": "norms",
        }
        return [
            ParameterDescriptor(
                name, family_by_suffix.get(name.rsplit(".", 1)[-1], "weights")
            )
            for name in self.values
        ]

    def configure_parameter_groups(self, groups) -> None:
        self.selected = {
            name: tuple(parameter.name for parameter in parameters)
            for name, parameters in groups.items()
        }

    def zero_grad(self) -> None:
        return None

    def train_microbatch(
        self, windows: tuple[int, ...], *, tokens: int, loss_scale: float
    ) -> dict[str, float]:
        self.microbatches.append((windows, loss_scale))
        return {
            "loss": float(sum(windows)),
            "forward_seconds": 0.01,
            "backward_seconds": 0.02,
            "comm_seconds": 0.003,
        }

    def optimizer_step(self, learning_rates: dict[str, float]) -> float:
        self.optimizer_lrs.append(dict(learning_rates))
        for names in self.selected.values():
            for name in names:
                self.values[name] += 1.0
        return 0.004

    def trainable_state_dict(self):
        return {
            name: self.values[name]
            for names in self.selected.values()
            for name in names
        }

    def load_trainable_state_dict(self, state) -> None:
        self.values.update({name: float(value) for name, value in state.items()})

    def optimizer_state_dict(self):
        selected_names = [name for names in self.selected.values() for name in names]
        return {
            "steps": len(self.optimizer_lrs),
            # Deliberately sparse: selected parameters with no gradient have no Adam row.
            "state": {
                "model.layers.0.qtip.lut": {
                    "exp_avg": 0.125,
                    "step": len(self.optimizer_lrs),
                }
            },
            "param_groups": [
                {"name": name, "params": list(names)}
                for name, names in self.selected.items()
            ],
            "selected_count": len(selected_names),
        }

    def load_optimizer_state_dict(self, state) -> None:
        self.loaded_optimizer_state = state
        self.optimizer_lrs = [{} for _ in range(int(state["steps"]))]

    def scheduler_state_dict(self):
        return {"last_update": len(self.optimizer_lrs)}

    def load_scheduler_state_dict(self, state) -> None:
        return None

    def deploy_export(self, checkpoint: Path, destination: Path):
        destination.write_text(
            json.dumps({"format": "toy-deploy", "checkpoint": checkpoint.name}) + "\n",
            encoding="utf-8",
        )
        return {"status": "EXPORTED", "path": str(destination)}


def test_resident_trainer_stages_once_and_reports_repeated_step_timings(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for name in ("model", "payload"):
        (tmp_path / name).mkdir()
    plan = ResidentTrainingPlan.from_dict(config)
    adapter = ToyResidentAdapter()
    trainer = ResidentTrainer(plan, adapter=adapter)

    residency = trainer.initialize()
    first = trainer.train_step()
    second = trainer.train_step()

    assert residency["resident_bytes"] == 128
    assert adapter.stage_calls == 1
    assert trainer.is_resident is True
    assert first.update == 0 and second.update == 1
    assert first.tokens == second.tokens == 4 * 128
    assert first.forward_seconds == pytest.approx(0.02)
    assert first.backward_seconds == pytest.approx(0.04)
    assert first.comm_seconds == pytest.approx(0.006)
    assert first.optimizer_seconds == pytest.approx(0.004)
    assert first.total_seconds >= 0.0
    assert adapter.microbatches == [
        ((20, 21), 0.5),
        ((22, 23), 0.5),
        ((20, 21), 0.5),
        ((22, 23), 0.5),
    ]
    assert adapter.optimizer_lrs[0]["repair"] == pytest.approx(0.001 / 4)
    assert adapter.optimizer_lrs[1]["repair"] == pytest.approx(0.001 / 2)
    assert (tmp_path / "run" / "TRAIN_STATUS.json").is_file()


def test_public_session_reuses_model_and_hot_swaps_checkpoint_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for name in ("model", "payload"):
        (tmp_path / name).mkdir()
    adapter = ToyResidentAdapter()
    phases: list[tuple[str, dict[str, object]]] = []
    session = ResidentTrainingSession.open(
        ResidentTrainingPlan.from_dict(config),
        adapter=adapter,
        phase_observer=lambda phase, event: phases.append((phase, dict(event))),
    )
    resident_model = session.model_instance

    first = session.continue_updates(2, checkpoint_every=1)
    first_checkpoint = Path(first["checkpoints"][0])
    session.continue_updates(1)
    loaded = session.hot_swap_checkpoint(first_checkpoint)
    resumed = session.continue_updates(1)

    assert loaded["next_update"] == 1
    assert resumed["next_update"] == 2
    assert session.model_instance is resident_model is adapter
    assert adapter.stage_calls == 1
    assert {name for name, _event in phases} >= {
        "resident_load",
        "forward",
        "backward",
        "communication",
        "optimizer",
        "update_total",
        "checkpoint_save",
        "checkpoint_hot_swap",
    }


def test_checkpoint_roundtrip_resume_and_deploy_hook(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for name in ("model", "payload"):
        (tmp_path / name).mkdir()
    plan = ResidentTrainingPlan.from_dict(config)
    original_adapter = ToyResidentAdapter()
    original = ResidentTrainer(plan, adapter=original_adapter)
    original.initialize()
    original.train_step()
    saved_values = dict(original_adapter.values)

    checkpoint = original.save_checkpoint()
    info = checkpoint_info(checkpoint)

    assert checkpoint.name == "UPDATE_00000001.safetensors"
    assert info["format"] == "banana-smasher-resident-checkpoint-v1"
    assert info["next_update"] == 1
    assert info["adapter"] == {"name": "toy-resident", "version": "1"}
    assert info["config"]["gradient_accumulation"] == 2
    assert info["trainable_count"] == 6
    assert info["parameter_groups"]["repair"] == [
        "model.layers.0.input_norm.weight",
        "model.layers.0.output_log_gain",
        "model.layers.0.proj.bias",
        "model.layers.0.qtip.lut",
        "model.layers.0.quant_scale",
    ]

    resumed_adapter = ToyResidentAdapter()
    resumed = ResidentTrainer(plan, adapter=resumed_adapter)
    loaded = resumed.load_checkpoint(checkpoint)

    assert loaded["next_update"] == 1
    assert resumed.update == 1
    assert resumed_adapter.values == saved_values
    assert len(resumed_adapter.optimizer_lrs) == 1
    assert resumed_adapter.loaded_optimizer_state is not None
    assert set(resumed_adapter.loaded_optimizer_state["state"]) == {
        "model.layers.0.qtip.lut"
    }
    assert (
        sum(
            len(group["params"])
            for group in resumed_adapter.loaded_optimizer_state["param_groups"]
        )
        == 6
    )
    advanced = resumed.train_step()
    assert advanced.update == 1
    assert resumed.update == 2

    deployed_path = tmp_path / "deploy.json"
    deployed = resumed.deploy_export(deployed_path, checkpoint=checkpoint)
    assert deployed["status"] == "EXPORTED"
    assert json.loads(deployed_path.read_text())["format"] == "toy-deploy"


def test_checkpoint_resume_rejects_optimizer_config_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for name in ("model", "payload"):
        (tmp_path / name).mkdir()
    original = ResidentTrainer(
        ResidentTrainingPlan.from_dict(config), adapter=ToyResidentAdapter()
    )
    original.initialize()
    original.train_step()
    checkpoint = original.save_checkpoint()

    config["optimizer"] = "adamw"
    resumed = ResidentTrainer(
        ResidentTrainingPlan.from_dict(config), adapter=ToyResidentAdapter()
    )
    with pytest.raises(ValueError, match="resume config mismatch.*optimizer"):
        resumed.load_checkpoint(checkpoint)


def test_packaged_k2_reference_math_has_no_external_private_module_dependency() -> None:
    torch = pytest.importorskip("torch")
    transformed = block_hadamard_128(torch.zeros((1, 128), dtype=torch.float32))
    decoded = direct_decode_matrix(
        torch.zeros((1, 1, 32), dtype=torch.int16),
        torch.zeros(1024, dtype=torch.float16),
    )

    assert transformed.shape == (1, 128)
    assert decoded.shape == (16, 16)


def test_official_k2_adapter_uses_stable_ids_and_sparse_adam_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    config = _config(tmp_path)
    config["model_source"] = "fixture:official_k2_backend"
    config["model_adapter"] = "official-k2-packed"
    config["topology"] = {"world_size": 1, "rank": 0, "layer_split": [[0, 0]]}
    config["windows"] = [20, 21]
    config["microbatch"] = 2
    config["gradient_accumulation"] = 1
    config["parameter_groups"] = [
        {"name": "norms", "lr": 0.01, "families": ["norms"], "include": ["*"]}
    ]
    model_root = tmp_path / "model"
    payload_root = tmp_path / "payload"
    model_root.mkdir()
    payload_root.mkdir()
    index = model_root / "model.safetensors.index.json"
    index.write_text('{"weight_map":{}}\n', encoding="utf-8")
    expected_index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    plan = ResidentTrainingPlan.from_dict(config)

    class Backend:
        def __init__(self) -> None:
            self.active = torch.nn.Parameter(torch.tensor(1.0))
            self.dormant = torch.nn.Parameter(torch.tensor(2.0))

        def resident_parameters(self):
            return [
                (
                    ParameterDescriptor(
                        "model.layers.0.input_norm.weight",
                        "norms",
                        "layer:0/norm:input",
                    ),
                    self.active,
                ),
                (
                    ParameterDescriptor(
                        "model.layers.0.dormant_norm", "norms", "layer:0/norm:dormant"
                    ),
                    self.dormant,
                ),
            ]

        def residency_metadata(self):
            return {"resident_bytes": 8, "payload_disk_reads": 1}

        def loss_for_windows(self, windows, *, tokens):
            return self.active.square(), 0.0

        def deploy_export(self, checkpoint, destination):
            destination.write_bytes(checkpoint.read_bytes())
            return {"status": "EXPORTED", "path": str(destination)}

    backend = Backend()

    def factory(**_kwargs):
        return backend

    adapter = OfficialK2PackedResidentAdapter(
        model_source=plan.model_source,
        backend_factory=factory,
        expected_model_index_sha256=expected_index_sha,
    )
    trainer = ResidentTrainer(plan, adapter=adapter)
    trainer.initialize()
    trainer.train_step()
    optimizer = adapter.optimizer_state_dict()

    assert optimizer["param_groups"][0]["params"] == [
        "layer:0/norm:dormant",
        "layer:0/norm:input",
    ]
    assert set(optimizer["state"]) == {"layer:0/norm:input"}
    assert adapter.metadata()["model_index_sha256"] == expected_index_sha

    checkpoint = trainer.save_checkpoint()
    resumed_backend = Backend()
    resumed_adapter = OfficialK2PackedResidentAdapter(
        model_source=plan.model_source,
        backend_factory=lambda **_kwargs: resumed_backend,
        expected_model_index_sha256=expected_index_sha,
    )
    resumed = ResidentTrainer(plan, adapter=resumed_adapter)
    resumed.load_checkpoint(checkpoint)
    resumed_optimizer = resumed_adapter.optimizer_state_dict()
    assert resumed_optimizer["param_groups"][0]["params"] == [
        "layer:0/norm:dormant",
        "layer:0/norm:input",
    ]
    assert set(resumed_optimizer["state"]) == {"layer:0/norm:input"}
