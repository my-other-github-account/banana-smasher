import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from repair_api import modern_green_resident as resident_module
from repair_api.api import ResidentRepairAPI, _validate_published_pre_resume_start
from repair_api.modern_green_resident import (
    BASE_LRS,
    ModernGreenResidentEngine,
    _admit_restored_optimizer_base_lrs,
    _configure_v7_lut_only_optimizer,
    _resident_optimizer_param_groups,
    _resolve_trainer_source,
)


def _rows(prefix: str, count: int):
    return [
        (f"{prefix}.{index}", torch.nn.Parameter(torch.zeros(1)))
        for index in range(count)
    ]


def test_lut_only_keeps_all43_admitted_but_only_names_optimizer_members():
    luts = _rows("layers", 43)
    norms = _rows("norm", 235)
    outputs = _rows("output", 43)
    selected_names = ["layers.3", "layers.17"]
    config = {
        "v7_lut_only_update": True,
        "trainable_luts": selected_names,
        "lut_lr": 0.00025,
    }

    optimizer_rows, manifest = _configure_v7_lut_only_optimizer(
        config, luts, norms, outputs
    )
    groups = _resident_optimizer_param_groups(config, optimizer_rows, BASE_LRS)

    assert len(luts) == manifest["admitted_plane_sources"] == 43
    assert [name for name, _ in optimizer_rows["luts"]] == selected_names
    assert {id(parameter) for group in groups for parameter in group["params"]} == {
        id(parameter) for name, parameter in luts if name in selected_names
    }
    assert groups[1]["params"] == [] and groups[1]["frozen"] is True
    assert groups[2]["params"] == [] and groups[2]["frozen"] is True
    assert manifest["frozen"] == {"norms": 235, "outputs": 43}
    assert all(parameter.requires_grad == (name in selected_names) for name, parameter in luts)
    assert not any(parameter.requires_grad for _name, parameter in [*norms, *outputs])


def test_existing_joint_all43_optimizer_path_is_unchanged():
    luts = _rows("layers", 43)
    norms = _rows("norm", 235)
    outputs = _rows("output", 43)
    rows, manifest = _configure_v7_lut_only_optimizer({}, luts, norms, outputs)
    groups = _resident_optimizer_param_groups({}, rows, BASE_LRS)

    assert [len(group["params"]) for group in groups] == [43, 235, 43]
    assert [group["group_name"] for group in groups] == ["luts", "norms", "outputs"]
    assert all("frozen" not in group for group in groups)
    assert manifest["mode"] == "joint_all43"
    assert all(parameter.requires_grad for _name, parameter in [*luts, *norms, *outputs])


def test_restored_optimizer_groups_admit_checkpoint_base_lrs_and_keep_existing_groups():
    restored_groups = [
        {"group_name": "all43_luts", "initial_lr": 2.5e-4},
        {"group_name": "norms", "initial_lr": 9.9},
        {"group_name": "outputs", "initial_lr": 9.9},
    ]

    admitted = _admit_restored_optimizer_base_lrs(BASE_LRS, restored_groups)

    assert admitted["all43_luts"] == 2.5e-4
    assert admitted["norms"] == BASE_LRS["norms"]
    assert admitted["outputs"] == BASE_LRS["outputs"]


def test_public_api_wrapper_pins_exactly_one_successor_update():
    api = Mock()
    api.artifact = Mock()
    api.artifact.checkpoint_key.return_value = "PRE"
    api._checkpoint_update.return_value = 0
    api.continue_two_spark_real.return_value = {"status": "PASS"}

    result = ResidentRepairAPI.continue_v7_lut_only_update(
        api,
        "PRE",
        trainable_luts=["layers.3"],
        lut_lr=0.00025,
        config={"authorized_api": True},
        receipt_path="receipt.json",
    )

    assert result == {"status": "PASS"}
    api.continue_two_spark_real.assert_called_once_with(
        "PRE",
        (1,),
        config={
            "authorized_api": True,
            "v7_lut_only_update": True,
            "trainable_luts": ["layers.3"],
            "lut_lr": 0.00025,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "layer_split": {"0": [0, 42]},
            "resident_validation_proof": False,
        },
        receipt_path="receipt.json",
    )


def test_public_api_wrapper_accepts_u0_alias_for_published_pre():
    api = Mock()
    api.artifact = Mock()
    api.artifact.checkpoint_key.return_value = "PRE"
    api._checkpoint_update.return_value = 0
    api.continue_two_spark_real.return_value = {"status": "PASS"}

    ResidentRepairAPI.continue_v7_lut_only_update(
        api,
        "U0",
        trainable_luts=["layers.3"],
        lut_lr=0.00025,
        config={"authorized_api": True},
        receipt_path="receipt.json",
    )

    api.artifact.checkpoint_key.assert_called_once_with("PRE")


def test_single_gpu_lut_only_pipeline_runs_real_autograd_without_collectives(monkeypatch):
    monkeypatch.setattr(resident_module, "_cuda_sync", lambda _torch: None)

    class NoDistributedCalls:
        def __getattr__(self, name):
            raise AssertionError(f"single-GPU LUT-only path called torch.distributed.{name}")

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.single_gpu_v7_lut_only = True
    engine.rank = 0
    engine.pipeline_microbatch = 2
    engine.training_physical_rows = 3
    engine.torch = torch
    engine.dist = NoDistributedCalls()
    engine.ids_cache = {
        20: torch.tensor([[0, 1, 2]]),
        21: torch.tensor([[2, 1, 0]]),
    }
    embedding = torch.nn.Embedding(3, 4).to(torch.bfloat16)
    engine.student = SimpleNamespace(
        config=SimpleNamespace(hc_mult=1, hidden_size=4),
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)),
        device=torch.device("cpu"),
    )
    engine._run_layers = lambda hidden, ids, train: hidden.square()
    engine._loss_group = lambda hidden, group: hidden.float().sum()
    engine._record_optimizer_diagnostic_boundary = lambda boundary: None

    loss, timings = ModernGreenResidentEngine._pipeline_pass(engine, [20, 21])

    assert loss is not None and loss > 0.0
    assert embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.weight.grad).item() > 0
    assert timings["forward_seconds"] >= 0.0
    assert timings["backward_seconds"] >= 0.0


def test_single_gpu_lut_only_skips_process_group_initialization():
    class NoDistributedCalls:
        def __getattr__(self, name):
            raise AssertionError(f"single-GPU LUT-only path called torch.distributed.{name}")

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.single_gpu_v7_lut_only = True
    engine.dist = NoDistributedCalls()

    ModernGreenResidentEngine._init_distributed(engine)


def test_single_gpu_full_surface_backend_disables_recompute_only():
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    config = {
        "execution_backend": "single_gpu_resident_no_recompute",
        "world_size": 1,
    }

    ModernGreenResidentEngine._configure_execution_backend(engine, config, rank=0)

    assert engine.single_gpu_resident is True
    assert engine.single_gpu_v7_lut_only is False
    assert engine.activation_checkpointing is False

    luts = _rows("layers", 43)
    norms = _rows("norm", 235)
    outputs = _rows("output", 43)
    rows, manifest = _configure_v7_lut_only_optimizer(config, luts, norms, outputs)
    assert [len(rows[name]) for name in ("luts", "norms", "outputs")] == [43, 235, 43]
    assert manifest["mode"] == "joint_all43"


def test_public_full_surface_wrapper_pins_one_successor_and_backend():
    api = Mock()
    api.artifact = Mock()
    api.artifact.checkpoint_key.return_value = "UPDATE_020"
    api._checkpoint_update.return_value = 20
    api.continue_two_spark_real.return_value = {"status": "PASS"}

    result = ResidentRepairAPI.continue_single_gpu_resident_update(
        api,
        "UPDATE_020",
        config={"authorized_api": True, "activation_checkpointing": True},
        receipt_path="receipt.json",
    )

    assert result == {"status": "PASS"}
    called = api.continue_two_spark_real.call_args
    assert called.args == ("UPDATE_020", (21,))
    configured = called.kwargs["config"]
    assert configured["execution_backend"] == "single_gpu_resident_no_recompute"
    assert configured["activation_checkpointing"] is False
    assert configured["world_size"] == 1
    assert configured["layer_split"] == {"0": [0, 42]}
    assert "v7_lut_only_update" not in configured


def test_single_gpu_checkpointed_backend_keeps_recompute_enabled():
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    config = {
        "execution_backend": "single_gpu_resident_checkpointed",
        "activation_checkpointing": True,
        "world_size": 1,
    }

    ModernGreenResidentEngine._configure_execution_backend(engine, config, rank=0)

    assert engine.single_gpu_resident is True
    assert engine.single_gpu_v7_lut_only is False
    assert engine.activation_checkpointing is True


def test_public_checkpointed_wrapper_pins_one_successor_and_backend():
    api = Mock()
    api.artifact = Mock()
    api.artifact.checkpoint_key.return_value = "UPDATE_020"
    api._checkpoint_update.return_value = 20
    api.continue_two_spark_real.return_value = {"status": "PASS"}

    result = ResidentRepairAPI.continue_single_gpu_checkpointed_update(
        api,
        "UPDATE_020",
        config={"authorized_api": True, "activation_checkpointing": False},
        receipt_path="receipt.json",
    )

    assert result == {"status": "PASS"}
    called = api.continue_two_spark_real.call_args
    assert called.args == ("UPDATE_020", (21,))
    configured = called.kwargs["config"]
    assert configured["execution_backend"] == "single_gpu_resident_checkpointed"
    assert configured["activation_checkpointing"] is True
    assert configured["world_size"] == 1
    assert configured["layer_split"] == {"0": [0, 42]}
    assert "v7_lut_only_update" not in configured


def test_exact_u20_checkpointed_full_surface_backend_is_authenticated():
    sha = "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
    _validate_published_pre_resume_start(
        20,
        {"sha256": sha, "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr"},
        config={
            "checkpoint_sha256": sha,
            "execution_backend": "single_gpu_resident_checkpointed",
            "activation_checkpointing": True,
            "world_size": 1,
            "rank": 0,
            "lr_scale": 0.5,
            "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
            "published_pre_checkpoint_sha256": "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
            "fresh_published_pre_lineage": True,
            "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        },
    )


def test_single_gpu_checkpointed_backend_loads_local_teacher_rows():
    source = inspect.getsource(ModernGreenResidentEngine._load_training_data)
    assert "if self.rank == 1 or self.single_gpu_resident:" in source


def test_single_gpu_checkpointed_backend_persists_its_direct_optimizer_state():
    source = inspect.getsource(ModernGreenResidentEngine._gather_state)
    assert 'elif self.single_gpu_resident:' in source
    assert 'optimizer = rows[0]["optimizer"]' in source


def test_single_gpu_checkpointed_optimizer_accepts_exact_empty_missing_state():
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.rank = 0
    engine.single_gpu_resident = True
    engine.config = {}
    engine.luts = _rows("lut", 43)
    engine.norms = _rows("norm", 235)
    engine.outputs = _rows("output", 43)
    engine.optimizer_rows = {
        "luts": engine.luts,
        "norms": engine.norms,
        "outputs": engine.outputs,
    }
    direct_state = {
        "state": {index: {"step": torch.tensor(1.0)} for index in range(321)},
        "param_groups": [
            {"params": list(range(43))},
            {"params": list(range(43, 278))},
            {"params": list(range(278, 321))},
        ],
    }
    engine.optimizer = Mock()
    engine.optimizer.state_dict.return_value = direct_state
    engine.scheduler = Mock()
    engine.scheduler.state_dict.return_value = {"last_epoch": 21}
    engine.trainer = Mock()
    engine.trainer.merge_optimizer_state.side_effect = AssertionError(
        "singleton checkpointed state must not use the two-rank dormant-norm validator"
    )

    _merged, optimizer, _report = ModernGreenResidentEngine._gather_state(engine)

    assert optimizer is not None
    assert set(optimizer["state"]) == set(range(321))
    engine.trainer.merge_optimizer_state.assert_not_called()


def test_single_gpu_checkpointed_rank_reports_do_not_self_reference():
    source = inspect.getsource(ModernGreenResidentEngine._step)
    assert 'local["rank_reports"] = [dict(row) for row in rows]' in source


def test_exact_u21_checkpointed_resume_is_authenticated():
    sha = "11df795d56d7f9210f20bb99e91b6518dc17d0e24cbfff6b96e120168ab64830"
    _validate_published_pre_resume_start(
        21,
        {"sha256": sha, "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr"},
        config={
            "checkpoint_sha256": sha,
            "execution_backend": "single_gpu_resident_checkpointed",
            "activation_checkpointing": True,
            "world_size": 1,
            "rank": 0,
            "lr_scale": 0.5,
            "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
            "published_pre_checkpoint_sha256": "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
            "fresh_published_pre_lineage": True,
            "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        },
    )


def test_public_checkpointed_boundary_wrapper_targets_u24():
    api = Mock()
    api.artifact = Mock()
    api.artifact.checkpoint_key.return_value = "UPDATE_021"
    api.continue_two_spark_real.return_value = {"status": "PASS"}
    result = ResidentRepairAPI.continue_single_gpu_checkpointed_to_boundary(
        api, "UPDATE_021", 24, config={"authorized_api": True}, receipt_path="r.json"
    )
    assert result == {"status": "PASS"}
    called = api.continue_two_spark_real.call_args
    assert called.args == ("UPDATE_021", (24,))
    assert called.kwargs["config"]["activation_checkpointing"] is True
    assert called.kwargs["config"]["world_size"] == 1


def test_resident_loader_releases_cpu_source_duplicate_after_gpu_consumer() -> None:
    source = (
        Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py"
    ).read_text()
    loop = source[source.index("for layer in range(first, last + 1):") :]
    materialize = loop.index("base.v3.materialize_layer")
    synchronize = loop.index("torch.cuda.synchronize()", materialize)
    model_release = loop.index("release_model_source_cache(layer)", synchronize)
    wire_release = loop.index("release_expert_source_cache(source)", model_release)
    empty_cache = loop.index("torch.cuda.empty_cache()", wire_release)
    status = loop.index("status_cb(", empty_cache)
    assert materialize < synchronize < model_release < wire_release < empty_cache < status
    assert "handles.clear()" in source
    assert "source.member_paths.values()" in source
    assert "POSIX_FADV_DONTNEED" in source


def test_u20_static_provider_rebinds_inherited_trainer_to_canonical_layerwise_loader() -> None:
    path, sha256 = _resolve_trainer_source({
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "static_w28_gate": {"windows": [28]},
        "trainer_source": "/sealed/warm/root/modern_green_clean_u0.py",
        "trainer_source_sha256": (
            "a55c2f5104b8d9dd06d845684d168be6f6e9dae637bac08443bd6ddbaf94201a"
        ),
    })

    assert path == (
        Path(resident_module.__file__).resolve().parent
        / "assets"
        / "static_w28_modern_green_clean_u0.py"
    )
    assert sha256 == "126c11f306a12ed35c1234bd12952a32662c3bd81fc2e74361f0a55ebdc21fc0"


def test_exact_u20_full_surface_backend_is_an_authenticated_resume():
    sha = "2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd"
    _validate_published_pre_resume_start(
        20,
        {"sha256": sha, "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr"},
        config={
            "checkpoint_sha256": sha,
            "execution_backend": "single_gpu_resident_no_recompute",
            "activation_checkpointing": False,
            "world_size": 1,
            "rank": 0,
            "lr_scale": 0.5,
            "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
            "published_pre_checkpoint_sha256": "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70",
            "fresh_published_pre_lineage": True,
            "shared_optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        },
    )
