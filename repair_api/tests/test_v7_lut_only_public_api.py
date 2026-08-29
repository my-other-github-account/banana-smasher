from types import SimpleNamespace
from unittest.mock import Mock

import torch

from repair_api import modern_green_resident as resident_module
from repair_api.api import ResidentRepairAPI, _validate_published_pre_resume_start
from repair_api.modern_green_resident import (
    BASE_LRS,
    ModernGreenResidentEngine,
    _configure_v7_lut_only_optimizer,
    _resident_optimizer_param_groups,
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
