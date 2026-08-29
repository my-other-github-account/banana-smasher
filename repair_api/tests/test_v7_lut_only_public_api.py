from types import SimpleNamespace
from unittest.mock import Mock

import torch

from repair_api import modern_green_resident as resident_module
from repair_api.api import ResidentRepairAPI
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
