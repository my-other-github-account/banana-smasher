from unittest.mock import Mock

import torch

from repair_api.api import ResidentRepairAPI
from repair_api.modern_green_resident import (
    BASE_LRS,
    _admit_restored_optimizer_base_lrs,
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
        },
        receipt_path="receipt.json",
    )
