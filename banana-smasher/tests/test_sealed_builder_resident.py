from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from banana_smasher.sealed_builder_resident import (
    ResidentSealedModel,
    bind_sealed_builder,
)


def _fake_builder(path: Path) -> str:
    path.write_text(
        """\
class PlaneSource:\n    pass\n\ndef build_layer_sd(layer, weight_map, get_tensor, mode, planes):\n    planes.calls.append(('build', layer, mode))\n    return {'mutable': planes.values[layer], 'static': get_tensor('static')}\n\ndef materialize_layer(model, layer, state, config):\n    planes_layer = model.model.layers[layer]\n    planes_layer.load_state_dict(state, strict=True, assign=True)\n    model.calls.append(('materialize', layer, config))\n    return planes_layer\n"""
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_binding_and_load_once_swap_many_preserve_model_and_static_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sealed.py"
    source_sha = _fake_builder(source)
    binding = bind_sealed_builder(source, expected_sha256=source_sha)
    assert binding.build_layer_sd is binding.module.build_layer_sd
    assert binding.materialize_layer is binding.module.materialize_layer
    assert binding.plane_source is binding.module.PlaneSource

    layer = torch.nn.Module()
    layer.register_parameter("mutable", torch.nn.Parameter(torch.tensor([0.0])))
    layer.register_parameter("static", torch.nn.Parameter(torch.tensor([0.0])))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        calls=[],
    )
    first = SimpleNamespace(calls=[], values={0: torch.tensor([1.0])})
    second = SimpleNamespace(calls=[], values={0: torch.tensor([9.0])})
    resident = ResidentSealedModel(
        binding=binding,
        model=model,
        config="sealed-config",
        weight_map={"static": "shard"},
        get_tensor=lambda _name: torch.tensor([4.0]),
        layers=(0,),
    )

    built = resident.materialize_once(first)
    original_identity = id(model)
    assert built["model_build_count"] == 1
    assert first.calls == [("build", 0, "planes")]
    assert torch.equal(layer.mutable, torch.tensor([1.0]))
    assert torch.equal(layer.static, torch.tensor([4.0]))

    swapped = resident.hot_swap(second, mutable_keys_by_layer={0: ("mutable",)})
    assert id(model) == original_identity == swapped["model_object_identity"]
    assert resident.model_build_count == 1
    assert swapped["swap_count"] == 1
    assert swapped["loaded_key_count"] == 1
    assert swapped["load_method"] == "torch.nn.Module.load_state_dict"
    assert swapped["assign"] is False
    assert second.calls == [("build", 0, "planes")]
    assert torch.equal(layer.mutable, torch.tensor([9.0]))
    assert torch.equal(layer.static, torch.tensor([4.0]))


def test_binding_refuses_source_sha_drift(tmp_path: Path) -> None:
    source = tmp_path / "sealed.py"
    _fake_builder(source)
    with pytest.raises(RuntimeError, match="sealed builder SHA mismatch"):
        bind_sealed_builder(source, expected_sha256="0" * 64)


def test_hot_swap_requires_materialized_model_and_declared_sealed_keys(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sealed.py"
    binding = bind_sealed_builder(source, expected_sha256=_fake_builder(source))
    layer = torch.nn.Module()
    layer.register_parameter("mutable", torch.nn.Parameter(torch.tensor([0.0])))
    layer.register_parameter("static", torch.nn.Parameter(torch.tensor([0.0])))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]), calls=[])
    planes = SimpleNamespace(calls=[], values={0: torch.tensor([1.0])})
    resident = ResidentSealedModel(
        binding=binding,
        model=model,
        config=None,
        weight_map={"static": "shard"},
        get_tensor=lambda _name: torch.tensor([4.0]),
        layers=(0,),
    )
    with pytest.raises(RuntimeError, match="must be materialized"):
        resident.hot_swap(planes, mutable_keys_by_layer={0: ("mutable",)})
    resident.materialize_once(planes)
    with pytest.raises(RuntimeError, match="keys absent"):
        resident.hot_swap(planes, mutable_keys_by_layer={0: ("unknown",)})
