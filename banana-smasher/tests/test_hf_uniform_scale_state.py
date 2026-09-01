from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher.hf_uniform_physical_provider import (
    HF_UNIFORM_PARAMETER_FAMILY,
    _HFUniformLayerStreamedBackend,
    _HFUniformTrainableSession,
    _TrainableArtifactTensorStore,
    _layer_split,
)
from banana_smasher.qtip1 import QTIP2_GEOMETRY, encode_qtip, gaussian_tlut


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path) -> tuple[dict[str, object], Path, Path, str]:
    root = tmp_path / "artifact"
    root.mkdir()
    matrix = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    encoded = encode_qtip(
        matrix,
        geometry=QTIP2_GEOMETRY,
        tlut=gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V),
    )
    trellis = root / "layer24.trellis.npy"
    scales = root / "layer24.scales.npy"
    np.save(trellis, encoded.packed, allow_pickle=False)
    np.save(scales, encoded.scales, allow_pickle=False)
    name = "model.layers.24.mlp.experts.0.gate_proj.weight"
    row = {
        "name": name,
        "shape": list(matrix.shape),
        "wire": {
            "geometry": QTIP2_GEOMETRY.as_mapping(),
            "trellis": {"path": trellis.name, "sha256": _sha(trellis)},
            "scales": {"path": scales.name, "sha256": _sha(scales)},
        },
    }
    return {
        "artifact_root": str(root),
        "routed_tensors": [row],
        "native_tensors": [],
    }, trellis, scales, name


class _CPUScaleSession:
    torch = torch
    device = "cpu"


def _backend(tmp_path: Path) -> tuple[_HFUniformLayerStreamedBackend, Path, Path, str]:
    artifact, trellis, scales, name = _artifact(tmp_path)
    backend = object.__new__(_HFUniformLayerStreamedBackend)
    backend.artifact = artifact
    backend.rank = 1
    backend.layer_split = {0: (0, 23), 1: (24, 44)}
    backend.session = _CPUScaleSession()
    backend.store = _TrainableArtifactTensorStore(artifact, tensor_overrides={})
    backend._parameters = {}
    backend._descriptors = ()
    backend._build_trainables()
    return backend, trellis, scales, name


def test_scale_only_backend_exposes_artifact_scale_member_not_dense_matrix(tmp_path: Path) -> None:
    backend, _trellis, scales, name = _backend(tmp_path)

    rows = backend.resident_parameters()
    assert len(rows) == 1
    descriptor, parameter = rows[0]
    assert descriptor.family == HF_UNIFORM_PARAMETER_FAMILY == "routed_q2_scales"
    assert descriptor.name == name
    assert descriptor.stable_id == f"artifact-scale:{scales.name}"
    assert tuple(parameter.shape) == (2,)
    assert backend.residency_metadata()["resident_bytes"] == 2 * np.dtype(np.float32).itemsize


def test_scale_compose_is_differentiable_and_never_mutates_artifact_members(tmp_path: Path) -> None:
    backend, trellis, scales, name = _backend(tmp_path)
    immutable_before = {_sha(trellis), _sha(scales)}

    decoded = backend.store.tensor(name)
    decoded.square().sum().backward()

    parameter = backend.resident_parameters()[0][1]
    assert parameter.grad is not None
    assert torch.count_nonzero(parameter.grad).item() == parameter.numel()
    assert {_sha(trellis), _sha(scales)} == immutable_before


def test_layer_materialization_preserves_scale_autograd_graph(tmp_path: Path) -> None:
    backend, _trellis, _scales, name = _backend(tmp_path)
    session = object.__new__(_HFUniformTrainableSession)
    session.torch = torch
    session.device = "cpu"
    session.store = backend.store
    session.combined_overrides = {}
    session._working_set_loads = 0
    session._prefix_for = lambda module: name.removesuffix("weight")
    module = torch.nn.Linear(32, 2, bias=False, device="meta")

    session._materialize(module)
    module(torch.ones((1, 32))).sum().backward()

    scale = backend.resident_parameters()[0][1]
    assert module.weight.grad_fn is not None
    assert scale.grad is not None
    assert torch.count_nonzero(scale.grad).item() == scale.numel()


def test_scale_checkpoint_state_restores_exact_values_used_by_compose(tmp_path: Path) -> None:
    backend, _trellis, _scales, name = _backend(tmp_path)
    original = backend.trainable_state_dict()
    baseline = backend.store.tensor(name).detach().clone()
    parameter = backend.resident_parameters()[0][1]
    with torch.no_grad():
        parameter.mul_(1.5)
    changed = backend.store.tensor(name).detach().clone()
    assert not torch.equal(changed, baseline)

    backend.load_trainable_state_dict(original)

    assert torch.equal(backend.store.tensor(name), baseline)
    assert backend.trainable_state_dict().keys() == original.keys()


def test_repair45_layer_split_is_exact_and_old_boundary_is_rejected() -> None:
    assert _layer_split({"0": [0, 23], "1": [24, 44]}, model_layer_count=45) == {
        0: (0, 23),
        1: (24, 44),
    }
    with pytest.raises(ValueError, match="exact repair45 rank layer ranges"):
        _layer_split({"0": [0, 22], "1": [23, 44]}, model_layer_count=45)
