"""Decode-device placement is explicit and cannot become a fallback."""
import torch
import pytest
from test_sealed_qtip_execution import unit_row
from banana_smasher import sealed_qtip_unit


def test_cpu_device_selection_preserves_authentic_unit(tmp_path):
    _, row = unit_row(tmp_path)
    expected = sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager")
    actual = sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager", device="cpu")
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="decode device"):
        sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager", device="meta")


def test_artifact_threads_explicit_device_without_retry(tmp_path, monkeypatch):
    import json
    from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore
    _, row = unit_row(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"expert.weight": "absent.safetensors"}}))
    artifact = dict(artifact_root=str(tmp_path), source=dict(model_root=str(tmp_path)),
                    geometry=dict(routed_layer_ids=[3]), routed_tensors=[row], native_tensors=[],
                    packed_decode_execution="eager", packed_decode_device="cuda:0")
    calls = []
    def fail_device(root, incoming, *, execution, device):
        calls.append((execution, str(device)))
        raise RuntimeError("injected device failure")
    monkeypatch.setattr(sealed_qtip_unit, "decode_sealed_unit", fail_device)
    store = ArtifactTensorStore(artifact)
    assert store.packed_decode_device == "cuda:0"
    with pytest.raises(RuntimeError, match="injected device failure"):
        store.tensor("expert.weight")
    assert calls == [("eager", "cuda:0")]
    assert store.model_reads == 0 and store.payload_reads == 0
    artifact.pop("packed_decode_device")
    assert ArtifactTensorStore(artifact).packed_decode_device == "cpu"
    with pytest.raises(ValueError, match="decode device"):
        ArtifactTensorStore(dict(artifact, packed_decode_device="meta"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA placement requires claimed accelerator")
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("normalization", ["default", "rounded"])
def test_real_cuda_unit_decode_preserves_values(tmp_path, bits, normalization):
    import hashlib
    unit, row = unit_row(tmp_path)
    unit["geometry"]["K"] = bits
    unit["trellis"] = torch.randint(0, 65536, (bits * 32 * 32 // 16,)).to(torch.uint16)
    unit["SU"] = torch.linspace(.7, 1.3, 32)
    unit["SV"] = torch.linspace(1.2, .8, 32)
    p = tmp_path / "unit.pt"
    torch.save(unit, p)
    row["wire"]["unit"].update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    cpu = sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager", device="cpu")
    gpu = sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager", device="cuda:0", normalization=normalization)
    if normalization == "rounded":
        assert torch.equal(gpu.cpu(), cpu)
    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-6, rtol=1e-5)
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash/size"):
        sealed_qtip_unit.decode_sealed_unit(tmp_path, row, execution="eager", device="cuda:0")
