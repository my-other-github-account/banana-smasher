from __future__ import annotations

from types import SimpleNamespace

import torch

from repair_api.api import _candidate_read_and_one_layer_delta_proof


def test_candidate_read_proof_gathers_multidimensional_uint8_nibbles(tmp_path) -> None:
    class Expert:
        packed_w1 = torch.empty(0)

        @staticmethod
        def _projection_plane(projection, plane):
            assert projection == "w1"
            assert plane in {"SU", "SV"}
            return torch.ones((1, 32), dtype=torch.float32)

        @staticmethod
        def _project(*_args):
            return torch.tensor([[0.2]], dtype=torch.float32)

    packed = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)

    class Student:
        payload_disk_reads = 3
        routed_payload_bytes = 4096
        device = torch.device("cpu")
        experts = {0: Expert()}
        sources = {0: SimpleNamespace(master=torch.empty(0))}

        @staticmethod
        def get_tensor(name):
            if name.endswith(".weight"):
                return packed
            if name.endswith(".scale"):
                return scale
            raise AssertionError(name)

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    engine = SimpleNamespace(torch=torch, student=Student())

    proof = _candidate_read_and_one_layer_delta_proof(
        engine,
        layer=0,
        checkpoint_path=checkpoint,
        checkpoint_sha256="a" * 64,
    )

    assert proof["candidate_payload_reads"] == 3
    assert proof["candidate_payload_bytes"] == 4096
    assert 0.0 < proof["relative_l2"] < 1.0
