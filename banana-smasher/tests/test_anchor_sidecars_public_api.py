from __future__ import annotations

from pathlib import Path

import torch

import banana_smasher


def test_public_api_writes_and_requires_revision_bound_teacher_support_v2(
    tmp_path: Path,
) -> None:
    write_manifest = banana_smasher.write_teacher_support_manifest
    load_manifest = banana_smasher.load_teacher_support_manifest
    manifest_path = tmp_path / "teacher-support.json"

    write_manifest(
        manifest_path,
        windows=[
            {
                "window_id": 10,
                "idx": torch.tensor([[7, 8]], dtype=torch.int32),
                "logprob": torch.tensor([[-0.1, -2.0]], dtype=torch.float16),
            }
        ],
        bank_sha256="1" * 64,
        teacher_sha256="2" * 64,
        basis_sha256="3" * 64,
        model_id="DeepSeek-V4-Flash-0731",
    )

    manifest = load_manifest(
        manifest_path,
        expected_bank_sha256="1" * 64,
        expected_teacher_sha256="2" * 64,
        expected_basis_sha256="3" * 64,
        expected_model_id="DeepSeek-V4-Flash-0731",
        require_revision_binding=True,
    )
    assert manifest["schema"] == "banana-smasher-anchor-teacher-sidecars-v2"
    assert manifest["window_ids"] == [10]
