from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import banana_smasher.anchor_sidecars as sidecars
from banana_smasher.anchor_sidecars import (
    CandidateSidecarWriter,
    load_candidate_manifest,
    load_teacher_support_manifest,
    load_teacher_window,
    score_anchor_sidecars,
    write_teacher_support_manifest,
)
from banana_smasher.hf_deepseek_v4_d4_adapter import _full_vocab_support_logprob


BANK_SHA = "1" * 64
TEACHER_SHA = "2" * 64
BASIS_SHA = "3" * 64
PACK_SHA = "4" * 64


def _teacher_manifest(root: Path, *, width: int = 8192) -> Path:
    manifest = root / "teacher-support.json"
    idx = torch.arange(width, dtype=torch.int32).repeat(2, 1)
    logprob = torch.log_softmax(
        torch.linspace(2.0, -2.0, width, dtype=torch.float32), dim=0
    ).to(torch.float16).repeat(2, 1)
    write_teacher_support_manifest(
        manifest,
        windows=[
            {"window_id": "window-a", "idx": idx, "logprob": logprob},
            {"window_id": "window-b", "idx": idx.flip(1), "logprob": logprob},
        ],
        bank_sha256=BANK_SHA,
        teacher_sha256=TEACHER_SHA,
    )
    return manifest


def test_width_8192_teacher_sidecars_roundtrip_without_inline_values(tmp_path: Path) -> None:
    manifest_path = _teacher_manifest(tmp_path)

    manifest = load_teacher_support_manifest(
        manifest_path,
        expected_bank_sha256=BANK_SHA,
        expected_teacher_sha256=TEACHER_SHA,
    )
    idx, logprob = load_teacher_window(manifest_path, "window-a")

    assert manifest["support_width"] == 8192
    assert manifest["window_ids"] == ["window-a", "window-b"]
    assert idx.shape == logprob.shape == (2, 8192)
    assert idx.dtype == torch.int32
    assert logprob.dtype == torch.float16
    assert "8191" not in manifest_path.read_text()
    assert manifest["windows"][0]["tensors"] == {
        "idx": {"dtype": "int32", "shape": [2, 8192]},
        "logprob": {"dtype": "float16", "shape": [2, 8192]},
    }


@pytest.mark.parametrize(
    ("idx", "logprob", "error"),
    [
        (
            torch.tensor([[7, 7]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.2]], dtype=torch.float16),
            "unique token IDs",
        ),
        (
            torch.tensor([[7, 8]], dtype=torch.int32),
            torch.tensor([[-0.2, -0.1]], dtype=torch.float16),
            "descending logprob order",
        ),
    ],
)
def test_teacher_sidecars_require_unique_descending_support_rows(
    tmp_path: Path, idx: torch.Tensor, logprob: torch.Tensor, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        write_teacher_support_manifest(
            tmp_path / "teacher.json",
            windows=[{"window_id": 0, "idx": idx, "logprob": logprob}],
            bank_sha256=BANK_SHA,
            teacher_sha256=TEACHER_SHA,
        )


def test_sidecar_loader_rejects_corruption_identity_and_window_order(tmp_path: Path) -> None:
    manifest_path = _teacher_manifest(tmp_path, width=2)
    manifest = json.loads(manifest_path.read_text())

    with pytest.raises(ValueError, match="bank_sha256 mismatch"):
        load_teacher_support_manifest(
            manifest_path,
            expected_bank_sha256="f" * 64,
        )

    first = manifest_path.parent / manifest["windows"][0]["path"]
    first.write_bytes(first.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_teacher_window(manifest_path, "window-a")

    clean_manifest = _teacher_manifest(tmp_path / "clean", width=2)
    writer = CandidateSidecarWriter(
        tmp_path / "candidate.json",
        teacher_manifest_path=clean_manifest,
        window_ids=["window-a", "window-b"],
        basis_sha256=BASIS_SHA,
        bank_sha256=BANK_SHA,
        model_id="fixture-model",
        pack_sha256=PACK_SHA,
    )
    with pytest.raises(ValueError, match="ordered window"):
        writer.write_window(
            "window-b",
            q_lp_at_ref=torch.zeros((2, 2), dtype=torch.float16),
            q_argmax=torch.zeros(2, dtype=torch.int32),
        )
    assert writer.write_window(
        "window-a",
        q_lp_at_ref=torch.zeros((2, 2), dtype=torch.float16),
        q_argmax=torch.zeros(2, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="pack_sha256 mismatch"):
        load_candidate_manifest(
            tmp_path / "candidate.json", expected_pack_sha256="f" * 64
        )


def test_sidecar_loaders_reject_symlink_payloads(tmp_path: Path) -> None:
    teacher_manifest = _teacher_manifest(tmp_path / "teacher", width=2)
    teacher = json.loads(teacher_manifest.read_text())
    teacher_sidecar = teacher_manifest.parent / teacher["windows"][0]["path"]
    external_teacher = tmp_path / "external-teacher.pt"
    external_teacher.write_bytes(teacher_sidecar.read_bytes())
    teacher_sidecar.unlink()
    teacher_sidecar.symlink_to(external_teacher)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_teacher_support_manifest(teacher_manifest)

    clean_teacher = _teacher_manifest(tmp_path / "candidate", width=2)
    candidate_manifest = tmp_path / "candidate.json"
    writer = CandidateSidecarWriter(
        candidate_manifest,
        teacher_manifest_path=clean_teacher,
        window_ids=["window-a", "window-b"],
        basis_sha256=BASIS_SHA,
        bank_sha256=BANK_SHA,
        model_id="fixture-model",
        pack_sha256=PACK_SHA,
    )
    writer.write_window(
        "window-a",
        q_lp_at_ref=torch.zeros((2, 2), dtype=torch.float16),
        q_argmax=torch.zeros(2, dtype=torch.int32),
    )
    candidate = json.loads(candidate_manifest.read_text())
    candidate_sidecar = candidate_manifest.parent / candidate["windows"][0]["path"]
    external_candidate = tmp_path / "external-candidate.pt"
    external_candidate.write_bytes(candidate_sidecar.read_bytes())
    candidate_sidecar.unlink()
    candidate_sidecar.symlink_to(external_candidate)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_candidate_manifest(candidate_manifest)


def test_candidate_sidecar_resume_preserves_completed_window_bytes(tmp_path: Path) -> None:
    teacher_manifest = _teacher_manifest(tmp_path, width=2)
    candidate_manifest = tmp_path / "candidate.json"
    kwargs = {
        "teacher_manifest_path": teacher_manifest,
        "window_ids": ["window-a", "window-b"],
        "basis_sha256": BASIS_SHA,
        "bank_sha256": BANK_SHA,
        "model_id": "fixture-model",
        "pack_sha256": PACK_SHA,
    }
    writer = CandidateSidecarWriter(candidate_manifest, **kwargs)
    assert writer.write_window(
        "window-a",
        q_lp_at_ref=torch.tensor([[-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float16),
        q_argmax=torch.tensor([0, 1], dtype=torch.int32),
    )
    manifest = json.loads(candidate_manifest.read_text())
    sidecar = candidate_manifest.parent / manifest["windows"][0]["path"]
    before = (sidecar.read_bytes(), sidecar.stat().st_mtime_ns)

    resumed = CandidateSidecarWriter(candidate_manifest, **kwargs)
    assert resumed.completed_window_ids == ["window-a"]
    assert not resumed.write_window(
        "window-a",
        q_lp_at_ref=torch.full((2, 2), 99.0, dtype=torch.float16),
        q_argmax=torch.tensor([1, 1], dtype=torch.int32),
    )
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before


def test_candidate_sidecar_adopts_committed_window_after_manifest_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_manifest = _teacher_manifest(tmp_path, width=2)
    candidate_manifest = tmp_path / "candidate.json"
    kwargs = {
        "teacher_manifest_path": teacher_manifest,
        "window_ids": ["window-a", "window-b"],
        "basis_sha256": BASIS_SHA,
        "bank_sha256": BANK_SHA,
        "model_id": "fixture-model",
        "pack_sha256": PACK_SHA,
    }
    writer = CandidateSidecarWriter(candidate_manifest, **kwargs)
    original_atomic_bytes = getattr(sidecars, "_atomic_bytes")

    def interrupt_manifest_commit(path: Path, payload: bytes) -> None:
        raise OSError("fixture interruption after sidecar commit")

    monkeypatch.setattr(sidecars, "_atomic_bytes", interrupt_manifest_commit)
    with pytest.raises(OSError, match="fixture interruption"):
        writer.write_window(
            "window-a",
            q_lp_at_ref=torch.tensor(
                [[-0.1, -1.2], [-0.2, -1.5]], dtype=torch.float16
            ),
            q_argmax=torch.tensor([7, 8], dtype=torch.int32),
        )
    monkeypatch.setattr(sidecars, "_atomic_bytes", original_atomic_bytes)

    sidecar = next((tmp_path / "candidate.sidecars").glob("*.pt"))
    before = (sidecar.read_bytes(), sidecar.stat().st_mtime_ns)
    resumed = CandidateSidecarWriter(candidate_manifest, **kwargs)
    assert resumed.completed_window_ids == ["window-a"]
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before


def test_sidecar_scorer_matches_historical_renorm_kld_and_full_vocab_top1(
    tmp_path: Path,
) -> None:
    teacher_manifest = tmp_path / "teacher.json"
    teacher_logits = torch.tensor(
        [[4.0, 3.0, 2.0, -1.0], [0.5, 3.0, 1.0, 2.0]], dtype=torch.float32
    )
    teacher_lp = torch.log_softmax(teacher_logits, dim=-1)
    teacher_idx = torch.tensor([[0, 1], [1, 3]], dtype=torch.int32)
    write_teacher_support_manifest(
        teacher_manifest,
        windows=[
            {
                "window_id": 7,
                "idx": teacher_idx,
                "logprob": teacher_lp.gather(1, teacher_idx.long()).to(torch.float16),
            }
        ],
        bank_sha256=BANK_SHA,
        teacher_sha256=TEACHER_SHA,
    )

    candidate_logits = torch.tensor(
        [[1.0, 2.5, 0.0, -4.0], [0.0, 2.0, 4.0, 1.0]], dtype=torch.float32
    )
    candidate_lp = torch.log_softmax(candidate_logits, dim=-1)
    gathered_lp = candidate_lp.gather(1, teacher_idx.long()).to(torch.float16)
    writer = CandidateSidecarWriter(
        tmp_path / "candidate.json",
        teacher_manifest_path=teacher_manifest,
        window_ids=[7],
        basis_sha256=BASIS_SHA,
        bank_sha256=BANK_SHA,
        model_id="fixture-model",
        pack_sha256=PACK_SHA,
    )
    assert writer.write_window(
        7,
        q_lp_at_ref=gathered_lp,
        q_argmax=candidate_logits.argmax(-1).to(torch.int32),
    )

    result = score_anchor_sidecars(teacher_manifest, tmp_path / "candidate.json")

    ref_lp = teacher_lp.gather(1, teacher_idx.long()).to(torch.float16).float()
    cand_lp = gathered_lp.float()
    lp_n = ref_lp - ref_lp.logsumexp(-1, keepdim=True)
    lq_n = cand_lp - cand_lp.logsumexp(-1, keepdim=True)
    historical = (lp_n.exp() * (lp_n - lq_n)).sum(-1)
    gathered_logits = candidate_logits.gather(1, teacher_idx.long())
    logits_normalized = gathered_logits - gathered_logits.logsumexp(-1, keepdim=True)
    full_lp_normalized = candidate_lp.gather(1, teacher_idx.long())
    full_lp_normalized -= full_lp_normalized.logsumexp(-1, keepdim=True)

    assert result["mean_kld"] == pytest.approx(historical.mean().item(), rel=1e-7)
    assert result["kld_sum"] == pytest.approx(historical.sum().item(), rel=1e-7)
    assert result["top1_matches"] == 0
    assert result["per_window"][0]["kld_sum"] == pytest.approx(
        historical.sum().item(), rel=1e-7
    )
    assert result["per_window"][0]["top1_matches"] == 0
    assert result["top1_agreement"] == 0.0
    assert torch.allclose(logits_normalized, full_lp_normalized, rtol=0, atol=1e-6)
    candidate = load_candidate_manifest(tmp_path / "candidate.json")
    assert candidate["identities"]["teacher_manifest_sha256"] == hashlib.sha256(
        teacher_manifest.read_bytes()
    ).hexdigest()


def test_sidecar_scorer_applies_historical_1024_position_cutoff(
    tmp_path: Path,
) -> None:
    teacher_manifest = tmp_path / "teacher.json"
    positions = 1025
    teacher_idx = torch.tensor([[5, 6]], dtype=torch.int32).repeat(positions, 1)
    teacher_lp = torch.tensor([[-0.1, -2.0]], dtype=torch.float16).repeat(
        positions, 1
    )
    write_teacher_support_manifest(
        teacher_manifest,
        windows=[
            {"window_id": 0, "idx": teacher_idx, "logprob": teacher_lp}
        ],
        bank_sha256=BANK_SHA,
        teacher_sha256=TEACHER_SHA,
    )
    candidate_lp = teacher_lp.clone()
    candidate_lp[-1] = torch.tensor([-2.0, -0.1], dtype=torch.float16)
    writer = CandidateSidecarWriter(
        tmp_path / "candidate.json",
        teacher_manifest_path=teacher_manifest,
        window_ids=[0],
        basis_sha256=BASIS_SHA,
        bank_sha256=BANK_SHA,
        model_id="fixture-model",
        pack_sha256=PACK_SHA,
    )
    writer.write_window(
        0,
        q_lp_at_ref=candidate_lp,
        q_argmax=torch.full((positions,), 5, dtype=torch.int32),
    )

    result = score_anchor_sidecars(teacher_manifest, tmp_path / "candidate.json")

    assert result["position_cutoff"] == 1024
    assert result["positions"] == 1024
    assert result["per_window"][0]["positions"] == 1024
    assert result["mean_kld"] == 0.0


def test_candidate_may_be_shorter_than_teacher_and_scores_historical_minimum(
    tmp_path: Path,
) -> None:
    teacher_manifest = tmp_path / "teacher.json"
    teacher_positions = 1025
    candidate_positions = 1024
    teacher_idx = torch.tensor([[5, 6]], dtype=torch.int32).repeat(
        teacher_positions, 1
    )
    teacher_lp = torch.tensor([[-0.1, -2.0]], dtype=torch.float16).repeat(
        teacher_positions, 1
    )
    write_teacher_support_manifest(
        teacher_manifest,
        windows=[
            {"window_id": 0, "idx": teacher_idx, "logprob": teacher_lp}
        ],
        bank_sha256=BANK_SHA,
        teacher_sha256=TEACHER_SHA,
    )
    writer = CandidateSidecarWriter(
        tmp_path / "candidate.json",
        teacher_manifest_path=teacher_manifest,
        window_ids=[0],
        basis_sha256=BASIS_SHA,
        bank_sha256=BANK_SHA,
        model_id="fixture-model",
        pack_sha256=PACK_SHA,
    )

    assert writer.write_window(
        0,
        q_lp_at_ref=teacher_lp[:candidate_positions].clone(),
        q_argmax=torch.full((candidate_positions,), 5, dtype=torch.int32),
    )
    result = score_anchor_sidecars(teacher_manifest, tmp_path / "candidate.json")

    assert result["position_cutoff"] == 1024
    assert result["positions"] == candidate_positions
    assert result["mean_kld"] == 0.0


def test_hf_terminal_helper_gathers_full_softmax_for_arbitrary_support_width() -> None:
    logits = torch.tensor([[4.0, 1.0, -2.0, 3.0]], dtype=torch.float32)
    support = torch.tensor([[3, 0, 2]], dtype=torch.int64)

    q_lp, q_argmax = _full_vocab_support_logprob(logits, support)

    expected = torch.log_softmax(logits, dim=-1).gather(1, support)
    assert q_lp.dtype == torch.float16
    assert q_argmax.dtype == torch.int32
    assert torch.equal(q_lp, expected.to(torch.float16))
    assert q_argmax.tolist() == [0]
