from __future__ import annotations

import inspect

import torch

from banana_smasher.hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime
from banana_smasher import backpack_runtime_exact64
from banana_smasher.resident_terminal_scorer import (
    ResidentScoreAccumulator,
    score_terminal_hidden,
)


def test_score_terminal_hidden_keeps_outputs_on_device_and_reuses_buffers() -> None:
    hidden = torch.arange(30, dtype=torch.float32).reshape(5, 6) / 10
    head = torch.nn.Linear(6, 11, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(66, dtype=torch.float32).reshape(11, 6) / 100)
    support = torch.tensor(
        [[0, 3, 7], [1, 4, 8], [2, 5, 9], [0, 6, 10], [3, 4, 5]],
        dtype=torch.int64,
    )
    q_lp = torch.empty((5, 3), dtype=torch.float16)
    q_argmax = torch.empty((5,), dtype=torch.int32)

    actual_lp, actual_argmax = score_terminal_hidden(
        hidden,
        support,
        head,
        chunk_size=2,
        q_lp_out=q_lp,
        q_argmax_out=q_argmax,
        compute_dtype=torch.float32,
    )

    logits = head(hidden).float()
    assert actual_lp.data_ptr() == q_lp.data_ptr()
    assert actual_argmax.data_ptr() == q_argmax.data_ptr()
    torch.testing.assert_close(actual_lp, logits.gather(1, support).half())
    torch.testing.assert_close(actual_argmax, logits.argmax(-1).int())
    assert ".cpu(" not in inspect.getsource(score_terminal_hidden)


def test_resident_accumulator_defers_cpu_finalization_and_matches_reference() -> None:
    accumulator = ResidentScoreAccumulator(torch)
    teacher_idx = torch.tensor([[4, 1, 0], [2, 3, 1]], dtype=torch.int32)
    teacher_lp = torch.tensor([[-0.1, -1.2, -2.1], [-0.2, -0.7, -2.3]], dtype=torch.float16)
    q_lp = torch.tensor([[-0.3, -0.9, -1.5], [-0.4, -0.8, -1.4]], dtype=torch.float16)
    q_argmax = torch.tensor([4, 3], dtype=torch.int32)

    accumulator.add("w0", teacher_idx, teacher_lp, q_lp, q_argmax)
    assert accumulator._rows[0].dtype == torch.float32
    result = accumulator.finalize()

    ref = teacher_lp.float()
    cand = q_lp.float()
    lp_n = ref - ref.logsumexp(-1, keepdim=True)
    lq_n = cand - cand.logsumexp(-1, keepdim=True)
    expected_sum = (lp_n.exp() * (lp_n - lq_n)).sum(dtype=torch.float64)
    assert result["positions"] == 2
    assert result["top1_matches"] == 1
    assert result["top1_agreement"] == 0.5
    assert abs(result["kld_sum"] - float(expected_sum)) < 1e-12
    assert result["per_window"][0]["window_id"] == "w0"
    add_source = inspect.getsource(ResidentScoreAccumulator.add)
    assert ".cpu(" not in add_source
    assert ".item(" not in add_source
    assert "bool(" not in add_source


def test_runtime_terminal_stage_delegates_to_resident_scorer_without_cpu_sync() -> None:
    source = inspect.getsource(DeepseekV4BackpackRuntime.terminal_stage)
    assert "score_terminal_hidden(" in source
    assert ".cpu()" not in source
    assert "q_lp_out" in source
    assert "q_argmax_out" in source


def test_exact64_terminal_keeps_teacher_and_reductions_resident() -> None:
    source = inspect.getsource(backpack_runtime_exact64._run_backpack_exact64)
    terminal = source[source.index("resident_teacher") :]
    assert "ResidentScoreAccumulator(" in terminal
    assert ".to(runtime.device)" in terminal
    assert "score_accumulator.add(" in terminal
    assert "resident_candidate" in terminal
    assert "resident_validated=True" in terminal
    assert "score_accumulator.finalize()" in terminal
    assert "_score_anchor_sidecars(" not in terminal
