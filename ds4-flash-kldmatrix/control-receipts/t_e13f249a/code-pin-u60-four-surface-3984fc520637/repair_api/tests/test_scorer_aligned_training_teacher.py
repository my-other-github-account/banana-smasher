import inspect
from pathlib import Path

import pytest

from repair_api.api import ResidentRepairAPI, _require_scorer_aligned_preupdate_gate
from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    ModernGreenResidentEngine,
    _resolve_scorer_aligned_training_corpus,
    _resolve_scorer_aligned_training_teacher_root,
)


def test_legacy_training_teacher_root_is_unchanged(tmp_path: Path) -> None:
    teacher = tmp_path / "legacy"
    assert _resolve_scorer_aligned_training_teacher_root({"teacher_root": str(teacher)}) == teacher


def test_scorer_aligned_teacher_selects_shared_bank(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    config = {
        "teacher_root": str(tmp_path / "legacy"),
        "scorer_aligned_training_teacher_root": str(accepted),
        "validation_teacher_root": str(accepted),
    }
    assert _resolve_scorer_aligned_training_teacher_root(config) == accepted


def test_scorer_aligned_teacher_rejects_support_source_divergence(tmp_path: Path) -> None:
    config = {
        "teacher_root": str(tmp_path / "legacy"),
        "scorer_aligned_training_teacher_root": str(tmp_path / "train"),
        "validation_teacher_root": str(tmp_path / "score"),
    }
    with pytest.raises(ArtifactError, match="teacher roots diverge"):
        _resolve_scorer_aligned_training_teacher_root(config)


def test_engine_does_not_overwrite_scorer_aligned_teacher_with_legacy_root() -> None:
    source = inspect.getsource(ModernGreenResidentEngine.__init__)
    assert source.count("self.teacher_root =") == 1
    assert "self.teacher_root = _resolve_scorer_aligned_training_teacher_root(config)" in source


def test_w28_training_corpus_selects_canonical_eval_fixture(tmp_path: Path) -> None:
    training = tmp_path / "windows_ds4_TRAIN.json"
    canonical_eval = tmp_path / "windows_ds4_eval.json"
    canonical_eval.write_text("[]")
    config = {
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "resident_validation_proof": True,
        "corpus": str(training),
        "w28_only_training_corpus_source": "canonical_eval",
        "validation_corpus": str(canonical_eval),
    }
    assert _resolve_scorer_aligned_training_corpus(config) == canonical_eval.resolve()


def test_w28_training_corpus_alignment_is_fail_closed(tmp_path: Path) -> None:
    base = {
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "resident_validation_proof": True,
        "corpus": str(tmp_path / "windows_ds4_TRAIN.json"),
        "validation_corpus": str(tmp_path / "windows_ds4_eval.json"),
    }
    assert _resolve_scorer_aligned_training_corpus(base) == Path(base["corpus"]).resolve()
    with pytest.raises(ArtifactError, match="must equal canonical_eval"):
        _resolve_scorer_aligned_training_corpus(
            {**base, "w28_only_training_corpus_source": "training"}
        )
    with pytest.raises(ArtifactError, match="requires the static W28 provider"):
        _resolve_scorer_aligned_training_corpus(
            {
                **base,
                "resident_validation_proof": False,
                "w28_only_training_corpus_source": "canonical_eval",
            }
        )


def test_preupdate_gate_requires_bitwise_support_identity_and_1e_12_kld() -> None:
    diagnostic = {
        "support_index_equal_count": 8_388_608,
        "support_index_total": 8_388_608,
        "new_loss_group_token_kld_mean": 0.1371363252401352,
        "old_tailfix_token_kld_mean": 0.1371363252401352,
    }
    assert _require_scorer_aligned_preupdate_gate(diagnostic)["status"] == "PASS"

    with pytest.raises(ArtifactError, match="support identity failed"):
        _require_scorer_aligned_preupdate_gate(
            {**diagnostic, "support_index_equal_count": 8_388_607}
        )
    with pytest.raises(ArtifactError, match="KLD equality failed"):
        _require_scorer_aligned_preupdate_gate(
            {**diagnostic, "new_loss_group_token_kld_mean": 0.1371363252421352}
        )


def test_w28_only_actual_backward_gate_is_before_backward_and_fail_closed() -> None:
    source = inspect.getsource(ModernGreenResidentEngine._pipeline_pass)
    gate = source.index('self.config.get("w28_only_expected_pre_backward_scalar")')
    backward = source.index(".backward()", gate)
    assert gate < backward
    assert 'group != [28] or loss_divisor != 1' in source
    assert 'self.status["w28_only_pre_backward_gate"]' in source
    assert '"optimizer_steps_before_gate": 0' in source


def test_w28_only_training_token_span_matches_static_scorer() -> None:
    source = inspect.getsource(ModernGreenResidentEngine._loss_group)
    assert 'self.config.get("w28_only_training_token_span")' in source
    assert 'group != [28]' in source
    assert 'length = min(length, training_token_span)' in source


def test_w28_only_training_readout_matches_static_full_vocab_quantization() -> None:
    source = inspect.getsource(ModernGreenResidentEngine._loss_group)
    flag = 'self.config.get("w28_only_training_readout_normalization")'
    full_logprob = "self.torch.log_softmax(logits, dim=-1)"
    quantized_gather = ".gather(1, idx[:length]).to(self.torch.float16).float()"
    assert flag in source
    assert 'group != [28]' in source[source.index(flag):]
    assert full_logprob in source
    assert quantized_gather in source
    assert source.index(full_logprob) < source.index(quantized_gather)


def test_w28_forward_mode_ab_is_layerwise_and_instrument_only() -> None:
    run_layers = inspect.getsource(ModernGreenResidentEngine._run_layers)
    assert "layer_capture" in run_layers
    assert "layer_capture(index, current)" in run_layers

    diagnostic = inspect.getsource(ModernGreenResidentEngine.diagnose_w28_forward_modes)
    assert 'self.config.get("w28_forward_mode_ab_probe") is not True' in diagnostic
    assert "training=True" not in diagnostic  # no synthetic result prose
    assert "_run_layers(hidden, ids, train, layer_capture=capture)" in diagnostic
    assert '"first_divergent_layer"' in diagnostic
    assert '"optimizer_steps": 0' in diagnostic


def test_public_continue_training_returns_forward_mode_ab_before_updates() -> None:
    source = inspect.getsource(ResidentRepairAPI.continue_two_spark_real)
    probe = source.index('config.get("w28_forward_mode_ab_probe") is True')
    advance = source.index("engine.advance_to(")
    assert probe < advance
    assert "engine.diagnose_w28_forward_modes()" in source
    assert 'old_tailfix_token_kld_mean' in source
    assert '"banana-smasher-w28-forward-mode-ab-v1"' in source


def test_w28_batch_context_ab_is_row0_only_and_instrument_only() -> None:
    diagnostic = inspect.getsource(ModernGreenResidentEngine.diagnose_w28_batch_context)
    assert 'self.config.get("w28_batch_context_ab_probe") is not True' in diagnostic
    assert "for windows in ((28,), (28, 56))" in diagnostic
    assert "current[0:1]" in diagnostic
    assert "self._loss_group(hidden[0:1], [28])" in diagnostic
    assert "self.preload_validation([28]" in diagnostic
    assert "validation_ids[window]" in diagnostic
    assert "ids.shape[1]" in diagnostic
    assert "self.base.T.T_TRAIN" not in diagnostic
    assert "56 not in self.ids_cache" not in diagnostic
    assert '"first_divergent_layer"' in diagnostic
    assert '"optimizer_steps": 0' in diagnostic

    source = inspect.getsource(ResidentRepairAPI.continue_two_spark_real)
    probe = source.index('config.get("w28_batch_context_ab_probe") is True')
    advance = source.index("engine.advance_to(")
    assert probe < advance
    assert "engine.diagnose_w28_batch_context()" in source
    assert '"banana-smasher-w28-batch-context-ab-v1"' in source
