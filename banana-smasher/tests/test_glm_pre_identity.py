"""Regression for the real 256-of-288 native-suffix admission defect (metadata only)."""
import hashlib
import json

import pytest

from banana_smasher.hf_balanced64 import score_balanced64_pre
from test_hf_balanced64_api import _FixtureRuntime, _lock


@pytest.mark.parametrize("selected_experts", [256, 288])
def test_public_pre_checks_routed_suffix_before_custom_runtime(tmp_path, selected_experts):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text(json.dumps({"text_config": {
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "n_routed_experts": 288,
    }}))
    names = {f"model.layers.1.mlp.experts.{e}.{p}_proj.weight"
             for e in range(288) for p in ("gate", "up", "down")}
    names.add("lm_head.weight")
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {n: "fixture.safetensors" for n in names}}))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    corpus = tmp_path / "corpus.json"
    corpus.write_text("fixture")
    lock = _lock(sha(corpus), sha(index))
    routed = {n for n in names if ".experts." in n and int(n.split(".experts.")[1].split(".")[0]) < selected_experts}
    native = names - routed
    artifact = {
        "schema": "banana-smasher-hf-moe-uniform-artifact-v1", "status": "PASS",
        "reload_verified": True,
        "intent": {"tier": "q2", "scope": "routed_only", "native_rest": True},
        "source": {"model_root": str(model), "model_index_sha256": sha(index),
                   "config_sha256": sha(config)},
        "routed_tensors": [{"name": n} for n in sorted(routed)],
        "native_tensors": [{"name": n} for n in sorted(native)],
        "coverage": {"duplicates": [], "gaps": []},
        "accounting": {"routed_tensor_count": len(routed), "planned_routed_tensor_count": len(routed),
                       "native_tensor_count": len(native), "planned_native_tensor_count": len(native)},
    }
    teacher = {"schema": "banana-smasher-balanced64-teacher-capture-v1", "status": "PASS",
               "suite_lock_sha256": lock["suite_lock_sha256"], "corpus_sha256": sha(corpus), "row_count": 64}
    def score():
        return score_balanced64_pre(artifact, teacher_capture=teacher, suite_lock=lock,
                                   corpus=corpus, receipt_path=tmp_path / "PRE.json", runtime=_FixtureRuntime())

    if selected_experts == 256:
        with pytest.raises(ValueError, match="routed-only inventory"):
            score()
        assert not (tmp_path / "PRE.json").exists()
    else:
        assert score()["rows_sealed"] == 64
