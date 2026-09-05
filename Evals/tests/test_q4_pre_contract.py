"""Real CUDA integration; Q4_PRE_CONFIG names local frozen physical inputs."""
import copy
import importlib.util
import json
import os

import pytest


def test_q4_pre_preserves_frozen_balanced64_contract(tmp_path):
    assert importlib.util.find_spec("banana_smasher.q4_pre_balanced64"), (
        "Balanced64 PRE requires the canonical original-Q4/native scoring path")
    from banana_smasher.q4_pre_balanced64 import run

    with open(os.environ["Q4_PRE_CONFIG"]) as handle:
        config = json.load(handle)
    # All independent drift cases enter the real path, before model allocation.
    mutations = {
        "window_ids": list(range(64)),
        "q4_inventory_sha256": "0" * 64,
        "teacher_bank": "NATIVE64",
        "forward_tokens": 1024,
        "microbatch": 1,
        "bank_dtype": "float32",
        "attention": "sdpa",
    }
    for key, value in mutations.items():
        wrong = copy.deepcopy(config)
        wrong["contract"][key] = value
        with pytest.raises(ValueError, match="Balanced64"):
            run(wrong, tmp_path / key, window_ids=[28, 56])

    result = run(config, tmp_path / "physical", window_ids=[28, 56])
    contract = result["contract"]
    assert contract["window_ids"] == [
        28, 56, 68, 71, 76, 99, 107, 122, 124, 130, 141, 156, 160, 171,
        180, 183, 185, 186, 196, 210, 212, 213, 218, 228, 232, 235, 249,
        270, 272, 273, 283, 288, 290, 295, 297, 306, 307, 309, 311, 328,
        331, 357, 362, 365, 368, 374, 376, 380, 384, 385, 391, 396, 413,
        429, 430, 437, 442, 447, 454, 462, 464, 475, 489, 499,
    ], "Balanced64 population changed"
    assert result["executed_window_ids"] == [28, 56]
    assert result["native"]["positions"] == result["q4"]["positions"] == 2048
    assert result["q4"]["decoded_units"] == 22016
    assert result["native"]["runtime"] == result["q4"]["runtime"]
    assert contract["teacher_bank"] == "TEACHER_0731_BALANCED64_V2"
    assert (contract["forward_tokens"], contract["scored_positions"],
            contract["microbatch"], contract["support"]) == (2048, 1024, 2, 8192)
    assert contract["bank_dtype"] == "float16"
    assert contract["score_dtype"] == "float64"
    print("BALANCED64_PRE_CONTRACT " + json.dumps(result, sort_keys=True))
