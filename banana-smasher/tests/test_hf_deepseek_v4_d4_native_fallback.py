from pathlib import Path

import torch
from safetensors.torch import save_file

from banana_smasher.hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime


def test_get_tensor_uses_complete_native_source_when_primary_shard_omits_key(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    complete = tmp_path / "complete"
    primary.mkdir()
    complete.mkdir()
    shard = "model-00001-of-00001.safetensors"
    key = "layers.10.ffn.experts.238.w2.weight"
    save_file({"other": torch.zeros(1)}, primary / shard)
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({key: expected}, complete / shard)

    runtime = DeepseekV4D4Runtime.__new__(DeepseekV4D4Runtime)
    runtime.model_root = primary
    runtime.weight_map = {key: shard}
    runtime.native_source_root = complete
    runtime.native_source_weight_map = {key: shard}
    runtime._counted_paths = set()
    runtime._bytes_read = 0

    observed = runtime._get_tensor(key)

    assert torch.equal(observed, expected)
    assert complete / shard in runtime._counted_paths
    assert primary / shard not in runtime._counted_paths
