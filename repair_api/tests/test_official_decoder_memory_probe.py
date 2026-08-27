import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from repair_api.official_k2_resident_score import OfficialK2ResidentRankEngine


def test_decoder_memory_probe_is_opt_in_and_records_allocator_state(tmp_path: Path) -> None:
    class Cuda:
        @staticmethod
        def mem_get_info(_device):
            return 11, 22

        @staticmethod
        def memory_allocated(_device):
            return 33

        @staticmethod
        def memory_reserved(_device):
            return 44

    path = tmp_path / "probe.jsonl"
    engine = cast(Any, OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine))
    engine.config = {"resident_validation_memory_probe_path": str(path)}
    engine.rank = 0
    engine.torch = SimpleNamespace(cuda=Cuda())
    engine.student = SimpleNamespace(device="cuda:0")

    engine._append_decoder_memory_probe(phase="before", layer=7)

    row = json.loads(path.read_text())
    assert row["schema"] == "banana-smasher-decoder-memory-probe-v1"
    assert (row["phase"], row["layer"]) == ("before", 7)
    assert row["device_free_bytes"] == 11
    assert row["allocated_bytes"] == 33
    assert row["reserved_bytes"] == 44


def test_decoder_memory_probe_does_nothing_when_unconfigured(tmp_path: Path) -> None:
    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.config = {}
    engine._append_decoder_memory_probe(phase="before", layer=0)
    assert list(tmp_path.iterdir()) == []


def test_modern_green_binds_probe_delegate_before_imported_decoder_call() -> None:
    source = (
        Path(__file__).parents[1] / "modern_green_resident.py"
    ).read_text()
    binding = source.index(
        "self._append_decoder_memory_probe = self._official_append_decoder_memory_probe"
    )
    workspace_delegate = source.index(
        "def _official_attention_workspace_for("
    )
    workspace_binding = source.index(
        "self._attention_workspace_for = self._official_attention_workspace_for"
    )
    release_delegate = source.index(
        "def _official_release_attention_output_workspace("
    )
    release_binding = source.index(
        "self._release_attention_output_workspace = ("
    )
    release_target = source.index(
        "self._official_release_attention_output_workspace", release_binding
    )
    dispatch = source.index(
        "OfficialK2ResidentRankEngine._run_layers)(self, hidden, ids)"
    )
    assert binding < dispatch
    assert workspace_delegate < workspace_binding < dispatch
    assert release_delegate < release_binding < release_target < dispatch
    assert "_official_observe_decoder_lifetime" not in source
