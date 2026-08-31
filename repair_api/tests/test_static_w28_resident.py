import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from repair_api.balanced64 import ScoreResult
from repair_api import static_w28_resident
from repair_api.api import _localize_official_k2_rank_seat


def test_static_wire_loader_uses_bounded_ordered_parallel_reads(tmp_path) -> None:
    source_path = (
        Path(__file__).parents[1] / "assets" / "static_w28_fast_v7_expert_base.py"
    )
    tree = ast.parse(source_path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_load_projection_payloads_into", "_load_projection_payloads"}
    ]
    seen: dict[str, object] = {}

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            seen["max_workers"] = kwargs.get("max_workers")
            seen["thread_name_prefix"] = kwargs.get("thread_name_prefix")
            super().__init__(*args, **kwargs)

    import numpy as np
    import torch

    namespace = {
        "Path": Path,
        "ThreadPoolExecutor": RecordingExecutor,
        "np": np,
        "torch": torch,
        "PACKED_BYTES": 64,
        "_managed_packed_allocation": lambda shape, device: (
            1,
            (tensor := torch.empty(shape, dtype=torch.int16)),
            tensor.numpy(),
        ),
        "_managed_packed_tensor": lambda pointer, owner, shape, device: owner,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    paths = []
    expected_packed = []
    for expert in range(3):
        packed = (np.arange(32, dtype="<i2") + expert * 100).reshape(1, 1, 32)
        su = np.arange(16, dtype="<f2") + expert
        sv = np.arange(16, dtype="<f2") + expert * 2
        path = tmp_path / f"E{expert:03d}_w1.q2v7wire"
        path.write_bytes(packed.tobytes() + su.tobytes() + sv.tobytes() + b"wire")
        paths.append(path)
        expected_packed.append(torch.from_numpy(packed.copy()))

    packed, su, sv, read_calls, read_bytes = namespace["_load_projection_payloads"](
        paths, m=16, k=16, packed_bytes=64, pin_memory=False,
        device=torch.device("cpu"),
    )

    assert seen == {"max_workers": 3, "thread_name_prefix": "w28-wire-read"}
    assert torch.equal(packed, torch.stack(expected_packed))
    assert tuple(su.shape) == (3, 16)
    assert tuple(sv.shape) == (3, 16)
    assert read_calls == 3
    assert read_bytes == 3 * 132

    source = source_path.read_text()
    assert "pin_memory: bool = True" in source
    assert source.count("non_blocking=True") == 2
    first_copy = source.index("su_cpu.to(device=device, non_blocking=True)")
    last_copy = source.index("sv_cpu.to(device=device, non_blocking=True)", first_copy)
    sync = source.index("stream.synchronize()", last_copy)
    managed_allocate = source.index("cudaMallocManaged")
    managed_prefetch = source.index("cudaMemPrefetchAsync", managed_allocate)
    managed_cpu_view = source.index("np.ctypeslib.as_array(owner)", managed_prefetch)
    managed_fill = source.index("packed[expert] =", managed_cpu_view)
    managed_cuda_alias = source.index(
        "packed_tensor = _managed_packed_tensor", managed_fill
    )
    assert "CudaMemLocation(2, 0)" in source
    assert (
        managed_allocate
        < managed_prefetch
        < managed_cpu_view
        < managed_fill
        < managed_cuda_alias
    )
    assert "_construct_storage_from_data_pointer" in source
    assert "_construct_CUDA_Tensor_From_Storage_And_Metadata" in source
    assert 'setattr(tensor, "_managed_cpu_owner", owner)' in source
    assert "packed_cpu.to(device=device" not in source
    assert "stream = torch.cuda.Stream(device=device)" in source
    assert "with torch.cuda.stream(stream):" in source
    assert 'thread_name_prefix="w28-h2d"' in source
    assert "transfer_pool.submit(" in source
    assert "for projection in PROJECTIONS:" in source[source.index("with ThreadPoolExecutor(") :]
    assert first_copy < last_copy < sync


def test_static_provider_constructs_one_managed_alias_per_layer() -> None:
    source = (
        Path(__file__).parents[1]
        / "assets"
        / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    constructor = source[source.index("class FullyResidentGroupedV7Experts") :]
    allocation = constructor.index("arena_pointer, arena_owner, arena_cpu")
    projection_loop = constructor.index(
        "for projection_index, projection in enumerate(PROJECTIONS)", allocation
    )
    fill = constructor.index("_load_projection_payloads_into(", projection_loop)
    alias = constructor.index("arena = _managed_packed_tensor(", fill)
    transfer = constructor.index("with ThreadPoolExecutor(", alias)
    assert allocation < projection_loop < fill < alias < transfer
    assert "arena_cpu[projection_index].reshape(packed_shape)" in constructor
    assert "arena[projection_index].reshape(packed_shape)" in constructor


def test_static_wire_cache_remains_kernel_reclaimable_without_startup_fadvise() -> None:
    source_path = (
        Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py"
    )
    source = source_path.read_text()
    release = source[source.index("        def release_expert_source_cache(") :]
    release = release[: release.index("\n\n        self.get_tensor")]
    assert "gc.collect()" in release
    assert "posix_fadvise" not in release
    assert "POSIX_FADV_DONTNEED" not in release
    assert "member_paths" not in release

    model_release = source[source.index("        def release_model_source_cache(") :]
    model_release = model_release[: model_release.index("\n\n        def release_expert_source_cache")]
    assert "handles.clear()" in model_release
    assert "gc.collect()" in model_release
    assert "posix_fadvise" not in model_release
    assert "POSIX_FADV_DONTNEED" not in model_release


def test_l006_identical_duplicate_wire_candidates_are_unambiguous(tmp_path) -> None:
    """Mirror the accepted PlaneSource rule: identical duplicates are one member."""
    source_path = (
        Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py"
    )
    tree = ast.parse(source_path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"sha256_file", "resolve_wire_candidate", "active_wire_templates"}
    ]
    selected.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "WIRE_MEMBER_TEMPLATES"
            for target in node.targets
        )
    )
    namespace = {"Path": Path, "Iterable": Iterable, "os": __import__("os"), "hashlib": hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)

    flat = tmp_path / "L006" / "E000_w1.q2v7wire"
    nested = tmp_path / "L006" / "wire" / "E000" / "w1.q2v7wire"
    flat.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    flat.write_bytes(b"accepted-wire")
    nested.write_bytes(flat.read_bytes())

    resolve = namespace["resolve_wire_candidate"]
    original_sha256_file = namespace["sha256_file"]
    namespace["sha256_file"] = lambda path: (_ for _ in ()).throw(
        AssertionError("unique candidate must not be rehashed")
    )
    assert resolve([flat], member="L006 E000/w1") == flat.resolve()
    namespace["sha256_file"] = original_sha256_file
    assert resolve([flat, nested], member="L006 E000/w1") == flat.resolve()
    active_templates = namespace["active_wire_templates"]
    unique_root = tmp_path / "unique-layout"
    unique_flat = unique_root / "E000_w1.q2v7wire"
    unique_flat.parent.mkdir(parents=True)
    unique_flat.write_bytes(b"wire")
    assert active_templates(unique_root) == ("E{expert:03d}_{projection}.q2v7wire",)
    duplicate_root = tmp_path / "duplicate-layout"
    duplicate_flat = duplicate_root / "E000_w1.q2v7wire"
    duplicate_nested = duplicate_root / "wire" / "E000" / "w1.q2v7wire"
    duplicate_flat.parent.mkdir(parents=True)
    duplicate_nested.parent.mkdir(parents=True)
    duplicate_flat.write_bytes(b"wire")
    duplicate_nested.write_bytes(b"wire")
    assert len(active_templates(duplicate_root)) == 2
    nested.write_bytes(b"conflicting-wire")
    try:
        resolve([flat, nested], member="L006 E000/w1")
    except RuntimeError as exc:
        assert str(exc) == "L006 E000/w1 conflicting duplicate"
    else:
        raise AssertionError("conflicting L006 duplicate was accepted")


def _write_reference(path: Path) -> str:
    path.write_text(json.dumps({
        "schema": "banana-smasher-sealed-2x2-cell-v1",
        "status": "PASS",
        "basis_sha256": static_w28_resident.BASIS_SHA256,
        "loaded_sha": static_w28_resident.CHECKPOINT_SHA256,
        "kld_mean": 0.1364830042977786,
        "top1": 880,
        "windows": 1,
    }))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_static_w28_calls_existing_resident_scorer_once_and_seals_truth(monkeypatch, tmp_path) -> None:
    reference = tmp_path / "B2_PUBLISHED_PRE.json"
    reference_sha = _write_reference(reference)
    seen = {}

    class API:
        artifact = SimpleNamespace(
            manifest={"score": {"official_k2_resident": {"provider_resolution_mode": "STATIC_W28_GROUPED"}}}
        )

        def score(self, checkpoint, windows):
            seen["checkpoint"] = checkpoint
            seen["windows"] = tuple(windows)
            return ScoreResult(
                checkpoint=checkpoint,
                windows=(28,),
                positions=1024,
                support=8192,
                kld=0.1364830042977786,
                top1=880,
                top1_rate=880 / 1024,
                artifact_root=str(tmp_path / "artifact"),
                spec="balanced64-v1",
                candidate_dir="fully-resident-official-k2",
                execution_mode="resident_in_memory",
                resident_load_seconds=179.0,
                timed_wall_seconds=23.0,
                identity={
                    "checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256,
                    "model_index_sha256": static_w28_resident.BASIS_SHA256,
                },
                runtime_counters={
                    "resident_engine_loads": 1,
                    "resident_checkpoint_rebinds": 0,
                    "timed_score_file_reads": 0,
                    "resident_ready": [{"checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256}],
                },
            )

    monkeypatch.setattr(
        static_w28_resident.ResidentRepairAPI,
        "open",
        lambda root, official_rank_seat=None: API(),
    )
    monkeypatch.setattr(
        static_w28_resident.sealed_pre_forward,
        "bind_sealed_pre_resident_config",
        lambda config: {"status": "PASS", "builder_sha256": "builder", "known_value_fixture": {
            "window": 28, "kld_mean": 0.1364830042977786, "top1": 880,
        }},
    )

    receipt = static_w28_resident.run_static_w28_resident_acceptance(
        task="t_test",
        root=tmp_path / "run",
        artifact_root=tmp_path / "artifact",
        checkpoint="PRE",
        canonical_pin="deadbeef",
        reference_receipt=reference,
        reference_sha256=reference_sha,
    )

    assert seen == {"checkpoint": "PRE", "windows": (28,)}
    assert receipt["status"] == "PASS"
    assert receipt["resident_state_persisted"] is True
    assert receipt["measurement"]["kld_mean"] == 0.1364830042977786
    assert receipt["measurement"]["top1"] == 880
    assert receipt["measurement"]["timed_wall_seconds"] == 23.0
    assert receipt["full64_launched"] is False
    assert receipt["sealed_truth_receipt_sha256"] == reference_sha
    assert receipt["source_binding"]["builder_sha256"] == "builder"


def test_rank_seat_localization_changes_only_runtime_rendezvous() -> None:
    original = {
        "rank": 0,
        "world_size": 2,
        "master_addr": "192.168.200.1",
        "master_port": 30391,
        "qsfp_host_ip_by_rank": {"0": "192.168.200.1", "1": "192.168.200.3"},
        "checkpoint_sha256": "sealed",
    }
    seat = {
        "rank": 0,
        "host": "spark-2",
        "local_qsfp_ip": "192.168.200.2",
        "peer_rank": 1,
        "peer_host": "spark-3",
        "peer_qsfp_ip": "192.168.200.3",
    }

    localized = _localize_official_k2_rank_seat(original, seat)

    assert original["master_addr"] == "192.168.200.1"
    assert localized["master_addr"] == "192.168.200.2"
    assert localized["qsfp_host_ip_by_rank"]["0"] == "192.168.200.2"
    assert localized["qsfp_host_ip_by_rank"]["1"] == "192.168.200.3"
    assert {
        key for key in localized if localized[key] != original[key]
    } == {"master_addr", "qsfp_host_ip_by_rank"}


def test_rank_seat_localization_refuses_peer_or_scientific_drift() -> None:
    original = {
        "rank": 0,
        "master_addr": "192.168.200.1",
        "qsfp_host_ip_by_rank": {"0": "192.168.200.1", "1": "192.168.200.3"},
    }
    seat = {
        "rank": 0,
        "local_qsfp_ip": "192.168.200.2",
        "peer_rank": 1,
        "peer_qsfp_ip": "192.168.200.4",
    }
    try:
        _localize_official_k2_rank_seat(original, seat)
    except Exception as exc:
        assert "authorized peer" in str(exc)
    else:
        raise AssertionError("peer drift was accepted")

    try:
        _localize_official_k2_rank_seat(
            original,
            {**seat, "peer_qsfp_ip": "192.168.200.3", "lr": 1e-4},
        )
    except Exception as exc:
        assert "fields refused" in str(exc)
    else:
        raise AssertionError("scientific field drift was accepted")


def test_static_w28_resident_acceptance_is_public_api() -> None:
    import repair_api

    assert (
        repair_api.run_static_w28_resident_acceptance
        is static_w28_resident.run_static_w28_resident_acceptance
    )
    assert "run_static_w28_resident_acceptance" in repair_api.__all__


def test_static_w28_refuses_nonresident_or_slow_result(monkeypatch, tmp_path) -> None:
    reference = tmp_path / "B2_PUBLISHED_PRE.json"
    reference_sha = _write_reference(reference)

    class API:
        artifact = SimpleNamespace(
            manifest={"score": {"official_k2_resident": {"provider_resolution_mode": "STATIC_W28_GROUPED"}}}
        )

        def score(self, checkpoint, windows):
            return ScoreResult(
                checkpoint=checkpoint,
                windows=(28,), positions=1024, support=8192,
                kld=0.1364830042977786, top1=880, top1_rate=880 / 1024,
                artifact_root=str(tmp_path), spec="balanced64-v1", candidate_dir="candidate",
                execution_mode="resident_in_memory", resident_load_seconds=1.0,
                timed_wall_seconds=300.001,
                identity={"checkpoint_sha256": static_w28_resident.CHECKPOINT_SHA256,
                          "model_index_sha256": static_w28_resident.BASIS_SHA256},
                runtime_counters={"resident_engine_loads": 1, "timed_score_file_reads": 0,
                                  "resident_ready": [{}]},
            )

    monkeypatch.setattr(
        static_w28_resident.ResidentRepairAPI,
        "open",
        lambda root, official_rank_seat=None: API(),
    )
    monkeypatch.setattr(
        static_w28_resident.sealed_pre_forward,
        "bind_sealed_pre_resident_config",
        lambda config: {"status": "PASS"},
    )

    try:
        static_w28_resident.run_static_w28_resident_acceptance(
            task="t_test", root=tmp_path / "run", artifact_root=tmp_path,
            checkpoint="PRE", canonical_pin="deadbeef",
            reference_receipt=reference, reference_sha256=reference_sha,
        )
    except RuntimeError as exc:
        assert "STATIC_W28_RESIDENT_RED" in str(exc)
    else:
        raise AssertionError("slow resident score must fail")
