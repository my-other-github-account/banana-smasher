import ast
import concurrent.futures
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import CheckpointError, checkpoint

from repair_api import api as api_module
from repair_api import resume_equivalence_official as official
from repair_api.modern_green_resident import (
    _checkpoint_route_replay_supported,
    _checkpoint_topk_route,
)


def test_official_resume_defers_all_scoring_to_the_one_static_w28_builder():
    source = inspect.getsource(official.run_official_resume_equivalence)
    assert "engine.score_resident" not in source
    assert "api.score(" not in source
    assert 'scores: dict[str, Any] = {}' in source


def test_official_resume_is_hard_bound_to_clean_u0():
    source = inspect.getsource(official.run_official_resume_equivalence)
    assert 'checkpoint_key("UPDATE_000")' in source
    assert 'PUBLISHED_PRE_SHA256' not in source
    assert '"UPDATE_000", replay=replay' in source
    assert "clean U0 must contain empty Adam state" in source


def test_resident_adam_uses_cuda_foreach_fast_path():
    source = inspect.getsource(official.ModernGreenResidentEngine.__init__)
    assert "foreach=True" in source
    assert "foreach=False" not in source


def test_resident_engine_can_disable_recompute_checkpointing_as_mechanics_only():
    init_source = inspect.getsource(official.ModernGreenResidentEngine.__init__)
    run_source = inspect.getsource(official.ModernGreenResidentEngine._run_layers)
    assert 'self.activation_checkpointing = bool(config.get("activation_checkpointing", True))' in init_source
    assert "if train and self.activation_checkpointing:" in run_source
    assert "hidden = layer_block(hidden, self.first, self.last + 1)" in run_source


def test_resident_engine_groups_reentrant_checkpoint_blocks_for_fast_low_memory_steps():
    init_source = inspect.getsource(official.ModernGreenResidentEngine.__init__)
    run_source = inspect.getsource(official.ModernGreenResidentEngine._run_layers)
    assert 'config.get("activation_checkpoint_interval", 1)' in init_source
    assert 'config.get("checkpoint_use_reentrant", False)' in init_source
    assert "range(self.first, self.last + 1, self.activation_checkpoint_interval)" in run_source
    assert "use_reentrant=self.checkpoint_use_reentrant" in run_source


def test_non_reentrant_route_replay_never_replaces_stateful_hash_router_path():
    class CanonicalHashRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 3
            self.weight = torch.nn.Parameter(torch.randn(4, self.hidden_dim))
            self.score_fn = torch.sigmoid
            self.routed_scaling_factor = 1.0
            self.register_buffer(
                "tid2eid",
                torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            )

        def forward(self, hidden_states, input_ids):
            flat = hidden_states.reshape(-1, self.hidden_dim)
            logits = torch.nn.functional.linear(flat, self.weight)
            # This extra differentiable path models the canonical hash router's
            # stateful forward boundary. Replacing only the recompute half with
            # a learned-top-k replay changes the saved-tensor count.
            scores = torch.sin(self.score_fn(logits))
            indices = getattr(self, "tid2eid")[input_ids.reshape(-1)].long()
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            return logits, weights, indices

    gate = CanonicalHashRouter()
    hidden = torch.randn(2, 2, 3, requires_grad=True)
    input_ids = torch.tensor([[0, 1], [2, 3]])
    fixed = {}
    invocation = {"count": 0}

    def mismatched_hash_replay(current):
        recompute = invocation["count"] > 0
        invocation["count"] += 1
        if not recompute:
            logits, weights, indices = gate(current, input_ids)
            fixed["indices"] = indices.detach()
        else:
            flat = current.reshape(-1, gate.hidden_dim)
            logits = torch.nn.functional.linear(flat, gate.weight)
            scores = gate.score_fn(logits)
            indices = fixed["indices"]
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (logits.gather(1, indices) * weights).sum()

    with pytest.raises(CheckpointError, match="different number of tensors"):
        mismatched_output = checkpoint(
            mismatched_hash_replay, hidden, use_reentrant=False
        )
        assert mismatched_output is not None
        mismatched_output.backward()

    assert _checkpoint_route_replay_supported(gate) is False
    calls = {"count": 0}

    def canonical_hash_path(current):
        calls["count"] += 1
        logits, weights, indices = gate(current, input_ids)
        return (logits.gather(1, indices) * weights).sum()

    hidden.grad = None
    canonical_output = checkpoint(
        canonical_hash_path, hidden, use_reentrant=False
    )
    assert canonical_output is not None
    canonical_output.backward()
    assert calls["count"] == 2
    assert hidden.grad is not None


def test_non_reentrant_canonical_topk_replay_preserves_343_tensor_save_path():
    class CanonicalTopKRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 3
            self.top_k = 2
            self.weight = torch.nn.Parameter(torch.randn(4, self.hidden_dim))
            self.score_fn = torch.sigmoid
            self.routed_scaling_factor = 1.0
            self.register_buffer("e_score_correction_bias", torch.zeros(4))

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, self.hidden_dim)
            logits = torch.nn.functional.linear(flat, self.weight)
            scores = self.score_fn(logits)
            indices = torch.topk(
                scores + self.e_score_correction_bias,
                self.top_k,
                dim=-1,
                sorted=False,
            ).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            return logits, weights * self.routed_scaling_factor, indices

    gate = CanonicalTopKRouter()
    hidden = torch.randn(2, 2, 3, requires_grad=True)

    # Model the production failure exactly: 331 surrounding saved tensors plus
    # the asymmetric router path produce the observed 343-vs-342 error.
    mismatched_fixed = {}
    mismatched_calls = {"count": 0}

    def mismatched_block(current):
        for _ in range(331):
            current = torch.sin(current)
        recompute = mismatched_calls["count"] > 0
        mismatched_calls["count"] += 1
        if not recompute:
            logits, weights, indices = gate(current)
            mismatched_fixed["indices"] = indices.detach()
        else:
            flat = current.reshape(-1, gate.hidden_dim)
            logits = torch.nn.functional.linear(flat, gate.weight)
            scores = gate.score_fn(logits)
            indices = mismatched_fixed["indices"]
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (logits.gather(1, indices) * weights).sum()

    with pytest.raises(CheckpointError) as mismatch:
        checkpoint(mismatched_block, hidden, use_reentrant=False).backward()
    message = str(mismatch.value)
    assert "Number of tensors saved during forward: 343" in message
    assert "Number of tensors saved during recomputation: 342" in message

    # The production helper must execute the canonical selection path in both
    # passes while binding recomputation's differentiable weights to the exact
    # forward indices. This keeps both tensor-saving paths identical.
    fixed = {}
    calls = {"count": 0}

    def matched_block(current):
        for _ in range(331):
            current = torch.sin(current)
        recompute = calls["count"] > 0
        calls["count"] += 1
        logits, weights, indices = _checkpoint_topk_route(
            torch,
            gate,
            current,
            fixed_indices=fixed.get("indices") if recompute else None,
        )
        if not recompute:
            fixed["indices"] = indices.detach()
        return (logits.gather(1, indices) * weights).sum()

    hidden.grad = None
    checkpoint(matched_block, hidden, use_reentrant=False).backward()
    assert calls["count"] == 2
    assert hidden.grad is not None


def test_published_pre_can_split_pipeline_microbatch_without_changing_window_dose():
    init_source = inspect.getsource(official.ModernGreenResidentEngine.__init__)
    step_source = inspect.getsource(official.ModernGreenResidentEngine._step)
    pipeline_source = inspect.getsource(
        official.ModernGreenResidentEngine._pipeline_update_1f1b
    )
    assert 'config.get("pipeline_microbatch", default_pipeline_microbatch)' in init_source
    assert "group_windows[index:index + self.pipeline_microbatch]" in step_source
    assert "self._pipeline_update_1f1b(" in step_source
    assert "loss_divisor=len(groups)" in step_source
    assert "loss / float(loss_divisor)" in pipeline_source
    assert "_batch_p2p_exchange" in pipeline_source
    assert "for index in range(1, len(groups))" in pipeline_source


def test_1f1b_exchange_complements_known_green_one_way_operations_by_rank():
    exchanges = {}
    for rank in (0, 1):
        engine = official.ModernGreenResidentEngine.__new__(
            official.ModernGreenResidentEngine
        )
        engine.rank = rank
        exchanges[rank] = []
        engine._batch_p2p_send = lambda tensor, *, dst, rank=rank: exchanges[rank].append(
            ("send", tensor, dst)
        )
        engine._batch_p2p_recv = lambda tensor, *, src, rank=rank: exchanges[rank].append(
            ("recv", tensor, src)
        )
        official.ModernGreenResidentEngine._batch_p2p_exchange(
            engine, f"out-{rank}", dst=1 - rank,
            incoming=f"in-{rank}", src=1 - rank,
        )

    assert exchanges[0] == [("send", "out-0", 1), ("recv", "in-0", 1)]
    assert exchanges[1] == [("recv", "in-1", 0), ("send", "out-1", 0)]
    warmup_source = inspect.getsource(official.ModernGreenResidentEngine._warm_p2p_communicator)
    assert "operations = [send, receive] if self.rank == 0 else [receive, send]" in warmup_source


def test_resume_equivalence_bootstrap_is_progress_marked_and_time_bounded():
    source = inspect.getsource(official.run_official_resume_equivalence)
    assert 'phase="dist_init_start"' in source
    assert 'phase="dist_init_complete"' in source
    assert "timeout=timedelta(seconds=600)" in source


def test_resume_equivalence_persists_complete_checkpoint_before_next_update():
    source = inspect.getsource(official.run_official_resume_equivalence)
    assert 'phase="checkpoint_persisted"' in source
    assert '"state": snapshot["state"]' in source
    assert '"optimizer": snapshot["optimizer"]' in source
    assert '"scheduler": snapshot["scheduler"]' in source
    assert "os.fsync(directory_fd)" in source
    assert source.index('phase="checkpoint_persisted"') < source.index("return {", source.index('phase="checkpoint_persisted"'))


def test_resident_wire_loader_batches_identical_projection_bytes_before_h2d(tmp_path):
    source_path = Path(official.__file__).parent / "assets" / "fast_v7_expert_train_batched.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_projection_payloads"
    )
    namespace = {
        "np": np, "torch": torch, "Path": Path, "Any": object,
        "concurrent": concurrent,
        "PACKED_BYTES": 2_097_152,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    loader = namespace["_load_projection_payloads"]

    m = k = 16
    packed_bytes = (k // 16) * (m // 16) * 32 * 2
    paths = []
    expected_packed = []
    expected_su = []
    expected_sv = []
    for expert in range(2):
        packed = (np.arange(32, dtype="<i2") + expert * 100)
        su = (np.arange(k, dtype="<f2") + expert * 10)
        sv = (np.arange(m, dtype="<f2") + expert * 20)
        path = tmp_path / f"expert-{expert}.wire"
        path.write_bytes(packed.tobytes() + su.tobytes() + sv.tobytes() + b"seal")
        paths.append(path)
        expected_packed.append(packed)
        expected_su.append(su)
        expected_sv.append(sv)

    packed, su, sv, read_calls, read_bytes = loader(
        paths, m=m, k=k, packed_bytes=packed_bytes
    )

    assert packed.device.type == su.device.type == sv.device.type == "cpu"
    assert tuple(packed.shape) == (2, 1, 1, 32)
    assert torch.equal(packed.reshape(2, 32), torch.from_numpy(np.stack(expected_packed)))
    assert torch.equal(su, torch.from_numpy(np.stack(expected_su)))
    assert torch.equal(sv, torch.from_numpy(np.stack(expected_sv)))
    assert read_calls == 2
    assert read_bytes == sum(path.stat().st_size for path in paths)
    assert "ThreadPoolExecutor" in ast.unparse(function)
    assert "pool.map(load_one" in ast.unparse(function)
    init_source = source[source.index("class FullyResidentGroupedV7Experts"):]
    assert "packed_cpu, su_cpu, sv_cpu" in init_source
    assert "packed_cpu.to(device=device)" in init_source
    assert "su_cpu.to(device=device)" in init_source
    assert "sv_cpu.to(device=device)" in init_source


def test_resident_score_moves_teacher_support_to_cpu_before_numpy():
    source = inspect.getsource(official.ModernGreenResidentEngine.score_resident)
    assert "teacher_logprob.detach().cpu().numpy()" in source
    assert "teacher_idx[:, 0].detach().cpu().numpy()" in source
    assert "teacher_logprob.numpy()" not in source
    assert "teacher_idx[:, 0].numpy()" not in source


def test_resident_score_has_independent_microbatch_from_training():
    init_source = inspect.getsource(official.ModernGreenResidentEngine.__init__)
    score_source = inspect.getsource(official.ModernGreenResidentEngine.score_resident)
    assert 'config.get("score_pipeline_microbatch", PIPELINE_MICROBATCH)' in init_source
    assert "range(0, len(ordered), self.score_pipeline_microbatch)" in score_source
    assert "ordered[offset:offset + self.score_pipeline_microbatch]" in score_source
    assert 'config.get("score_head_window_microbatch", 1)' in init_source
    assert "range(0, len(batch), self.score_head_window_microbatch)" in score_source
    assert ".reshape(len(head_windows) * count" in score_source


def test_distributed_init_pins_the_authorized_qsfp_interface():
    source = inspect.getsource(official.run_official_resume_equivalence)
    pin = source.index('socket_ifname = str(config.get("nccl_socket_ifname", ""))')
    nccl = source.index('os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname')
    gloo = source.index('os.environ["GLOO_SOCKET_IFNAME"] = socket_ifname')
    init = source.index("dist.init_process_group")
    assert pin < nccl < gloo < init


def test_modern_green_distributed_init_pins_the_authorized_qsfp_interface():
    source = inspect.getsource(official.ModernGreenResidentEngine._init_distributed)
    pin = source.index('socket_ifname = str(self.config.get("nccl_socket_ifname", ""))')
    nccl = source.index('os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname')
    gloo = source.index('os.environ["GLOO_SOCKET_IFNAME"] = socket_ifname')
    init = source.index("self.dist.init_process_group")
    assert pin < nccl < gloo < init


def test_distributed_init_eagerly_warms_the_two_rank_p2p_communicator():
    init_source = inspect.getsource(official.ModernGreenResidentEngine._init_distributed)
    warm_source = inspect.getsource(official.ModernGreenResidentEngine._warm_p2p_communicator)
    assert "self._warm_p2p_communicator()" in init_source
    assert "batch_isend_irecv" in warm_source
    assert "P2POp" in warm_source
    assert "request.wait()" in warm_source


class _RankSensitiveTorch:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.save_calls = 0

    def save(self, payload, stream) -> None:
        self.save_calls += 1
        stream.write(f"rank={self.rank};payload={payload['value']}".encode())


class _BroadcastBus:
    def __init__(self) -> None:
        self.canonical = None
        self.sources = []

    def for_rank(self, rank: int):
        bus = self

        class _Dist:
            def broadcast_object_list(self, row, src: int) -> None:
                bus.sources.append(src)
                if rank == src:
                    bus.canonical = row[0]
                else:
                    row[0] = bus.canonical

        return _Dist()


def test_checkpoint_bytes_are_serialized_only_on_rank0_and_broadcast_exactly():
    bus = _BroadcastBus()
    rank0_torch = _RankSensitiveTorch(rank=0)
    rank1_torch = _RankSensitiveTorch(rank=1)

    rank0_bytes = official._canonical_checkpoint_bytes(
        {"value": "same-logical-state"}, rank=0, dist=bus.for_rank(0), torch_module=rank0_torch
    )
    rank1_bytes = official._canonical_checkpoint_bytes(
        {"value": "same-logical-state"}, rank=1, dist=bus.for_rank(1), torch_module=rank1_torch
    )

    assert rank0_bytes == b"rank=0;payload=same-logical-state"
    assert rank1_bytes == rank0_bytes
    assert rank0_torch.save_calls == 1
    assert rank1_torch.save_calls == 0
    assert bus.sources == [0, 0]


class _Optimizer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {
            "state": {},
            "param_groups": [
                {"params": [0], "lr": 1.0},
                {"params": [1], "lr": 2.0},
                {"params": [2], "lr": 3.0},
            ],
        }

    def load_state_dict(self, value):
        self.loaded = value


def test_global_u0_optimizer_is_projected_to_the_rank_local_parameter_ids():
    class _Engine:
        optimizer = _Optimizer()
        luts = [("l1", object())]
        norms = [("n1", object())]
        outputs = [("o1", object())]
        state = {
            "luts": {"l0": None, "l1": None},
            "norms": {"n0": None, "n1": None},
            "outputs": {"o0": None, "o1": None},
        }

    global_state = {
        "state": {
            10: {"exp_avg": "l0"}, 11: {"exp_avg": "l1"},
            20: {"exp_avg": "n0"}, 21: {"exp_avg": "n1"},
            30: {"exp_avg": "o0"}, 31: {"exp_avg": "o1"},
        },
        "param_groups": [
            {"params": [10, 11], "lr": 0.1},
            {"params": [20, 21], "lr": 0.2},
            {"params": [30, 31], "lr": 0.3},
        ],
    }

    official._load_rank_local_optimizer_state(_Engine, global_state)

    loaded = _Engine.optimizer.loaded
    assert [group["params"] for group in loaded["param_groups"]] == [[0], [1], [2]]
    assert [group["lr"] for group in loaded["param_groups"]] == [0.1, 0.2, 0.3]
    assert loaded["state"] == {
        0: {"exp_avg": "l1"}, 1: {"exp_avg": "n1"}, 2: {"exp_avg": "o1"}
    }


def test_public_resume_equivalence_releases_continuous_before_resume_instantiation():
    source = inspect.getsource(api_module.ResidentRepairAPI.resume_equivalence)
    continuous_loop = source.index('step("continuous"')
    release = source.index('release_fn(continuous_model)')
    resume_instantiate = source.index(
        'resume_model, resume_optimizer, resume_scheduler = instantiate()'
    )
    assert continuous_loop < release < resume_instantiate


def test_public_resume_equivalence_releases_pre_midpoint_engine_before_reload_instantiation():
    source = inspect.getsource(api_module.ResidentRepairAPI.resume_equivalence)
    midpoint_reload = source.index(
        'resume_model, resume_optimizer, resume_scheduler = instantiate()',
        source.index('reloaded_payload = torch.load('),
    )
    release = source.index('release_fn(resume_model)', source.index('midpoint_pre_save_fingerprint'))
    assert release < midpoint_reload


def test_public_resume_equivalence_replays_rng_and_restores_midpoint_rng_state():
    source = inspect.getsource(api_module.ResidentRepairAPI.resume_equivalence)
    assert source.count("\n        reset_rng()") == 2
    assert '"python_rng_state": random.getstate()' in source
    assert '"torch_rng_state": torch.get_rng_state()' in source
    assert '"cuda_rng_state_all": torch.cuda.get_rng_state_all()' in source
    assert 'random.setstate(reloaded_payload["python_rng_state"])' in source
    assert 'torch.set_rng_state(reloaded_payload["torch_rng_state"])' in source
    assert 'torch.cuda.set_rng_state_all(reloaded_payload["cuda_rng_state_all"])' in source


def test_official_config_maps_sealed_binrepair_paths_to_resident_engine_schema():
    config = {
        "binrepair_manifest": "/sealed/manifest.json",
        "binrepair_delta_dir": "/sealed/delta",
        "binrepair_vq3b_dir": "/sealed/planes",
    }

    engine_config = official._resident_engine_config(config)

    assert engine_config["manifest"] == config["binrepair_manifest"]
    assert engine_config["delta_dir"] == config["binrepair_delta_dir"]
    assert engine_config["vq3b_dir"] == config["binrepair_vq3b_dir"]
    assert config == {
        "binrepair_manifest": "/sealed/manifest.json",
        "binrepair_delta_dir": "/sealed/delta",
        "binrepair_vq3b_dir": "/sealed/planes",
    }


def test_official_progress_receipt_is_atomic_and_phase_specific(tmp_path):
    destination = tmp_path / "PROGRESS.json"

    official._write_progress(destination, rank=1, arm="resume", update=3, phase="after_step")

    assert json.loads(destination.read_text()) == {
        "schema": "banana-smasher-resume-equivalence-progress-v1",
        "rank": 1,
        "arm": "resume",
        "update": 3,
        "phase": "after_step",
    }
    assert not destination.with_suffix(".tmp").exists()


def test_official_engine_warm_load_persists_per_layer_progress():
    source = inspect.getsource(official.run_official_resume_equivalence)
    assert 'phase="engine_init_start"' in source
    assert 'engine_config["progress_callback"] = progress_callback' in source
    assert 'phase=f"engine_{phase}"' in source
    assert 'phase="engine_init_complete"' in source
    assert 'details={"load_seconds": engine.student.load_seconds}' in source
