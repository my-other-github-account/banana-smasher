"""Public HF MoE capability contracts closed for this campaign (G8/G9/G12/G13/G15/G16).

These tests are model-family semantic, not campaign specific: every fixture is a tiny
synthetic MoE tree whose routed scope is decided from config + index semantics alone.
No DeepSeek fixture, no numeric baseline, and no GLM roster is borrowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

REVISION = "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"

# A generic MoE geometry with a dense prefix, a routed stack, and one trailing
# auxiliary (multi-token-prediction) layer above num_hidden_layers.
CONFIG = {
    "architectures": ["FixtureMoeForCausalLM"],
    "model_type": "fixture_moe",
    "text_config": {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_nextn_predict_layers": 1,
        "vocab_size": 16384,
    },
}

ROUTED = {
    "model.layers.1.mlp.experts.0.up_proj.weight",
    "model.layers.1.mlp.experts.1.up_proj.weight",
}
NON_ROUTED = {
    "model.embed_tokens.weight",                          # embedding
    "model.layers.0.mlp.gate_proj.weight",                # dense-prefix layer
    "model.layers.1.mlp.gate.weight",                     # router
    "model.layers.1.mlp.shared_experts.up_proj.weight",   # shared expert
    "model.layers.1.input_layernorm.weight",              # norm
    "model.layers.2.mlp.experts.0.up_proj.weight",        # AUXILIARY layer expert
    "model.layers.2.eh_proj.weight",                      # auxiliary head marker
    "lm_head.weight",                                     # output head
}


def _write_tree(root: Path, *, config: dict | None = None, extra: dict | None = None) -> Path:
    """Materialize a plain (non-symlinked) synthetic HF MoE repository tree.

    Routed matrices are 2x8 so the QTIP2 cyclic trellis closes (L=16, V=2 -> the
    sequence needs at least 8 steps); non-routed tensors keep an arbitrary shape,
    which also proves the two classes are handled by different code paths.
    """

    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        name: np.arange(16, dtype=np.float32).reshape(2, 8) for name in ROUTED
    }
    payload.update({name: np.ones((2, 4), dtype=np.float32) for name in NON_ROUTED})
    if extra:
        payload.update(extra)
    shard = "model-00001-of-00001.safetensors"
    save_file(payload, str(root / shard))
    (root / "config.json").write_text(
        json.dumps(config if config is not None else CONFIG, sort_keys=True) + "\n"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": 1}, "weight_map": {n: shard for n in payload}},
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _write_hf_cache_tree(base: Path) -> Path:
    """Materialize the canonical `hf download` layout: symlinks into blobs/ + .cache."""

    blobs = base / "blobs"
    _write_tree(blobs)
    snapshot = base / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for entry in sorted(blobs.iterdir()):
        os.symlink(entry, snapshot / entry.name)
    bookkeeping = snapshot / ".cache" / "huggingface" / "download"
    bookkeeping.mkdir(parents=True)
    (bookkeeping / "config.json.metadata").write_text("downloaded\n")
    (snapshot / ".cache" / "huggingface" / ".gitignore").write_text("*\n")
    return snapshot


# --------------------------------------------------------------------------------------
# G8 — canonical HF cache/snapshot layout is admissible; identity is by content hash
# --------------------------------------------------------------------------------------


def test_admission_accepts_symlinked_hf_snapshot_and_binds_content_identity(
    tmp_path: Path,
) -> None:
    from banana_smasher import admit_hf_source

    snapshot = _write_hf_cache_tree(tmp_path / "hub")

    receipt = admit_hf_source(
        snapshot, revision=REVISION, receipt_path=tmp_path / "ADMISSION.json"
    )

    assert receipt["status"] == "PASS"
    assert receipt["binding"]["identity"] == "content-sha256"
    assert receipt["binding"]["symlinks_resolved"] is True
    members = {row["path"]: row for row in receipt["binding"]["members"]}
    assert set(members) == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    }
    for row in members.values():
        # every member is bound by its RESOLVED target, not by the link itself
        assert row["symlink"] is True
        assert Path(row["realpath"]).is_file()
        assert not Path(row["realpath"]).is_symlink()
        assert len(row["sha256"]) == 64
        assert row["bytes"] == Path(row["realpath"]).stat().st_size
    # a dereferenced copy of the same bytes has the same identity
    plain = admit_hf_source(
        _write_tree(tmp_path / "plain"),
        revision=REVISION,
        receipt_path=tmp_path / "PLAIN.json",
    )
    assert plain["config_sha256"] == receipt["config_sha256"]
    assert plain["model_index_sha256"] == receipt["model_index_sha256"]


def test_admission_rejects_unresolvable_and_non_regular_members(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source

    snapshot = _write_hf_cache_tree(tmp_path / "hub")
    (snapshot / "config.json").unlink()
    os.symlink(tmp_path / "does-not-exist.json", snapshot / "config.json")

    with pytest.raises(ValueError, match="missing or unresolvable"):
        admit_hf_source(snapshot, revision=REVISION, receipt_path=tmp_path / "R.json")

    directory = _write_tree(tmp_path / "dir-shard")
    (directory / "model-00001-of-00001.safetensors").unlink()
    (directory / "model-00001-of-00001.safetensors").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        admit_hf_source(directory, revision=REVISION, receipt_path=tmp_path / "R2.json")


def test_admission_rejects_a_non_immutable_revision(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source

    with pytest.raises(ValueError, match="40- or 64-digit hex"):
        admit_hf_source(
            _write_tree(tmp_path / "m"), revision="main", receipt_path=tmp_path / "R.json"
        )


# --------------------------------------------------------------------------------------
# G16 — the authoritative repository roster is published, not inferred
# --------------------------------------------------------------------------------------


def test_admission_publishes_repository_roster_and_excluded_bookkeeping(
    tmp_path: Path,
) -> None:
    from banana_smasher import HF_CLIENT_BOOKKEEPING_PREFIXES, admit_hf_source

    snapshot = _write_hf_cache_tree(tmp_path / "hub")

    boundary = admit_hf_source(
        snapshot, revision=REVISION, receipt_path=tmp_path / "ADMISSION.json"
    )["roster_boundary"]

    assert boundary["excluded_prefixes"] == list(HF_CLIENT_BOOKKEEPING_PREFIXES)
    assert boundary["repository_files"] == [
        "config.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    ]
    assert boundary["repository_file_count"] == 3
    # the client-side download stamps are excluded, named, and counted separately
    assert boundary["excluded_files"] == [
        ".cache/huggingface/.gitignore",
        ".cache/huggingface/download/config.json.metadata",
    ]
    assert boundary["excluded_file_count"] == 2
    # both partitions are covered: a naive walk equals repository + excluded
    naive = sum(1 for p in snapshot.rglob("*") if not p.is_dir())
    assert naive == boundary["repository_file_count"] + boundary["excluded_file_count"]
    assert boundary["repository_bytes"] > 0
    assert boundary["excluded_bytes"] > 0


# --------------------------------------------------------------------------------------
# G12 — standalone, read-only routed-scope discovery
# --------------------------------------------------------------------------------------


def test_routed_scope_discovery_is_standalone_and_reads_no_tensor_bytes(
    tmp_path: Path,
) -> None:
    from banana_smasher import discover_hf_moe_routed_scope

    scope = discover_hf_moe_routed_scope(
        _write_tree(tmp_path / "m"), revision=REVISION, receipt_path=tmp_path / "SCOPE.json"
    )

    assert scope["status"] == "PASS"
    assert scope["scope"] == "routed_only"
    assert scope["reads_tensor_bytes"] is False
    assert scope["adapter"]["id"] == "hf-numeric-experts-v1"
    assert scope["routed_tensor_names"] == sorted(ROUTED)
    assert scope["native_tensor_names"] == sorted(NON_ROUTED)
    assert scope["accounting"] == {
        "source_tensor_count": len(ROUTED) + len(NON_ROUTED),
        "routed_tensor_count": len(ROUTED),
        "native_tensor_count": len(NON_ROUTED),
    }
    assert scope["mechanisms"]["fallback"] == 0
    assert json.loads((tmp_path / "SCOPE.json").read_text()) == scope


def test_routed_scope_discovery_excludes_every_non_routed_surface(tmp_path: Path) -> None:
    from banana_smasher import discover_hf_moe_routed_scope

    scope = discover_hf_moe_routed_scope(
        _write_tree(tmp_path / "m"), revision=REVISION, receipt_path=tmp_path / "S.json"
    )
    routed = set(scope["routed_tensor_names"])
    native = set(scope["native_tensor_names"])

    # routers, shared experts, dense-prefix MLPs, norms, embeddings, output heads, and
    # auxiliary-layer experts are all non-routed.
    for excluded in NON_ROUTED:
        assert excluded not in routed, excluded
        assert excluded in native, excluded
    # routed ∪ native covers everything exactly once
    assert routed | native == ROUTED | NON_ROUTED
    assert not routed & native


def test_routed_scope_discovery_agrees_exactly_with_the_plan(tmp_path: Path) -> None:
    from banana_smasher import discover_hf_moe_routed_scope, plan_hf_moe_uniform

    model = _write_tree(tmp_path / "m")
    scope = discover_hf_moe_routed_scope(
        model, revision=REVISION, receipt_path=tmp_path / "S.json"
    )
    plan = plan_hf_moe_uniform(
        model,
        revision=REVISION,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=tmp_path / "P.json",
    )

    assert scope["routed_tensor_names"] == sorted(r["name"] for r in plan["routed_tensors"])
    assert scope["native_tensor_names"] == sorted(r["name"] for r in plan["native_tensors"])
    assert scope["adapter"] == plan["adapter"]
    assert scope["geometry"] == plan["geometry"]


def test_routed_scope_discovery_refuses_a_source_with_no_routed_experts(
    tmp_path: Path,
) -> None:
    from banana_smasher import discover_hf_moe_routed_scope

    dense = {
        "architectures": ["DenseOnly"],
        "text_config": {"num_hidden_layers": 2, "n_routed_experts": 4},
    }
    root = tmp_path / "dense"
    root.mkdir()
    save_file(
        {"model.layers.0.mlp.gate_proj.weight": np.ones((2, 4), dtype=np.float32)},
        str(root / "s.safetensors"),
    )
    (root / "config.json").write_text(json.dumps(dense) + "\n")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.mlp.gate_proj.weight": "s.safetensors"}})
        + "\n"
    )

    with pytest.raises(ValueError, match="adapter selection must resolve exactly once"):
        discover_hf_moe_routed_scope(
            root, revision=REVISION, receipt_path=tmp_path / "S.json"
        )


# --------------------------------------------------------------------------------------
# G15 — the auxiliary-layer bound is data-driven and stated inline in the receipt
# --------------------------------------------------------------------------------------


def test_auxiliary_layer_rule_is_published_and_excludes_trailing_head_experts(
    tmp_path: Path,
) -> None:
    from banana_smasher import HF_AUXILIARY_LAYER_RULE, discover_hf_moe_routed_scope

    geometry = discover_hf_moe_routed_scope(
        _write_tree(tmp_path / "m"), revision=REVISION, receipt_path=tmp_path / "S.json"
    )["geometry"]

    assert geometry["expected_model_layers"] == 2
    assert geometry["dense_prefix_layers"] == 1
    assert geometry["routed_experts"] == 2
    assert geometry["model_layer_ids"] == [0, 1]
    # layer 2 lives above num_hidden_layers: it is auxiliary and NEVER routed
    assert geometry["auxiliary_layer_ids"] == [2]
    assert geometry["routed_layer_ids"] == [1]
    assert geometry["routed_auxiliary_layers"] == []
    assert geometry["model_layer_gaps"] == []
    # the semantics are inline, not left for the caller to infer from a bare id list
    assert geometry["auxiliary_layer_rule"] == HF_AUXILIARY_LAYER_RULE
    assert "first_k_dense_replace" in geometry["auxiliary_layer_rule"]
    assert geometry["auxiliary_layer_deciding_config_keys"] == [
        "num_hidden_layers",
        "first_k_dense_replace",
        "n_routed_experts",
        "num_nextn_predict_layers",
    ]


def test_auxiliary_layer_bound_follows_config_not_a_hardcoded_roster(
    tmp_path: Path,
) -> None:
    """Raising num_hidden_layers moves the boundary; nothing is model-name specific."""
    from banana_smasher import discover_hf_moe_routed_scope

    promoted = json.loads(json.dumps(CONFIG))
    promoted["text_config"]["num_hidden_layers"] = 3  # layer 2 is now a real stack layer
    promoted["text_config"]["num_nextn_predict_layers"] = 0

    geometry = discover_hf_moe_routed_scope(
        _write_tree(tmp_path / "m", config=promoted),
        revision=REVISION,
        receipt_path=tmp_path / "S.json",
    )["geometry"]

    assert geometry["auxiliary_layer_ids"] == []
    assert geometry["routed_layer_ids"] == [1, 2]


def test_plan_rejects_a_gap_in_the_declared_model_layer_stack(tmp_path: Path) -> None:
    from banana_smasher import discover_hf_moe_routed_scope

    gapped = json.loads(json.dumps(CONFIG))
    gapped["text_config"]["num_hidden_layers"] = 5  # layers 3,4 are absent from the index

    with pytest.raises(ValueError, match="layer coverage defects"):
        discover_hf_moe_routed_scope(
            _write_tree(tmp_path / "m", config=gapped),
            revision=REVISION,
            receipt_path=tmp_path / "S.json",
        )
    # the FAILED receipt is still durable for the caller to inspect
    assert json.loads((tmp_path / "S.json").read_text())["status"] == "FAILED"


# --------------------------------------------------------------------------------------
# G9 — declared, testable dependency boundary between planning and encoding
# --------------------------------------------------------------------------------------


def test_planning_tier_runs_under_the_base_install(tmp_path: Path) -> None:
    """discover/plan/preflight are metadata-only and must not need the solve extra."""
    from banana_smasher import (
        discover_hf_moe_routed_scope,
        plan_hf_moe_uniform,
        preflight_hf_moe_output_fit,
    )

    model = _write_tree(tmp_path / "m")
    assert discover_hf_moe_routed_scope(
        model, revision=REVISION, receipt_path=tmp_path / "S.json"
    )["status"] == "PASS"
    plan = plan_hf_moe_uniform(
        model,
        revision=REVISION,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=tmp_path / "P.json",
    )
    assert plan["status"] == "PASS"
    assert preflight_hf_moe_output_fit(
        plan,
        free_bytes=1 << 40,
        reserve_bytes=1 << 20,
        receipt_path=tmp_path / "F.json",
    )["status"] == "PASS"


def test_encoding_beyond_the_fixture_bound_fails_closed_naming_the_solve_extra() -> None:
    from banana_smasher import HF_SOLVE_EXTRA_REQUIREMENT
    from banana_smasher.hf_moe import _encode_hf_q2
    from banana_smasher.qtip1 import QTIP2_GEOMETRY, gaussian_tlut

    try:
        import torch as _torch

        if _torch.cuda.is_available():
            pytest.skip("CUDA encoder is present; the base-install boundary does not apply")
    except ModuleNotFoundError:
        pass

    oversized = np.ones((512, 1024), dtype=np.float32)  # 2 MiB > the 1 MiB fixture bound
    with pytest.raises(RuntimeError) as excinfo:
        _encode_hf_q2(
            oversized,
            geometry=QTIP2_GEOMETRY,
            tlut=gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V),
        )
    message = str(excinfo.value)
    assert "[solve]" in message
    assert HF_SOLVE_EXTRA_REQUIREMENT in message
    # the refusal is explicit: no silent slower fallback was taken
    assert "refusing slower fallback" in message


def test_solve_extra_requirement_names_the_base_install_capabilities() -> None:
    from banana_smasher import HF_SOLVE_EXTRA_REQUIREMENT

    for metadata_only in (
        "admit_hf_source",
        "discover_hf_moe_routed_scope",
        "plan_hf_moe_uniform",
        "preflight_hf_moe_output_fit",
    ):
        assert metadata_only in HF_SOLVE_EXTRA_REQUIREMENT


# --------------------------------------------------------------------------------------
# G13 — the teacher/PRE hardware contract is published and enforced before execution
# --------------------------------------------------------------------------------------


def test_hardware_contract_is_discoverable_without_staging_a_source() -> None:
    from banana_smasher import balanced64_hardware_contract

    report = balanced64_hardware_contract()

    assert report["schema"] == "banana-smasher.balanced64-hardware-contract.v1"
    assert set(report["host"]["detected"]) == {"torch", "cuda", "cuda_device_count", "mps"}
    shipped = [row for row in report["runtimes"] if row.get("entry_point") == "hf-sharded"]
    assert len(shipped) == 1
    contract = shipped[0]["contract"]
    assert contract["required_accelerator"] == "cuda"
    assert contract["minimum_ranks"] == 1
    assert contract["reason"]
    assert isinstance(contract["satisfied"], bool)


def test_hardware_contract_probe_never_raises_on_an_inadmissible_host() -> None:
    from banana_smasher import balanced64_hardware_contract
    from banana_smasher.hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime

    report = balanced64_hardware_contract(ShardedHFBalanced64Runtime())
    contract = report["runtimes"][0]["contract"]

    detected = report["host"]["detected"]
    assert contract["satisfied"] is bool(
        detected["cuda"] and detected["cuda_device_count"] >= 1
    )
    if not detected["cuda"]:
        # an MPS/CPU host is reported as inadmissible rather than silently accepted
        assert contract["satisfied"] is False
        assert "cpu-only hosts" in contract["not_admissible"]


def test_capability_gate_fails_closed_with_the_contract_stated() -> None:
    from banana_smasher import Balanced64CapabilityError
    from banana_smasher.hf_balanced64 import _require_balanced64_capability
    from banana_smasher.hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime

    runtime = ShardedHFBalanced64Runtime()
    try:
        import torch

        if torch.cuda.is_available():
            pytest.skip("host satisfies the contract; the refusal path does not apply")
    except ModuleNotFoundError:
        pass

    with pytest.raises(Balanced64CapabilityError) as excinfo:
        _require_balanced64_capability(runtime, "capture_balanced64_teacher")
    message = str(excinfo.value)
    assert "capture_balanced64_teacher" in message
    assert "required_accelerator=cuda" in message
    assert "hf-sharded-balanced64-v1" in message
    assert "detected=" in message


def test_a_runtime_declaring_no_contract_is_not_gated() -> None:
    from banana_smasher.hf_balanced64 import _require_balanced64_capability

    class _Unconstrained:
        runtime_id = "fixture-unconstrained-v1"

    result = _require_balanced64_capability(_Unconstrained(), "score_balanced64_pre")
    assert result["satisfied"] is True
    assert result["declared"] is False


def test_contract_belongs_to_the_executor_actually_selected() -> None:
    """A caller-supplied executor owns its own requirement; the seam is not the gate."""
    from banana_smasher.hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime

    package_owned = ShardedHFBalanced64Runtime()
    caller_supplied = ShardedHFBalanced64Runtime(executor_factory=lambda **_: None)

    assert package_owned.hardware_contract["required_accelerator"] == "cuda"
    assert caller_supplied.hardware_contract is None


def test_contract_rejects_an_unknown_accelerator_declaration() -> None:
    from banana_smasher.hf_balanced64 import _evaluate_contract

    with pytest.raises(ValueError, match="unknown accelerator"):
        _evaluate_contract(
            {"required_accelerator": "quantum"},
            {"cuda": True, "cuda_device_count": 8, "torch": True, "mps": False},
        )
    with pytest.raises(ValueError, match="minimum_ranks"):
        _evaluate_contract(
            {"required_accelerator": "cuda", "minimum_ranks": 0},
            {"cuda": True, "cuda_device_count": 8, "torch": True, "mps": False},
        )


# --------------------------------------------------------------------------------------
# Native-byte preservation + serialization/reload contracts (uniform Q2 routed-only)
# --------------------------------------------------------------------------------------


def test_build_preserves_native_bytes_exactly_and_reloads_by_hash(tmp_path: Path) -> None:
    from banana_smasher import build_hf_moe_uniform, open_hf_moe_uniform

    model = _write_tree(tmp_path / "m")
    built = build_hf_moe_uniform(
        model,
        revision=REVISION,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        output=tmp_path / "artifact",
        native_spill_root=tmp_path / "spill",
    )

    assert built["status"] == "PASS"
    assert built["reload_verified"] is True
    assert built["mechanisms"] == {
        "fallback": 0,
        "reconstruction": 0,
        "relay": 0,
        "streaming": 0,
    }
    # every non-routed tensor is stored as EXACT source data bytes
    assert {row["name"] for row in built["native_tensors"]} == NON_ROUTED
    for row in built["native_tensors"]:
        assert row["representation"] == "exact-source-data-bytes"
        assert row["artifact_sha256"] == row["source_sha256"]
    # only adapter-selected routed matrices carry Q2 wire members
    assert {row["name"] for row in built["routed_tensors"]} == ROUTED
    for row in built["routed_tensors"]:
        assert row["wire"]["encoder"]["fallback"] == 0
    # reload is byte-verifying and idempotent
    assert open_hf_moe_uniform(tmp_path / "artifact") == built


def test_reload_refuses_a_tampered_native_member(tmp_path: Path) -> None:
    from banana_smasher import build_hf_moe_uniform, open_hf_moe_uniform

    built = build_hf_moe_uniform(
        _write_tree(tmp_path / "m"),
        revision=REVISION,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        output=tmp_path / "artifact",
        native_spill_root=tmp_path / "spill",
    )
    victim = Path(built["storage"]["native_root"]) / built["native_tensors"][0]["path"]
    victim.write_bytes(b"\x00" * victim.stat().st_size)

    with pytest.raises(ValueError, match="native member hash mismatch"):
        open_hf_moe_uniform(tmp_path / "artifact")


def test_build_refuses_a_non_routed_only_intent(tmp_path: Path) -> None:
    from banana_smasher import plan_hf_moe_uniform

    for scope, native_rest in (("all", True), ("routed_only", False)):
        with pytest.raises(ValueError, match="routed_only with native_rest=True"):
            plan_hf_moe_uniform(
                _write_tree(tmp_path / "m"),
                revision=REVISION,
                tier="q2",
                scope=scope,
                native_rest=native_rest,
                receipt_path=tmp_path / f"P-{scope}-{native_rest}.json",
            )


def test_plan_rejects_a_duplicate_tensor_outside_the_index(tmp_path: Path) -> None:
    """A shard header carrying an unindexed tensor is a duplicate/coverage defect."""
    from banana_smasher import plan_hf_moe_uniform

    model = _write_tree(tmp_path / "m")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    del index["weight_map"]["lm_head.weight"]  # present in the shard, absent from the index
    (model / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="coverage or routed-scope defects"):
        plan_hf_moe_uniform(
            model,
            revision=REVISION,
            tier="q2",
            scope="routed_only",
            native_rest=True,
            receipt_path=tmp_path / "P.json",
        )
    receipt = json.loads((tmp_path / "P.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["coverage"]["duplicates"] == ["lm_head.weight"]


def test_plan_rejects_an_index_gap(tmp_path: Path) -> None:
    """A tensor declared in the index but missing from every shard header is a gap."""
    from banana_smasher import plan_hf_moe_uniform

    model = _write_tree(tmp_path / "m")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.layers.1.mlp.experts.9.up_proj.weight"] = (
        "model-00001-of-00001.safetensors"
    )
    (model / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="coverage or routed-scope defects"):
        plan_hf_moe_uniform(
            model,
            revision=REVISION,
            tier="q2",
            scope="routed_only",
            native_rest=True,
            receipt_path=tmp_path / "P.json",
        )
    receipt = json.loads((tmp_path / "P.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["coverage"]["gaps"] == [
        "model.layers.1.mlp.experts.9.up_proj.weight"
    ]


def test_admission_rejects_an_unsafe_shard_binding(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source

    model = _write_tree(tmp_path / "m")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "../escape.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="unsafe shard binding"):
        admit_hf_source(model, revision=REVISION, receipt_path=tmp_path / "A.json")


def test_admission_publishes_architecture_and_model_type(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source

    admitted = admit_hf_source(
        _write_tree(tmp_path / "m"),
        revision=REVISION,
        receipt_path=tmp_path / "A.json",
    )

    assert admitted["config_semantics"] == {
        "architectures": ["FixtureMoeForCausalLM"],
        "model_type": "fixture_moe",
    }


def test_discovery_and_plan_reuse_sealed_admission_without_rehashing_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher.hf_moe as hf_moe

    model = _write_tree(tmp_path / "m")
    admission = hf_moe.admit_hf_source(
        model, revision=REVISION, receipt_path=tmp_path / "A.json"
    )
    original_sha256 = hf_moe._sha256

    def metadata_only_hash(path: Path) -> str:
        if path.suffix == ".safetensors":
            raise AssertionError("sealed payload shard was rehashed")
        return original_sha256(path)

    monkeypatch.setattr(hf_moe, "_sha256", metadata_only_hash)
    scope = hf_moe.discover_hf_moe_routed_scope(
        model,
        revision=REVISION,
        source_admission=admission,
        receipt_path=tmp_path / "S.json",
    )
    plan = hf_moe.plan_hf_moe_uniform(
        model,
        revision=REVISION,
        source_admission=tmp_path / "A.json",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=tmp_path / "P.json",
    )

    assert scope["source"]["reuse_verified"] is True
    assert plan["source"]["reuse_verified"] is True
    assert scope["routed_tensor_names"] == sorted(ROUTED)


def test_sealed_admission_reuse_rejects_payload_stat_drift(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source, discover_hf_moe_routed_scope

    model = _write_tree(tmp_path / "m")
    admission = admit_hf_source(
        model, revision=REVISION, receipt_path=tmp_path / "A.json"
    )
    shard = model / admission["shards"][0]
    shard.write_bytes(shard.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="sealed admission member stat drift"):
        discover_hf_moe_routed_scope(
            model,
            revision=REVISION,
            source_admission=admission,
            receipt_path=tmp_path / "S.json",
        )
