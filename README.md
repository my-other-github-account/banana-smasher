# Banana Smasher
Canonical package code lives in the nested path `banana-smasher/src/banana_smasher/`; this is intentional—do not create a second package tree.
Install from `banana-smasher/pyproject.toml`; the single command is `smash` (`banana_smasher.cli:main`).

## Supported Python and the two dependency tiers
Python 3.11 or newer is required; older interpreters cannot install the pinned numeric
stack (the default macOS `python3` 3.9 fails on `numpy==2.3.5`). Verified working: 3.11–3.13.

The public API has two declared, testable dependency tiers:

- **Base install** — `python -m pip install ./banana-smasher`. Supports the complete
  metadata-only planning tier: `admit_hf_source`, `discover_hf_moe_routed_scope`,
  `plan_hf_moe_uniform`, `preflight_hf_moe_output_fit`, and
  `balanced64_hardware_contract`. These read config, index, and safetensors headers
  only; they never read tensor bytes and never need torch.
- **`[solve]` extra** — `python -m pip install './banana-smasher[solve]'`. Required by
  every call that encodes or executes: `estimate_hf_moe_uniform` and
  `build_hf_moe_uniform*` for production-sized routed tensors (the encoder refuses a
  slower fallback and names this extra), and the whole teacher-capture / PRE path.

Calls that need the extra fail closed naming it; see `HF_SOLVE_EXTRA_REQUIREMENT`.

## The complete public routed-only Q2 journey
Every entry point a fresh caller needs is named below and detailed with full signatures
and receipt contracts in [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md). No internal module
map, `CODEBASE_MAP.md`, `runtime/` internals, or fleet-specific path is required.

| Stage | Public call | Tier |
| --- | --- | --- |
| source admission | `admit_hf_source(model, *, revision, receipt_path)` | base |
| routed-scope discovery | `discover_hf_moe_routed_scope(model, *, revision, receipt_path)` | base |
| build plan | `plan_hf_moe_uniform(...)` | base |
| output fit | `preflight_hf_moe_output_fit(plan, ...)` | base |
| bounded estimate | `estimate_hf_moe_uniform(...)` | solve |
| build | `ResidentRepairAPI.build_uniform(...)` / `build_hf_moe_uniform(...)` | solve |
| horizontal build | `build_hf_moe_uniform_shard(...)`, `union_hf_moe_uniform_shards(...)` | solve |
| reload / admit | `open_hf_moe_uniform(...)`, `open_hf_moe_uniform_shard(...)` | base |
| hardware contract | `balanced64_hardware_contract()` | base |
| eval inputs | `recover_balanced64_source_text(...)`, `build_balanced64_token_ledger(...)` | base |
| teacher capture | `capture_balanced64_teacher(...)` | solve + CUDA |
| canonical PRE | `score_balanced64_pre(...)` | solve + CUDA |

Source admission accepts the canonical HuggingFace cache/snapshot layout: symlinked
members are resolved and bound by content SHA-256, and the receipt publishes the
authoritative repository roster plus the excluded client-side `.cache/huggingface/`
bookkeeping subtree, so file/byte identity is reproducible without guesswork.

Routed scope is derived from config and tensor-name semantics, never from a model name
or a hardcoded roster: routed layer ids are `[first_k_dense_replace, num_hidden_layers)`,
and any layer id at or above `num_hidden_layers` is an auxiliary prediction head
(multi-token-prediction / `num_nextn_predict_layers`) whose experts stay native rest.
Each plan and discovery receipt states this rule inline as `geometry.auxiliary_layer_rule`.

Teacher capture and PRE execute a real forward pass. Call `balanced64_hardware_contract()`
**before** staging a large source: it reports each registered runtime's declared
requirement (the shipped `hf-sharded` runtime requires CUDA, minimum 1 rank) and whether
this host satisfies it. The public calls fail closed with that contract stated.

## Checkpoint identity
Canonical checkpoint identity: the published QTIP2 V7 pre-repair artifact is
`f9bffe04…` (the [Evals row](Evals/README.md#results): KLD `0.229392`, Top-1
`56,533/65,536`). Raw U0 `7978d100…` is a different state (about `0.2356`
KLD), not an alias for the published PRE artifact.
Every checkpoint-loading operation requires the same explicit `checkpoint_sha`
keyword and refuses identity drift. Lower-level provider integrations build with
`ResidentRepairAPI.build_uniform(model, tier, checkpoint_sha=...)` and mix with
`ResidentRepairAPI.backpack_mix(builds, bpw_target, checkpoint_sha=...)`. For the public routed-only Q2 production journey, use the one-command isolated
phase runner in [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md): `smash improve` launches
`score_pre()`, `repair_train()`, and `score_post()` as three fresh processes,
with a hash-bound receipt between each process. The build call binds the SHA once, so
later phase calls need no repeated identity arguments. Every receipt echoes the
bound value as `input_checkpoint_sha256`.
Artifact identity, layer-tier provenance, canary reference, and tolerance live in each artifact's `identity.json`.

Every digest published in this repository states what it hashes — file bytes versus a
canonicalized internal identity — so shipped locks can be verified deterministically.

Use `smash --help` for the public CLI. Contributors (not fresh API callers) will also
find `CODEBASE_MAP.md`, the pinned `runtime/v7/` rail, image helpers at
`docker/examples/build_image.sh` / `docker/examples/serve.sh`, contributor laws in
`AGENTS.md`, and lineage in `ACCELERATIONS.md` / `PROVENANCE.md`.

