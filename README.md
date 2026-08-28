# Banana Smasher
Canonical package code lives in the nested path `banana-smasher/src/banana_smasher/`; this is intentional—do not create a second package tree.
Install from `banana-smasher/pyproject.toml`; the single command is `smash` (`banana_smasher.cli:main`).
Python 3.11 or newer is required; older interpreters cannot install the pinned numeric stack.
For the public routed-only Q2 path, start with [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md):
plan and fit-check a pinned HF MoE source, run the bounded estimate, build through
`ResidentRepairAPI.build_uniform(...)`, reload with `open_hf_moe_uniform(...)`,
and only then score PRE. No internal module map or fleet-specific path is required.
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
Use `smash --help` for the public CLI; `runtime/v7/` is the pinned rail, with image helpers at `docker/examples/build_image.sh` and `docker/examples/serve.sh`.
Contributor laws are in `AGENTS.md`; accelerations and source lineage are in `ACCELERATIONS.md` and `PROVENANCE.md`.
