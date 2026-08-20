# Banana Smasher
Canonical package code lives in the nested path `banana-smasher/src/banana_smasher/`; this is intentional—do not create a second package tree.
Install from `banana-smasher/pyproject.toml`; the single command is `smash` (`banana_smasher.cli:main`).
Start with `CODEBASE_MAP.md` for the authoritative module and repository map.
Build uniform QTIP-V7 tiers with `ResidentRepairAPI.build_uniform(model, tier)` in `resident_repair_api.py`.
Mix prebuilt tiers to a BPW target with `ResidentRepairAPI.backpack_mix(builds, bpw_target)`; mixing never re-solves anchors.
Score and train through `score_pre()`, `repair_train()`, and `score_post()` on that same API.
Artifact identity, layer-tier provenance, canary reference, and tolerance live in each artifact's `identity.json`.
Use `smash --help` for the public CLI; `runtime/v7/` is the pinned rail, with image helpers at `docker/examples/build_image.sh` and `docker/examples/serve.sh`.
Contributor laws are in `AGENTS.md`; accelerations and source lineage are in `ACCELERATIONS.md` and `PROVENANCE.md`.
