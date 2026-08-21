# CODEBASE MAP — read before changing campaign code

## Canonical repository
`github.com/my-other-github-account/banana-smasher` is the single source of truth. Deploy only pinned commits. The local canonical checkout is normally `~/clawd/banana-smasher-runtime`.

## Intentional nested layout
The reusable Python distribution is intentionally under `banana-smasher/`; its package is `banana-smasher/src/banana_smasher/`. Do not create a second root-level package.

## Unified build, mix, score, and repair path
- Identity boundary: published PRE is `f9bffe04…` (KLD `0.229392`, Top-1
  `56,533/65,536` in the authoritative [Evals table](Evals/README.md#results)).
  Raw U0 `7978d100…` is a different checkpoint state at about `0.2356` KLD.
  Never substitute one identity for the other.
- `resident_repair_api.py`: public two-stage API: `build_uniform()` × QTIP{1,2,3,4}-V7, `backpack_mix()`, `score_pre()`, `repair_train()`, `score_post()`.
- `artifact_identity.py`: fail-closed identity loaded from each artifact's `identity.json`; no artifact-specific module constants.
- `loader.py`, `hf_deepseek_v4_backpack_adapter.py`, `qtip_v7_routes.py`: canonical mixed-backpack loading and uniform V7 route adaptation.
- `knapsack.py`, `backpack_dimensions.py`, `backpack_exact64.py`, `backpack_contextual_prepare.py`: measured tier selection and exact64 evaluation; mixing consumes uniform packs and never re-solves.
- `repair.py`, `resident_training.py`: canonical repair and resident training implementations.
- `contract.py`, `validation.py`: fail-closed public contracts.
- `cli.py`: the single `smash` command surface.

## Other maintained roots
- `banana-smasher-plugin/`: stock-vLLM integration.
- `runtime/v7/`: pinned deployment rail and vendored runtime snapshot; reusable changes still originate in the package.
- `Evals/`: evaluation protocols, results, tests, and MMLU tooling.
- `docker/`: image construction plus deployment examples.
- `provenance/`: release source inventory required by image admission.
- `archive/`: historical notes and superseded Backpack/kernel-development material; never use it as product code.
- `tools/w328_recovery/`: retained W328 hidden-boundary reconstruction and GB10
  memory-gate utilities. They are recovery tools, not product runtime paths.

## Porting closure
The interim `~/clawd/ds4-flash-kldmatrix/repair_api/` scratch tree is retired:
its maintained identity, resident score/train, cache, and geometry behavior now
lives in the package modules above. The generic W328 checkpoint and memory-gate
changes were imported; the later single-layer stream-retry patch is historical
and deliberately not part of the uniform product runtime. Nothing in either
scratch location is a deployment dependency.

## Governing laws
One path for scoring; identity belongs to artifacts, not constants; canary gates are mandatory; QTIP-V7 {1,2,3,4}+native is the v1 tier space; non-routed tensors remain native; uniform builds precede backpack mixing; fixes land as tested commits on `main`.
