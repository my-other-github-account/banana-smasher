# CODEBASE MAP — read before changing campaign code

## Canonical repository
`github.com/my-other-github-account/banana-smasher` is the single source of truth. Deploy only pinned commits. The local canonical checkout is normally `~/clawd/banana-smasher-runtime`.

## Intentional nested layout
The reusable Python distribution is intentionally under `banana-smasher/`; its package is `banana-smasher/src/banana_smasher/`. Do not create a second root-level package.

## Unified build, mix, score, and repair path
- `resident_repair_api.py`: public two-stage API: `build_uniform()` × QTIP{1,2,3,4}-V7, `backpack_mix()`, `score_pre()`, `repair_train()`, `score_post()`.
- `artifact_identity.py`: fail-closed identity loaded from each artifact's `identity.json`; no artifact-specific module constants.
- `loader.py`, `hf_deepseek_v4_backpack_adapter.py`, `qtip_v7_routes.py`: canonical mixed-backpack loading and V7 route adaptation, including dense L034.
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

## Governing laws
One path for scoring; identity belongs to artifacts, not constants; canary gates are mandatory; QTIP-V7 {1,2,3,4}+native is the v1 tier space; non-routed tensors remain native; uniform builds precede backpack mixing; fixes land as tested commits on `main`.
