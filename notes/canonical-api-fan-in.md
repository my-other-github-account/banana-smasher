# Canonical API fan-in

PR #4 is the focused PoC integration branch for the reusable Banana Smasher command and runtime surface:

- `smash solve`, `smash export`, `smash update`, `smash bank`, `smash evaluate`, and `smash anchor`
- exact Backpack and dynamic-dimension commands
- the stock-vLLM `banana_smasher_plugin:register` entry point and native-plane runtime closure

## Related pull requests

- PR #2's plugin/runtime source paths are already byte-identical to PR #4. It is superseded for this focused fan-in.
- PR #1 remains separate. Its claim-bound route controller is operational tooling, not canonical package/runtime closure required by this PoC.
- PR #3 remains separate. Its repository-level BALANCED64 publication surface is not required by the current package API/runtime path.

## Anchor closure

The four-bank anchor API from commit `3d8630ae14ceaa7e832b182bda49698f99a5315e` is integrated into the current command surface. The wheel includes the anchor schemas, bundled bank manifests, and `ANCHOR_EVALUATION.md`.

Focused Python 3.13 package smoke:

```text
python3.13 -m pytest -q banana-smasher/tests/test_anchor_evaluation.py banana-smasher/tests/test_cli.py
20 passed, 3 skipped
```

The skips are the existing macOS limitation for Linux `renameat2(RENAME_EXCHANGE)` metadata-refresh tests.

Real CLI validation:

```text
python3.13 -m banana_smasher.cli anchor validate \
  --manifest banana-smasher/anchor_banks/train_balanced64.bank.json
status=PASS window_count=64
membership_sha256=3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3
manifest_payload_sha256=37dd16bb5e6b0ac6ce954a31b65fed12a7f5f17915808d01d8bf10bcab3116d9
```

The plugin module also loads its declared `banana_smasher_plugin:register` callable from source. This local import is not a CUDA, stock-vLLM boot, API, latency, or quality result; those hardware-dependent gates remain pending until run on the target runtime.
