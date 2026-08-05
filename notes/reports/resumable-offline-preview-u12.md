# Resumable candidate materialization and Preview-U12 parity

## Source scope

This report describes the Banana Smasher source revision containing this file, based on parent commit `da60675bf7059c1a64e50a9b503355398a6051dd`. It covers the public package and `smash` CLI changes for bounded candidate materialization, offline weight-offloaded inference, and the F521 Preview-U12 six-class objective.

Hardware receipts and commands are delivered through the task receipt rather than copied into this public repository, so private host paths and host identity remain outside the source tree.

## Candidate materialization

`materialize_candidate_producer` now treats candidate production as an ordered 64-window operation with bounded chunks. After each successful chunk it atomically publishes:

- consumable interim JSONL rows in manifest order;
- a progress receipt with completed and total window counts;
- bindings for the bank content, candidate, model manifest, producer config, basis, and selected execution mode.

Resume accepts only an ordered prefix whose progress bindings and completed count match. A resumed invocation computes only the missing suffix. A completed rerun validates the final producer and receipt and returns without invoking the producer again.

The Python API and `smash anchor materialize-candidate` expose `auto`, `vllm`, and `offline-layerwise` execution modes plus a positive `chunk_size`. The bundled offline configuration uses vLLM's public positive `cpu_offload_gb` weight-offload path; the offline backend refuses a configuration that does not actually enable weight offload. `auto` follows the producer configuration rather than silently changing its backend.

## Preview-U12 math

The Preview-U12 option API requires authenticated per-option inputs for:

- six-class predictions;
- class-specific expert-routing importance;
- projection weight and class-specific or scalar projection correction;
- exact physical bytes.

For class `c`, the option cost is:

`max(0, (prediction[c] + projection_correction[c]) * routing_importance[c] * projection_weight)`

Parity uses explicit raw weights of one for each of `agentic`, `chat`, `code`, `multilingual`, `prose`, and `reasoning`, normalized to a uniform one-sixth mean. The earlier `1/1/1.5/2/1.5/1` weighting remains available only as the `legacy-preview` preset. The public solve adapter feeds authenticated option costs into the exact class-balanced solver, which enforces the byte envelope and all six aggregate class caps. Pareto pruning removes an option only when another option in the same cell is no worse in bytes and every one of the six class costs, and strictly better in at least one dimension.

Tier menus remain generic. Callers can include and exclude arbitrary declared tiers. The current campaign policy selects `qtip2.5` by default without embedding a model-family condition in the solver.

## Focused validation

Executed from `banana-smasher/` with `PYTHONPATH=src`:

```text
python3.13 -m pytest -q \
  tests/test_candidate_materializer_resume.py \
  tests/test_offline_layerwise_dispatch.py \
  tests/test_backpack_preview_u12.py \
  tests/test_class_balanced_knapsack.py \
  tests/test_fixed_d4_public_closure.py
12 passed
```

Additional gates:

```text
python3.13 -m ruff check src tests
All checks passed!

git diff --check
PASS

PYTHONPATH=src python3.13 \
  -m banana_smasher.cli anchor materialize-candidate --help
PASS: help exposes --execution-mode {auto,vllm,offline-layerwise} and --chunk-size
```

A diagnostic full repository run reached `391 passed, 5 skipped` and retained one existing release-surface assertion failure because the README contains additional command examples beyond that test's literal three-command expectation. The requested changed-path tests and lint gates are green; this PoC does not alter the release README contract.

The sealed fixture is `banana-smasher/tests/fixtures/f521_preview_u12.json`, SHA-256 `2ebf895ce77e8b99ae7390892dc6cd31c587f360e2032ee2eed97682b91adfc4`.
