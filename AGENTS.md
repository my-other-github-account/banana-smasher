# Canonical Banana Smasher repository

This repository is the sole source of truth for Banana Smasher code, tests, package metadata, runtime integration, Docker assets, public documentation, and public-safe reporting.

## Product scope

- The product is the mixed-QTIP Backpack pipeline and its QTIP2, QTIP3, D4, and native-MXFP4 serving runtime.
- New code, tests, documentation, reports, and deployment work must serve that product scope directly. Unrelated historical quantization formats are not product inputs, compatibility targets, or preservation targets.
- V4 is the containerized U004 mixed-QTIP Backpack on the F521 assignment. V5 is U012 on the same assignment and pipeline basis.

## Standalone deployment law

1. A deployment release starts from one fresh, self-contained pipeline run root and may consume only the declared base-model inputs plus artifacts generated inside that run root.
2. Backpack selection, pre-repair materialization and evaluation, repair, final export, image build, plugin installation, and performance evidence must form one hash-bound lineage.
3. Final export must not depend on a pack, overlay, repair checkpoint, image, cache, runtime module, or receipt borrowed from an earlier run. Optional machine caches may accelerate recomputation only when their producer, key, validation, and disposable nature are explicit.
4. The supported serving boundary is pinned stock vLLM plus the installable `banana-smasher-plugin` wheel and a verified self-contained `/model` pack. No out-of-tree Python overlay, host-side monkey patch, hidden environment setup, or manually edited model artifact is allowed.
5. Functional boot is not release acceptance. The same exported artifact must pass coherent API behavior and the frozen decode, concurrency, and prefill performance gates recorded under `notes/`.

## Repository boundary

- Canonical remote: `git@github.com:my-other-github-account/banana-smasher.git`
- Canonical local checkout on the primary development Mac: `~/clawd/banana-smasher-runtime`
- The historical `spark-bench-reproducers/glm52-ds4-bq3-ptq-opd` tree and its refs are read-only evidence. Import reviewed behavior from it; never land new Banana Smasher work there.
- Do not copy built wheels, compiled kernels, caches, model artifacts, private receipts, internal host paths, hostnames, IP addresses, or credentials into this public repository.

## Change discipline

1. Fetch `origin/main` and work in this repository.
2. Preserve the stock-vLLM pip-plugin boundary and every accepted acceleration. A missing accelerated dependency or authority input must fail closed; do not introduce silent fallback.
3. Port legacy fixes semantically, with source-ref provenance and focused tests. Do not merge a legacy tree wholesale.
4. Keep lightweight pack operations importable without Torch/vLLM/CUDA. Heavy producer/update dependencies must remain lazy and optional.
5. Run focused tests, the complete static suite, both wheel builds/ZIP checks, Ruff, privacy/credential scans, and license-surface checks before publication.
6. For runtime/acceleration changes, static tests are not a hardware acceptance claim. Record the required Linux ARM64 SM120/SM121 image/boot gate in `notes/` until it is physically rerun.
7. Use commit identity `banana_bae <banana_bae@users.noreply.github.com>` for public history.

## Reporting

All reports, result tables, benchmark summaries, methodology receipts, migration audits, and decision records belong under `notes/`. Product API documentation may stay beside the package or component it documents, but evidence and status claims belong in `notes/` and must bind their source/artifact revisions.

## Publication posture

No license has been assigned to the original work. Do not add a `LICENSE`/`COPYING` file, package license metadata/classifier, SPDX header, or prose granting a license unless the owner explicitly chooses one. Upstream dependency license facts may be stated only with clearly limited scope.
