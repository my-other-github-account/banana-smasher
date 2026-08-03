# Provenance

The standalone extraction is rooted at immutable source commit:

`c00714c6803f7e2de7a95d103dbe172236b22adf`

Runtime corrections proven after that extraction are ported from immutable successor snapshots:

- `50468029e846c926e8f0aaeb6c9efc1c1a1ac0de` — DeepGEMM and native-plane runtime closure;
- `12157a76b11f172081882b9b5efbc8281b65f74b` — explicit CUDA-graph exclusion tag for the native-plane custom op.

The canonical port adds one fail-closed integration adaptation after independent review: the runtime image pins vLLM's breakable `PIECEWISE` CUDA-graph mode, and native-plane registration rejects opt-out, full capture, or a missing active vLLM configuration. This preserves the successor's eager-break behavior without treating `splitting_ops` as a silent fallback.

No extraction bytes were copied from an uncommitted working tree. `provenance/SOURCE_INVENTORY.json` records, for every retained source file:

- its path in this repository;
- its path relative to the authoritative source subtree;
- the immutable source commit for successor-derived files;
- SHA-256 of the source object bytes;
- SHA-256 of the current repository bytes; and
- whether the bytes remain identical.

Most retained files are byte-identical. Intentional adaptations are limited to standalone documentation/examples and source tests or package documentation that had to reflect the current pinned public runtime and mandatory caller-supplied P1016 runtime-floor value. The inventory makes each changed source-derived file explicit.

Upstream components remain separately identified by immutable inputs in `docker/Dockerfile`: stock vLLM image digest and revision, FlashInfer source and fix commits, public DeepGEMM source commit, and the CUDA runtime package versions. Runtime package provenance is generated inside the image at `/opt/banana-smasher/provenance/`.

This repository contains no statement granting rights to original work. Consult each upstream project for the terms that apply to upstream components.
