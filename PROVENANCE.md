# Provenance

The retained implementation and binary assets were extracted directly from Git objects at immutable source commit:

`c00714c6803f7e2de7a95d103dbe172236b22adf`

No bytes were copied from a working tree. `provenance/SOURCE_INVENTORY.json` records, for every retained source file:

- its path in this repository;
- its path relative to the authoritative source subtree;
- SHA-256 of the source object bytes;
- SHA-256 of the current repository bytes; and
- whether the bytes remain identical.

Most retained files are byte-identical. Intentional adaptations are limited to standalone documentation/examples and source tests or package documentation that had to reflect the current pinned public runtime and mandatory caller-supplied P1016 runtime-floor value. The inventory makes each changed source-derived file explicit.

Upstream components remain separately identified by immutable inputs in `docker/Dockerfile`: stock vLLM image digest and revision, FlashInfer source and fix commits, public DeepGEMM source commit, and the CUDA runtime package versions. Runtime package provenance is generated inside the image at `/opt/banana-smasher/provenance/`.

The independently authored anchor-evaluation product surface is classified as
generated product code rather than falsely inheriting the runtime extraction
commit. `banana-smasher/anchor_banks/FOUR_BANK_PROVENANCE.json` binds the
recovered public-safe bank memberships and source-receipt hashes. Its companion
`UNRESOLVED_PROVENANCE.json` names tokenizer, teacher, and historical scorer
identities that could not be proven from the recovered receipts; those fields
remain unresolved rather than inferred.

This repository contains no statement granting rights to original work. Consult each upstream project for the terms that apply to upstream components.
