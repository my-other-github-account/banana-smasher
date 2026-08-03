# Reports and evidence

This directory is the public reporting surface for Banana Smasher. It contains curated human-readable summaries, comparison tables, migration notes, and compact scrubbed evidence.

## Evidence rules

- Bind each claim to a model/artifact basis and same-work receipt hashes.
- Separate measured results from estimates and pending hardware gates.
- Prefer aggregate metrics and reproducible commands over raw logs.
- Record failures and rejected comparisons; do not present a missing gate as a pass.
- Keep raw private receipts, machine paths, host identities, task identifiers, credentials, and internal orchestration outside Git.
- Do not add model weights, generated packs, checkpoints, or other large binary payloads.

Use `report-template.md` for new performance or quality reports.
