# Reports and evidence

This directory is the public reporting surface for Banana Smasher. It contains curated human-readable reports, comparison tables, migration notes, and compact scrubbed receipt summaries. Keep each artifact under `notes/`; do not place reporting material beside product code.

## Evidence rules

- Bind each claim to a model/artifact basis and same-work receipt hashes.
- Separate measured results from estimates and pending hardware gates.
- Prefer aggregate metrics and reproducible commands over raw logs.
- Record failures and rejected comparisons; do not present a missing gate as a pass.
- Mark newly added measurements as `NEW`, and leave the owner as `TBD` until a named public maintainer accepts it.
- Every comparison row must state method, exact basis, KLD, top-1, GB, packed bpw, floating-point format, instrument, sample count (`n`), status, and source. Record FP8 as 8 bits.
- Keep raw private receipts, machine paths, host identities, task identifiers, credentials, and internal orchestration outside Git.
- Do not add model weights, generated packs, checkpoints, or other large binary payloads.

Use `report-template.md` for new performance or quality reports.
