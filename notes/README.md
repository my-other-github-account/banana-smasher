# Notes

`notes/` is the canonical home for public-safe Banana Smasher evidence and operating records.

## Layout

- `reports/` — revision-bound implementation, validation, release, and benchmark reports.
- `tables/` — decision-grade KLD, top-1, size/GB, packed-bpw, FP-baseline, throughput, and timing tables.
- `migrations/` — source-of-truth cutovers and semantic port audits.
- `decisions/` — durable API, compatibility, acceleration, and publication decisions.

## Rules

1. Every quantitative row identifies the exact code revision, artifact/basis, instrument, sample size, and whether it is predicted, measured, accepted, or pending.
2. Packed VQ bpw is computed from packed wire bytes plus declared overhead; never from an unpacked integer container. FP8 is reported as 8 bpw, not 16.
3. Report KLD and top-1 from the same instrument, include GB and bpw, retain an FP column, and use one basis per column.
4. Never publish raw host receipts, private paths, hostnames, private IPs, credentials, or unsanitized command logs. Store scrubbed summaries with hashes of immutable public artifacts where useful.
5. Static tests, image builds, GPU boots, API correctness, and performance measurements are separate gates. Do not promote one into evidence for another.
6. Reports and tables are append-only evidence records once published; corrections get a dated superseding note rather than silent historical rewriting.
