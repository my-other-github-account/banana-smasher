# Serial activation-cache batching

## Scope

This change extracts two reusable activation-cache mechanisms from sealed production evidence into the public package without importing model wrappers or orchestration:

- `build_activation_cache` amortizes fixed cache-builder work across a caller-selected batch while keeping persistence ordered and strictly serial.
- `run_shape_stable_batch` batches only a caller-declared shape-stable path, then invokes the shape-sensitive path once per item in original order.

The source evidence diff is SHA-256 `02b0b11aac20c8445756b16b15fde0221849ca1a843512e29e880edb4b6e543d`. The public API is a clean generic rewrite rather than a copy of architecture-specific code.

## Integrity and acceleration sentinels

The focused tests require:

- one batch-stable call for a multi-item batch and one shape-sensitive call per item;
- exact requested-key order before any payload is written;
- unique cache keys;
- a positive, non-boolean integer batch size;
- a non-negative, non-boolean integer byte count from every writer;
- serial writes on the caller thread;
- monotonic completed-key and persisted-byte progress.

Any contract drift fails closed. No thread pool, process pool, asynchronous writer, or hidden fallback is used.

## Public local microbenchmark

Receipt: `notes/activation-cache-serial-benchmark.json`

Method:

- 24 keys;
- 262,144 persisted bytes per key (6,291,456 total bytes per arm);
- 50,000 fixed builder rounds per batch;
- baseline batch size 1 versus candidate batch size 8;
- five repetitions;
- identical serial atomic file writes in both arms;
- SHA-256 equality for every payload.

Measured medians:

| Arm | Builder calls | Median seconds | Exact payload equality |
|---|---:|---:|---|
| Batch size 1 | 24 | 0.249735 | yes |
| Batch size 8 | 3 | 0.036646 | yes |

Measured local speedup: **6.8147x**. This isolates fixed per-batch builder amortization; it is not a GPU throughput claim.

## Production evidence boundary

The sealed source evidence reported a 16.208425 to 3.102164 seconds-per-window cold-cache change (**5.224877x**) with exact cache equality and a worst scorer absolute delta of `5.8300793e-06`. Those numbers remain provenance for the mechanism, not a claim that this generic public microbenchmark reproduced the model workload. No Spark GPU payload was launched for this change.
