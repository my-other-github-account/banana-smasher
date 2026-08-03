# Activation-cache write overlap benchmark

## Scope

This benchmark isolates one reusable activation-cache build primitive: build the next deterministic batch while prior batch payloads are durably persisted by a bounded worker pool. It does not claim a model-training or end-to-end repair speedup.

The implementation preserves input key order, rejects missing/duplicate builder rows before persistence, propagates write failures, bounds pending batches, and reports monotonic completed-key and byte counters.

## Physical benchmark

A same-machine serial baseline and overlap candidate each produced 24 deterministic cache payloads of 16 MiB (384 MiB per arm). Both arms used two-key build batches, two I/O workers, identical payload generation, atomic rename, and `fsync` per payload. Progress-write overhead was applied once per persisted batch in both arms.

| Arm | Wall time |
|---|---:|
| Serial baseline | 3.679824 s |
| Bounded overlap | 3.638765 s |
| Same-work speedup | 1.011284× |

All 24 candidate payload SHA-256 values matched the corresponding baseline values exactly, and the candidate wrote 402,653,184 bytes. An earlier fair-overhead repetition measured 1.034896×. A preceding subsecond probe measured 0.853717× because candidate-only progress persistence dominated the tiny workload; that negative result was retained and the comparison was corrected rather than discarded.

## Verification

Focused unit tests cover:

- real overlap between the next build and prior writes;
- exact persisted payload bytes and monotonic progress;
- fail-closed builder-key drift;
- invalid batch, worker, and pending-depth settings.

This primitive requires consumer integration and a representative model-cache gate before any end-to-end acceleration claim.
