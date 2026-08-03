# Update12 readiness and launch boundary

This candidate admits two independently sealed public changes:

- portable segmented update core: `55c5c5899a9a67b2b9062ecb4c91baa966b48e5b`;
- serial exact-equal activation-cache batching: `f16a9f11ca42a18e44a1825f4848496a3c40f0c1`.

The combined focused gate covers physical-token memory selection, strict batch-one geometry, segmented accumulation, one optimizer step, checkpoint interruption/resume, directory relocation and rebind, completed replay, immutable identity drift, activation-cache exact byte equality, and serial ordered persistence.

## Launch boundary

The repository now contains `notes/update12-dry-run-launch-manifest.json`. It is intentionally marked `BLOCKED_PREREQUISITES` and `launch_authorized=false`. It is not evidence that model training, a persistent worker, or any update has run.

A physical Update12 sequence may begin only after all of these are sealed:

1. the exact pre-repair pack, including complete member bytes, assignment, basis, and structural verification;
2. the same-basis own-teacher BALANCED64 baseline with instrument, KLD, top-1, sample-count, and packed-byte evidence;
3. an installed accelerated update backend that proves persistent `WAITING`, begins segment timing at `SEGMENT_START`, preserves relocation-safe resume, and refuses fallback;
4. a fresh physical memory budget that includes resident frozen, trainable, optimizer, staging, and calibrated activation bytes while preserving at least 4 GiB for the operating system.

The public command template requests batch one and 1,024 physical tokens. The core may shrink that window before compute when measured memory requires it. Each public update transaction must produce exactly one optimizer step and a `PASS_UPDATE` receipt with observed physical shape, logical geometry, teacher geometry, finite required gradients, peak memory, immutable identities, and complete timing. Twelve such sequential, identity-consistent transactions are required for Update12.

Any missing gate, basis drift, ambiguous backend, failed resume/rebind, impossible memory geometry, forbidden fallback, wrong observed shape, or wrong optimizer count is a terminal refusal. No warm start or reduced-work substitute is permitted.
