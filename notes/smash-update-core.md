# Portable `smash update` core

The update core separates physical tensor geometry from logical accumulation:

- `--tokens N` requests the token dimension seen by each model forward.
- `--segments M` controls the number of contiguous physical slices accumulated into one logical update.
- batch size is currently one.
- every declared segment contributes a summed loss divided by the total logical item count, followed by exactly one optimizer step.

## Memory sizing

Sizing runs before backend loading or model compute. The budget includes available memory, the 4 GiB operating-system floor, resident frozen tensors, trainable tensors, optimizer state, staging storage, and calibrated activation bytes per token. The selected token count is the largest value not exceeding the request that fits the budget. There is no fixed token cap for small models. If even the minimum physical shape does not fit, the command fails before forward execution.

## Backend contract

The CLI resolves an installed entry point in the `banana_smasher.update_backends` group. A backend receives the selected physical tokens, segment count, batch size, immutable identity, request path, output paths, and the complete memory-sizing record. It must execute through the public tensor/update-engine API and return a passing durable receipt. Missing, ambiguous, malformed, fallback, wrong-shape, or multi-step backends fail closed.

The Python API `run_tensor_update` accepts a model's trainable parameters, optimizer, batch-1 input tensor, teacher targets, teacher mask, positions, a summed-loss callback, and a `MemoryBudget`. Teacher targets may include feature dimensions after token axis 1. Masks and positions use `[1, tokens]` geometry.

## Durable resume and relocation

Each committed segment snapshots trainable parameters, accumulated gradients, optimizer state, CPU/CUDA random state, phase timings, and peak-memory evidence. The checkpoint manifest admits only a contiguous completed prefix. Payload publication precedes manifest publication, with file and directory synchronization around atomic replacement.

Payload, output, receipt, and rebind records use paths relative to the checkpoint root. Moving a run directory changes only its root binding. On reopen, the loader verifies immutable identity and payload bytes, loads only tensor/primitive checkpoint values through restricted weights-only deserialization, verifies the manifest HMAC against the checkpoint's separately published 32-byte authentication key, creates an atomic rebind receipt, then updates the manifest binding. Joint payload/manifest substitution without the authentication key, payload corruption, unauthorized serialized globals, rebind-receipt corruption, geometry drift, and changes to any content, config, assignment, AOT, runtime, or code SHA fail closed. The checkpoint directory, including its authentication key, is one trust unit and must be moved together. A completed checkpoint is bound to the caller's resolved output and receipt paths and replays its verified receipt without another forward or optimizer mutation.

## Receipt fields

A passing receipt includes:

- requested and selected physical tokens;
- logical tokens and segment item counts;
- observed model input shape;
- teacher target, mask, and position geometry;
- forward, backward, and optimizer counts;
- finite required-gradient confirmation;
- peak-memory bytes and the complete sizing breakdown;
- immutable content/config/assignment/AOT/runtime/code identities;
- segment, optimizer, artifact, and durable receipt/manifest publication timing boundaries;
- parameter hashes before and after the single optimizer step;
- backend and explicit no-fallback status.

Contiguous causal slicing is labeled `causal-segmented-no-equivalence-claim`. The core rejects `exact` or `equal-work` labels unless the caller supplies a passing semantic parity result.
