# Production update acceleration

This lane composes the portable segmented update core with a model-agnostic full-depth production helper. It adds grouped K-major update primitives, a bounded FWHT seam, and relocation-safe frozen SafeTensor descriptors without changing packed-wire export or runtime artifacts.

## Provenance

| Surface | Source identity | Port decision |
| --- | --- | --- |
| Bounded FWHT | repository commit `555fb20fd80e8733c9c7184412c840665d3c3dbf`; file SHA-256 `16eea326db3a40fffa579e7b75c5ac1a7719d4dfed912ba04c98932249115f50` | Accepted exactly, with its bounded-scratch and autograd tests. |
| K-major batched VJP | same commit; file SHA-256 `af335d1b0d6dd91f918f67216543a19d94ac0121f13c3190ead4243d63182ec2` | Accepted, then tightened so the CPU reference reduction requires explicit opt-in. |
| Fused and grouped K-major VJP | same commit; file SHA-256 `4fc10bb4e238bae693df8671722791dd1af997d91c8b54633573b5afed7cd97d` | Ported the kernels and added fail-closed code-range, packed-layout, and BF16-contract validation. Triton/CUDA absence fails loudly. |
| Layer graph | semantic source at the same commit | Ported the generic grouped BMM/VJP and unbalanced-routing behavior. Model-specific adapters were omitted. The CPU reference reduction requires explicit opt-in. |
| File-backed frozen tensors | source file SHA-256 `97631f2ab99a655218200463b4976413aabe176bc8fad9d3a5beb596684a4029`; focused source test SHA-256 `05db7b5e82cbf57a72034116f88a181d63d6ccae6e9c7945e64c6bbb01f28a41` | Generalized to relative index/member paths, tensor key, shape, dtype, byte count, immutable index and member SHA-256 identities, lazy-open data, explicit execution device, and hash-bound relocation. No historical model path or default was retained. |

## Contract

`run_full_depth_update` receives an ordered layer surface, explicit frozen modules, batch-1 input and teacher tensors, a memory budget, input/loss adapters, and a callable acceleration-counter probe. It:

- applies memory autosizing before model compute;
- preserves the selected physical token axis at every depth;
- creates a fresh optimizer over only the explicit trainable surface;
- performs exactly one optimizer step through the portable update core;
- refuses missing, non-finite, or frozen gradients;
- refuses absent or zero acceleration counters unless the caller explicitly opts into a reference run;
- records depth shapes, requested and selected physical geometry, teacher geometry, observed counter deltas, immutable identities, peak memory, phase counts, and an explicit semantic claim that defaults to no equivalence claim.

`FileBackedTensorDescriptor` stores no tensors. A binding verifies the index and selected SafeTensor member by SHA-256 and validates the indexed key, shape, dtype, and logical byte count. Rebinding to a relocated root succeeds only when those immutable identities match; old or substituted members fail closed. `FileBackedFrozenLinear` reopens the member independently for forward and backward and retains no frozen parameter or buffer.

## TDD evidence

The focused tests were first run with the new modules absent and failed during collection for the missing production surfaces. After the implementation they passed. Additional RED/GREEN cycles proved that static path labels are insufficient, CPU K-major reductions require explicit reference opt-in, non-divisible VJP tails must fail before optimizer use, fused code indices/layouts/dtypes must be checked before launch, and file-backed loads must stay bound to the verified open member.

Current CPU-feasible focused coverage includes descriptor identity/rebind/substitution and open-file race resistance, zero persistent tensor storage, forward/backward lazy reopen, K-major same-work gradient parity and tail refusal, grouped unbalanced routing, fused pre-launch validation, bounded FWHT scratch, full-depth token geometry at depths 1/3/5, fresh optimizer behavior, frozen parameter/buffer preservation, honest semantic defaults, observed sentinel deltas, and slow/reference-path refusal.

## Deliberate omissions and pending gates

- No private runner, coordination mechanism, model path, artifact identity, or operational default is included.
- No model-specific mixed-tier or QTIP adapter is included in this isolated update lane; existing solver/kernel ownership remains untouched.
- No dense or CPU fallback is selected automatically.
- CUDA/Triton fused-kernel parity and a real full-depth production model update remain hardware-only gates. CPU tests do not claim those gates passed.
