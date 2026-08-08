# Native-V4 distinct-expert batch acceptance — 2026-08-07

## Result

**PASS:** the public native-V4 batch encoder processed eight distinct FF0731 layer-18 experts below two seconds per complete expert while reproducing all historical physical-SSE references exactly.

| Scope | Solve wall time | Per cell | Max physical-SSE delta | Status |
|---|---:|---:|---:|---|
| 8 distinct `down` cells | 5.114572312973905 s | 0.6393215391217382 s | 0.0 | PASS |
| 8 distinct `fused13` cells | 10.710486008028965 s | 1.3388107510036207 s | 0.0 | PASS |
| 8 complete experts (`down + fused13`) | 15.82505832100287 s | **1.9781322901253588 s** | 0.0 across all 16 cells | **PASS** |

## Binding

- Source commit: `33d71997967dee23e48f6c20fb75b7a227f33837`
- Source tree: `df94c83f4a608ceb5757895ae0ddb9c908cf0863`
- FF0731 basis SHA-256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Receipt SHA-256: `b34cbf1dc3e94e91b813ce9935b8e97ce8ae9d53d9b00ea41b7d477b3a780a91`
- Layer: 18
- Experts: 0–7
- Maximum solver sequence batch: 2,048
- Receipt schema: `banana-smasher-native-v4-distinct-expert-batch-v1`

## Method

Each expert used its own authentic `down` and `fused13` source tensors, controls, and activation-derived Hessian. This is not a repeated-cell throughput estimate. The batch path preserved the historical reverse-16 LDLQ recurrence, selected scale, native-V4 geometry, decoder, and physical tensor reconstruction.

The timing is encoder solve time. It excludes one-time model loading and Hessian preparation. Preparation took 12.463230384048074 seconds for all eight `down` cells and 28.52323631296167 seconds for all eight `fused13` cells.

Physical quality was checked after native-V4 decoding and reconstruction into the original projection geometry. Every one of the 16 direct SSE values exactly matched its historical single-cell reference; every recorded SSE delta was `0.0`.

## Focused checks

- Scalar-arithmetic regression: PASS
- Public CUDA-cell focused tests: `2 passed in 0.81s`
- Authentic E000 `fused13` code parity: 0 differing tiles and 0 differing bytes
- Eight-expert terminal status: PASS
