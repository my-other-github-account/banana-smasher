# QTIP2 quality and bpw root-cause report — 2026-08-03

## Verdict

The quarantined QTIP2 result was produced by a K2 solver that evaluated 8 predecessors per retained prefix and labeled that reduced graph `exact`. Canonical QTIP for `(L=16, k=2, V=2)` evaluates `2^(kV) = 16` predecessors. This is a recurrence of the previously measured half-branch QTIP2 search defect.

The reported `1.142857 bpw` was a separate accounting error. The packed trellis stream is exactly 2 bits per quantized value. The receipt divided its bytes by a quantized-value count that was 1.75 times too large.

The `0.5067100527587314` KLD row and its `1.142857 bpw` label remain quarantined.

## External specification

The QTIP paper defines an `(L,k,V)` trellis as having `2^(kV)` incoming and outgoing edges per node. It also states that a length-`T` sequence requires `kT + L - kV` bits before machine-word alignment. Therefore:

- `k=2`, `V=2` requires 16 predecessor choices per vector step.
- The selected edge stream costs 2 bits per scalar weight.
- State/alignment and transform metadata can make the complete wire slightly larger than 2 bpw, never 1.142857 bpw for a complete K2 representation.

Sources:

- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235)
- [Cornell QTIP canonical bitshift implementation](https://github.com/Cornell-RelaxML/qtip/blob/main/lib/codebook/bitshift.py): `sumdelta` and `state_cand` enumerate `2**(K*V)` transitions; `update()` minimizes over that complete axis.

## Quality defect

### Current bad solver

The sealed V46 receipt states:

- geometry: `L16/K2/V2`
- retained prefixes: 4,096
- branches per prefix: 8
- branch sampling: `alternating-parity-full`
- implementation: `qtip-trellis-v2-graph-replay-b256-chunked-batch-exact-v46`

It is exact only on its reduced 8-branch graph. It is not exact for the canonical 16-branch K2 recurrence.

### Previously measured same defect

The prior P823 paired probe re-encoded 24 existing half-branch K2 units with the full-16 exact recurrence:

| Cohort | Units | Median SSE reduction | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Rep-16 | 16 | 50.1761309343% | 49.4944928080% | 51.0277162113% |
| Overnight | 8 | 49.9047162916% | 49.3177727201% | 50.1437945240% |

The old unit receipts explicitly recorded `branches_per_prefix=8` and `full_branches_per_prefix=16`. The probe verdict was mandatory re-encode and QTIP2 price refresh.

Validated proof implementation SHA-256:

`96b83c837a017c36f6630ac7b6b7a3be16888ea15a03593f3c4709b0675c3a50`

The later production adaptation that generated the accepted P851 exact-QTIP2 menu has SHA-256:

`379a24289514ead53de1415fdddc9cf77026d46c7b8d9ffef783fe5632a9319b`

The production diff changes only the Triton launch width from 32 warps to 8 and updates the implementation label. The recurrence and output semantics are unchanged.

### New matched-fit discriminator

A fresh current-basis L009/E000/fused13 discriminator held the solver and source weight fixed while expanding the fit from 32 to 128 windows:

| Metric | V46 / 32 windows | V46 / 128 windows | Improvement |
|---|---:|---:|---:|
| Candidate SSE | 3085.421258130882 | 3067.6585868179013 | 0.5756967956% |
| Relative RMS | 0.5436206514282734 | 0.5420535894562948 | 0.2882638781% |
| Fit rows | 1,146 | 4,737 | — |
| Fit route mass | 219.57373046875 | 985.437255859375 | — |

The 128-window artifact used the matching model-index basis, a fresh solve with no warm start, and exact canonical pack round-trip. It remained on the 8-branch V46 recurrence and is quarantined.

This rules out fit-window count as the primary cause. More calibration improved V46 by less than one percent in SSE, while restoring the missing K2 branches previously improved matched units by approximately fifty percent.

### Ruled out as primary causes

- **Different trellis source wheel:** inspected accepted and bad `trellis_v2` source bytes were identical.
- **32 versus 128 fit windows:** matched discriminator improved SSE by only 0.5756967956%.
- **Current pack corruption:** the 32- and 128-window artifacts both reported exact canonical pack round-trip.
- **Model basis drift:** the matched discriminator bound the intended current model-index SHA.

The downstream materializer/decompressor must still pass an independent consumer parity gate before a full rebuild, but the first large numerical divergence is already localized to the K2 search recurrence.

## BPW defect

The sealed aggregate receipt contains:

- packed trellis bytes: `69,256,347,648`
- declared rate: `2` bits per value
- incorrect `quantized_weight_elements`: `484,794,433,536`

The quantized-value count follows directly from the packed stream and declared rate:

`69,256,347,648 bytes × 8 bits/byte ÷ 2 bits/value = 277,025,390,592 values`

Therefore:

| Accounting scope | Bytes | Denominator | BPW |
|---|---:|---:|---:|
| Trellis code stream | 69,256,347,648 | 277,025,390,592 weights | 2.0000000000 |
| Complete expert wire excluding amortized shared TLUT | 69,572,057,088 | 277,025,390,592 weights | 2.0091171265 |
| Matching target FP8 reference | 277,025,390,592 | 277,025,390,592 weights | 8.0000000000 |

The complete-wire number includes the required row/column sign-scale fields and matrix scale. A shared TLUT adds only an amortized model-level term and must be listed explicitly when included.

The bad denominator is exactly 1.75 times the physical quantized-value count:

`484,794,433,536 ÷ 277,025,390,592 = 1.75`

That produces the bogus label:

`69,256,347,648 × 8 ÷ 484,794,433,536 = 1.142857142857... bpw`

The packed stream was not magically smaller than two bits. The reporting code mixed a whole/reference model count with the QTIP expert-wire numerator.

## Corrective path and acceptance

1. Bind the production full-16 P851 implementation (`379a2428…`) in the reusable Banana Smasher QTIP2 API.
2. Prove assignment/trajectory parity against the independently validated P823 recurrence (`96b83c83…`) and Cornell's canonical `2**(K*V)` Viterbi, with and without overlap.
3. Re-run one current-basis L009/E000/fused13 unit using the already sealed 128-window fit.
4. Independently decode and reconstruct it through the actual consumer; compare assignment hash, reconstructed weight hash/trajectory, and one anchor-window output.
5. Only after those gates pass, rebuild the QTIP2 bank and measure KLD/top-1.
6. Report both `code_bpw=2.0` and full expert-wire bpw with an explicit byte ledger and quantized-weight denominator.

No corrected current-basis QTIP2 KLD is claimed by this report.
