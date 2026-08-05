# BALANCED64 IQ2/IQ3/IQ4/DwarfStar/QTIP2.5 results

This is the first repository-native compact evaluation bundle for the
DeepSeek-V4-Flash-0731 BALANCED64 comparison, now including the exact-lock
BANANA-SMASHER QTIP2.5 result and its six-category breakdown. The machine receipt is
[`results/deepseek-v4-flash-0731-balanced64-v1.json`](results/deepseek-v4-flash-0731-balanced64-v1.json).

## Global comparison

All rows use the same own-base FP8 teacher, 64 ordered windows, 65,536 scored
positions, teacher top-8,192 support, and normalized BPW denominator of
`284,334,567,511` parameters.

| Candidate | Top-1 agreement ↑ | KLD ↓ | Wire GB | Wire BPW | FP |
|---|---:|---:|---:|---:|---|
| UD-IQ4_XS | 92.44% | 0.068349 | 136.662 | 3.845 | FP8 e4m3 dynamic own-base teacher |
| **BANANA-SMASHER QTIP2.5** | **89.09%** | **0.181971** | **106.623** | **3.000** | FP8 e4m3 dynamic own-base teacher |
| UD-IQ3_XXS | 87.95% | 0.177708 | 104.208 | 2.932 | FP8 e4m3 dynamic own-base teacher |
| UD-IQ2_XXS | 84.57% | 0.276747 | 90.861 | 2.556 | FP8 e4m3 dynamic own-base teacher |
| DwarfStar Q2 0731 | 83.69% | 0.309521 | 93.691 | 2.636 | FP8 e4m3 dynamic own-base teacher |

Top-1 ordering is:

1. UD-IQ4_XS
2. BANANA-SMASHER QTIP2.5
3. UD-IQ3_XXS
4. UD-IQ2_XXS
5. DwarfStar Q2 0731

KLD orders IQ3 immediately ahead of QTIP2.5, so the two global metrics are not
uniform after adding QTIP2.5. Exact rates, counts, KLD decimals, bytes, and BPW
remain in the machine receipt rather than the rounded human table.

The complete IQ2 run exceeds DwarfStar by 577 Top-1 matches, or
0.88043212890625 percentage points. DwarfStar's earlier win on one IQ2 canary
must not be reported as a complete-run win.

## Basis and denominator

- BALANCED64 is `64 × 1,024 = 65,536` scored positions.
- The normalized BPW column uses one exact denominator for every row and is
  derived from packed wire bytes, not an in-memory integer container or nominal
  quant label.
- DwarfStar wire bytes include both its base GGUF and drafter GGUF.
- The public suite lock is `d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9`.

## QTIP2.5 category breakdown

Projecting the same 64 global window IDs through the canonical 512-window
provenance gives `19/7/9/10/10/9` for
agentic/chat/code/multilingual/prose/reasoning.

The earlier protected `610e13dd…` map changed 52 labels and is invalid for
subgroup reporting. QTIP2.5 was reaggregated from complete per-window evidence
using the corrected map:

| Category | Top-1 agreement ↑ | Matches / positions | KLD ↓ | Windows |
|---|---:|---:|---:|---:|
| Agentic | 89.27% | 17,368 / 19,456 | 0.233114 | 19 |
| Chat | 93.22% | 6,682 / 7,168 | 0.049606 | 7 |
| Code | 93.09% | 8,579 / 9,216 | 0.083579 | 9 |
| Multilingual | 83.30% | 8,530 / 10,240 | 0.323280 | 10 |
| Prose | 82.26% | 8,423 / 10,240 | 0.264734 | 10 |
| Reasoning | 95.56% | 8,807 / 9,216 | 0.026375 | 9 |

The category Top-1 counts fan into exactly `58,389/65,536`. The weighted
category KLD differs from the sealed global binary64 result by only
`1.803125e-17`, inside the verifier's `1e-15` decimal reaggregation tolerance.
The other four historical candidates do not have complete corrected-map
category evidence in this bundle, so no cross-model category comparison is
claimed.

The exact public-safe QTIP2.5 source receipt is
[`evidence/qtip25-competitive-balanced64-v1.json`](evidence/qtip25-competitive-balanced64-v1.json),
SHA-256 `0811769ba4888ab7ef9737d78c7741e3dfddcc452f9021ec5245c701d9b14644`.

## Source-identity scope

The machine receipt records SHA-256 identifiers for the protected terminal
results, manifests, teacher identity, independent checks, and row collections.
Those protected objects are not copied or retrievable from this public
repository, so their digests do not independently authenticate the historical
KLD claims.

Artifact identity is complete as recorded for IQ2, DwarfStar, and QTIP2.5. IQ3 and IQ4
remain explicitly partial because public candidate revisions and artifact
manifest/tree digests were not retained. Operational run/host labels, private
paths, prompts, tensors, model weights, and credentials are excluded.

## Validate

```bash
python3 -m evaluations.tools.receipts verify \
  notes/evaluations/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock evaluations/configs/balanced64-v1.json
```

This validates the tracked lock, structure, Top-1/GB/BPW arithmetic, and
rankings. It does not execute models or recompute historical KLD.

For standardized per-position row aggregation and exact limitations, see
[`../../evaluations/protocols/balanced64-v1.md`](../../evaluations/protocols/balanced64-v1.md).

## Replay status

Full GPU replay is blocked for all five rows because teacher/corpus
payloads and the historical scorer environment are not public. IQ3/IQ4 also
lack complete public candidate identities. The receipt records these blockers
per row. No HOLDOUT512 result is included or implied.
