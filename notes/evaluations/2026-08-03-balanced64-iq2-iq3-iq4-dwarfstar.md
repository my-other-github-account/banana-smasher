# BALANCED64 IQ2/IQ3/IQ4/DwarfStar global results (NEW)

This is the first repository-native compact evaluation bundle for the
DeepSeek-V4-Flash-0731 BALANCED64 comparison. The machine receipt is
[`results/deepseek-v4-flash-0731-balanced64-v1.json`](results/deepseek-v4-flash-0731-balanced64-v1.json).

## Global comparison

All rows use the same own-base FP8 teacher, 64 ordered windows, 65,536 scored
positions, teacher top-8,192 support, and normalized BPW denominator of
`284,334,567,511` parameters.

| Candidate | FP | Exact wire GB | Normalized BPW | KLD ↓ | Top-1 agreement ↑ |
|---|---|---:|---:|---:|---:|
| UD-IQ4_XS | FP8 e4m3 dynamic own-base teacher | 136.662446656 | 3.845116627283469 | 0.0683488486737012 | 60,584/65,536 (92.44384765625%) |
| UD-IQ3_XXS | FP8 e4m3 dynamic own-base teacher | 104.207848032 | 2.931978308348837 | 0.17770788160865483 | 57,638/65,536 (87.9486083984375%) |
| UD-IQ2_XXS | FP8 e4m3 dynamic own-base teacher | 90.860736928 | 2.556445745541928 | 0.2767474104898907 | 55,422/65,536 (84.5672607421875%) |
| DwarfStar Q2 0731 | FP8 e4m3 dynamic own-base teacher | 93.691352992 | 2.636087586877748 | 0.30952134732070036 | 54,845/65,536 (83.68682861328125%) |

Quality ordering is uniform on the two measured global metrics:

1. UD-IQ4_XS
2. UD-IQ3_XXS
3. UD-IQ2_XXS
4. DwarfStar Q2 0731

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

## Corrected subgroup map

Projecting the same 64 global window IDs through the canonical 512-window
provenance gives `19/7/9/10/10/9` for
agentic/chat/code/multilingual/prose/reasoning.

The earlier protected `610e13dd…` map changed 52 labels and is invalid for
subgroup reporting. No class-level KLD or Top-1 from that map is published
here. This regrouping does not alter the global rows above.

## Source-identity scope

The machine receipt records SHA-256 identifiers for the protected terminal
results, manifests, teacher identity, independent checks, and row collections.
Those protected objects are not copied or retrievable from this public
repository, so their digests do not independently authenticate the historical
KLD claims.

Artifact identity is complete as recorded for IQ2 and DwarfStar. IQ3 and IQ4
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

Full GPU replay is blocked for all four historical rows because teacher/corpus
payloads and the historical scorer environment are not public. IQ3/IQ4 also
lack complete public candidate identities. The receipt records these blockers
per row. No HOLDOUT512 result is included or implied.
