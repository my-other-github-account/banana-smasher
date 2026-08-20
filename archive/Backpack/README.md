# Backpack QTIP anchor calculations

This page preserves the internal `train_balanced64` uniform-QTIP anchor measurements used for Backpack family and solver calculations.

> **Population warning:** these anchors are not competitive `BALANCED64_V1` results. The internal bank has **0/64 window-ID overlap** with the competitive bank. Do not place these values in, or infer a competitive ranking from, the [Evals table](../Evals/README.md).

Both candidates use the same ordered 64 internal `(window_id, category)` pairs, 1,024 positions per window, 65,536 positions total, and the same DeepSeek-V4-Flash-0731 FP8 own-base teacher. The internal class mix is agentic/chat/code/multilingual/prose/reasoning = `12/10/11/10/10/11`.

## Exact global anchors

| Uniform anchor | Top-1 ↑ | KLD ↓ | FP basis |
|---|---:|---:|---|
| **QTIP3 uniform exact** | 59,796/65,536 (0.91241455078125) | 0.047175884822847125 | FP8 own-base teacher |
| **QTIP2 genuine corrected all-43** | 57,124/65,536 (0.87164306640625) | 0.20088080907074077 | FP8 own-base teacher |

QTIP2's complete internal anchor exceeds its historical `0.2` KLD target by exactly `0.00088080907074077`; the measurement remains a valid family anchor, not a competitive row.

## Exact Top-1 counts by internal category

| Uniform anchor | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **QTIP3 exact** | 11,146/12,288 | 9,593/10,240 | 10,578/11,264 | 8,917/10,240 | 8,814/10,240 | 10,748/11,264 |
| **QTIP2 all-43** | 10,670/12,288 | 9,290/10,240 | 10,266/11,264 | 8,189/10,240 | 8,186/10,240 | 10,523/11,264 |

## Exact KLD by internal category

| Uniform anchor | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **QTIP3 exact** | 0.06856082258666941 | 0.016910390892254247 | 0.030649311925740972 | 0.0898051557328945 | 0.07053989790676105 | 0.00789362555635767 |
| **QTIP2 all-43** | 0.26116463296024256 | 0.07059443210562813 | 0.1278824905213041 | 0.423045817064655 | 0.3008114294217323 | 0.03374290939545452 |

## Machine result and identities

The [machine-readable anchor result](results/deepseek-v4-flash-0731-train-balanced64-qtip-anchors-v1.json) retains the exact global/category decimals, Top-1 numerators and denominators, internal bank manifests, candidate/source hashes, FP8/scorer basis, and explicit population separation.

The competitive result remains exclusively under [`Evals/results/`](../Evals/results/deepseek-v4-flash-0731-balanced64-v1.json).
