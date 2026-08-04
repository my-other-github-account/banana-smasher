# Anchor64 v6: exact status, reconstruction, and measurement contract

## Decision

An anchor is a completed uniform quantization **with an observed KLD score**. A layer/unit coverage manifest alone is not an anchor score.

The exact-0731 QTIP2, QTIP3, D4K2048, and D4K4096 uniform-tier artifacts are complete, but their manifests contain no overall or per-class KLD fields. Therefore:

| Exact-0731 uniform tier | Quantization artifacts | Overall KLD | Six-class KLD | Current anchor-score status |
|---|---:|---:|---:|---|
| QTIP2 | complete | not measured | not measured | missing |
| QTIP3 | complete | not measured | not measured | missing |
| D4K2048 | complete | not measured | not measured | missing |
| D4K4096 | complete | not measured | not measured | missing |

Historical KLD rows from another basis do not fill these cells.

## Separate evaluation instruments

- `BALANCED64_V1` remains the separate quick final-result panel. It is not used as Anchor64.
- `ANCHOR64_V6_HISTORICAL_CALIBRATION` is the reconstructed uniform-tier anchor panel.
- The two panels have zero window overlap.

## Anchor64 v6 panel

- Parent bank: 512 windows, 1,024 scored positions per window.
- Parent class counts: agentic 154, chat 52, code 76, multilingual 76, prose 78, reasoning 76.
- Anchor64 quota: agentic 12, chat 10, code 11, multilingual 10, prose 10, reasoning 11.
- Selection: six class-specific minimax binary optimizations over every recovered historical full-512 KLD vector.
- Historical calibration corpus: 15 vectors—11 uniform VQ anchors plus four mixed/repaired wires.
- Exact window IDs and class grouping are frozen in `ANCHOR64_V6_HISTORICAL_CALIBRATION.json`.

### Frozen class corrections

| Class | Multiplier |
|---|---:|
| agentic | 1.0012260434 |
| chat | 1.0034874621 |
| code | 1.0049846293 |
| multilingual | 0.9975504068 |
| prose | 1.0043267042 |
| reasoning | 0.9955673075 |

### Estimator

For each uniform quant:

1. Measure the mean KLD for each class on its selected Anchor64 windows.
2. Multiply each measured class mean by its frozen class correction.
3. Reconstruct the estimated full-512 overall KLD with the parent class proportions:

```text
overall_estimate =
    154/512 * agentic_estimate
  +  52/512 * chat_estimate
  +  76/512 * code_estimate
  +  76/512 * multilingual_estimate
  +  78/512 * prose_estimate
  +  76/512 * reasoning_estimate
```

The unweighted mean over the 64 windows is diagnostic only because Anchor64 deliberately oversamples the thin parent classes.

## Historical reconstruction result

Across all 15 historical full-512 vectors used for calibration:

| Metric | Result |
|---|---:|
| Worst absolute per-class error | 1.5382% |
| RMS per-class error | 0.6860% |
| Worst absolute overall error | 0.5313% |
| RMS overall error | 0.2518% |

Status: `PASS_HISTORICAL_RECONSTRUCTION`.

## Critical limitation

This panel does **not** carry a universal ±4% class-accuracy claim on an unseen base or quant family. Earlier omitted/new-vector tests failed that gate:

- Leave-one-vector-out worst class error: 8.97–11.12%.
- Leave-one-vector-out worst overall error: 3.85–3.86%.
- A weighted-coreset variant reached 10.74% class and 5.01% overall on a new vector.

Consequently, reports must distinguish:

- **Measured Anchor64 KLD:** the raw observed class/window values on the exact candidate.
- **Estimated full-512 KLD:** the corrected class values and parent-weighted overall estimate.
- **Measured full-512 KLD:** unavailable until a 512-window run is physically executed.

Anchor64 v6 is useful for ranking and pricing the four exact-0731 uniform tiers cheaply. It is not permission to relabel estimates as full-512 measurements.

## Exact-0731 scoring contract

Every exact-0731 row must bind:

1. Exact basis SHA `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.
2. The completed uniform-tier artifact manifest.
3. Anchor64 manifest SHA below.
4. The exact-0731 own-teacher bank and scorer configuration.
5. Raw per-window KLD, raw per-class means, corrected per-class estimates, and parent-weighted overall estimate.
6. Coverage and failure receipts; no silent fallback to old rows, BALANCED64, or a different base.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `ANCHOR64_V6_HISTORICAL_CALIBRATION.json` | `5c731c28c0096f245fc7109016f353071f59f617c376401e473513fe1a6a6bc5` |
| `ANCHOR64_V6_HISTORICAL_CALIBRATION.METHOD.json` | `3aba1493d31cde16d65ef53c348c7ece0ca7f16cbd80a9b63470b5785d016934` |
| `ANCHOR64_V6_HISTORICAL_CALIBRATION_VALIDATION.json` | `b2eb6270d5321106ce2d6db02ce304bededfc7751f427a7c37f30afe1242ab15` |

## Next measurement table

The following values remain `TBD` until exact-0731 Anchor64 scoring runs:

| Tier | Measured Anchor64 global | Agentic | Chat | Code | Multilingual | Prose | Reasoning | Estimated full-512 overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QTIP2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| QTIP3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D4K2048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D4K4096 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
