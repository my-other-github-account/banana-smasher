# L034 Banana Q2 versus authentic EXL K2

Date: 2026-08-10

## Scope

This report records the complete layer-34 assignment-physical comparison for the
Banana Q2 candidate at commit
`bb807b9f6ffcae211f9dc779b5b576198c3ac6da` against the authentic EXL K2
control.

- Model/input basis SHA-256:
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Coverage: 256 experts × `w1`/`w2`/`w3` = 768 physical members
- Roster: 768 members, 0 gaps, 0 duplicates, 0 pass-through bytes
- Roster SHA-256:
  `13aaa61931aa362a355854aad7bfdb78db328833dfcb83f2444435d058ad2140`
- Ordered physical file-set SHA-256:
  `cdbbaa738d658758c75df711185b1d57f87ed612b1518e9173957517ed5d7052`
- Evaluation rail: canonical rows `[10, 12]`, 2,048 full-vocabulary token
  positions
- KLD convention: natural logarithm

## Result

| Variant | Top-1 | Top-1 rate | Mean KLD | Complete accounting |
|---|---:|---:|---:|---:|
| Authentic EXL K2 | 1968/2048 | 96.09375% | 0.018689766940723482 | 2.0117225646972656 BPW |
| Banana Q2 current variant | **1978/2048** | **96.58203125%** | **0.01788801344325433** | 2.035162607828776 BPW |

Banana Q2 improved Top-1 by 10 positions (0.48828125 percentage points) and
reduced mean KLD by `0.000801753497469152` (4.289799332501019%). Its complete
accounting is `0.0234400431315104` BPW (1.165172750102238%) larger, including
one deduplicated 2,048-byte FP16 `[1024]` LUT.

## Immutable evidence

- Final terminal SHA-256:
  `146eb45a6c6b6ced321e2bc85e453d3ac6f7f987bce23025b972b9cee958c0aa`
- Scientific terminal SHA-256:
  `8f1bad8cb0f52557dabcbe43cbe2cb5936a936c6a92ab9e30e48d47f3f51d138`
- Complete score terminal SHA-256:
  `068cf52b8d8f1f8984bf84dea2d216bb527d95253111648b261ae0e2556c2fb7`
- Public evidence receipt:
  `notes/reports/2026-08-10-l034-banana-q2-vs-exl-k2.json`

## Interpretation and limits

This is a complete 768-member assignment-physical layer result, not a proxy or
partial-expert score. It establishes that this Banana Q2 candidate beats the
K2 control on both measured quality metrics for the stated rail.

The complete 256-expert compact-wire deployment decode was not part of this
receipt. The PR therefore lands the independently authored native Q2 codec,
solver, and exact physical reconstruction machinery plus this result; it does
not claim that a trainable LUT optimizer or the final full compact-wire
serving integration is complete.
