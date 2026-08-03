# Protected Source Durability Scrub — 2026-08-03

Status: PASS (measured, full population)

## Scope

A protected historical source tree and its independently verified archive were checked against the same sealed reference manifest. The source scrub was strictly read-only: it performed no repair, transfer, rename, permission change, deletion, model boot, pack operation, solve, or allocation.

Reference manifest SHA-256:
`9bcfda44273a5b0b574625d43d34ff17d846c6a09b176664016d8e8d2be76771`

## Measured results

| Metric | Protected source | Verified archive | Comparison |
|---|---:|---:|---|
| Manifest entries | 76 | 76 | exact |
| Directories | 12 | 12 | exact |
| Regular files hashed | 64 | 64 | exact |
| Regular logical bytes hashed | 146,593,983,631 | 146,593,983,631 | exact |
| Source allocated inventory | 146,594,226,176 | not applicable | source unchanged |
| Missing members | 0 | 0 | exact |
| Unexpected members | 0 | 0 | exact |
| Type/mode/size mismatches | 0 | 0 | exact |
| SHA-256 content mismatches | 0 | 0 | exact |
| Observed manifest SHA-256 | `9bcfda44273a5b0b574625d43d34ff17d846c6a09b176664016d8e8d2be76771` | `9bcfda44273a5b0b574625d43d34ff17d846c6a09b176664016d8e8d2be76771` | exact |

Source terminal receipt SHA-256:
`6734318272e0bf40412a77042560c45cb87c91888432fea4465f78f8f29aa24f`

Archive terminal receipt SHA-256:
`67d8f21957e19563460ba3dc82d686d2e7a9ea75427ff02db50dd5ab340e4aa4`

## Instrument and verification method

- Instrument identity SHA-256: `433c7fe75bcd5714126d5c8a54ee347585a7ae88796e9f2211aedbad0217b6f1`.
- I/O ran at idle scheduling priority and CPU ran at reduced scheduling priority.
- Every regular member was streamed through SHA-256.
- Relative path, file type, mode, size, and content SHA-256 were compared with the sealed reference.
- Each regular file was stat-checked before and after its read to fail closed on an in-read mutation.
- The complete population was measured; this was not a sample or prediction.

## Manifest-order audit note

The first full physical read completed all 64 files and 146,593,983,631 bytes with empty semantic mismatch lists, but its generated manifest used a different traversal order from the sealed reference. The already-hashed path-keyed map was then serialized in sealed reference order without rereading source content. The resulting manifest was byte-identical to the reference.

Order-only correction receipt SHA-256:
`23be0604a1035a38914886b1b064e1ac5a255da4bf9041d07c35953509655cb7`

## Conclusion and limits

The protected source and verified archive match exactly for the full sealed population. The source remained present with its original logical and allocated inventory. Source deletion was not authorized and did not occur.

This report establishes storage durability and source/archive identity only. It makes no claim about model serving, model correctness, package correctness, runtime performance, or deletion eligibility.
