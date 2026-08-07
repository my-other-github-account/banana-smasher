# Native-V4 fine-rate build and Anchor64 API

Implementation revision: `d0b2977bdd46af826ae9c9c9ba9cdbf7ec9ecaf8`

## Contract

Native V4 keeps one homogeneous geometry per candidate:

- `L=16`
- `V=4`
- shared Q9×V2 TLUT
- integer transition width `B`
- exact code rate `B / 4` BPW

`native_v4_geometry(bpw)` accepts exact quarter rates from 1.00 through 4.00 BPW. It rejects non-quarter rates rather than averaging members or introducing an assignment map.

Examples:

| Code BPW | B | Bytes per 16×16 block |
|---:|---:|---:|
| 1.75 | 7 | 56 |
| 2.25 | 9 | 72 |
| 2.50 | 10 | 80 |
| 2.75 | 11 | 88 |

## Public Python API

Build one cell at any exact quarter rate:

```python
from banana_smasher import build_qtip_native_v4_cell

receipt = build_qtip_native_v4_cell(
    source="cell.npy",
    control="cell-control.npz",
    tlut="tlut.npy",
    output="candidate-b7",
    bpw=1.75,
    intended_basis_sha256=BASIS_SHA256,
    observed_basis_sha256=BASIS_SHA256,
    backend="cuda",
)
```

Build and Anchor64 any declared rate menu:

```python
from banana_smasher import build_qtip_native_v4_anchor_set

anchor_set = build_qtip_native_v4_anchor_set(
    source="cell.npy",
    control="cell-control.npz",
    tlut="tlut.npy",
    output="native-v4-anchor-menu",
    bpws=(1.75, 2.25, 2.50, 2.75),
    anchor_bank="anchor64.npz",
    teacher="cell.npy",
    intended_basis_sha256=BASIS_SHA256,
    observed_basis_sha256=BASIS_SHA256,
    backend="cuda",
)
```

The order is declaration order. Empty menus and duplicate equivalent quarter rates fail.

The existing `build_qtip25_native_v4_cell(...)` and `anchor_qtip25_native_v4_cell(...)` remain fixed-B10 compatibility wrappers.

## CLI

```bash
smash qtip-native-v4 build-cell \
  --source cell.npy \
  --control cell-control.npz \
  --tlut tlut.npy \
  --output candidate-b9 \
  --bpw 2.25 \
  --intended-basis-sha256 "$BASIS_SHA256" \
  --observed-basis-sha256 "$BASIS_SHA256"
```

Omitting `--bpw` preserves the existing 2.50-BPW command behavior.

## Backpack declarations

Fine-rate candidates use a distinct family so they cannot silently route through the historical mixed-ring QTIP implementation:

```json
{
  "id": "native-v4-225",
  "family": "qtip_native_v4",
  "bpw": 2.25,
  "backend": "cuda",
  "control_root": "controls",
  "tlut": "tlut.npy",
  "tlut_sha256": "<sha256>",
  "basis_sha256": "<sha256>"
}
```

The provider is resolved dynamically as `qtip-native-v4@2.25`. A plan may declare any finite set of valid quarter-rate tiers without adding provider code. The shared TLUT is represented by one content-addressed activation artifact across those tiers.

Candidate generation expects one `<cell_id>.npz` compact control under `control_root`. Candidate generation, candidate verification, pricing, prediction, and Anchor64 stages are implemented. Selected serving-pack materialization remains deliberately fail-loud until the stock-runtime pack contract learns native-V4 transforms; this document makes no serving claim.

## Size and batching

There is no fixed block-count limit. Physical cells are partitioned into 16×16 blocks and streamed through `solve_batch`; packed bytes scale exactly as:

```text
code_bytes = weight_count × B / 32
```

The compact FWHT transform requires each physical matrix axis to be a positive power of two and divisible by 16. CUDA batch size controls working memory; it does not change the code stream.

## Hardware smoke

One NVIDIA GB10 built two synthetic 8,388,608-weight cells (`32768×256`, identity compact transform, shared Gaussian Q9×V2 TLUT) through the public CUDA API with `solve_batch=2048`.

| Tier | Code bytes | Full-cell wire BPW | Encode | Encode rate | Installed decode | Fallbacks |
|---:|---:|---:|---:|---:|---:|---:|
| B7 / 1.75 | 1,835,008 | 1.816898 | 1.0631 s | 7.891M weights/s | 311.1M weights/s | 0 |
| B9 / 2.25 | 2,359,296 | 2.316898 | 1.1797 s | 7.111M weights/s | 310.5M weights/s | 0 |

Receipt SHA-256:

- B7: `d16dd608a03ee1e7be8e450525c1ae0e365058f652af311ffb8bd9ed6a83a49d`
- B9: `c15baee7b7aa3309f47cee110da78053ef86863f12c4fa211fdc0e66ac40fd5e`

Both receipts passed CPU/CUDA decode parity and exact packed-byte closure. These are mechanism and throughput smokes, not model-quality or HOLDOUT results.

## Focused validation

```text
19 passed in 2.63s
Ruff: all checks passed
compileall: passed
wheel: banana_smasher-1.0.0-py3-none-any.whl
CLI --bpw help smoke: passed
```
