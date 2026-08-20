# QTIP 2.5 native V4 API handoff

## Status and scope

Use the homogeneous native QTIP 2.5 codec through the public Banana Smasher API introduced at source revision `8308119`.

This integration covers the current PoC gate:

1. Take one physical expert/projection weight cell.
2. Bind it to the compact QTIP transform from the **same model basis**.
3. Build exact homogeneous `L16/B10/V4` codes on CUDA.
4. Reconstruct and seal the physical decoded cell.
5. Measure that cell with Banana's standard 64-window candidate-anchor metric.

It deliberately does not claim a complete serving pack or a whole-model model-logit verdict. The cell anchor is the fast candidate gate. Whole-model exact KLD still requires a materialized model producer and the canonical Anchor/train8 scorer.

## Bound evidence

- API integration source: `8308119`
- Native codec source: `e2ec6ba35f1e8aeba07d8fa573666eaf38efc438`
- CUDA packed-state acceleration: `7a26f983ef9c709b3156834415674094d384d997`
- Corrected two-cell train8 quality receipt SHA-256: `d22ba3faac8acd97712b1d59904ae901f12336ee7581affcfd4bb33d95353ad4`
- API CUDA smoke receipt SHA-256: `3edae40270e70723811a2e05c65144c91f49022c0b243d7c5ec05de603ed2154`

The API smoke exercised one full 8,388,608-weight cell:

| Metric | Result |
|---|---:|
| Status | PASS |
| Code BPW | 2.500000 |
| Code bytes | 2,621,440 |
| Compact transform bytes | 12,288 |
| Full cell wire BPW including Wscale and shared TLUT | 2.515629 |
| CUDA encode wall time | 1.943 s |
| Installed CUDA decode rate | 315.4M weights/s |
| Fallback calls | 0 |

## Inputs

Every build requires four exact inputs:

- `source.npy`: finite physical `float32 [rows, columns]` weights. Both dimensions must be divisible by 16.
- Compact control: `.pt` or `.npz` containing `SU`, `SV`, `Wscale`, and `shape`. Optional `qtip_k` is preserved as provenance.
- `tlut.npy`: finite `float32 [512, 2]` shared Q9/V2 TLUT.
- Basis SHA-256: the intended and observed model-basis hashes must match exactly.

The compact control must belong to the same cell and basis as `source.npy`. It supplies the transform only; V4 does **not** average K2/K3 members and does not alternate assignments.

## Python API: build quickly

```python
from banana_smasher import build_qtip25_native_v4_cell

cell = build_qtip25_native_v4_cell(
    "inputs/L000_E000_down.npy",
    "controls/E000_down_QTIP_TRANSFORM.pt",
    "shared/qtip_q9_v2_tlut.npy",
    "run/candidates/L000_E000_down",
    intended_basis_sha256=BASIS_SHA256,
    observed_basis_sha256=BASIS_SHA256,
    backend="cuda",
    solve_batch=2048,
    decode_batch=2048,
    decode_repeats=1,
)

assert cell["status"] == "PASS"
assert cell["accounting"]["exact_code_bpw"] == 2.5
assert cell["installed_cuda_decode"]["counters"]["fallback_calls"] == 0
```

The measured fast-path defaults are intentionally shallow:

- `backend="cuda"`
- `solve_batch=2048`
- `decode_batch=2048`
- `decode_repeats=1`

Use `backend="reference"` only for tiny tests. It is not a model-scale producer.

The output directory contains:

- `codes.npy`: exact packed B10 stream
- `decoded.npy`: reconstructed physical `float32` cell
- `SU.npy`, `SV.npy`, `Wscale.npy`: preserved compact transform storage
- `NATIVE_V4_CELL_RECEIPT.json`: low-level CUDA mechanics receipt
- `CELL_RECEIPT.json`: public basis, artifact, byte, speed, decode, and direct-error receipt

## Python API: anchor the built cell

```python
from banana_smasher import anchor_qtip25_native_v4_cell

anchor = anchor_qtip25_native_v4_cell(
    "run/candidates/L000_E000_down",
    anchor_bank="banks/cell_anchor64.npz",
    teacher="inputs/L000_E000_down.npy",
    output="run/anchors/L000_E000_down.json",
)

assert anchor["status"] == "PASS"
assert anchor["same_instrument"] is True
assert anchor["windows"] == 64
```

The anchor bank must contain exactly:

- `features`: finite `float32 [64, source_weight_count]`
- `classes`: 64 labels covering `agentic`, `chat`, `code`, `multilingual`, `prose`, and `reasoning`

The anchor receipt binds the candidate, teacher, bank, and measured metrics by SHA-256.

## CLI equivalent

Build:

```bash
smash qtip-native-v4 build-cell \
  --source inputs/L000_E000_down.npy \
  --control controls/E000_down_QTIP_TRANSFORM.pt \
  --tlut shared/qtip_q9_v2_tlut.npy \
  --output run/candidates/L000_E000_down \
  --intended-basis-sha256 "$BASIS_SHA256" \
  --observed-basis-sha256 "$BASIS_SHA256" \
  --backend cuda \
  --solve-batch 2048 \
  --decode-batch 2048 \
  --decode-repeats 1
```

Anchor:

```bash
smash qtip-native-v4 anchor-cell \
  --candidate run/candidates/L000_E000_down \
  --anchor-bank banks/cell_anchor64.npz \
  --teacher inputs/L000_E000_down.npy \
  --output run/anchors/L000_E000_down.json
```

## Batch pattern

Keep each cell independently sealed so a stopped producer can resume by skipping existing validated `CELL_RECEIPT.json` files:

```python
import json
from pathlib import Path

from banana_smasher import build_qtip25_native_v4_cell

for job in jobs:
    output = Path(run_root) / "candidates" / job["cell_id"]
    terminal = output / "CELL_RECEIPT.json"
    if terminal.is_file() and json.loads(terminal.read_text()).get("status") == "PASS":
        continue
    build_qtip25_native_v4_cell(
        job["source"],
        job["control"],
        shared_tlut,
        output,
        intended_basis_sha256=basis_sha256,
        observed_basis_sha256=basis_sha256,
        backend="cuda",
        solve_batch=2048,
        decode_batch=2048,
        decode_repeats=1,
    )
```

Do not parallelize multiple cells onto one GPU until a measured single-process batch requires it. The measured producer already completes one full cell in roughly two seconds; source/control I/O and model extraction can dominate.

## Acceptance rules for an agent

A cell is usable only when all of these hold:

1. `CELL_RECEIPT.json.status == "PASS"`.
2. Intended and observed basis SHA-256 are identical.
3. Geometry is exactly `L=16`, `B=10`, `V=4`, `phase_count=1`.
4. `exact_code_bpw == 2.5` and packed bytes close exactly.
5. `alternation == false` and `member_averaging == false` in geometry.
6. Installed CUDA decode parity passes and `fallback_calls == 0`.
7. The cell anchor uses the same bank and teacher as its matched control.

Direct SSE alone is not an admission result. PERIODIC and TWOSTEP both improved direct SSE while worsening train8 KLD. Advance V4 because it preserved Top-1 and improved matched KLD, not merely because it compressed well.

## Pasteable agent brief

> Use Banana Smasher source revision `8308119` or later. Do not reimplement QTIP 2.5 V4. Build each physical expert/projection cell with `build_qtip25_native_v4_cell(..., backend="cuda", solve_batch=2048, decode_batch=2048, decode_repeats=1)`, using the exact same-basis compact QTIP control and shared float32 `[512,2]` TLUT. Require exact 2.5 code BPW, L16/B10/V4, phase_count 1, zero alternation/member averaging, installed CUDA parity, and zero fallback calls. Then call `anchor_qtip25_native_v4_cell` with the matched teacher and common 64-window bank. Treat this as the fast cell gate; do not report a whole-model quality win until the materialized model has passed the canonical model-logit Anchor/train8 scorer.
