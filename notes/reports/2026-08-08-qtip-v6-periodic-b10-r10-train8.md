# Periodic QTIP2.5 B10 fixed-row and Train8 result

## Decision

The canonical periodic PR31 QTIP2.5 codec passes the full product gate with the fixed B10 geometry:

- vector width: 4
- transition bits: 10
- phase widths: `[2, 3, 2, 3]`
- exact code BPW: `2.500000`
- assignment and routing bytes: `0`
- final-state closure: PASS
- installed CUDA fallback calls: `0`

The accepted candidate uses one deterministic midpoint warmup cycle, fixed-path physical-Hessian scale refinement, and a frozen 2,048-byte PR31 table trained on a disjoint two-cell source bank. The table was frozen before fresh held-out E000/E001 encoding and fixed-row evaluation. No evaluation rows were used for training.

## Full Train8 product gate

Basis: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`

| Arm | Top-1 | Directional KLD | Code bits | Full wire bytes |
| --- | ---: | ---: | ---: | ---: |
| Canonical V4 Q9 | 7972 / 8192 | 0.006965687346973616 | 41,943,040 | 5,271,560 |
| Periodic B10 r10 | **7985 / 8192** | **0.006884747486994660** | 41,943,040 | 5,269,512 |

The periodic candidate improves Top-1 by 13 matches and reduces directional KLD by `0.000080939859978956`. The full evaluation terminal is SHA-256 `379695432bbe5515be33707ca5b5fe4039e0ea3a3a532766f79d5afb6bb1e6c1`; its manifest is SHA-256 `a00ce63c6e7809c78e8f885b7290a411f566902df1cfba383811810a4348e132`.

## Fixed-row ladder

| Rows | Candidate Top-1 | V4 Top-1 | Candidate KLD | V4 KLD | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 10, 12 | 1978 / 2048 | 1974 / 2048 | 0.011373581359976877 | 0.011671909253017003 | GREEN |
| 10, 12, 19, 24 | 3980 / 4096 | 3966 / 4096 | 0.009390345044349852 | 0.009617942974689809 | GREEN |
| Train8 | 7985 / 8192 | 7972 / 8192 | 0.006884747486994660 | 0.006965687346973616 | GREEN |

Receipt hashes:

- rows 10/12: `6424435fe1409f434323efb1e8bd650dc7c4a4ab7e5502df0b6ec722016620e9`
- rows 10/12/19/24: `06d5cf9f5641ab1697d9555813b189077a20b8a734715e47ef7538931b2b2dbe`
- Train8: `379695432bbe5515be33707ca5b5fe4039e0ea3a3a532766f79d5afb6bb1e6c1`

## K2/K3 bracket

On the same immutable Train8 evidence, authentic ordinary Q2 scores 7970 Top-1 with KLD `0.0070511504768758495`, while authentic ordinary Q3 scores 7988 Top-1 with KLD `0.006669683517957591`. Periodic B10 r10 scores 7985 Top-1 with KLD `0.006884747486994660`, placing it between the authentic K2 and K3 endpoints on both metrics and near K3.

## Frozen-table provenance

The training bank contains two basis-matched L018/E000 source cells (`down` and `fused13`) and excludes held-out L000/E000, L000/E001, and every evaluation row.

- training terminal: `b43ca1947614ff36d5702c9cc3a676d452a17400a28a9286f3ec30415993b5a4`
- frozen NPY artifact: `917534a1b01d652e0cf8454c767f8c6a7ecc5e9ba01222e18cedfd003ec3d899`
- frozen tensor bytes: `3ca6e37fbdbc2b5bbc69ecf51e47eda09d69f5f68e85c36e3e69b02e1b14f634`
- hardware parity terminal: `1889e4a70a331d56b2ecf89ef6807cd05636dc4842199c5e460b26f0ee897f4a`
- held-out E000 terminal: `f88bd7b55b3ff8a5d1a9ec6619ec4b7f085e944c04954e08881a72ab0093b2d6`
- held-out E001 terminal: `61fc884ee90138236ec941515129bc0f996348acd6ddd7e090e16bd90c56b2cf`

The frozen table is shipped as `banana-smasher/src/banana_smasher/assets/periodic_b10_rate_specific_pr31.npy` and exposed by `periodic_b10_rate_specific_codebook()`.

## Source and held-out artifact identity

The accepted hardware path binds these source hashes:

- codec: `25a87d3d683937dd1c673212378c276050f8b08d249e05c5dd98ea0e454d59ba`
- evaluated build API: `fbf6d8cae4cbb425b554218cf7dc123d579fc3ca8ef165eae261f2042863f295`
- CUDA cell runner: `30c7b8895de475540bd0e26b8a66eacd25328ff809d07b95118d1262da4f24cf`
- installed runtime bridge: `19a03e40ef9749ba8025282663529cf5505d4548e86097905ca1d8414f20d4bc`

The committed build API is `508d9f30c86932100a32eb6c70b9af92a4ec651bb18ad999f3df75443e2569d4`; its only delta from the evaluated API is the package-local frozen-table accessor named above. Held-out identities are:

| Cell | Source SHA-256 | Control SHA-256 | Packed-code SHA-256 |
| --- | --- | --- | --- |
| E000/down | `8ac827143d0b599074f38d65bd7a774d561798fb1435563e0df4984ca4c9e002` | `0288deb8c5a395dd385e248d3046d634baf65837a259993e0361bfc23ef78e47` | `0833c10b5496feb2bae5f1d309aab8e5f47176fa0d955b42fa9d9e411abaed18` |
| E001/down | `9209441edb03102b00538286cfbc09e2323adadf05bacb56eeca3553c17cdddf` | `632dbf6ca23eb4100dce22d764e03776c0e944a138028f9f2b43d51349604356` | `80ab8089a6167c8c52f63ab9a5db680de3e8cf5f5e52199974d8b88cdfe94fae` |

## Verification

Focused source-contract tests: 30 passed.

Focused installed-runtime tests: 8 passed.

Hardware evidence proves reference/CUDA encoder parity, reference/Torch/installed-CUDA decode parity, exact final-state closure, exact 2.5 code BPW, and zero installed-CUDA fallback for the accepted table and geometry.
