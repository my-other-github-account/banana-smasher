# Pre-V4 mixed-QTIP container lineage reconstruction

Date: 2026-08-04

## Scope correction

The historical “V3 Golden Container” is not the later P1321/P1423 learned-VQ/IQ3 serving line and is not an unrelated image recently tagged `v3`. It belongs to the earlier mixed-QTIP reproducibility campaign.

The retained receipts do not use literal `V1`, `V2`, and `V3` tags consistently. The defensible operational mapping is therefore based on chronology and capability transitions:

| Reconstructed version | Campaign artifact | Image identity | Outcome |
|---|---|---|---|
| V1 | P602 reproducible mixed-tier container | `sha256:f4a366150a6e5a01c978ba6f6654c4f427777f14b97c1a1c704292320266fbb7` | Reproducible four-family systems instrument. Approximately 17.0–17.2 decode tok/s and 1.1K–2.2K prefill tok/s were reproduced across cold boots. It used QTIP but was a synthetic performance instrument, not the trained F521 product and not a quality oracle. |
| V2 | P1191 actual Wire-C image | `sha256:8f955adf64714cb55b0bb14864e184b817a28d91cf883ef8c3ddc98f4ee81be2` | First statically complete actual-Wire image. It included the mixed packed runtime, QTIP extensions, warmed cache, and read-only model/plane mounts. Boot failed after all 46 model shards loaded with `KeyError: 'E'` because the child Engine interpreter did not install the runtime seam. No valid performance claim. |
| V3 | P1235 corrected actual Wire-C golden image | `sha256:2908b791d7cbc435fa8437a751f172ecd798683860929c76345b5c92719d0c02` | Corrected the child-interpreter import seam with a baked `.pth` hook. Reached HTTP ready, coherent generation, zero swap, and stable measured C1/prefill performance on the exact mixed pack. |

This mapping is high confidence for the functional lineage, but the surviving artifacts do not provide a literal version-label receipt that says “P602=V1, P1191=V2, P1235=V3.”

## V3 artifact identity

The P1235 golden run binds:

- image: `sha256:2908b791d7cbc435fa8437a751f172ecd798683860929c76345b5c92719d0c02`
- pack manifest: `3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220`
- planes manifest: `b524c5a67bbcad6aef14d70b464b46097302bf004bb75c1265f2ff683bae083d`
- active overlay: `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62`
- packed-runtime source tree: `0265cfde5ad4552d1879ce536f553aa4072e26992726433fb9981987ff5c5b13`
- performance receipt: `59ca2a800e559a164e3413099f1862a9aaea1c392381463825e3ac49e780f91b`

The sealed pack has 22,016 projection rows:

| Family | Rows |
|---|---:|
| QTIP3 | 14,979 |
| QTIP2 | 2,266 |
| D4 | 4,717 |
| native MXFP4 | 54 |
| **Total** | **22,016** |

D4 is further split into 197 K1024 rows, 2,610 K2048 rows, and 1,910 K4096 rows.

## QTIP execution evidence and limitation

The recovered runtime is genuinely QTIP-aware rather than an IQ3/W2/W3 substitute:

- `p1016_packed_kernels.py` implements direct QTIP2/QTIP3 blocked-trellis decode, including separate rate-2 and rate-3 offset maps.
- The C1 path applies QTIP SU pre-scaling, FWHT, packed raw trellis GEMV, Wscale, post-FWHT, SV, and family-select finalization.
- Dynamic dispatch reads the selected expert's family on device and executes QTIP2, QTIP3, D4, or native-MXFP4 from expert-specific pointer tables without dense expert materialization.
- Boot receipts armed QTIP dynamic widths 2048 and 4096 for both projections across all 43 layers.
- The image contained the exact QTIP dynamic and packed-kernel closure and loaded the QTIP-dominant pack before the measured decode requests.

The historical run predates the current physical-counter contract. It does not report request-level `qtip2_kernel_launches` and `qtip3_kernel_launches` separately. Its `P1016_QTIP_CUDA=0` setting disabled an older optional compiled-extension branch; the retained P1016 Triton mixed-exact kernel still contains and dispatches direct QTIP trellis decode. Treat the source, armed boot, pack identity, and performance as a strong implementation oracle, but do not substitute them for a fresh current per-family counter seal.

## Measured V3 result

All rows came from one corrected container boot:

- decode256: 10 serial rows
- minimum decode: 15.878221 tok/s
- mean decode: 15.888635 tok/s
- decode CV: 0.000555
- exact-2048 prefill: 872.158635 tok/s
- semantic coherence: pass
- temperature-0 logit check: pass
- peak residency: 76,414,812,160 bytes
- swap: zero
- model and planes: read-only mounts
- model bytes in image: none

V3 therefore clears the current C1 throughput target and supplies a useful exact-2K prefill reference. It does not supply a valid C2/C4/C8/C16 ladder because its accepted run used max-sequences one.

## U004 relationship

The V3 expert pack has the same F521 mixed family census and active overlay used by the later U004 product basis. However, the P1235 container and performance receipts explicitly identify the P943 **pre-repair** lineage and do not bind the `UPDATE_004` checkpoint.

Consequences:

1. V3 is a valid runtime and performance oracle for the mixed QTIP expert machinery.
2. V3 is not evidence that repaired U004 itself was served.
3. V4 remains the U004 product on this mixed basis, and V5 remains U012 on the same basis.
4. Current V5 acceptance must bind the exact U012 repair state and produce fresh QTIP2/QTIP3 counters.

## Reusable implementation oracle

The highest-value reviewed mechanisms to port are:

1. The baked every-interpreter runtime-install seam that fixed P1191's `KeyError: 'E'` failure.
2. Device-resident family and pointer tables with graph-safe dynamic expert selection.
3. Direct QTIP2/QTIP3 trellis decode with the complete SU/FWHT/Wscale/SV transform sequence.
4. The specialized six-routed-row C1 path and caller-owned workspaces.
5. Lossless boot consolidation with exact index/codebook/scale validation.
6. Read-only external model and packed-plane mounts with no model payload baked into the image.
7. Warmed runtime/cache behavior sufficient for stable 15.89 tok/s C1.

The historical dense-all prefill path is a performance reference, not automatic modern acceptance: its logs include transient per-family dequantization during long prefill. The clean V5 product must satisfy the current no-fallback and physical-counter contract.

Historical images, compiled extensions, cubins, caches, and mission trees remain comparison oracles only. The deliverable must regenerate required native assets from reviewed canonical source and serve the exact V5/U012 artifact through the ordinary stock-vLLM plugin path.
