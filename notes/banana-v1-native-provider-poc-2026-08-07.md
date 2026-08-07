# Banana V1 native provider PoC — 2026-08-07

## Decision

The Banana-native scalar V1 path is now implemented as a standalone provider and builder. It contains no imports or references to another codec implementation.

The compact default remains:

```text
level = (((state * 48917 + 50631) & 0xFFFF) >> 6)
value = gaussian_quantile_lut[level]
```

`48917` is odd, so the affine map permutes all 65,536 states before balanced grouping into 1,024 levels. The raw FP16 codebook is exactly 2,048 bytes.

## Implemented path

- fixed scalar `L16/B2/V1` geometry at exactly 2 code bits per weight;
- deterministic 1,024-value FP16 Gaussian-quantile codebook;
- balanced affine state labeling;
- exact two-pass cyclic/tail-biting Viterbi;
- global scale search over nine bounded factors;
- deterministic two-sided randomized normalized Hadamard transform;
- reverse 16-column LDLQ feedback interface;
- exact 2-bpw circular packing, unpacking, and closure checks;
- NumPy correctness decoder and Torch on-device packed decoder;
- build, write, verify, materialize, predict, and generic Backpack pricing seams;
- raw `codebook.fp16` activation artifact so its wire charge is exactly 2,048 bytes rather than a container-file size.

## Real E104 product-path smoke

Exact source: FF0731, layer 40, expert 104. The smoke used the first physical 16×16 source-weight slice of each projection, the full native transform/build/pack/materialize/predict path, and a zero LDLQ lower matrix because no routed activation Hessian was staged locally.

| Projection | MSE | Relative MSE | Scale factor | Scale-search gain | Build time |
|---|---:|---:|---:|---:|---:|
| fused13 | 0.000042072213 | 0.06633348 | 1.00 | 0.000% | 1.045 s |
| down | 0.000044406175 | 0.06676931 | 1.05 | 0.261% | 1.025 s |

For each 256-weight slice:

- trellis codes: 64 bytes = exactly 2.0 bpw;
- per-cell codes + FP32 scale + FP16 transform signs: 132 bytes;
- shared raw FP16 codebook: 2,048 bytes;
- tiny-smoke complete wire: 2,180 bytes, with the shared codebook intentionally unamortized;
- artifact verification: PASS;
- provider verification: PASS;
- materialized prediction equals build reconstruction: PASS.

## Verification

```text
PYTHONPATH=src python3.13 -m pytest \
  tests/test_banana_v1.py tests/test_backpack_family_providers.py -q
```

Result: `12 passed`.

Receipt: `notes/receipts/BANANA_V1_E104_NATIVE_POC.json`.

## Boundary

This closes the portable provider/build/pack/decode PoC. It does not yet claim routed-Hessian improvement, model-level KLD/Update12 quality, or fused server-kernel throughput. Those require the real routed E104 gate and hardware runtime measurement.
