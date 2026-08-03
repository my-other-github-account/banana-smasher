# Report title

Status: NEW | pending | measured | rejected

Owner: TBD

## Scope

State the command, runtime mode, hardware class, and question being tested.

## Basis

| Item | Identity |
| --- | --- |
| Model/artifact basis | SHA-256 or immutable release identifier |
| Banana Smasher revision | Git commit |
| Runtime dependency | Stock-vLLM version or commit |
| Configuration | Scrubbed config hash |

## Method

Provide the exact public command and enough portable configuration to reproduce the run. Describe warmup, sample count, and acceptance threshold. Do not include private paths or credentials.

## Results

Use one row per same-work comparison. `FP` is bits per floating-point element; write FP8 as `8`. `GB` and packed bpw must describe physical storage, not marketing labels.

| Status | Method | Exact basis | KLD | Top-1 | GB | Packed bpw | FP | Instrument | n | Source | Owner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| NEW | pending | pending | pending | pending | pending | pending | 8 | pending | pending | pending | TBD |

## Evidence

List concise receipt names and SHA-256 values. Raw private receipts stay outside Git.

## Limitations and pending gates

Name unrun hardware tests, unsupported environments, and any reason a comparison is not valid.
