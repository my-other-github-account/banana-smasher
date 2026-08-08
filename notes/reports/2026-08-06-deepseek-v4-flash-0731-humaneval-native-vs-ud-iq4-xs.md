# DeepSeek-V4-Flash-0731 HumanEval: official release vs UD-IQ4_XS

Status: MEASURED — causal audit finds a native prompt-route mismatch; matched physical replay pending

Owner: TBD

## Scope

This report compares two exact DeepSeek-V4-Flash-0731 artifacts on the frozen `HUMANEVAL_0731_V1` code-correctness protocol. Both rows used exactly two NVIDIA DGX Spark nodes over QSFP, target-only decoding, no speculative decoding, and effective BF16 K-cache plus BF16 V-cache. The official row used TP2 through Ray/vLLM; the GGUF row used a true two-node llama.cpp RPC layer split with a 1:1 tensor split.

## Top-1 results

| Artifact | HumanEval pass@1 | HumanEval+ pass@1 | Empty/null | Canonical rows |
| --- | ---: | ---: | ---: | ---: |
| Official DeepSeek release | **154/164 (93.90%)** | **147/164 (89.63%)** | 0 | 164 |
| Unsloth `UD-IQ4_XS` | **159/164 (96.95%)** | **152/164 (92.68%)** | 4 | 164 |

## Causal-audit correction

The five-task direction in each aggregate is not an established major gap. Exact paired recount gives:

| Suite | Both pass | UD-IQ4_XS only | Official only | Both fail | Exact two-sided McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 151 | 8 | 3 | 2 | 0.2265625 |
| HumanEval+ | 142 | 10 | 5 | 7 | 0.3017578125 |

Both rows used byte-identical user messages for all 164 tasks, but they did **not** use the same rendered model prompt. The official vLLM 0.24.0 DeepSeek-V4 tokenizer route defaulted to chat mode and rendered every request as:

```text
<｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜></think>
```

The stock GGUF/llama.cpp route defaulted to thinking mode and rendered:

```text
<｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜><think>
```

This is not a vocabulary-conversion artifact. The official HF tokenizer and GGUF metadata have exactly the same 129,280 tokens by ID and the same 127,741 BPE merges. Re-tokenization matched both sealed prompt-token counts on 164/164 tasks, while rendered bytes and token IDs differed on 164/164: the terminal route token is 128822 (`</think>`) for official and 128821 (`<think>`) for GGUF.

The exact discordance audit also does not attribute the IQ4 direction to empties, truncation, syntax, or response-format failures: such route-failure classes net two tasks *against* IQ4 in both suites. The global reasoning-route mismatch remains the decisive instrument confound.

Plain answer: both published scores are real measurements of their named artifacts and native stacks. They do not establish a major reproducible quality gap, and they do not establish that IQ4 quantization caused the observed direction. Until native-route and prompt-normalized repeat replays are complete, the defensible classification is **prompt-render/tokenization mismatch**, not quantization causality.

## Basis

| Item | Official release | Unsloth UD-IQ4_XS |
| --- | --- | --- |
| Repository | `deepseek-ai/DeepSeek-V4-Flash-0731` | `unsloth/DeepSeek-V4-Flash-0731-GGUF` |
| Immutable revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | `fbbb5b93fb787c21338159b0af3318bb3f4d9768` |
| Weight recipe | Official block-FP8: e4m3 weights, ue8m0 128×128 scales; BF16 model dtype | Stock `UD-IQ4_XS` four-file GGUF |
| Artifact bytes | Index-declared weights: 166,878,536,440; runtime physical weights: 166,886,535,336 (155.425198 GiB) | 136,662,446,656 (127.276822 GiB) |
| Wire bpw | Not derived: this eval contract does not seal a whole-model element denominator | Not derived: this eval contract does not seal a whole-model element denominator |
| Runtime | vLLM 0.24.0; backend SHA-256 `95d15035a273595cfbb896010c2ed7e7290da8465ee901b2ef3c3c70b18fbb33` | llama.cpp `e3546c7948e3af463d0b401e6421d5a4c2faf565` plus RPC memset fix `d1549b0e1f22f84b655d7323f5ca36638bafd95c` |
| Banana Smasher source | `49a54dc5ac8a5c329307f2830cb950d0eb48c253` | `49a54dc5ac8a5c329307f2830cb950d0eb48c253` |

The official release is the native reference, but its weights are not unquantized BF16 weights: its configuration resolves the published block-FP8 recipe above. BF16 applies to the runtime model dtype and, critically for this comparison, both K and V cache dtypes.

### Official release identity

- `model.safetensors.index.json` SHA-256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- `config.json` SHA-256: `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 48 weight shards, bound by the immutable revision and index above

### UD-IQ4_XS file identity

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4-Flash-0731-UD-IQ4_XS-00001-of-00004.gguf` | 5,257,664 | `b68fd7fa0579916d7b91bacdbdabece42789b4716be7131f32f59b58642c9472` |
| `DeepSeek-V4-Flash-0731-UD-IQ4_XS-00002-of-00004.gguf` | 49,431,060,672 | `d8a00e31a38313ca122687a8d0bcb26da744b27d5aaf966c6a7256cae31fb03c` |
| `DeepSeek-V4-Flash-0731-UD-IQ4_XS-00003-of-00004.gguf` | 49,605,340,544 | `86b27d95a0e8b16be7cc2ad60e3840c3523b4061b8f92b940522e3cfa523059d` |
| `DeepSeek-V4-Flash-0731-UD-IQ4_XS-00004-of-00004.gguf` | 37,620,787,776 | `24e6c819db7df257c93acbbd0e1aec6c03fd9f04036b92be065bfbe5539ca9fc` |

## Method

The frozen protocol used EvalPlus `0.4.0.dev44` at commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, HumanEvalPlus `v0.1.10` dataset hash `fe585eb4df8c88d844eeb463ea4d0302`, and exactly 164 unique tasks. Each request used temperature `0`, top-p `0.95`, N=1, one user message, and provider-enforced `max_completion_tokens=4096`. Only `message.content` was scored. Semantic empty/null responses were retained as failures; only transport failures without a canonical row were retried.

Both canonical JSONL files were scored in the same frozen container (`sha256:c341bd2c45cfd56885b132ed403c80e19b002ffceadb3d40e059eccb05a92543`) with networking disabled and a read-only root filesystem.

## BF16 KV and target-only proof

- Official release: effective argv included `--dtype bfloat16 --kv-cache-dtype bfloat16`; engine introspection resolved `kv_cache_dtype=bfloat16` / `torch.bfloat16` on both TP ranks. The terminal runtime reserved 2,147,483,648 KV bytes per rank with `max_num_seqs=1`. The runtime configuration recorded `speculative_config=None` and target-only decoding.
- UD-IQ4_XS: effective argv included `--cache-type-k bf16 --cache-type-v bf16`; runtime cache metadata recorded both cache types as `bf16`. The server used a two-node RPC layer split with `--tensor-split 1,1`, and the runtime recorded target-only decoding with speculation disabled.

## Evidence

| Evidence | SHA-256 |
| --- | --- |
| Official score seal | `90a7b23dfb42809523f959ed66f996ecd6a9889820423265b18fee8a0a558ba3` |
| Official canonical JSONL | `f5ad9013abe6013891f745969a2e9bf7ce66fccf03cc8e7ea950f3b2f519b541` |
| Official result JSON | `fa4e4cf3deb74d50bb95e4c143a28519292b49b59b8dcf6b2e38b05865ae77e6` |
| Official generation handoff | `e2eab4821de678461555acc1c8d97b3f67a0a93d20533fecf29c2ee3b4257ee2` |
| Official runtime identity | `eead902fb5836c5ce7fb0a1168ee1aa839964f277d75d64d6b380daff3387dbc` |
| Official terminal runtime instance | `2d6005f8ed29156e237f01a023ebd617098ee69412ae29140a129f77dca36d76` |
| UD-IQ4_XS score seal | `701149a113c5705c1c27a6adf44a901ee41a6a076b72a2ba4050f16956397864` |
| UD-IQ4_XS canonical JSONL | `46f499f8ff03ca9bf944269c76c222b52f05c61db469d1d30955e71dbcb48fcd` |
| UD-IQ4_XS result JSON | `c85670d4999ad8c5eb87abb4a78884d639427e6268f3051c08aa0dae1bdd05d3` |
| UD-IQ4_XS generation handoff | `6fee67e9f80b253c0ede0bbf91223738556ae4bbb159c52e84e87cd64e9b2da6` |
| UD-IQ4_XS runtime identity | `d0e8547ba3565187436dbde1b86dffbd5b29a6f2d33dfbac0d231607b5bf8bbc` |
| UD-IQ4_XS four-file identity manifest | `3df9c62e534bc03a4d05ef303dd2f14e0d83bb0c530f2cc7ee974dd3bbd0ef07` |
| Stage-A causal-audit public summary | `2b9d9a0c28dabf8953755e21ff5faae670fbc8cda9b273c85843064b12b9dbf8` |

## Limitations

These are one generation per task under two different native runtime/render routes. Exact paired uncertainty is reported above and does not reject equal discordance probabilities at conventional thresholds. The causal audit proves that the native routes differ at the final reasoning-mode token on all 164 prompts; it does not yet measure what happens when both stacks receive a normalized rendered prompt. A stable repeated discordant/control replay under native and normalized routes is required before considering a full aggregate rerun. Even a reproduced normalized-route direction would remain a stack-level result unless a common runtime/token path isolates weight representation.
