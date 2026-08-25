# Teacher Capture V2 — exact lineage and frame proof

Status date: 2026-08-25 UTC  
Task: `t_6048c45b`  
Canonical implementation pin: `8747d9a9920a2386cb15e533d4592559dbfdb7da`

## Result

Teacher Capture V2 uses the production Balanced64 frame: candidate token ids from `windows_ds4_eval.json` and teacher rows captured from the same eval corpus. The published PRE checkpoint reproduces its historical W28 fixture against the V2 row:

- PRE checkpoint: `f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70`
- V2 W28 teacher row: `812b408e414ddfb1f3fd2c3c94d3982565fe6f02034e147d08f1e797f0ec9aef`
- Eval corpus: `5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd`
- measured PRE W28: KLD `0.14050351244454912`, top-1 `887/1024`
- historical singleton fixture: KLD `0.14062129470098408`, top-1 `884/1024`
- KLD delta: `-0.00011778225643496` (`-0.08376%`)

The latest U56 student in that same frame scores W28 KLD `0.13404867801690837`, top-1 `884/1024`. That singleton initially appeared to end the `+2.7%` question, but the subsequently sealed PRE Full64 V2 aggregate is KLD `0.23567034601287126`, or `+2.819875%` versus the V1-bank `0.22920699467439512` target. Therefore W28 was not representative of Full64 teacher-bank replacement. The earlier `0.4476229084034261` result used TRAIN corpus-local window 28, which is a different token sequence and therefore a different experiment.

## Frame matrix: ids corpus, teacher corpus, checkpoint

| Measurement | Candidate ids corpus | Teacher corpus / row | Checkpoint | Receipt-backed result |
|---|---|---|---|---|
| Published Evals row | eval SHA `5aadaacb…` | V1 eval bank; W28 SHA `56175348…` | published PRE `f9bffe04…` | KLD `0.229392`, top-1 `56,533` |
| B1 raw-U0 reproduction | eval SHA `5aadaacb…` | V1 eval bank; W28 SHA `56175348…` | raw U0 `7978d100…` | KLD `0.2292069946743951`, top-1 `56,534` |
| Four-host PRE fan-out | eval SHA `5aadaacb…` | V1 eval bank; W28 SHA `56175348…` | published PRE `f9bffe04…` | KLD `0.22920699467439512`, top-1 `56,534` |
| V1 W28 singleton fixture | eval SHA `5aadaacb…` | V1 eval W28 SHA `56175348…` | raw U0 `7978d100…` | KLD `0.14062129470098408`, top-1 `884` |
| V2 aligned, PRE | eval SHA `5aadaacb…` | V2 eval W28 SHA `812b408e…` | published PRE `f9bffe04…` | KLD `0.14050351244454912`, top-1 `887` |
| V2 aligned, latest student | eval SHA `5aadaacb…` | V2 eval W28 SHA `812b408e…` | U56 `589031d3…` | KLD `0.13404867801690837`, top-1 `884` |
| V2 TRAIN diagnostic | TRAIN SHA `16575db7…` | V2 TRAIN W28 SHA `8217bdb9…` | U56 `589031d3…` | KLD `0.4476229084034261`, top-1 `857` |

Operational proof for the published/B1/fan-out frame is the sealed launcher from task `t_896a422b`: `production_shard_launcher.py` lines 27–28 hash-gate eval corpus SHA `5aadaacb…`; line 49 passes that same eval corpus to the candidate builder, alongside the V1 eval teacher bank and PRE checkpoint. The phrase “score corpus: TRAIN; builder corpus: eval” in `BALANCED64_PRE_REPRO.md` describes campaign/training lineage, not the actual candidate ids supplied to the sealed production scorer. The executable receipt is authoritative.

TRAIN window 28 and eval window 28 are corpus-local ordinals. Their first token ids are `582` and `81944`, respectively; the sequences are unequal. Consequently the TRAIN diagnostic is not numerically comparable to the eval/eval W28 fixture.

## FP8 teacher source

- Model: DeepSeek-V4-Flash-0731, model's own FP8 base
- Payload path on capture host: `/home/dnola/models/hf/DeepSeek-V4-Flash-0731`
- `model.safetensors.index.json` SHA256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- `config.json` SHA256: `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `swiglu_limit`: `10.0`
- Positive payload header coverage: `72,317/72,317` mapped keys on Spark-3
- Spark-1 warning: its same-named base path was a stale partial payload (`18,557/72,317` keys) and was quarantined from capture.

## Corpus and window geometry

Production corpus:

- path: `/home/dnola/missions/P1T_0731_TABLE_t_7c53c08a_s3/inputs/windows_ds4_eval.json`
- SHA256: `5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd`
- sequence length: 2,048 tokens
- scored positions: first 1,024
- teacher support: top 8,192 full-softmax log probabilities
- microbatch: 1 for V2 capture
- W28 id: `28`
- Balanced64 ids: `28,56,68,71,76,99,107,122,124,130,141,156,160,171,180,183,185,186,196,210,212,213,218,228,232,235,249,270,272,273,283,288,290,295,297,306,307,309,311,328,331,357,362,365,368,374,376,380,384,385,391,396,413,429,430,437,442,447,454,462,464,475,489,499`

Rejected diagnostic corpus:

- path: `/home/dnola/missions/DS4_TEACHER/static/windows_ds4_TRAIN.json`
- SHA256: `16575db7fd180ca193aa13c4e642400b9ed416dbd0c36c3c5302422b31f5cbae`

## Forward, attention, mask, and dtype map

Teacher builder:

- repository path: `ds4-flash-kldmatrix/repair_api/assets/t8192_w28_sdpa_teacher_builder.py`
- SHA256: `6f6d61c1151d92bb8047a6623ae6b2a55a6a18e0f37dc069659f812c3f554b7c`
- repair_api git pin: `8747d9a9920a2386cb15e533d4592559dbfdb7da`

Runtime:

- Python: `/home/dnola/humming_env/bin/python3`
- PyTorch: `2.11.0+cu130`
- Transformers: `5.12.1`
- Transformers release tag: `v5.12.1`; annotated tag object `a030302dcd4777bbf042ee46c30c5dbe6d2a2eb2`, dereferenced git commit `ddb849abe009d1089e6c691bfc897f27211c663c`
- installed distribution `RECORD` SHA256: `1b27df1d2749df60ec5d416ca0d6eab09bc9dada674bae6e57a4d6059c67d93d`
- modeling file: `/home/dnola/humming_env/lib/python3.12/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`
- modeling file SHA256: `3be3c5211507ddd1b37ac9dbb27f47533c39d7922779c124517e1d3a7a9c4253`
- capture host: `spark-3`
- attention selector: `sink-sdpa`, registered as `official_k2_sink_corrected_sdpa`
- adapter source: `repair_api/modern_green_resident.py::_sink_corrected_sdpa_forward`
- mask path: `transformers.masking_utils.create_sliding_window_causal_mask`; config is temporarily set to eager while constructing the additive sliding-window mask, then restored to `official_k2_sink_corrected_sdpa` for each layer forward

A one-variable eval-corpus eager capture produced the exact same serialized row SHA `812b408e…` as sink-SDPA, proving equivalence for W28 under this pin.

Dtypes:

- bf16: attention projection weights, compressor/indexer linear weights, gate weight, shared and routed expert weights, embedding, LM head, RMSNorm weights
- fp32: learned attention sinks, compressor/indexer position bias, hyper-connection tensors, expert score correction bias
- int64: `tid2eid`
- output `idx`: int32 `[T, 8192]`
- output `logprob`: fp16 `[T, 8192]`

## Commands

W28 production-frame capture:

```bash
PYTHONPATH=/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/code/ds4-flash-kldmatrix \
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
/home/dnola/humming_env/bin/python3 \
  /home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/code/ds4-flash-kldmatrix/repair_api/assets/t8192_w28_sdpa_teacher_builder.py \
  --mode bf16 \
  --meta-dir /home/dnola/models/hf/DeepSeek-V4-Flash-0731 \
  --local-dir /home/dnola/models/hf/DeepSeek-V4-Flash-0731 \
  --corpus /home/dnola/missions/P1T_0731_TABLE_t_7c53c08a_s3/inputs/windows_ds4_eval.json \
  --out /home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/teacher/CANDIDATE_EVAL_CORPUS_SINK_SDPA_W28 \
  --windows 28 --mb 1 --chunk 1 \
  --attention-implementation sink-sdpa \
  --tag causal-eval-corpus-sink-sdpa-pin-8747d9a
```

Full64 uses the same command and identity with `--windows <Balanced64 csv>`, output root `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/teacher/V2_FULL64`, and chunking recorded in per-launch receipts.

## Output identities and durable receipts

W28:

- production-frame row: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/teacher/CANDIDATE_EVAL_CORPUS_SINK_SDPA_W28/t8192_win28.pt`
- row SHA256: `812b408e414ddfb1f3fd2c3c94d3982565fe6f02034e147d08f1e797f0ec9aef`
- launch receipt SHA256: `df9e09d656aef46e8ea270b2824b97e188c4aa4f06bad8f7c328d64a8e886eca`
- result receipt SHA256: `aa4fc444083dcb42909f7a07c6dd77472c9ddec6f351e3b7a2f66c26f9813aa5`

Frame proof and student score:

- PRE W28 receipt: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-1/receipts/PRE_W28_V2_EVAL_ALIGNED.rank0.json`
- latest-student aligned receipt: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-1/receipts/STUDENT_W28_V2_EVAL_ALIGNED.rank0.json`
- TRAIN diagnostic receipt: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-1/receipts/STUDENT_W28_V2_TRAIN.rank0.json`
- immutable V1-vs-V2 static diff: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/receipts/STATIC_DIFF.V1_V2.W28.json`, SHA256 `a5bfa607d732d33d0179b2bd4e159b9129b29d0ff325b7ba3975470185e152ca`

Full64 bank:

- durable root: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/teacher/V2_FULL64`
- row count: `64`; DONE count: `64`
- manifest: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/receipts/TEACHER_V2_FULL64_MANIFEST.json`
- manifest SHA256: `69751b0353709bbf6e560af802be802d66e0d7f6f6f75d456e0a10afabfe3c98`
- canonical ordered `(window_id,row_sha256)` list SHA256: `3b0350dcecd02c1d969dae9c8385ddcc110cb655ce23d9e39dbf61816d1469d9`
- `DONE.jsonl` SHA256: `9683b34e6cb4eccd8223f3864333bbb5677b1db92fb2ff9f9d10e1eee4c2ab7b`
- initial 16-row launch receipt SHA256: `38975f8f4a0d6c95bca9dfb3e15321dc083ce3e75020b34c1439578af422fa87`
- remaining 48-row launch receipt SHA256: `d29f53ddcc72a21a3c8eaf7228e1002f2afe56396e09f765f35a31e44d780e4b`

The manifest contains the full per-window path, byte size, and SHA256 table. It is the canonical row-level lineage artifact; the compact ordered-list digest above binds the entire bank without duplicating that 64-row table here.

The PRE Full64 aggregate was measured through the unchanged public `ResidentRepairAPI.validate` path:

- result: KLD `0.23567034601287126`, top-1 `56,498/65,536`
- timed wall: `885.7171241130002` seconds
- sealed V1-bank reproduction target: KLD `0.22920699467439512`, top-1 `56,534/65,536`
- delta: `+0.006463351338476142` (`+2.819875260638449%`)
- rank-0 receipt: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-1/receipts/PRE_FULL64_V2_PUBLIC_API.rank0.json`, SHA256 `d12ccd116a60152b096bf6b7a085acf227e3a9ac72e1fcd7fc392b7a3da054c6`
- rank-1 receipt: `/home/dnola/missions/TEACHER_CAPTURE_V2_t_6048c45b_spark-3/receipts/PRE_FULL64_V2_PUBLIC_API.rank1.json`, SHA256 `6e645062b611817b6f7eaf9ba12807f5cf2701b588f4468cbdfde1fe92ac7053`
- coverage: 64 windows, 65,536 positions; both ranks emitted identical aggregate metrics
- mechanism gate: zero fallback calls, zero CPU-relay bytes, zero reconstruction calls, zero timed score/model reads

This result does **not** reproduce the V1-bank `0.229207` target. It shows that W28 agreement alone did not establish Full64 teacher-target invariance: replacing the V1 eval teacher bank with the recaptured V2 eval teacher bank changes the aggregate by `+2.819875%`. The result is a quality RED against the reproduction gate and must not be relabeled as an accepted reproduction.
