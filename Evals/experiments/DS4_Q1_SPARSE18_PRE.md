# DS4 Q1 sparse 18-cell PRE — not uniform-Q1 evaluation

The retained experiment replaces 18 original Q1 cells (nine expert pairs in
layers 0, 1, 2, 15, 16, 17, 29, 30, 31); all other weights remain on the native
BF16 teacher path. No repair or calibration is performed.

Frozen Balanced64 measurement: 64 windows, 2048-token forward context, first
1024 positions per window, identical ordered teacher top-8192 support.

- KL(teacher || candidate): **0.009883195606578363**
- Common-support Top-1: **64148 / 65536 = 97.882080078125%**
- Measurement validity: verified.
- Numerical quality acceptance: not adjudicated; no pre-existing Q1 threshold.
- Uniform-Q1 quality: **not measured by this sparse intervention**.

[Machine-readable measurements](ds4-q1-sparse18-pre.json) preserve the original
64 per-window and six class values. These are the existing measurements from
archive SHA256 `0c066f41b95e492800ad766771043bc725d9fee31ecd09e0b849ef634b0f996d`;
publication/controller-format conversion did not rerun the forward. The
original runtime commit is `6bf59df320a76f06bf6f58adddc0dcf9b759a31a`.

Both distributions were normalized in binary64 on the common frozen support,
with no KL clamping. The original reduction used ordered `math.fsum` over all
65536 per-position values; equal-length per-window means reaggregate within
binary64 tolerance. Top-1 is deterministic first-index argmax on common support,
not the candidate's full-vocabulary argmax.

This result is deliberately outside the uniform model comparison table.
Sparse intervention cannot predict uniform-Q1 quality, nor isolate historical
runtime drift without a paired fresh native baseline. It does not authorize
quality GREEN by comparison with a Q3 historical score. Retained FP16 logprobs
and BF16 execution constrain numerical precision. The teacher index/config
were bound, but native checkpoint shard hashes were not independently rederived.
The initial BF16 telemetry failure is retained in the source archive and is
not counted as a completed measurement.

## Production recovery observation

The stopped full producers reached the canonical exact-memory preflight and
refused below the 4-GiB reserve. Recovery used the unchanged existing recipe,
retained captures/TLUT/seeds and exact K1 kernel from canonical main
`06eccc69c2686676fa601f4cae1602ef89633127`, with fresh bounded per-cell processes.
The stale runner's memory-contract monkeypatch was not propagated. Two missing
cells per shard completed and passed runtime packed-decode FP16 conformance;
no already sealed cell was recomputed. All 5665 previous receipts and artifacts
were independently hash-read back unchanged. Counts advanced 2038→2040,
1325→1327 and 2302→2304. This is a bounded recovery result, not an all-22016-cell
completion or an end-to-end quality measurement of the new cells.
