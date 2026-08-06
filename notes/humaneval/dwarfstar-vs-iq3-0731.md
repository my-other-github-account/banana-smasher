# DwarfStar vs IQ3 — standardized HumanEval 0731

Both rows were scored sequentially on the same physical scorer and immutable container image. Results apply only to the exact artifacts and target-only serving stacks identified below; they do not establish quantization causality.

| Route | HumanEval pass@1 | HumanEval+ pass@1 | Semantic empty/null failures | Canonical JSONL SHA-256 |
|---|---:|---:|---:|---|
| DwarfStar 0731 target-only | 93.90% (154/164) | 87.80% (144/164) | 0/0 | `7432000317d8d3f78301554d7b769b7a5faa5a303bcc401970e6cccbad3c3c8a` |
| IQ3 0731 target-only | 94.51% (155/164) | 90.24% (148/164) | 8/0 | `42ba91dfbf73898f82e043a6cf511f85ef2e6bf9f5211bee7bcf2c9a9068d92a` |

Semantic empty/null outputs are counted as failures.

## Reproducibility identities

- Source revision: `49a54dc5ac8a5c329307f2830cb950d0eb48c253`
- Suite-lock SHA-256: `ea5037f5f77b8208eeb3ac8b8a0d41138d96dbf5790cba87243e8b584f233308`
- HumanEvalPlus dataset hash: `fe585eb4df8c88d844eeb463ea4d0302`
- Scorer image ID: `sha256:c341bd2c45cfd56885b132ed403c80e19b002ffceadb3d40e059eccb05a92543`
- Scorer image digest(s): none recorded
- Instrument: network disabled, read-only root, all capabilities dropped, no-new-privileges, 8 GiB memory, PID limit 512, eight EvalPlus workers.
- Decoding route: target-only for both rows.

### DwarfStar 0731 target-only

- Generation handoff SHA-256: `28396a7115577c653745cfbe41a0560f0859227f7604b4dc026e3bf844f3c76a`
- Canonical JSONL SHA-256: `7432000317d8d3f78301554d7b769b7a5faa5a303bcc401970e6cccbad3c3c8a`
- Raw EvalPlus result SHA-256: `fb5ba180494c21a0eb2862b26d0a71b0f7c78c14cf56e10b9341c30b3066407e`
- Artifact `base.sha256` SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Artifact `drafter_identity_only.sha256` SHA-256: `8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686`
- Artifact `engine.sha256` SHA-256: `46110fcc47d59e387c040555bb1edd8e83e8bef4786f89726caeb5842e52565a`

### IQ3 0731 target-only

- Generation handoff SHA-256: `57cb976b1a9b60593df9f799ee98139a77b0f123d2711da833d33b42fa9c9751`
- Canonical JSONL SHA-256: `42ba91dfbf73898f82e043a6cf511f85ef2e6bf9f5211bee7bcf2c9a9068d92a`
- Raw EvalPlus result SHA-256: `fabb3bdc808abe83bc18f45e2168c877ebc838d7f7cb4f540a62c94f539bc482`
- Artifact `model_index_sha256` SHA-256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Artifact `quant_artifact.artifact_preflight_receipt_sha256` SHA-256: `4171beecc3d69b0eced34c3a261b3780fbde53dfcc39cc3e6dac36bfd610763d`
- Artifact `runtime.server_launch_receipt_sha256` SHA-256: `050c5fa089496d5b97a12360bf48ab3c96384f6df838d76510f806198507b6d9`
- Artifact `runtime.server_ready_receipt_sha256` SHA-256: `373ff81ceb40b804d9bc4b207a9ee65989bc116f2aadf49b1ed68e2dfee14bdb`
