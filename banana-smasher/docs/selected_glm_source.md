# Selected GLM FP8 source localization

A model-root `SELECTED_TENSORS.json` explicitly opts the canonical GLM source
adapter into local selected-range payloads. The original config and complete
model.safetensors.index.json remain byte-for-byte intact. No sparse or slim
safetensors shard is invented. Missing, uncovered or corrupt selected reads
raise; the adapter does not fall back to full shards.

Schema `banana-smasher.selected-tensor-source.v1` contains:

- `index_sha256`, `config_sha256`: actual original file digests.
- `descriptor_path`, `descriptor_sha256`: root-confined original owner descriptor;
  `source_basis` must equal the index digest. Its unique `rows` carry `key`,
  original `parent`, `parent_bytes`, `header_sha256` (JSON header bytes excluding
  length prefix), absolute `offset`, `bytes`, `dtype`, `shape`, `data_sha256`.
- `parents`: original shard basenames mapped to `header_path`, `header_sha256`,
  `bytes`. The header file contains the original eight-byte length prefix and
  complete unmodified header JSON, not a reserialized subset.
- `payloads`: exact tensor keys mapped to root-confined local raw payload paths.

The public batch launch requires every config to set
`selected_source_manifest_sha256` to the actual sidecar digest, in addition to
the existing model-index and import-closure gates. Pin the original owner
handoff/descriptor in the experiment admission. The loader cannot establish
external provenance from a caller-authored manifest alone.

The existing canonical FP8 dequantization and blockwise 128x128 inverse-scale
operation are reused; this is not a second encoder or decoder. The bounded
selected source mode supports F8_E4M3 weights and F32 scales, including scales
in other original shards. It validates original header offsets, dtype/shape,
original shard length, exact payload length/digest and confinement at use time.
Source receipts label selected coverage and payload digests separately; they
never claim a complete parent shard hash from a range read.

A fixture test first failed with absent parent shards and now compares actual
FP8 down/fused results against the full-source path. Corruption, missing ranges,
header/index/config/offset mutation, path escape and missing manifest pins fail
closed. Physical source localization is separate from build speed/held-out
quality measurement; passing these tests does not promote an accelerator.
