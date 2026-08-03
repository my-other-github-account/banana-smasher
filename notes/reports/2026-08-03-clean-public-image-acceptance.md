# Clean public image acceptance — 2026-08-03

## Scope

This report records a clean-box build and inspection of the public Banana Smasher
Docker image. It covers source identity, immutable dependency pins, wheel construction,
image verification, package inventory, and the documented build/serve interface.
It does **not** claim a model boot, model-pack execution, allocation execution, or
performance acceptance.

## Source and build context

- Canonical repository: `https://github.com/my-other-github-account/banana-smasher`
- Source commit: `7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad`
- Source tree: `2eab417e581c27324f5097187095f52423d5eee2`
- Source bundle SHA-256: `02b996d94478369d8dec21c1e9e2196d3c6d0634b20a3b50647eb5412275493c`
- Source checkout status: clean, detached at the commit above
- Build host role: allocated Spark-3 clean-box lane (`linux/arm64`, SM121)
- Build context gate: `.dockerignore` excludes `.git`, `.worktrees`, `notes`, caches,
  local build outputs, credentials, receipts, and model artifacts.
- Accepted command:

```console
docker buildx build --no-cache --load --progress=plain --platform linux/arm64 -f docker/Dockerfile -t banana-smasher-public:t_644dd18a .
```

No registry push was performed.

## Immutable upstream sources

| Component | Public source identity |
| --- | --- |
| Base runtime | `vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b` |
| vLLM source revision | `ee0da84a` (image revision label `ee0da84ab9e04ac7610e28580af62c365e898389`) |
| DeepGEMM | `https://github.com/deepseek-ai/DeepGEMM.git`, `refs/tags/nv_dev_f8e8fb5`, commit `f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca`, package `2.6.1` |
| FlashInfer | `https://github.com/flashinfer-ai/flashinfer.git`, commit `d020372b068f335e2fe427372e134977a2235c49` |
| FlashInfer SM120 changes | `b34f49255f1640542da91665f58558a3e5e308f1`, `76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9`, `0c5fda59bb6fa71eae875693a024bb0fb37ba7d6` |

Banana Smasher, its plugin, DeepGEMM, and FlashInfer were built as wheels from the
pinned public checkouts. No private wheel or internal source tree was used.

## Accepted result

- Build status: **PASS**
- No-cache build log: 2,518,609 bytes; SHA-256
  `c7ee76abae569647d6bc6e4dc0505c98ab3c7da9d3dfb5b2c0b2abd8ea67b9e9`
- In-build package tests: **88 passed, 8 skipped**
- Local image tag: `banana-smasher-public:t_644dd18a`
- Local OCI image configuration digest:
  `sha256:b3e68602acad0c4f12da3bfdda21a838d59dd664a957b03569497b66a79e5293`
- Image platform: `linux/arm64`
- Image size reported by Docker: 21,507,233,758 bytes
- Repository digests: none, as expected for a local-only `--load` build with no push
- Image-inspection SHA-256:
  `72e044cec01aee08e9443a89698027d73971e6d9e1b8eda623beb04dbad7ab44`
- Runtime image verifier: **PASS**; receipt SHA-256
  `731ee3ba791f3c0ac11b4d628e730a0d50476f70f89b4415012376f88386b515`
- Package inventory/SBOM: **PASS**; SHA-256
  `3abeb99d9b148d41c251f721aa6929a50b7d16977e97cb75c0c0cb4e47d78c2e`
- Embedded source receipt SHA-256:
  `ff42dfd9469488bacc6b3f36861d3674739f6ee26e177386e470e3f4602f766f`
- Terminal acceptance receipt SHA-256:
  `f773c9a2fcc60bb5d8391b84e46bdc902b5bb1a10a385bbcdfb11fe113a4b978`

The verifier confirmed the real CUDA 13 runtime-library binding, all provenance
manifests, the QTIP lookup table, 26 SM120 cubins, and 6 E43 cubins.

## Package inventory

| Package | Version |
| --- | --- |
| `banana-smasher` | `1.0.0` |
| `banana-smasher-plugin` | `0.2.0` |
| `deep-gemm` | `2.6.1` |
| `flashinfer-python` | `0.6.17` |
| `numpy` | `2.2.6` |
| `quack-kernels` | `0.5.0` |
| `safetensors` | `0.8.0` |
| `tilelang` | `0.1.9` |
| `torch` | `2.11.0+cu130` |
| `triton` | `3.6.0` |
| `vllm` | `0.24.0` |

The package-version receipt SHA-256 is
`c0bbf885a3226046dea766246f39c3e0005a269611719fcbcf8f93f7fcb25084`.

## Repository gates

- Python 3.13 static suite: **107 passed, 15 skipped**
- Ruff on the changed Python files: **PASS**
- Both public wheels built and passed ZIP integrity checks.
- `banana_smasher-1.0.0-py3-none-any.whl` SHA-256:
  `45881d167bed80be5316526dbc4faf89f12441b44857292be694a48f57fad6bd`
- `banana_smasher_plugin-0.2.0-py3-none-any.whl` SHA-256:
  `83613cc1eb988fc0209701e580cc2ca19d00707c597b2e5209bcb54efc77592e`

The first registry-auth attempt failed loudly on a transient DNS lookup. The next
no-cache build exposed a stale sibling source-pin assertion before any image could be
accepted. That assertion, the extraction sentinel, and source-inventory hashes were
updated, and the complete no-cache build was rerun from the new clean source commit.
Only the final full rerun is accepted above.

## Literal documented interface

The repository README documents the build command as:

```console
IMAGE=banana-smasher-runtime:local examples/build_image.sh
```

It documents the serve command as:

```console
MODEL_DIR="$MODEL_OUT" IMAGE=banana-smasher-runtime:local examples/serve.sh
```

Those commands are reported verbatim for interface review; the serve command was
**not** executed in this task because model boot and pack/allocation work were outside
the card's stop boundary.
