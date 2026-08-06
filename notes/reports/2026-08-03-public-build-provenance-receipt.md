# Public build provenance receipt — 2026-08-03

## Scope and result

This receipt resolves the direct source and base-image authorities used by the accepted
Linux ARM64 clean-box build. It records only public-safe identifiers. Private fleet
paths, hostnames, addresses, task identifiers, and raw control-plane receipts are
intentionally excluded.

Result: the vLLM, base-image, DeepGEMM, and FlashInfer authorities resolve to public
immutable objects. The accepted Banana Smasher source object is exact locally but is
**not yet reachable from the canonical public remote**, so publication of that commit
is the remaining direct-source provenance blocker. The Dockerfile also pins package
versions but not every fetched package byte; that narrower reproducibility limitation
is recorded below. The exact build/capability host claim was observed released; a
subsequent exact-CAS successor claim appeared before final verification, so this
receipt does not assert that the builder is currently unclaimed.

## Accepted Banana Smasher source

The accepted no-cache image build used a clean detached checkout of:

- canonical repository: <https://github.com/my-other-github-account/banana-smasher>
- commit: `7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad`
- tree: `2eab417e581c27324f5097187095f52423d5eee2`
- recorded source-bundle SHA-256:
  `02b996d94478369d8dec21c1e9e2196d3c6d0634b20a3b50647eb5412275493c`

Local verification:

```console
$ git show --no-patch --format=fuller 7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad
commit 7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad
...
    test: align DeepGEMM source provenance
$ git rev-parse '7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad^{tree}'
2eab417e581c27324f5097187095f52423d5eee2
$ git rev-list --count origin/main..7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad
2
$ git log --oneline --reverse origin/main..7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad
097aa04 fix: pin official SM120 DeepGEMM source
7626236 test: align DeepGEMM source provenance
```

The shared development checkout was dirty during this audit and is not asserted as a
build source. Its committed `HEAD` was
`0d6121b086b1cd9fd593e5aaec9df348953a3227`, tree
`6c42d85aad9f4a614adbd75698bd76d50480ca04`; those identifiers do not include its
uncommitted bytes.

### Public-reachability blocker

After a fresh fetch, the canonical origin advertised only `main` at
`21cd7964e9eb6ead7fb354af00ca9c44a909636a` and the separate public operations ref.
No origin remote-tracking ref contained the accepted source commit. Anonymous checks
returned HTTP 404 for the expected commit page and HTTP 422 from the commit API:

<https://github.com/my-other-github-account/banana-smasher/commit/7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad>

Expected public object: commit
`7626236ad2bd8488cc4caf1b8be7f51e32dfa1ad`, tree
`2eab417e581c27324f5097187095f52423d5eee2`. Until a reviewed public ref reaches that
object, a stranger cannot reproduce the accepted build solely from the canonical
remote. No push was performed in this audit.

## vLLM revision `ee0da84a`

`ee0da84a` is not a Banana Smasher revision. It is the abbreviated form of the exact
vLLM source commit used for the official vLLM `v0.24.0` release image:

- public repository: <https://github.com/vllm-project/vllm>
- public tag: <https://github.com/vllm-project/vllm/releases/tag/v0.24.0>
- full commit: `ee0da84ab9e04ac7610e28580af62c365e898389`
- commit URL:
  <https://github.com/vllm-project/vllm/commit/ee0da84ab9e04ac7610e28580af62c365e898389>
- source tree: `b9c60750e4c524f4445528bd72451ca75896162b`
- parent: `217c64a976869883fcf0c52a8cf8bc3c954d285b`
- subject: `[KV-Offloading] Fix tensors_per_block stride (#46888)`
- author date: `2026-06-28T01:01:45Z`
- committer date: `2026-06-28T07:04:08Z`

Resolution command and output:

```console
$ git ls-remote https://github.com/vllm-project/vllm.git refs/tags/v0.24.0
ee0da84ab9e04ac7610e28580af62c365e898389 refs/tags/v0.24.0
```

The Linux ARM64 image config independently reports the same full commit in all three
places: `ai.vllm.build.commit`, `org.opencontainers.image.revision`, and
`VLLM_BUILD_COMMIT`. Its source label is
`https://github.com/vllm-project/vllm`. Therefore `VLLM_UPSTREAM_REV=ee0da84a` in
`docker/Dockerfile` means the official tag's source commit, not an unresolvable local
short hash.

## Official vLLM base image

Registry reference: `docker.io/vllm/vllm-openai:v0.24.0`.

Resolved anonymously from Docker Hub:

- tag manifest-list digest:
  `sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f`
- unique `linux/arm64` manifest digest:
  `sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b`
- `linux/arm64` config digest:
  `sha256:730a973ed3917e4eb96cb5c3a195272fe2712d291d86001ceba2f91053f41d4e`
- exact build pin:
  `vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b`

```console
$ docker buildx imagetools inspect vllm/vllm-openai:v0.24.0
Name:      docker.io/vllm/vllm-openai:v0.24.0
MediaType: application/vnd.docker.distribution.manifest.list.v2+json
Digest:    sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
...
Platform:  linux/arm64
Name:      docker.io/vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b
```

A direct anonymous Registry V2 read returned one Linux ARM64 manifest and reproduced
both the manifest and config digests above. The Dockerfile pin is therefore the
platform manifest itself, not the mutable tag or only the multi-platform list.

## Other direct public source authorities

| Input | Public immutable identity |
| --- | --- |
| DeepGEMM | <https://github.com/deepseek-ai/DeepGEMM>, tag `refs/tags/nv_dev_f8e8fb5` -> commit `f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca`, tree `e7df13b9c9607ac77e0c739087b52ecd9c1323e7` |
| FlashInfer base | <https://github.com/flashinfer-ai/flashinfer>, commit `d020372b068f335e2fe427372e134977a2235c49`, tree `5ae786b7b7676b69a5e777b9cff36ebd60ed675b` |
| FlashInfer decode change | commit `b34f49255f1640542da91665f58558a3e5e308f1`, tree `7a632127d1ce469063249c76dca40c2d21748430` |
| FlashInfer decode test | commit `76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9`, tree `3c1569ccb08185c3d74894acce11de4e38158f7d` |
| FlashInfer prefill change | commit `0c5fda59bb6fa71eae875693a024bb0fb37ba7d6`, tree `04a870206d052373be537a869077beedc0006127` |

Tag verification:

```console
$ git ls-remote https://github.com/deepseek-ai/DeepGEMM.git refs/tags/nv_dev_f8e8fb5
f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca refs/tags/nv_dev_f8e8fb5
```

Each FlashInfer commit was also resolved successfully through the anonymous public
GitHub commit API. The local FlashInfer patch and all Banana Smasher/plugin sources are
members of the accepted Banana Smasher source tree above; their public reachability is
therefore blocked by the same unadvertised canonical commit, not by an unknown source
path.

## Clean-box host-claim release check

At `2026-08-03T15:50:42Z`, the current private fleet registry was fresh-read without
claiming the builder or starting a workload. The registry itself had SHA-256
`cf0851c0d5db4507c5ffeab970a8a7b6c016a661c25eccb975318b1b5c818a4e` and a source
mtime of `2026-08-03T15:49:28.968701Z`. The allocated clean-box builder row reported:

```text
host_claim_state=RELEASED_PUBLIC_IMAGE_CAPABILITY_PASS
active_claim=false
active_payload=null
controller_pid=null
controller_startticks=null
claim_preimage_sha256=8d11e703549da1b0ecb5ab615fc3ef8a1c49c009ac8fc73f45bd53390a2359a5
current_claim_sha256=0969a6c4d971979907c7111e6c361543fe74421d9f6b8933b972c8aad8f7c223
release_receipt_sha256=5d8d0b809999da28ee131dc821c7a9efe1160128f6de70c5a4983ec7cc8f46de
```

A final read at `2026-08-03T15:58:02Z` found that the registry had advanced to
SHA-256 `d4c58b3f6e4a5a0960493e5de30ccf1cb5514fce9a34106cf7bc56da0887db2c`
and the builder row now reported:

```text
host_claim_state=CLAIMED
active_payload=TARGETED_DYNAMIC_BACKPACK_AUTHORITY_SOURCE_INDEX
controller_pid=null
controller_startticks=null
claim_preimage_sha256=0969a6c4d971979907c7111e6c361543fe74421d9f6b8933b972c8aad8f7c223
current_claim_sha256=400b272a89ca23d1748fb1e62515a96d37a0eb9eb51ab18c0061a6ae2bd62130
```

The successor claim's preimage exactly equals the released build/capability postimage.
This confirms the requested exact-CAS release occurred; it also proves the builder is
not currently available for another claim. No release, claim, or workload action was
taken. The private registry location and release-receipt path are deliberately omitted
from this public repository.

## Remaining byte-lock limitation

The direct source/base authorities above are immutable. The accepted Dockerfile still
uses version-pinned but hash-unlocked package-manager fetches at
`docker/Dockerfile:9-10`, `docker/Dockerfile:37-41`, and
`docker/Dockerfile:81-89`. It does not bind every selected PyPI wheel/sdist to a
SHA-256 or use an immutable Debian repository snapshot/package-file digest. The
accepted package inventory/SBOM receipt has SHA-256
`3abeb99d9b148d41c251f721aa6929a50b7d16977e97cb75c0c0cb4e47d78c2e`, but its
contents are not a public lock file in the canonical origin.

Consequently this receipt establishes source identity and the exact base-image bytes;
it does not claim a bit-for-bit repeatable transitive package resolution. A complete
byte lock would need public requirements/constraints containing each fetched artifact
SHA-256 plus a pinned Debian snapshot and package-file hashes.
