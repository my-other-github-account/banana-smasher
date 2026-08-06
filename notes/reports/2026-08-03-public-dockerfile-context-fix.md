# Public Dockerfile and build-context correction — 2026-08-03

## Result

The active public container path now has a fail-closed, immutable Linux ARM64 vLLM
base reference and a clean repository-root build context. Every stage names
`vllm/vllm-openai:v0.24.0` at the resolved ARM64 manifest digest
`sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b`;
the base can no longer be replaced through a build argument. The public tag,
full upstream commit `ee0da84ab9e04ac7610e28580af62c365e898389`, and short revision
`ee0da84a` are checked during the build and retained in image labels and the
image-local source receipt.

The two Banana Smasher wheels continue to be built and tested from the repository
source copied into the builder stage. No host wheel, virtual environment,
`PYTHONPATH`, bind-mounted patch tree, or alternate launcher is part of the build.
The reviewed repository patch under `docker/patches/`, the TileLang-to-real-CUDA
runtime link, the AOT assets, and the runtime defaults remain explicit image inputs.

## Changed files

| File | Change | Rationale |
| --- | --- | --- |
| `.dockerignore` | Expanded the root-context denylist for version-control state, reports, credentials, host environments, caches, model/export data, private or frozen trees, host patch trees, local wheels, and unpublished binaries. | A clean checkout is the only intended build input; unrelated host state must not enter or enlarge the context. The reviewed `docker/patches/` source remains available. |
| `docker/Dockerfile` | Hard-coded the digest-pinned official base in all four stages and added exact public tag/full-commit checks plus labels and receipt fields. | Prevents base-image build-argument substitution and makes the `v0.24.0` / `ee0da84a` authority inspectable in the completed image. |
| `docker/tests/test_public_source_dockerfile.py` | Updated the runtime-stage locator for the now-literal pinned `FROM` line. | Preserves the existing install-order regression check after removing the overrideable image argument. |
| `tests/test_public_docker_context.py` | Added five static contract tests covering the immutable base/source identity, repository-source wheel builds, TileLang/runtime defaults, context exclusions, and retired-path naming/host dependency bans. | Converts the public build and context requirements into a focused regression gate. |
| `provenance/SOURCE_INVENTORY.json` | Refreshed the byte counts and output hashes for the changed retained Docker source files. | Keeps retained-source admission exact; unrelated pre-existing inventory edits were not rewritten. |
| `notes/reports/2026-08-03-public-dockerfile-context-fix.md` | Added this changed-files table, rationale, verification, and limitations. | Keeps the public correction evidence under `notes/` without sending reports into the image build context. |

## Verification

- `python3 -m pytest -q tests/test_public_docker_context.py docker/tests/test_public_source_dockerfile.py`
  — **11 passed**.
- `python3 -m ruff check tests/test_public_docker_context.py docker/tests/test_public_source_dockerfile.py`
  — **passed**.
- `docker buildx build --check --platform linux/arm64 --file docker/Dockerfile .`
  — **passed with no warnings**; BuildKit loaded the root `.dockerignore` and resolved
  metadata for the exact ARM64 vLLM manifest digest.
- A BuildKit `FROM scratch` / `COPY . /context` context audit — **passed** with
  106 files / 1,394,463 bytes, zero forbidden host/report/model paths, and the
  reviewed FlashInfer patch plus AOT assets present.
- `git diff --check` — **passed**.

The broader `python3 -m pytest -q docker/tests tests` shared-worktree run finished
with **29 passed, 2 failed**. Both failures predate and are outside this correction:
one retained-source entry for a concurrently modified native-plane implementation
has a stale hash, and the repository-wide privacy test traverses a host `.venv`
instead of excluding it. The two task-owned retained Docker entries were checked
directly and match their recorded byte counts and SHA-256 values exactly.

## Scope and remaining publication gates

This is a static macOS build-graph/context correction, not a Linux ARM64 image-build
or SM120/SM121 boot acceptance claim. The TileLang linkage and accelerator imports
still require the documented clean-box image build and physical GPU boot gate.

The companion provenance receipt also records two publication limitations this
change does not conceal: the accepted Banana Smasher source commit is not yet
advertised by the canonical public remote, and transitive package-manager downloads
are version-pinned but not fully artifact-hash/debian-snapshot locked. Accordingly,
the Docker graph is now self-contained with respect to a clean repository checkout
and excludes host artifacts, but a stranger's bit-for-bit public reproduction remains
blocked until those source-publication and byte-lock items are resolved.
