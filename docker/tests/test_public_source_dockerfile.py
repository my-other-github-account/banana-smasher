from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/Dockerfile"
DEPLOY = ROOT / "README.md"


def test_public_source_dockerfile_contract() -> None:
    text = DOCKERFILE.read_text()
    lower = text.lower()
    assert "vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b" in text
    assert "ee0da84a" in text
    assert "COPY banana-smasher /src/banana-smasher" in text
    assert "COPY banana-smasher-plugin /src/banana-smasher-plugin" in text
    assert "COPY docker /src/docker" in text
    assert "python3 -m build --wheel" in text
    assert "ARG BANANA_SMASHER_SOURCE_COMMIT" in text
    assert "--stamp-provenance" in text
    package_builder = text.split("FROM ${VLLM_IMAGE} AS flashinfer-builder", 1)[0]
    cuda_devel_packages = (
        "cuda-nvrtc-dev-13-0=13.0.88-1",
        "libcublas-dev-13-0=13.1.1.3-1",
        "libcusolver-dev-13-0=12.0.4.66-1",
        "libcusparse-dev-13-0=12.6.3.3-1",
    )
    for package in cuda_devel_packages:
        assert package in package_builder
        assert package_builder.index(package) < package_builder.index(
            "python3 -m build --wheel"
        )
    assert (
        'CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"'
        in package_builder
    )
    assert "scipy==1.16.1" in package_builder
    assert package_builder.index("scipy==1.16.1") < package_builder.index(
        "python3 -m pytest -q"
    )
    assert "python3 -m pytest -q" in text
    assert "/src/banana-smasher/tests" in text
    assert "/src/banana-smasher-plugin/tests" in text
    assert "mkdir -p /wheel" in text
    assert "https://github.com/flashinfer-ai/flashinfer.git" in text
    assert "d020372b068f335e2fe427372e134977a2235c49" in text
    assert "b34f49255f1640542da91665f58558a3e5e308f1" in text
    assert "76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9" in text
    assert "0c5fda59bb6fa71eae875693a024bb0fb37ba7d6" in text
    assert "BUILD_NVEP=0" in text
    uninstall = "pip uninstall -y flashinfer-cubin flashinfer-jit-cache"
    install = "/tmp/wheels/flashinfer_python-0.6.17-py3-none-any.whl"
    assert uninstall in text
    assert text.index(uninstall) < text.index(install, text.index("FROM ${VLLM_IMAGE} AS runtime"))
    assert 'find_spec("flashinfer_cubin") is None' in text
    assert 'find_spec("flashinfer_jit_cache") is not None' in text
    assert '"flashinfer-cubin" not in names' in text
    assert 'm.version("flashinfer-jit-cache")=="0.6.17+cu130"' in text
    assert "FLASHINFER_DISABLE_VERSION_CHECK" not in text
    assert "flashinfer-python==0.6.12" not in text
    assert "https://github.com/deepseek-ai/DeepGEMM.git" in text
    assert "refs/tags/nv_dev_f8e8fb5" in text
    assert "f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca" in text
    assert "DG_FORCE_BUILD=1" in text
    assert "cuda-nvrtc-dev-13-0=13.0.88-1" in text
    assert "deep_gemm-2.6.1" in text
    assert "banana_smasher_plugin:register" not in text  # verified by the image script
    assert "libcudart_stub.so" in text
    assert "libcudart.so.13" in text
    assert "runtime_defaults.json" in text
    assert "cubins-sm120" in text and "cubins-e43" in text
    assert "runtime/ASSET_MANIFEST.json" in text
    assert "runtime/ACCELERATION_MANIFEST.json" in text
    assert "provenance/SOURCE_INVENTORY.json" in text
    assert "flashinfer-autotune/0.6.14" not in text
    assert "flashinfer_autotune_cache/0.6.14" not in text
    assert 'CMD ["vllm", "serve", "/model"' in text
    forbidden = (
        "vllm_runtime",
        "pyoverlay",
        "pythonpath",
        "ld_preload",
        "lic" + "ense",
        "sp" + "dx",
        "gene" + "sis",
        "hf_token",
    )
    for token in forbidden:
        assert token not in lower


def test_pinned_deepgemm_source_is_publicly_fetchable_and_sm120_capable(
    tmp_path: Path,
) -> None:
    """Reject official DeepGEMM tags that drift or omit required SM120 sources."""
    text = DOCKERFILE.read_text()
    repo_match = re.search(r"^ARG DEEPGEMM_SOURCE_REPO=(\S+)$", text, re.MULTILINE)
    ref_match = re.search(r"^ARG DEEPGEMM_SOURCE_REF=(\S+)$", text, re.MULTILINE)
    commit_match = re.search(
        r"^ARG DEEPGEMM_SOURCE_COMMIT=([0-9a-f]{40})$", text, re.MULTILINE
    )
    assert repo_match is not None
    assert ref_match is not None
    assert commit_match is not None
    repo = repo_match.group(1)
    ref = ref_match.group(1)
    commit = commit_match.group(1)

    checkout = tmp_path / "deepgemm-fetch"
    subprocess.run(
        ["git", "init", str(checkout)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", repo],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "fetch", "--depth=1", "origin", ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    fetched = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "FETCH_HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert fetched == commit

    tree = set(
        subprocess.run(
            ["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.splitlines()
    )
    required_sm120_sources = {
        "csrc/jit_kernels/heuristics/sm120.hpp",
        "csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp",
        "deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh",
    }
    assert required_sm120_sources <= tree

    version_source = subprocess.run(
        ["git", "-C", str(checkout), "show", "FETCH_HEAD:deep_gemm/__init__.py"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "__version__ = '2.6.1'" in version_source


def test_source_receipt_writer_emits_one_valid_json_document(tmp_path: Path) -> None:
    line = next(
        line.strip()
        for line in DOCKERFILE.read_text().splitlines()
        if 'p="/opt/banana-smasher/provenance/source.json"' in line
    )
    argv = shlex.split(line)
    assert argv[:2] == ["python3", "-c"]
    receipt_path = tmp_path / "source.json"
    code = argv[2].replace(
        "/opt/banana-smasher/provenance/source.json",
        str(receipt_path),
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    raw = receipt_path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\\n")
    receipt = json.loads(raw)
    assert receipt["vllm_upstream_revision"] == "ee0da84a"
    assert receipt["deep_gemm_source_ref"] == "refs/tags/nv_dev_f8e8fb5"
    assert receipt["deep_gemm_source_commit"] == (
        "f8e8fb5830fa5cda6e4ea73d360bb3f21f87a3ca"
    )
    assert receipt["deep_gemm_version"] == "2.6.1"
    assert receipt["flashinfer_source_commit"] == (
        "d020372b068f335e2fe427372e134977a2235c49"
    )


def test_runtime_removes_stale_flashinfer_binary_provider_namespaces() -> None:
    text = DOCKERFILE.read_text()
    uninstall = "pip uninstall -y flashinfer-cubin flashinfer-jit-cache"
    remove_cubin = "rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin"
    remove_jit_cache = "rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache"
    install_source = "/tmp/wheels/flashinfer_python-0.6.17-py3-none-any.whl"

    assert uninstall in text
    assert remove_cubin in text
    assert remove_jit_cache in text
    assert text.index(uninstall) < text.index(remove_cubin) < text.index(install_source, text.index(remove_cubin))
    assert text.index(uninstall) < text.index(remove_jit_cache) < text.index(install_source, text.index(remove_jit_cache))


def test_source_build_includes_required_flashinfer_aot_closure() -> None:
    text = DOCKERFILE.read_text()
    patch = (ROOT / "docker/patches/flashinfer-u12-aot.patch").read_text()
    defaults = json.loads((ROOT / "docker/runtime_defaults.json").read_text())
    verifier = (ROOT / "docker/scripts/verify_public_image.py").read_text()
    smoke_path = ROOT / "docker/scripts/smoke_flashinfer_sm121.py"
    smoke = smoke_path.read_text()
    flashinfer_builder = text.index("FROM ${VLLM_IMAGE} AS flashinfer-builder")
    flashinfer_build_pattern = re.compile(
        r"python3 -m pip wheel --no-cache-dir --no-build-isolation --no-deps \\\n"
        r"\s+--wheel-dir /wheel ./flashinfer-jit-cache"
    )
    flashinfer_build_match = flashinfer_build_pattern.search(text)
    assert flashinfer_build_match is not None
    flashinfer_build = flashinfer_build_match.start()

    assert "FLASHINFER_CUDA_ARCH_LIST=\"12.0f 12.1a\"" in text
    assert "TORCH_CUDA_ARCH_LIST=\"12.0;12.1+PTX\"" in text
    assert "FLASHINFER_ENABLE_PTX=1" in text
    assert "flashinfer-u12-aot.patch" in text
    assert "python3 -m build --wheel --no-isolation --outdir /wheel ./flashinfer-jit-cache" not in text
    assert "flashinfer_jit_cache-0.6.17+cu130-cp39-abi3-manylinux_2_28_aarch64.whl" in text
    assert "/tmp/wheels/flashinfer_jit_cache-0.6.17+cu130-cp39-abi3-manylinux_2_28_aarch64.whl" in text
    assert "FLASHINFER_DISABLE_JIT=1" in text
    assert "VLLM_HAS_FLASHINFER_CUBIN=1" in text
    assert "test -x /usr/local/cuda/bin/nvcc" in text[flashinfer_builder:flashinfer_build]
    for package in (
        "cuda-nvrtc-dev-13-0=13.0.88-1",
        "libcublas-dev-13-0=13.1.1.3-1",
        "libcusolver-dev-13-0=12.0.4.66-1",
        "libcusparse-dev-13-0=12.6.3.3-1",
    ):
        assert flashinfer_builder < text.index(package, flashinfer_builder) < flashinfer_build

    assert '"fa2_head_dim": [' in patch
    assert "(512, 512)" in patch
    assert "code=compute_" in patch
    assert '"flashinfer-jit-cache": "0.6.17+cu130"' in verifier
    assert '"sampling"' in verifier
    assert '"sparse_mla_sm120"' in verifier
    assert '"head_dim_qk_512_head_dim_vo_512"' in verifier
    assert "smoke_flashinfer_sm121.py" in text
    compile(smoke, str(smoke_path), "exec")
    assert 'FLASHINFER_DISABLE_JIT") != "1"' in smoke
    for sampling_api in (
        "top_p_sampling_from_probs",
        "top_k_sampling_from_probs",
        "top_k_top_p_sampling_from_logits",
    ):
        assert sampling_api in smoke
    assert "single_prefill_with_kv_cache" in smoke
    assert "flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4" in smoke
    assert "128 * 1024 * 1024" in smoke
    assert "(num_tokens, num_heads, 512)" in smoke
    assert "swa_topk, extra_topk = 128, 2048" in smoke
    assert "num_swa_blocks * page_block_size >= swa_topk" in smoke
    assert "compressed_kv_cache=compressed_kv_cache" in smoke
    assert "extra_sparse_indices=extra_sparse_indices" in smoke
    assert "extra_sparse_topk_lens=extra_sparse_topk_lens" in smoke
    assert "out=sparse_out" in smoke
    assert "sinks=sinks" in smoke
    assert "from flashinfer.autotuner import autotune as flashinfer_autotune" in smoke
    assert "trtllm_batch_decode_with_kv_cache_mla" in smoke
    assert "callable(flashinfer_autotune)" in smoke
    assert "callable(trtllm_batch_decode_with_kv_cache_mla)" in smoke
    assert "torch.cuda.synchronize()" in smoke
    assert defaults["environment"]["FLASHINFER_DISABLE_JIT"] == "1"
    assert defaults["environment"]["VLLM_HAS_FLASHINFER_CUBIN"] == "1"


def test_flashinfer_sm121_smoke_matches_vllm_autotuner_boot_import() -> None:
    smoke = (ROOT / "docker/scripts/smoke_flashinfer_sm121.py").read_text()

    assert "from flashinfer.autotuner import AutoTuner" in smoke
    assert "callable(AutoTuner)" in smoke


def test_flashinfer_sm121_smoke_matches_vllm_c4_decode_index_rank() -> None:
    smoke = (ROOT / "docker/scripts/smoke_flashinfer_sm121.py").read_text()

    assert ".reshape(num_tokens, 1, extra_topk)" in smoke


def test_native_plugin_build_has_pinned_cuda_development_toolchain() -> None:
    text = DOCKERFILE.read_text()
    package_builder = text.index("FROM ${VLLM_IMAGE} AS package-builder")
    plugin_build = text.index(
        "python3 -m build --wheel --no-isolation --outdir /wheels /src/banana-smasher-plugin"
    )

    for package in (
        "cuda-nvrtc-dev-13-0=13.0.88-1",
        "libcublas-dev-13-0=13.1.1.3-1",
        "libcusolver-dev-13-0=12.0.4.66-1",
        "libcusparse-dev-13-0=12.6.3.3-1",
    ):
        assert package_builder < text.index(package, package_builder) < plugin_build
    assert 'CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"' in text
    assert '"torch==2.11.0"' in (ROOT / "banana-smasher-plugin/pyproject.toml").read_text()
    assert "/wheels/banana_smasher_plugin-0.2.0-*.whl" in text
    assert "/tmp/wheels/banana_smasher_plugin-0.2.0-*.whl" in text


def test_runtime_defaults_are_baked_and_parseable() -> None:
    defaults = json.loads((ROOT / "docker/runtime_defaults.json").read_text())
    assert defaults["model"] == "/model"
    assert defaults["serve"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
    assert defaults["serve"]["max_num_seqs"] == 16
    assert defaults["serve"]["kv_cache_dtype"] == "fp8"
    assert defaults["environment"]["VLLM_USE_DEEP_GEMM"] == "1"
    assert defaults["environment"]["VLLM_USE_DEEP_GEMM_E8M0"] == "1"
    dockerfile = DOCKERFILE.read_text()
    assert "VLLM_USE_DEEP_GEMM=1" in dockerfile
    assert "VLLM_USE_DEEP_GEMM_E8M0=1" in dockerfile


def test_readme_uses_release_helpers_and_no_runtime_environment_flags() -> None:
    text = DEPLOY.read_text()
    assert "examples/build_image.sh" in text
    assert "examples/serve.sh" in text
    build = (ROOT / "examples/build_image.sh").read_text()
    serve = (ROOT / "examples/serve.sh").read_text()
    assert "docker buildx build" in build
    assert "--platform linux/arm64" in build and "--no-cache" in build
    assert 'SOURCE_COMMIT="${SOURCE_COMMIT:-$(git rev-parse HEAD)}"' in build
    assert '--build-arg "BANANA_SMASHER_SOURCE_COMMIT=$SOURCE_COMMIT"' in build
    assert "docker run --rm --gpus all" in serve
    assert "8000:8000" in serve
    assert "/root/.cache/vllm/flashinfer_autotune_cache" in serve
    assert "smash export" in text and "smash verify" in text
