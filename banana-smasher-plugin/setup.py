from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
# setuptools requires CUDAExtension sources to be setup-root-relative.  Keep
# package-data discovery anchored by ROOT, but pass only relative source paths
# to the native builder.
CSRC = Path("src") / "banana_smasher_plugin" / "csrc"

# The public CUDA image may override this with an even narrower list, but a
# source build must never silently omit either GB10 target or forward PTX.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1+PTX")


COMMON_NVCC = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
    "--ptxas-options=-v",
]

setup(
    ext_modules=[
        CUDAExtension(
            "banana_smasher_plugin._v4_moe",
            [
                str(CSRC / "route_compaction.cu"),
                str(CSRC / "qtip_transforms.cu"),
                str(CSRC / "vq_warp_gemv.cu"),
                str(CSRC / "qtip" / "wrapper.cpp"),
                str(CSRC / "qtip" / "qtip_dynamic_torch.cu"),
            ],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": COMMON_NVCC},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    package_data={
        "banana_smasher_plugin": [
            "*.json",
            "*.npy",
            "csrc/*.cu",
            "csrc/qtip/*.cu",
            "csrc/qtip/*.cpp",
            "csrc/qtip/*.h",
        ]
    },
    zip_safe=False,
)
