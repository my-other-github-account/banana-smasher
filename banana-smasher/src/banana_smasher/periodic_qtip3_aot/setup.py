from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="banana-smasher-periodic-qtip3-exact",
    ext_modules=[
        CUDAExtension(
            name="periodic_qtip3_cuda_exact",
            sources=["csrc/binding.cpp", "csrc/periodic_qtip3_exact.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-lineinfo", "--fmad=false"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
