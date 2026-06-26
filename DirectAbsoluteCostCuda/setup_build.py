"""Build direct_abs_cost_cuda from source (for non-pilot / newer PyTorch envs)."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="direct_abs_cost_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="direct_abs_cost_cuda",
            sources=["DirectAbsoluteCost_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
