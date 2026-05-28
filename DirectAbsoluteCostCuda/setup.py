import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def _split_paths(env_name):
    value = os.environ.get(env_name, "")
    return [path for path in value.split(os.pathsep) if path]


setup(
    name='direct_abs_cost_cuda',
    version='0.1.0',
    description='CUDA extension for DirectAbsoluteCost',
    packages=[],               # <— 明确告诉它没有纯 Python 包
    ext_modules=[
        CUDAExtension(
            name='direct_abs_cost_cuda',
            sources=['DirectAbsoluteCost_cuda.cu'],
            include_dirs=_split_paths('PILOT_CUDA_EXTRA_INCLUDE_DIRS'),
            library_dirs=_split_paths('PILOT_CUDA_EXTRA_LIBRARY_DIRS'),
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,           # <— 禁用 zip 包安装
)
