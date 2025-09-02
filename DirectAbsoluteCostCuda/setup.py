from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='direct_abs_cost_cuda',
    version='0.1.0',
    description='CUDA extension for DirectAbsoluteCost',
    packages=[],               # <— 明确告诉它没有纯 Python 包
    ext_modules=[
        CUDAExtension(
            name='direct_abs_cost_cuda',
            sources=['DirectAbsoluteCost_cuda.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,           # <— 禁用 zip 包安装
)
