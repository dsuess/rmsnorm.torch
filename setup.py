from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm",
    ext_modules=[
        CUDAExtension("rmsnorm_cuda", [
            "rmsnorm/rmsnorm_cuda.cpp",
            "rmsnorm/rmsnorm_cuda_kernel.cu",
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })