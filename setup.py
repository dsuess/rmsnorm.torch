from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path


pkg_root = Path(__file__).parent

setup(
    name="rmsnorm",
    ext_modules=[
        CUDAExtension(
            "rmsnorm.rmsnorm_cuda",
            sources=[
                "rmsnorm/rmsnorm_cuda.cpp",
                "rmsnorm/rmsnorm_cuda_kernel.cu",
            ],
            #include_dirs=[
                #pkg_root / "third_party" / "cutlass" / "include",
                #pkg_root / "third_party" / "cutlass" / "tools" / "util" / "include"
            #],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False)
    }
)