import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from pathlib import Path


pkg_root = Path(__file__).parent

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

setup(
    name="rmsnorm",
    ext_modules=[
        CUDAExtension(
            "rmsnorm.rmsnorm_cuda",
            sources=[
                "rmsnorm/rmsnorm_cuda.cpp",
                "rmsnorm/rmsnorm_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": append_nvcc_threads(["-O3", "--use_fast_math"]),
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False)
    }
)