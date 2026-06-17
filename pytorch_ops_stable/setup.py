import glob
import os
import platform
import sys

import torch

def _check_platform():
    if platform.system() == "Windows":
        print(
            "Error: Building pytorch_ops_stable on Windows is not supported.\n"
            "Please build on Linux.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Detect WSL: /proc/version contains "Microsoft" or "WSL"
    try:
        with open("/proc/version") as f:
            proc_version = f.read().lower()
        if "microsoft" in proc_version or "wsl" in proc_version:
            print(
                "Error: Building pytorch_ops_stable on WSL is not supported.\n"
                "Please build on native Linux.",
                file=sys.stderr,
            )
            sys.exit(1)
    except FileNotFoundError:
        pass  # Not Linux, already handled above

_check_platform()

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    use_debug = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("CUDA", "1") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    if use_debug:
        print("Compiling in debug mode")

    extension = CUDAExtension if use_cuda else CppExtension
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not use_debug else "-O0",
            # Define Py_LIMITED_API with min version 3.9 to expose only the stable
            # limited API subset from torch 
            "-DPy_LIMITED_API=0x03090000",
            # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
            # stable API subset from torch
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ],
        "nvcc": [
            "-O3" if not use_debug else "-O0",
            # NVCC also needs TORCH_TARGET_VERSION for stable ABI in CUDA code
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            # USE_CUDA is currently needed for aoti_torch_get_current_cuda_stream
            # declaration in shim.h. This will be improved in a future release.
            "-DUSE_CUDA=1",
        ]
    }

    if use_debug:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    pwd = os.path.dirname(os.path.curdir)
    sources = list(glob.glob(os.path.join(pwd, "pytorch_ops_stable", "csrc", "*.cpp")))
    if use_cuda:
        sources += list(glob.glob(os.path.join(pwd, "pytorch_ops_stable", "csrc", "cuda", "*.cu")))

    ext_modules = [
        extension(
            "pytorch_ops_stable._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            # Build 1 wheel across multiple Python versions
            py_limited_api=True,
        )
    ]

    return ext_modules


setup(
    name="pytorch_ops_stable",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension,
    },
    options={
        "bdist_wheel": {
            "py_limited_api": "cp39",   # 3.9 is minimum supported Python version
        }
    },
)
