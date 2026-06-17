import glob
import os

import torch
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
            "-DUSE_CUDA=1",
        ],
        "nvcc": [
            "-O3" if not use_debug else "-O0",
            "-DUSE_CUDA=1",
        ]
    }
    if use_cuda:
        extra_compile_args["cxx"].append("-DUSE_CUDA=1")

    pwd = os.path.dirname(os.path.curdir)
    sources = [os.path.join(pwd, "ext.cpp")]
    sources += list(glob.glob(os.path.join(pwd, "pybind_ops", "csrc", "*.cpp")))
    if use_cuda:
        sources += list(glob.glob(os.path.join(pwd, "pybind_ops", "csrc", "cuda", "*.cu")))


    ext_modules = [
        extension(
            "pybind_ops._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules

setup(
    name="pybind_ops",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension,
    },
)