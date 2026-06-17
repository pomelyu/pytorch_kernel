#Pytorch Kernel

Minimal implementation for pytorch custom kernels based on pytorch [guideline](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) and pybind

```

# === Non-ABI-Stable version (i.e. build a package for every pytorch version) === #
# install
cd pytorch_ops
pip install --no-build-isolation .

# test
pytest tests/test_pytorch_ops.py


# === ABI-Stable version (i.e. build one package for all pytorch version >= 2.10) === #
# !!! - Linux-only. see pytorch_ops_stable/pytorch_ops_stable/csrc/cuda/fused_muladd.cu:L37
# install
cd pytorch_ops_stable
pip install --no-build-isolation .

# test
pytest tests/test_pytorch_ops_stable.py

```