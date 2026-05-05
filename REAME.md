#Pytorch Kernel

Minimal implementation for pytorch custom kernels based on pytorch [guideline](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) and pybind

```
# install
cd pytorch_ops
pip install --no-build-isolation .

# test
pytest tests/test_pytorch_ops.py
```