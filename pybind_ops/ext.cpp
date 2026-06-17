#include <torch/extension.h>

namespace pybind_ops {
at::Tensor fused_muladd(const at::Tensor& a, const at::Tensor& b, double c);
} // namespace pybind_ops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_muladd",
        &pybind_ops::fused_muladd,
        "Fused multiply-add: result[i] = a[i] * b[i] + c  (CPU and CUDA)"
    );
}