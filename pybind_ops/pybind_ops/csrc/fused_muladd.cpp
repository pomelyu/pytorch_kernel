#include <torch/extension.h>

// Forward declaration of CUDA kernel (only linked when compiled with CUDA)
#ifdef USE_CUDA
at::Tensor fused_muladd_cuda(const at::Tensor& a, const at::Tensor& b, double c);
#endif

namespace pybind_ops {

at::Tensor fused_muladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dtype() == at::kFloat, "a must be float32");
    TORCH_CHECK(b.dtype() == at::kFloat, "b must be float32");
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + static_cast<float>(c);
    }
    return result;
}

at::Tensor fused_muladd(const at::Tensor& a, const at::Tensor& b, double c) {
    if (a.device().type() == at::DeviceType::CUDA) {
#ifdef USE_CUDA
        return fused_muladd_cuda(a, b, c);
#else
        TORCH_CHECK(false, "fused_muladd: this build was compiled without CUDA support");
#endif
    }
    return fused_muladd_cpu(a, b, c);
}

} // namespace pybind_ops
