#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace pytorch_ops {

__global__ void fused_muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        result[idx] = a[idx] * b[idx] + c;
    }
}

at::Tensor fused_muladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int numel = a_contig.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_muladd_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);
    return result;
}

TORCH_LIBRARY_IMPL(pytorch_ops, CUDA, m) {
    m.impl("fused_muladd", &fused_muladd_cuda);
}

}
