#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pytorch_ops {

__global__ void fused_muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        result[idx] = a[idx] * b[idx] + c;
    }
}

torch::stable::Tensor fused_muladd_cuda(const torch::stable::Tensor& a, const torch::stable::Tensor& b, double c) {
    STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

    torch::stable::Tensor a_contig = torch::stable::contiguous(a);
    torch::stable::Tensor b_contig = torch::stable::contiguous(b);
    torch::stable::Tensor result = torch::stable::empty_like(a_contig);
    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* result_ptr = result.mutable_data_ptr<float>();

    int numel = a_contig.numel();

    // NOTE: On Windows and WSL, passing the stream obtained from
    // aoti_torch_get_current_stream to the CUDA kernel causes silent wrong
    // results (e.g. all-zero outputs). The root cause is that StreamHandle
    // is a heap-allocated opaque object (freed via aoti_torch_delete_stream),
    // not the raw cudaStream_t value itself. reinterpret_cast<cudaStream_t>
    // therefore yields the struct's memory address rather than the actual CUDA
    // stream handle, producing undefined kernel behavior on these platforms.
    // Until the stable API exposes a proper way to extract the raw cudaStream_t,
    // the stream argument must be omitted so the kernel runs on the default stream.
    StreamHandle stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_stream(a.get_device_index(), &stream_ptr));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    fused_muladd_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);
    return result;
}

STABLE_TORCH_LIBRARY_IMPL(pytorch_ops_stable, CUDA, m) {
    m.impl("fused_muladd", TORCH_BOX(&fused_muladd_cuda));
}

}
