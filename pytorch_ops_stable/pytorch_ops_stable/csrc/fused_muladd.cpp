#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

extern "C" {
    PyMODINIT_FUNC PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace pytorch_ops_stable {

torch::stable::Tensor fused_muladd_cpu(const torch::stable::Tensor& a, const torch::stable::Tensor& b, double c) {
    STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
    STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

    torch::stable::Tensor a_contig = torch::stable::contiguous(a);
    torch::stable::Tensor b_contig = torch::stable::contiguous(b);
    torch::stable::Tensor result = torch::stable::empty_like(a_contig);
    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* result_ptr = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
}

STABLE_TORCH_LIBRARY(pytorch_ops_stable, m) {
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("fused_muladd(Tensor a, Tensor b, float c) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(pytorch_ops_stable, CPU, m) {
    m.impl("fused_muladd", TORCH_BOX(&fused_muladd_cpu));
}

}
