#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

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

namespace pytorch_ops {

at::Tensor fused_muladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
}

TORCH_LIBRARY(pytorch_ops, m) {
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("fused_muladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(pytorch_ops, CPU, m) {
    m.impl("fused_muladd", &fused_muladd_cpu);
}

}
