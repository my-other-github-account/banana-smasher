#include <torch/extension.h>

std::vector<torch::Tensor> periodic_qtip_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const c10::optional<torch::Tensor>& overlap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("viterbi", &periodic_qtip_exact_cuda,
             "Full-branch exact PERIODIC K2/K3 Viterbi (CUDA)");
}
