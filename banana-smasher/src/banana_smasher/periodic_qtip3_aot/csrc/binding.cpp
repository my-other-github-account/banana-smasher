#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> periodic_qtip3_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& scalar_lut,
    const c10::optional<torch::Tensor>& overlap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("viterbi", &periodic_qtip3_exact_cuda,
        "Periodic QTIP3 L16/K3/V1 exact paired-step CUDA graph solve");
}
