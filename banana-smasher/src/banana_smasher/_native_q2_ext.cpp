#include <torch/extension.h>

void quantize_tiles_q2_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indices,
    torch::Tensor temp_costs,
    torch::Tensor temp_edges,
    torch::Tensor lut);

void quantize_tiles_q2(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indices,
    torch::Tensor temp_costs,
    torch::Tensor temp_edges,
    torch::Tensor lut) {
    TORCH_CHECK(input.is_cuda(), "native Q2 input must be CUDA");
    quantize_tiles_q2_cuda(input, output, indices, temp_costs, temp_edges, lut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("quantize_tiles_q2", &quantize_tiles_q2, "Banana native Q2 trellis quantizer");
}
