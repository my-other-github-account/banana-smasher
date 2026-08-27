#include <torch/extension.h>

#include <vector>

namespace py = pybind11;

namespace {

void check_common(
    const torch::Tensor& x,
    const torch::Tensor& offsets,
    const torch::Tensor& work_experts,
    const torch::Tensor& work_starts,
    const torch::Tensor& packed,
    const torch::Tensor& lut
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2,
                "x must be CUDA float32[rows,k]");
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt32 && offsets.dim() == 1,
                "offsets must be CUDA int32[experts+1]");
    TORCH_CHECK(work_experts.is_cuda() && work_experts.scalar_type() == torch::kInt32 && work_experts.dim() == 1,
                "work_experts must be CUDA int32[work]");
    TORCH_CHECK(work_starts.is_cuda() && work_starts.scalar_type() == torch::kInt32 && work_starts.dim() == 1,
                "work_starts must be CUDA int32[work]");
    TORCH_CHECK(work_experts.numel() == work_starts.numel(), "work tensor length mismatch");
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kInt16 && packed.dim() == 4 && packed.size(3) == 32,
                "packed must be CUDA int16[experts,tiles_k,tiles_m,32]");
    TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kFloat32 && lut.dim() == 1 && lut.numel() == 1024,
                "lut must be CUDA float32[1024]");
    TORCH_CHECK(offsets.numel() == packed.size(0) + 1, "offset count mismatch");
    TORCH_CHECK(x.size(1) == packed.size(1) * 16, "x K mismatch");
    TORCH_CHECK(x.is_contiguous() && offsets.is_contiguous() && work_experts.is_contiguous() &&
                work_starts.is_contiguous() && packed.is_contiguous() && lut.is_contiguous(),
                "grouped K2 inputs must be contiguous");
    TORCH_CHECK(x.device() == offsets.device() && x.device() == work_experts.device() &&
                x.device() == work_starts.device() && x.device() == packed.device() && x.device() == lut.device(),
                "grouped K2 device mismatch");
}

}  // namespace

torch::Tensor grouped_inner_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor packed,
    torch::Tensor lut
);

std::vector<torch::Tensor> grouped_inner_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor active_experts,
    torch::Tensor packed,
    torch::Tensor lut
);

torch::Tensor grouped_inner_forward(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor packed,
    torch::Tensor lut
) {
    check_common(x, offsets, work_experts, work_starts, packed, lut);
    return grouped_inner_forward_cuda(x, offsets, work_experts, work_starts, packed, lut);
}

std::vector<torch::Tensor> grouped_inner_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor active_experts,
    torch::Tensor packed,
    torch::Tensor lut
) {
    check_common(x, offsets, work_experts, work_starts, packed, lut);
    TORCH_CHECK(grad_output.is_cuda() && grad_output.scalar_type() == torch::kFloat32 &&
                grad_output.dim() == 2 && grad_output.size(0) == x.size(0) &&
                grad_output.size(1) == packed.size(2) * 16 && grad_output.is_contiguous(),
                "grad_output geometry mismatch");
    TORCH_CHECK(active_experts.is_cuda() && active_experts.scalar_type() == torch::kInt32 &&
                active_experts.dim() == 1 && active_experts.is_contiguous(),
                "active_experts must be CUDA int32[active]");
    TORCH_CHECK(active_experts.device() == x.device(), "active_experts device mismatch");
    return grouped_inner_backward_cuda(
        grad_output, x, offsets, work_experts, work_starts, active_experts, packed, lut
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("grouped_inner_forward", &grouped_inner_forward, "Grouped packed K2 forward");
    module.def("grouped_inner_backward", &grouped_inner_backward, "Grouped packed K2 backward");
}
