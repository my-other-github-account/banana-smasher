#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "inference_dynamic.cu"

namespace {

template <uint32_t R, uint32_t K>
void decompress_matvec_dynamic(
    torch::Tensor& out,
    const torch::Tensor& sources,
    const torch::Tensor& family_block_count,
    const torch::Tensor& block_experts,
    const torch::Tensor& block_valid_m,
    const torch::Tensor& block_route_rows,
    const torch::Tensor& x,
    const torch::Tensor& codebook,
    const torch::Tensor& physical_counters) {
  CHECK_INPUT(out);
  CHECK_INPUT(sources);
  CHECK_INPUT(family_block_count);
  CHECK_INPUT(block_experts);
  CHECK_INPUT(block_valid_m);
  CHECK_INPUT(block_route_rows);
  CHECK_INPUT(x);
  CHECK_INPUT(codebook);
  CHECK_INPUT(physical_counters);

  TORCH_CHECK(out.scalar_type() == torch::kFloat32,
              "out must be contiguous CUDA float32");
  TORCH_CHECK(sources.scalar_type() == torch::kInt64 && sources.dim() == 1,
              "sources must be a contiguous CUDA int64 pointer table");
  TORCH_CHECK(family_block_count.scalar_type() == torch::kInt32 &&
                  family_block_count.numel() == 1,
              "family_block_count must be contiguous CUDA int32 [1]");
  TORCH_CHECK(block_experts.scalar_type() == torch::kInt32 &&
                  block_experts.dim() == 1,
              "block_experts must be contiguous CUDA int32 [max_blocks]");
  TORCH_CHECK(block_valid_m.scalar_type() == torch::kInt32 &&
                  block_valid_m.sizes() == block_experts.sizes(),
              "block_valid_m must match block_experts");
  TORCH_CHECK(block_route_rows.scalar_type() == torch::kInt32 &&
                  block_route_rows.dim() == 2 &&
                  block_route_rows.size(0) == block_experts.numel(),
              "block_route_rows must be int32 [max_blocks, route_stride]");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.dim() == 2,
              "x must be contiguous CUDA float16 [routes, K]");
  TORCH_CHECK(codebook.scalar_type() == torch::kFloat16 && codebook.dim() == 1,
              "codebook must be contiguous CUDA float16");

  constexpr int64_t M = 4096;
  const int64_t routes = x.size(0);
  TORCH_CHECK(routes >= 1 && routes <= 49152, "routes must be in [1, 49152]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == routes && out.size(1) == M,
              "out must have shape [routes, 4096]");
  TORCH_CHECK(x.size(0) == routes && x.size(1) == K,
              "x shape does not match routes/template K");
  TORCH_CHECK(block_route_rows.size(1) >= 1 && block_route_rows.size(1) <= 16,
              "QTIP compact route stride must be in [1, 16]");
  TORCH_CHECK(codebook.numel() == (1 << (9 + 1)),
              "canonical QTIP TLUT must contain 1024 float16 values");
  TORCH_CHECK(physical_counters.scalar_type() == torch::kInt64 &&
                  physical_counters.numel() >= 24,
              "physical_counters must be contiguous CUDA int64[24+]");
  TORCH_CHECK(out.get_device() == x.get_device() &&
                  sources.get_device() == x.get_device() &&
                  family_block_count.get_device() == x.get_device() &&
                  block_experts.get_device() == x.get_device() &&
                  block_valid_m.get_device() == x.get_device() &&
                  block_route_rows.get_device() == x.get_device() &&
                  codebook.get_device() == x.get_device() &&
                  physical_counters.get_device() == x.get_device(),
              "all tensors must be on the same CUDA device");

  const c10::cuda::CUDAGuard guard(x.device());
  decompress_matvec_dynamic_ptr<16U, 9U, R, 1U, 4096U, 1U, K>(
      reinterpret_cast<float*>(out.data_ptr<float>()),
      reinterpret_cast<const int64_t*>(sources.data_ptr<int64_t>()),
      reinterpret_cast<const half2*>(x.data_ptr<c10::Half>()),
      reinterpret_cast<const half2*>(codebook.data_ptr<c10::Half>()),
      family_block_count.data_ptr<int32_t>(),
      block_experts.data_ptr<int32_t>(), block_valid_m.data_ptr<int32_t>(),
      block_route_rows.data_ptr<int32_t>(),
      physical_counters.data_ptr<int64_t>(), static_cast<int>(R) - 2,
      static_cast<int>(block_experts.numel()),
      static_cast<int>(block_route_rows.size(1)),
      at::cuda::getCurrentCUDAStream(x.get_device()).stream());
}

}  // namespace

#define DEFINE_COMPACT_QTIP(R, K)                                                \
  void decompress_matvec_compact_##R##_##K(                                     \
      torch::Tensor& out, const torch::Tensor& sources,                          \
      const torch::Tensor& family_block_count,                                   \
      const torch::Tensor& block_experts, const torch::Tensor& block_valid_m,     \
      const torch::Tensor& block_route_rows, const torch::Tensor& x,              \
      const torch::Tensor& codebook, const torch::Tensor& physical_counters) {     \
    decompress_matvec_dynamic<R##U, K##U>(                                        \
        out, sources, family_block_count, block_experts, block_valid_m,           \
        block_route_rows, x, codebook, physical_counters);                        \
  }

DEFINE_COMPACT_QTIP(2, 4096)
DEFINE_COMPACT_QTIP(3, 4096)
DEFINE_COMPACT_QTIP(2, 2048)
DEFINE_COMPACT_QTIP(3, 2048)

#undef DEFINE_COMPACT_QTIP
