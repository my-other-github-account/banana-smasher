#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "inference_dynamic.cu"

namespace {

template <uint32_t R, uint32_t M, uint32_t K, int Variant, bool HostCodebook = false>
void specialized_qtip(
    torch::Tensor& out, const torch::Tensor& sources,
    const torch::Tensor& family_block_count, const torch::Tensor& block_experts,
    const torch::Tensor& block_valid_m, const torch::Tensor& block_route_rows,
    const torch::Tensor& x, const torch::Tensor& codebook,
    const torch::Tensor& physical_counters, int64_t specialized_counter_index) {
  CHECK_INPUT(out); CHECK_INPUT(sources); CHECK_INPUT(family_block_count);
  CHECK_INPUT(block_experts); CHECK_INPUT(block_valid_m); CHECK_INPUT(block_route_rows);
  CHECK_INPUT(x); CHECK_CONTIGUOUS(codebook); CHECK_INPUT(physical_counters);
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be CUDA float32");
  TORCH_CHECK(sources.scalar_type() == torch::kInt64 && sources.dim() == 1, "sources must be CUDA int64 pointers");
  TORCH_CHECK(family_block_count.scalar_type() == torch::kInt32 && family_block_count.numel() == 1, "family count must be int32[1]");
  TORCH_CHECK(block_experts.scalar_type() == torch::kInt32 && block_experts.dim() == 1, "block experts must be int32");
  TORCH_CHECK(block_valid_m.scalar_type() == torch::kInt32 && block_valid_m.sizes() == block_experts.sizes(), "block valid_m mismatch");
  TORCH_CHECK(block_route_rows.scalar_type() == torch::kInt32 && block_route_rows.dim() == 2 && block_route_rows.size(0) == block_experts.numel(), "route descriptors invalid");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.dim() == 2, "x must be CUDA float16");
  TORCH_CHECK(codebook.scalar_type() == torch::kFloat16 && codebook.numel() == 1024, "QTIP TLUT must be float16[1024]");
  TORCH_CHECK(codebook.is_cuda() || HostCodebook, "QTIP TLUT must be CUDA unless it aliases a V7 coherent envelope");
  constexpr int64_t N = M;
  const int64_t routes = x.size(0);
  TORCH_CHECK(routes >= 1 && routes <= 49152 && x.size(1) == K, "QTIP route/K shape invalid");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == routes && out.size(1) == N, "QTIP output shape invalid");
  TORCH_CHECK(block_route_rows.size(1) >= 1 && block_route_rows.size(1) <= 16, "route tile must be in [1,16]");
  TORCH_CHECK(physical_counters.scalar_type() == torch::kInt64 && physical_counters.numel() >= 160, "physical_counters must be CUDA int64[160+]");
  TORCH_CHECK(specialized_counter_index >= 32 && specialized_counter_index < 160, "specialized counter is outside matrix layout");
  const c10::cuda::CUDAGuard guard(x.device());
  decompress_matvec_specialized_ptr<16U, 9U, R, 1U, M, 1U, K, Variant>(
      out.data_ptr<float>(), sources.data_ptr<int64_t>(),
      reinterpret_cast<const half2*>(x.data_ptr<c10::Half>()),
      reinterpret_cast<const half2*>(codebook.data_ptr<c10::Half>()),
      family_block_count.data_ptr<int32_t>(), block_experts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(), block_route_rows.data_ptr<int32_t>(),
      physical_counters.data_ptr<int64_t>(), static_cast<int>(R) - 2,
      static_cast<int>(specialized_counter_index), static_cast<int>(block_experts.numel()),
      static_cast<int>(block_route_rows.size(1)),
      at::cuda::getCurrentCUDAStream(x.get_device()).stream());
}

}  // namespace

#define DEFINE_QTIP_SPECIALIZATION(NAME, R, K, VARIANT, COUNTER) \
  void NAME(torch::Tensor& out, const torch::Tensor& sources, \
      const torch::Tensor& family_block_count, const torch::Tensor& block_experts, \
      const torch::Tensor& block_valid_m, const torch::Tensor& block_route_rows, \
      const torch::Tensor& x, const torch::Tensor& codebook, \
      const torch::Tensor& physical_counters, int64_t specialized_counter_index) { \
    TORCH_CHECK(specialized_counter_index == COUNTER, "QTIP matrix counter mismatch"); \
    specialized_qtip<R##U, 4096U, K##U, VARIANT>(out, sources, family_block_count, \
        block_experts, block_valid_m, block_route_rows, x, codebook, \
        physical_counters, specialized_counter_index); \
  }

void qtip2_v7_direct(
    torch::Tensor& out, const torch::Tensor& sources,
    const torch::Tensor& family_block_count, const torch::Tensor& block_experts,
    const torch::Tensor& block_valid_m, const torch::Tensor& block_route_rows,
    const torch::Tensor& x, const torch::Tensor& embedded_codebook,
    const torch::Tensor& physical_counters, int64_t variant,
    int64_t specialized_counter_index) {
  TORCH_CHECK(variant >= 0 && variant <= 8, "QTIP V7 variant must be in [0,8]");
#define V7_LAUNCH(M, K, VARIANT) \
  specialized_qtip<2U, M##U, K##U, VARIANT, true>( \
      out, sources, family_block_count, block_experts, block_valid_m, \
      block_route_rows, x, embedded_codebook, physical_counters, \
      specialized_counter_index)
#define V7_VARIANTS(M, K) \
  if (variant == 0) V7_LAUNCH(M, K, 0); \
  else if (variant == 1) V7_LAUNCH(M, K, 1); \
  else if (variant == 2) V7_LAUNCH(M, K, 2); \
  else if (variant == 3) V7_LAUNCH(M, K, 3); \
  else if (variant == 4) V7_LAUNCH(M, K, 4); \
  else if (variant == 5) V7_LAUNCH(M, K, 5); \
  else if (variant == 6) V7_LAUNCH(M, K, 6); \
  else if (variant == 7) V7_LAUNCH(M, K, 7); \
  else V7_LAUNCH(M, K, 8)
  if (x.size(1) == 4096 && out.size(1) == 2048) {
    V7_VARIANTS(2048, 4096);
  } else if (x.size(1) == 2048 && out.size(1) == 4096) {
    V7_VARIANTS(4096, 2048);
  } else {
    TORCH_CHECK(false, "QTIP V7 direct shape must be 4096->2048 or 2048->4096");
  }
#undef V7_VARIANTS
#undef V7_LAUNCH
}

DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c1, 2, 4096, 0, 32)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c2, 2, 4096, 1, 33)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c4, 2, 4096, 2, 34)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c8, 2, 4096, 3, 35)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c16, 2, 4096, 4, 36)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_bm16, 2, 4096, 5, 37)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_large, 2, 4096, 6, 38)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_exact_2k, 2, 4096, 7, 39)
DEFINE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_large_8192, 2, 4096, 8, 128)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c1, 2, 2048, 0, 40)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c2, 2, 2048, 1, 41)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c4, 2, 2048, 2, 42)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c8, 2, 2048, 3, 43)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c16, 2, 2048, 4, 44)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_bm16, 2, 2048, 5, 45)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_large, 2, 2048, 6, 46)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_exact_2k, 2, 2048, 7, 47)
DEFINE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_large_8192, 2, 2048, 8, 129)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c1, 3, 4096, 0, 48)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c2, 3, 4096, 1, 49)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c4, 3, 4096, 2, 50)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c8, 3, 4096, 3, 51)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c16, 3, 4096, 4, 52)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_bm16, 3, 4096, 5, 53)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_large, 3, 4096, 6, 54)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_exact_2k, 3, 4096, 7, 55)
DEFINE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_large_8192, 3, 4096, 8, 130)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c1, 3, 2048, 0, 56)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c2, 3, 2048, 1, 57)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c4, 3, 2048, 2, 58)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c8, 3, 2048, 3, 59)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c16, 3, 2048, 4, 60)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_bm16, 3, 2048, 5, 61)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_large, 3, 2048, 6, 62)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_exact_2k, 3, 2048, 7, 63)
DEFINE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_large_8192, 3, 2048, 8, 131)

#undef DEFINE_QTIP_SPECIALIZATION
