#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <limits>

namespace {

constexpr int kFamilies = 7;
constexpr int kDecodeBlockRows = 4;

// One deterministic device thread builds stable family/expert descriptors.  The
// descriptor tensors are caller-owned and shape-stable, so their addresses do
// not change between CUDA graph capture and replay.  No descriptor cardinality
// is copied to the host.
__global__ void compact_routes_kernel(
    const int64_t* __restrict__ expert_ids,
    const int8_t* __restrict__ family_codes,
    int32_t* __restrict__ family_block_counts,
    int32_t* __restrict__ block_experts,
    int32_t* __restrict__ block_valid_m,
    int32_t* __restrict__ block_route_rows,
    int32_t* __restrict__ expert_route_counts,
    int32_t* __restrict__ expert_last_block,
    int64_t* __restrict__ physical_counters,
    int rows,
    int experts,
    int max_blocks,
    int block_rows) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  for (int family = 0; family < kFamilies; ++family) {
    family_block_counts[family] = 0;
    for (int block = 0; block < max_blocks; ++block) {
      const int descriptor = family * max_blocks + block;
      block_experts[descriptor] = -1;
      block_valid_m[descriptor] = 0;
      for (int lane = 0; lane < block_rows; ++lane) {
        block_route_rows[descriptor * block_rows + lane] = -1;
      }
    }
  }
  for (int expert = 0; expert < experts; ++expert) {
    expert_route_counts[expert] = 0;
    expert_last_block[expert] = -1;
  }

  for (int route = 0; route < rows; ++route) {
    const int64_t expert_id = expert_ids[route];
    if (expert_id == -1) continue;
    if (expert_id < 0 || expert_id >= experts) continue;
    const int expert = static_cast<int>(expert_id);
    const int family = static_cast<int>(family_codes[expert]);
    if (family < 0 || family >= kFamilies) continue;

    int block = expert_last_block[expert];
    if (block < 0 || block_valid_m[family * max_blocks + block] == block_rows) {
      block = family_block_counts[family]++;
      if (block >= max_blocks) {
        family_block_counts[family] = max_blocks;
        continue;
      }
      const int descriptor = family * max_blocks + block;
      block_experts[descriptor] = expert;
      block_valid_m[descriptor] = 0;
      expert_last_block[expert] = block;
    }
    const int descriptor = family * max_blocks + block;
    const int lane = block_valid_m[descriptor]++;
    block_route_rows[descriptor * block_rows + lane] = route;
    ++expert_route_counts[expert];
  }
  for (int family = 0; family < kFamilies; ++family) {
    physical_counters[family] = family_block_counts[family];
    int family_rows = 0;
    for (int block = 0; block < family_block_counts[family]; ++block) {
      family_rows += block_valid_m[family * max_blocks + block];
    }
    if (family < 4) {
      physical_counters[4 + family] = family_rows;
    } else {
      physical_counters[28 + family - 4] = family_rows;
    }
  }
  physical_counters[8] = block_rows;
  ++physical_counters[22];
}

__global__ void finalize_output_kernel(
    const float* __restrict__ out,
    const int64_t* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ result,
    int64_t total,
    int output_width,
    int experts) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int route = static_cast<int>(index / output_width);
    const int64_t expert_id = expert_ids[route];
    const bool valid_route = expert_id >= 0 && expert_id < experts;
    result[index] = valid_route ? __float2bfloat16_rn(out[index])
                                : __float2bfloat16_rn(0.0f);
  }
}

at::Tensor compact_routes_cuda(
    const at::Tensor& expert_ids,
    const at::Tensor& family_codes,
    at::Tensor out,
    at::Tensor family_block_counts,
    at::Tensor block_experts,
    at::Tensor block_valid_m,
    at::Tensor block_route_rows,
    at::Tensor expert_route_counts,
    at::Tensor expert_last_block,
    at::Tensor physical_counters,
    int64_t block_rows64) {
  TORCH_CHECK(expert_ids.is_cuda() && family_codes.is_cuda() && out.is_cuda(),
              "routing inputs and output must be CUDA tensors");
  TORCH_CHECK(family_block_counts.is_cuda() && block_experts.is_cuda() &&
                  block_valid_m.is_cuda() && block_route_rows.is_cuda() &&
                  expert_route_counts.is_cuda() && expert_last_block.is_cuda() &&
                  physical_counters.is_cuda(),
              "compaction descriptors must be CUDA tensors");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong && expert_ids.dim() == 1,
              "expert_ids must be contiguous int64 [rows]");
  TORCH_CHECK(family_codes.scalar_type() == at::kChar && family_codes.dim() == 1,
              "family_codes must be contiguous int8 [experts]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == expert_ids.numel(),
              "out must be [rows, output_width]");
  TORCH_CHECK(block_rows64 == 1 || block_rows64 == 2 ||
                  block_rows64 == kDecodeBlockRows || block_rows64 == 16,
              "block_rows must be one of 1, 2, 4, or 16");
  TORCH_CHECK(family_block_counts.sizes() == at::IntArrayRef({kFamilies}),
              "family_block_counts must be int32 [7]");
  TORCH_CHECK(block_experts.dim() == 2 && block_experts.size(0) == kFamilies,
              "block_experts must be int32 [7, max_blocks]");
  TORCH_CHECK(block_valid_m.sizes() == block_experts.sizes(),
              "block_valid_m must match block_experts");
  TORCH_CHECK(block_route_rows.dim() == 3 &&
                  block_route_rows.size(0) == kFamilies &&
                  block_route_rows.size(1) == block_experts.size(1) &&
                  block_route_rows.size(2) == block_rows64,
              "block_route_rows must be int32 [7, max_blocks, block_rows]");
  TORCH_CHECK(expert_route_counts.numel() == family_codes.numel() &&
                  expert_last_block.numel() == family_codes.numel(),
              "expert descriptor tables must cover all experts");
  TORCH_CHECK(physical_counters.scalar_type() == at::kLong &&
                  physical_counters.numel() >= 153,
              "physical_counters must be int64 with at least 153 entries");
  for (const at::Tensor* tensor : std::array<const at::Tensor*, 10>{
           &expert_ids, &family_codes, &out, &family_block_counts,
           &block_experts, &block_valid_m, &block_route_rows,
           &expert_route_counts, &expert_last_block, &physical_counters}) {
    TORCH_CHECK(tensor->is_contiguous(), "all compaction tensors must be contiguous");
    TORCH_CHECK(tensor->get_device() == expert_ids.get_device(),
                "all compaction tensors must share one CUDA device");
  }
  for (const at::Tensor* tensor : std::array<const at::Tensor*, 6>{
           &family_block_counts, &block_experts, &block_valid_m,
           &block_route_rows, &expert_route_counts, &expert_last_block}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "all descriptor tensors must be int32");
  }
  TORCH_CHECK(expert_ids.numel() <= std::numeric_limits<int>::max() &&
                  family_codes.numel() <= std::numeric_limits<int>::max() &&
                  out.size(1) <= std::numeric_limits<int>::max() &&
                  block_experts.size(1) <= std::numeric_limits<int>::max(),
              "compaction shape exceeds int32 launch limits");

  const c10::cuda::CUDAGuard guard(expert_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream(expert_ids.get_device()).stream();
  const int rows = static_cast<int>(expert_ids.numel());
  compact_routes_kernel<<<1, 1, 0, stream>>>(
      expert_ids.data_ptr<int64_t>(), family_codes.data_ptr<int8_t>(),
      family_block_counts.data_ptr<int32_t>(), block_experts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(), block_route_rows.data_ptr<int32_t>(),
      expert_route_counts.data_ptr<int32_t>(), expert_last_block.data_ptr<int32_t>(),
      physical_counters.data_ptr<int64_t>(),
      rows, static_cast<int>(family_codes.numel()),
      static_cast<int>(block_experts.size(1)), static_cast<int>(block_rows64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor finalize_output_cuda(
    const at::Tensor& out,
    const at::Tensor& expert_ids,
    int64_t experts64,
    at::Tensor result) {
  TORCH_CHECK(out.is_cuda() && expert_ids.is_cuda() && result.is_cuda(),
              "finalize tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::kFloat &&
                  result.scalar_type() == at::kBFloat16,
              "finalize requires FP32 input and BF16 result");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong && expert_ids.dim() == 1 &&
                  expert_ids.is_contiguous(),
              "finalize expert_ids must be contiguous int64 [rows]");
  TORCH_CHECK(out.sizes() == result.sizes() && out.is_contiguous() &&
                  result.is_contiguous(),
              "finalize tensors must be shape-matched and contiguous");
  TORCH_CHECK(out.size(0) == expert_ids.numel(),
              "finalize expert_ids must cover every output row");
  TORCH_CHECK(out.get_device() == result.get_device() &&
                  out.get_device() == expert_ids.get_device(),
              "finalize tensors must share one CUDA device");
  TORCH_CHECK(experts64 >= 0 && experts64 <= std::numeric_limits<int>::max() &&
                  out.size(1) <= std::numeric_limits<int>::max(),
              "finalize shape exceeds int32 launch limits");
  const c10::cuda::CUDAGuard guard(out.device());
  const int64_t total = out.numel();
  const int blocks = static_cast<int>((total + 255) / 256);
  const auto stream = at::cuda::getCurrentCUDAStream(out.get_device()).stream();
  finalize_output_kernel<<<blocks, 256, 0, stream>>>(
      out.data_ptr<float>(),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(result.data_ptr<at::BFloat16>()), total,
      static_cast<int>(out.size(1)), static_cast<int>(experts64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(banana_smasher_v4, m) {
  m.def("compact_routes(Tensor expert_ids, Tensor family_codes, Tensor(a!) out, "
        "Tensor(b!) family_block_counts, Tensor(c!) block_experts, "
        "Tensor(d!) block_valid_m, Tensor(e!) block_route_rows, "
        "Tensor(f!) expert_route_counts, Tensor(g!) expert_last_block, "
        "Tensor(h!) physical_counters, int block_rows) -> Tensor(a!)");
  m.def("finalize_output(Tensor out, Tensor expert_ids, int experts, "
        "Tensor(a!) result) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(banana_smasher_v4, CUDA, m) {
  m.impl("compact_routes", &compact_routes_cuda);
  m.impl("finalize_output", &finalize_output_cuda);
}
