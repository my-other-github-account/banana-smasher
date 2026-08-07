#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kLegacyFamilies = 4;
constexpr int kNativeV4Families = 3;
constexpr int kThreads = 256;
constexpr int kCounterBase = 140;
constexpr int kDequantCounter = 152;

__device__ __forceinline__ uint16_t native_v4_state(
    const uint8_t* code, int step, int transition_bits) {
  const int stream_bits = 64 * transition_bits;
  const int start = step * transition_bits;
  uint16_t state = 0;
#pragma unroll
  for (int bit = 0; bit < 16; ++bit) {
    const int position = (start + bit) % stream_bits;
    const int value = (code[position >> 3] >> (7 - (position & 7))) & 1;
    state = static_cast<uint16_t>((state << 1) | value);
  }
  return state;
}

__device__ __forceinline__ float native_v4_value(
    const uint8_t* expert_codes,
    const half* tlut,
    int64_t weight_index,
    int transition_bits) {
  const int64_t block = weight_index >> 8;
  const int within = static_cast<int>(weight_index & 255);
  const int step = within >> 2;
  const int lane = within & 3;
  const uint8_t* code = expert_codes + block * (8 * transition_bits);
  const uint16_t state = native_v4_state(code, step, transition_bits);
  const uint16_t rotated = static_cast<uint16_t>((state >> 8) | (state << 8));
  const uint16_t selected = lane < 2 ? state : rotated;
  const uint32_t hash = (static_cast<uint32_t>(selected) + 1u) * selected;
  const int index = static_cast<int>((hash >> 6) & 511u);
  float value = __half2float(tlut[index * 2 + (lane & 1)]);
  if (lane == 0 || lane == 2) {
    const float sign = 1.0f - 2.0f * static_cast<float>((hash >> 15) & 1u);
    value *= sign;
  }
  return value;
}

__global__ void native_v4_fused_gemv_kernel(
    float* out,
    const int64_t* sources,
    const half* x,
    const half* tlut,
    const int32_t* family_block_counts,
    const int32_t* block_experts,
    const int32_t* block_valid_m,
    const int32_t* block_route_rows,
    int max_blocks,
    int block_rows,
    int n,
    int k) {
  const int native_family = static_cast<int>(blockIdx.y) / max_blocks;
  const int descriptor = static_cast<int>(blockIdx.y) % max_blocks;
  const int family = kLegacyFamilies + native_family;
  if (native_family >= kNativeV4Families ||
      descriptor >= family_block_counts[family]) {
    return;
  }
  const int valid_m = block_valid_m[family * max_blocks + descriptor];
  const int route_lane = static_cast<int>(blockIdx.z);
  if (route_lane >= valid_m) {
    return;
  }
  const int expert = block_experts[family * max_blocks + descriptor];
  const int route = block_route_rows[
      (family * max_blocks + descriptor) * block_rows + route_lane];
  if (expert < 0 || route < 0) {
    return;
  }
  const int transition_bits = native_family == 0 ? 7 : (native_family == 1 ? 9 : 10);
  const auto* codes = reinterpret_cast<const uint8_t*>(sources[expert]);
  float sum = 0.0f;
  const int output_row = static_cast<int>(blockIdx.x);
  for (int input_column = threadIdx.x; input_column < k; input_column += blockDim.x) {
    const int64_t weight_index =
        static_cast<int64_t>(output_row) * k + input_column;
    sum += __half2float(x[static_cast<int64_t>(route) * k + input_column]) *
           native_v4_value(codes, tlut, weight_index, transition_bits);
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(route) * n + output_row] = reduction[0];
  }
}

__global__ void native_v4_receipt_kernel(
    const int32_t* family_block_counts,
    const int32_t* block_valid_m,
    int max_blocks,
    int n,
    int k,
    int64_t* counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int native_family = 0; native_family < kNativeV4Families; ++native_family) {
    const int family = kLegacyFamilies + native_family;
    int64_t rows = 0;
    for (int descriptor = 0; descriptor < family_block_counts[family]; ++descriptor) {
      rows += block_valid_m[family * max_blocks + descriptor];
    }
    if (rows == 0) {
      continue;
    }
    const int transition_bits = native_family == 0 ? 7 : (native_family == 1 ? 9 : 10);
    const int base = kCounterBase + native_family * 4;
    atomicAdd(counters + base, int64_t{1});
    atomicAdd(counters + base + 1, rows);
    atomicAdd(counters + base + 2, rows * static_cast<int64_t>(n) * k * transition_bits / 32);
  }
  // This operator consumes packed planes directly.  The dedicated dequant counter
  // remains zero and is exported beside the three physical-rate receipts.
  atomicAdd(counters + kDequantCounter, int64_t{0});
}

void native_v4_gemv_cuda(
    torch::Tensor out,
    torch::Tensor sources,
    torch::Tensor transformed_x,
    torch::Tensor tlut,
    torch::Tensor family_block_counts,
    torch::Tensor block_experts,
    torch::Tensor block_valid_m,
    torch::Tensor block_route_rows,
    torch::Tensor physical_counters) {
  TORCH_CHECK(out.is_cuda() && transformed_x.is_cuda() && tlut.is_cuda(),
              "native V4 GEMV tensors must be CUDA resident");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(transformed_x.scalar_type() == at::kHalf,
              "transformed_x must be float16");
  TORCH_CHECK(tlut.scalar_type() == at::kHalf && tlut.numel() == 1024,
              "tlut must be float16 [512,2]");
  TORCH_CHECK(sources.scalar_type() == at::kLong && sources.dim() == 1,
              "sources must be int64 [experts]");
  TORCH_CHECK(family_block_counts.sizes() == at::IntArrayRef({7}),
              "family_block_counts must be int32 [7]");
  TORCH_CHECK(block_experts.dim() == 2 && block_experts.size(0) == 7,
              "block_experts must be int32 [7,max_blocks]");
  TORCH_CHECK(block_valid_m.sizes() == block_experts.sizes(),
              "block_valid_m must match block_experts");
  TORCH_CHECK(block_route_rows.dim() == 3 && block_route_rows.size(0) == 7 &&
                  block_route_rows.size(1) == block_experts.size(1),
              "block_route_rows must be int32 [7,max_blocks,block_rows]");
  TORCH_CHECK(physical_counters.scalar_type() == at::kLong &&
                  physical_counters.numel() > kDequantCounter,
              "physical_counters must cover native V4 receipts");
  TORCH_CHECK(out.dim() == 2 && transformed_x.dim() == 2 &&
                  out.size(0) == transformed_x.size(0),
              "out and transformed_x route dimensions must match");
  const int rows = static_cast<int>(out.size(0));
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(transformed_x.size(1));
  const int max_blocks = static_cast<int>(block_experts.size(1));
  const int block_rows = static_cast<int>(block_route_rows.size(2));
  TORCH_CHECK(rows > 0 && n > 0 && k > 0 && max_blocks > 0,
              "native V4 GEMV dimensions must be positive");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<unsigned int>(n),
      static_cast<unsigned int>(kNativeV4Families * max_blocks),
      static_cast<unsigned int>(block_rows));
  native_v4_fused_gemv_kernel<<<grid, kThreads, 0, stream>>>(
      out.data_ptr<float>(),
      sources.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(transformed_x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(tlut.data_ptr<at::Half>()),
      family_block_counts.data_ptr<int32_t>(),
      block_experts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(),
      block_route_rows.data_ptr<int32_t>(),
      max_blocks,
      block_rows,
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  native_v4_receipt_kernel<<<1, 1, 0, stream>>>(
      family_block_counts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(),
      max_blocks,
      n,
      k,
      physical_counters.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(banana_smasher_v4, m) {
  m.def("native_v4_gemv(Tensor out, Tensor sources, Tensor transformed_x, Tensor tlut, Tensor family_block_counts, Tensor block_experts, Tensor block_valid_m, Tensor block_route_rows, Tensor physical_counters) -> ()");
  m.impl("native_v4_gemv", torch::kCUDA, &native_v4_gemv_cuda);
}
