#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTransformThreads = 256;

__global__ void qtip_pre_transform_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ su,
    half* __restrict__ transformed,
    const int32_t* __restrict__ family_block_count,
    const int32_t* __restrict__ block_experts,
    const int32_t* __restrict__ block_valid_m,
    const int32_t* __restrict__ block_route_rows,
    int width,
    int route_stride) {
  const int block = blockIdx.x;
  const int block_row = blockIdx.y;
  if (block >= family_block_count[0] || block_row >= block_valid_m[block]) return;
  const int route = block_route_rows[block * route_stride + block_row];
  if (route < 0) return;
  const int expert = block_experts[block];
  extern __shared__ float values[];
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    values[column] = __bfloat162float(x[static_cast<int64_t>(route) * width + column]) *
                     su[static_cast<int64_t>(expert) * width + column];
  }
  __syncthreads();
  for (int span = 1; span < width; span <<= 1) {
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      const int group = pair / span;
      const int lane = pair - group * span;
      const int left = group * span * 2 + lane;
      const int right = left + span;
      const float a = values[left];
      const float b = values[right];
      values[left] = a + b;
      values[right] = a - b;
    }
    __syncthreads();
  }
  const float normalization = rsqrtf(static_cast<float>(width));
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    transformed[static_cast<int64_t>(route) * width + column] =
        __float2half_rn(values[column] * normalization);
  }
}

__global__ void qtip_post_transform_kernel(
    float* __restrict__ out,
    const float* __restrict__ wscale,
    const float* __restrict__ sv,
    const int32_t* __restrict__ family_block_count,
    const int32_t* __restrict__ block_experts,
    const int32_t* __restrict__ block_valid_m,
    const int32_t* __restrict__ block_route_rows,
    int width,
    int route_stride) {
  const int block = blockIdx.x;
  const int block_row = blockIdx.y;
  if (block >= family_block_count[0] || block_row >= block_valid_m[block]) return;
  const int route = block_route_rows[block * route_stride + block_row];
  if (route < 0) return;
  const int expert = block_experts[block];
  const float scale = wscale[expert];
  extern __shared__ float values[];
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    values[column] = out[static_cast<int64_t>(route) * width + column] * scale;
  }
  __syncthreads();
  for (int span = 1; span < width; span <<= 1) {
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      const int group = pair / span;
      const int lane = pair - group * span;
      const int left = group * span * 2 + lane;
      const int right = left + span;
      const float a = values[left];
      const float b = values[right];
      values[left] = a + b;
      values[right] = a - b;
    }
    __syncthreads();
  }
  const float normalization = rsqrtf(static_cast<float>(width));
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    out[static_cast<int64_t>(route) * width + column] =
        values[column] * normalization *
        sv[static_cast<int64_t>(expert) * width + column];
  }
}

void check_descriptors(
    const at::Tensor& family_block_count,
    const at::Tensor& block_experts,
    const at::Tensor& block_valid_m,
    const at::Tensor& block_route_rows) {
  TORCH_CHECK(family_block_count.is_cuda() && block_experts.is_cuda() &&
                  block_valid_m.is_cuda() && block_route_rows.is_cuda(),
              "QTIP descriptors must be CUDA tensors");
  TORCH_CHECK(family_block_count.scalar_type() == at::kInt &&
                  block_experts.scalar_type() == at::kInt &&
                  block_valid_m.scalar_type() == at::kInt &&
                  block_route_rows.scalar_type() == at::kInt,
              "QTIP descriptors must be int32");
  TORCH_CHECK(family_block_count.numel() == 1 && block_experts.dim() == 1 &&
                  block_valid_m.sizes() == block_experts.sizes() &&
                  block_route_rows.dim() == 2 &&
                  block_route_rows.size(0) == block_experts.numel(),
              "invalid QTIP descriptor shapes");
}

at::Tensor qtip_pre_transform(
    const at::Tensor& x,
    const at::Tensor& su,
    at::Tensor transformed,
    const at::Tensor& family_block_count,
    const at::Tensor& block_experts,
    const at::Tensor& block_valid_m,
    const at::Tensor& block_route_rows) {
  check_descriptors(family_block_count, block_experts, block_valid_m, block_route_rows);
  TORCH_CHECK(x.is_cuda() && su.is_cuda() && transformed.is_cuda(),
              "QTIP pre-transform tensors must be CUDA-resident");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && su.scalar_type() == at::kFloat &&
                  transformed.scalar_type() == at::kHalf,
              "QTIP pre-transform requires BF16/FP32/FP16 tensors");
  TORCH_CHECK(x.dim() == 2 && transformed.sizes() == x.sizes() &&
                  su.dim() == 2 && su.size(1) == x.size(1),
              "invalid QTIP pre-transform shapes");
  const c10::cuda::CUDAGuard guard(x.device());
  const int width = static_cast<int>(x.size(1));
  TORCH_CHECK(width == 2048 || width == 4096, "QTIP width must be 2048 or 4096");
  const int stride = static_cast<int>(block_route_rows.size(1));
  const dim3 grid(block_experts.numel(), stride, 1);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  qtip_pre_transform_kernel<<<grid, kTransformThreads, width * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      su.data_ptr<float>(), reinterpret_cast<half*>(transformed.data_ptr<at::Half>()),
      family_block_count.data_ptr<int32_t>(), block_experts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(), block_route_rows.data_ptr<int32_t>(),
      width, stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return transformed;
}

at::Tensor qtip_post_transform(
    at::Tensor out,
    const at::Tensor& wscale,
    const at::Tensor& sv,
    const at::Tensor& family_block_count,
    const at::Tensor& block_experts,
    const at::Tensor& block_valid_m,
    const at::Tensor& block_route_rows) {
  check_descriptors(family_block_count, block_experts, block_valid_m, block_route_rows);
  TORCH_CHECK(out.is_cuda() && wscale.is_cuda() && sv.is_cuda(),
              "QTIP post-transform tensors must be CUDA-resident");
  TORCH_CHECK(out.scalar_type() == at::kFloat && wscale.scalar_type() == at::kFloat &&
                  sv.scalar_type() == at::kFloat,
              "QTIP post-transform requires FP32 tensors");
  TORCH_CHECK(out.dim() == 2 && sv.dim() == 2 && sv.size(1) == out.size(1) &&
                  wscale.dim() == 1 && wscale.size(0) == sv.size(0),
              "invalid QTIP post-transform shapes");
  const c10::cuda::CUDAGuard guard(out.device());
  const int width = static_cast<int>(out.size(1));
  TORCH_CHECK(width == 4096, "QTIP post-transform width must be 4096");
  const int stride = static_cast<int>(block_route_rows.size(1));
  const dim3 grid(block_experts.numel(), stride, 1);
  const auto stream = at::cuda::getCurrentCUDAStream(out.get_device()).stream();
  qtip_post_transform_kernel<<<grid, kTransformThreads, width * sizeof(float), stream>>>(
      out.data_ptr<float>(), wscale.data_ptr<float>(), sv.data_ptr<float>(),
      family_block_count.data_ptr<int32_t>(), block_experts.data_ptr<int32_t>(),
      block_valid_m.data_ptr<int32_t>(), block_route_rows.data_ptr<int32_t>(),
      width, stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(banana_smasher_v4, m) {
  m.def("qtip_pre_transform(Tensor x, Tensor su, Tensor(a!) transformed, "
        "Tensor family_block_count, Tensor block_experts, Tensor block_valid_m, "
        "Tensor block_route_rows) -> Tensor(a!)");
  m.def("qtip_post_transform(Tensor(a!) out, Tensor wscale, Tensor sv, "
        "Tensor family_block_count, Tensor block_experts, Tensor block_valid_m, "
        "Tensor block_route_rows) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(banana_smasher_v4, CUDA, m) {
  m.impl("qtip_pre_transform", &qtip_pre_transform);
  m.impl("qtip_post_transform", &qtip_post_transform);
}
