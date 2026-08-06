#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {
constexpr int STEPS = 128;
constexpr int STATES = 65536;
constexpr int BATCH = 256;
constexpr int MAX_PREFIXES = 4096;
constexpr int FINAL_PREFIXES = 1024;
constexpr int THREADS = 256;

__device__ __forceinline__ float emission(
    float x0, float x1, float l0, float l1) {
  const float d0 = __fsub_rn(l0, x0);
  const float d1 = __fsub_rn(l1, x1);
  return __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
}

__device__ __forceinline__ void update_best(
    float value, uint8_t q, float& best, uint8_t& best_q) {
  if (value < best) {
    best = value;
    best_q = q;
  }
}

template <int WIDTH, int PREVIOUS_WIDTH, bool FIRST, bool HAS_OVERLAP>
__global__ void periodic_step(
    const float* __restrict__ x,
    const float* __restrict__ lut_aos,
    const float* __restrict__ previous,
    float* __restrict__ current,
    const int32_t* __restrict__ overlap,
    uint8_t* __restrict__ backpointer,
    int step) {
  constexpr int PREFIXES = 1 << (16 - WIDTH);
  constexpr int BRANCHES = 1 << WIDTH;
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = BATCH * PREFIXES;
  if (linear >= total) return;
  const int seq = linear / PREFIXES;
  const int prefix = linear - seq * PREFIXES;
  const float x0 = x[(step * 2) * BATCH + seq];
  const float x1 = x[(step * 2 + 1) * BATCH + seq];
  float best = INFINITY;
  uint8_t best_q = 0;
#pragma unroll
  for (int q = 0; q < BRANCHES; ++q) {
    const int state = q * PREFIXES + prefix;
    if constexpr (FIRST && HAS_OVERLAP) {
      if ((state >> 6) != overlap[seq]) continue;
    }
    const float l0 = lut_aos[state * 2];
    const float l1 = lut_aos[state * 2 + 1];
    float value = emission(x0, x1, l0, l1);
    if constexpr (!FIRST) {
      value = __fadd_rn(previous[seq * MAX_PREFIXES + (state >> PREVIOUS_WIDTH)], value);
    }
    update_best(value, static_cast<uint8_t>(q), best, best_q);
  }
  current[seq * MAX_PREFIXES + prefix] = best;
  backpointer[(static_cast<int64_t>(step) * BATCH + seq) * MAX_PREFIXES + prefix] = best_q;
}

template <bool HAS_OVERLAP>
__global__ void periodic_backtrack(
    const float* __restrict__ final_cost,
    const uint8_t* __restrict__ backpointer,
    const int32_t* __restrict__ overlap,
    int32_t* __restrict__ states) {
  const int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= BATCH) return;
  int prefix = 0;
  if constexpr (HAS_OVERLAP) {
    prefix = overlap[seq];
  } else {
    float best = INFINITY;
    for (int candidate = 0; candidate < FINAL_PREFIXES; ++candidate) {
      const float value = final_cost[seq * MAX_PREFIXES + candidate];
      if (value < best) {
        best = value;
        prefix = candidate;
      }
    }
  }
  for (int step = STEPS - 1; step >= 0; --step) {
    const int width = (step & 1) ? 6 : 4;
    const int prefixes = 1 << (16 - width);
    const uint8_t q = backpointer[
        (static_cast<int64_t>(step) * BATCH + seq) * MAX_PREFIXES + prefix];
    const int state = static_cast<int>(q) * prefixes + prefix;
    states[step * BATCH + seq] = state;
    if (step > 0) {
      const int previous_width = ((step - 1) & 1) ? 6 : 4;
      prefix = state >> previous_width;
    }
  }
}

#define CHECK_CUDA(value) TORCH_CHECK((value).is_cuda(), #value " must be CUDA")
#define CHECK_CONTIGUOUS(value) TORCH_CHECK((value).is_contiguous(), #value " must be contiguous")
}  // namespace

std::vector<torch::Tensor> periodic_qtip_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const c10::optional<torch::Tensor>& overlap) {
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CUDA(lut_aos);
  CHECK_CONTIGUOUS(lut_aos);
  TORCH_CHECK(x.device() == lut_aos.device(), "x and LUT must share one CUDA device");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && x.dim() == 2 &&
          x.size(0) == STEPS * 2 && x.size(1) == BATCH,
      "PERIODIC CUDA input must be contiguous float32 [256,256]");
  TORCH_CHECK(
      lut_aos.scalar_type() == torch::kFloat32 && lut_aos.dim() == 2 &&
          lut_aos.size(0) == STATES && lut_aos.size(1) == 2,
      "PERIODIC CUDA LUT must be contiguous float32 [65536,2]");

  const bool has_overlap = overlap.has_value();
  torch::Tensor overlap_tensor;
  if (has_overlap) {
    overlap_tensor = overlap.value();
    CHECK_CUDA(overlap_tensor);
    CHECK_CONTIGUOUS(overlap_tensor);
    TORCH_CHECK(overlap_tensor.device() == x.device(), "overlap device mismatch");
    TORCH_CHECK(
        overlap_tensor.scalar_type() == torch::kInt32 &&
            overlap_tensor.dim() == 1 && overlap_tensor.numel() == BATCH,
        "PERIODIC overlap must be contiguous int32 [256]");
    TORCH_CHECK(
        overlap_tensor.min().item<int32_t>() >= 0 &&
            overlap_tensor.max().item<int32_t>() < FINAL_PREFIXES,
        "PERIODIC overlap must be in [0,1024)");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  auto cost0 = torch::empty({BATCH, MAX_PREFIXES}, x.options());
  auto cost1 = torch::empty_like(cost0);
  auto backpointer = torch::empty(
      {STEPS, BATCH, MAX_PREFIXES}, x.options().dtype(torch::kUInt8));
  auto states = torch::empty({STEPS, BATCH}, x.options().dtype(torch::kInt32));
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  float* previous = cost0.data_ptr<float>();
  float* current = cost1.data_ptr<float>();
  uint8_t* backpointer_ptr = backpointer.data_ptr<uint8_t>();
  const int32_t* overlap_ptr = has_overlap ? overlap_tensor.data_ptr<int32_t>() : nullptr;

  {
    const int total = BATCH * MAX_PREFIXES;
    const int blocks = (total + THREADS - 1) / THREADS;
    if (has_overlap) {
      periodic_step<4, 6, true, true><<<blocks, THREADS, 0, stream>>>(
          x.data_ptr<float>(), lut_aos.data_ptr<float>(), previous, current,
          overlap_ptr, backpointer_ptr, 0);
    } else {
      periodic_step<4, 6, true, false><<<blocks, THREADS, 0, stream>>>(
          x.data_ptr<float>(), lut_aos.data_ptr<float>(), previous, current,
          nullptr, backpointer_ptr, 0);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    float* temporary = previous;
    previous = current;
    current = temporary;
  }

  for (int step = 1; step < STEPS; ++step) {
    if (step & 1) {
      constexpr int prefixes = 1024;
      const int total = BATCH * prefixes;
      const int blocks = (total + THREADS - 1) / THREADS;
      periodic_step<6, 4, false, false><<<blocks, THREADS, 0, stream>>>(
          x.data_ptr<float>(), lut_aos.data_ptr<float>(), previous, current,
          nullptr, backpointer_ptr, step);
    } else {
      const int total = BATCH * MAX_PREFIXES;
      const int blocks = (total + THREADS - 1) / THREADS;
      periodic_step<4, 6, false, false><<<blocks, THREADS, 0, stream>>>(
          x.data_ptr<float>(), lut_aos.data_ptr<float>(), previous, current,
          nullptr, backpointer_ptr, step);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    float* temporary = previous;
    previous = current;
    current = temporary;
  }

  if (has_overlap) {
    periodic_backtrack<true><<<1, BATCH, 0, stream>>>(
        previous, backpointer_ptr, overlap_ptr, states.data_ptr<int32_t>());
  } else {
    periodic_backtrack<false><<<1, BATCH, 0, stream>>>(
        previous, backpointer_ptr, nullptr, states.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {states};
}
