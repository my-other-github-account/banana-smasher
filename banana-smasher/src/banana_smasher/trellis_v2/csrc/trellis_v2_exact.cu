#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {
constexpr int PREFIXES = 4096;
constexpr int STATES = 65536;
constexpr int BRANCHES = 16;
constexpr int PREFIX_PAIRS = PREFIXES / 2;
constexpr int THREADS = 512;
constexpr size_t SHARED_BYTES = 2 * PREFIXES * sizeof(float);

__device__ __forceinline__ float exact_emission(
    float x0, float x1, float l0, float l1) {
  const float d0 = __fsub_rn(l0, x0);
  const float d1 = __fsub_rn(l1, x1);
  return __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
}

__device__ __forceinline__ float exact_candidate(
    float previous, float x0, float x1, float l0, float l1) {
  return __fadd_rn(previous, exact_emission(x0, x1, l0, l1));
}

__device__ __forceinline__ uint8_t packed_q(
    const uint8_t* __restrict__ backpointer,
    int steps,
    int batch,
    int step,
    int seq,
    int prefix) {
  const int64_t index =
      (static_cast<int64_t>(step) * batch + seq) * PREFIX_PAIRS + (prefix >> 1);
  const uint8_t pair = backpointer[index];
  return static_cast<uint8_t>((pair >> ((prefix & 1) * 4)) & 15u);
}

// One CTA owns one complete source row. Both FP32 cost banks remain resident for
// every step; only exact four-bit q winners leave the CTA. This is the bounded
// full-row specialization of the package QTIP/K2 producer's packed traceback.
__global__ __launch_bounds__(THREADS, 2) void full_row_k2_viterbi(
    const float* __restrict__ x,
    const float* __restrict__ lut_aos,
    const int32_t* __restrict__ overlap,
    uint8_t* __restrict__ backpointer,
    int32_t* __restrict__ states,
    int steps,
    int batch,
    bool has_overlap) {
  extern __shared__ float storage[];
  float* previous = storage;
  float* current = storage + PREFIXES;
  const int seq = blockIdx.x;

  const float first_x0 = x[seq];
  const float first_x1 = x[batch + seq];
  const int required = has_overlap ? overlap[seq] : -1;
  const int required_q = has_overlap ? required >> 8 : 0;
  const int required_residue = has_overlap ? required & 255 : 0;

  for (int pair = threadIdx.x; pair < PREFIX_PAIRS; pair += blockDim.x) {
    const int j0 = pair * 2;
    const int j1 = j0 + 1;
    float best0 = CUDART_INF_F;
    float best1 = CUDART_INF_F;
    uint8_t q0 = 0;
    uint8_t q1 = 0;
    if (has_overlap) {
      if ((j0 >> 4) == required_residue) {
        const int state = required_q * PREFIXES + j0;
        best0 = exact_emission(
            first_x0, first_x1, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      if ((j1 >> 4) == required_residue) {
        const int state = required_q * PREFIXES + j1;
        best1 = exact_emission(
            first_x0, first_x1, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      q0 = static_cast<uint8_t>(required_q);
      q1 = static_cast<uint8_t>(required_q);
    } else {
#pragma unroll
      for (int q = 0; q < BRANCHES; ++q) {
        const int state0 = q * PREFIXES + j0;
        const int state1 = q * PREFIXES + j1;
        const float candidate0 = exact_emission(
            first_x0, first_x1,
            lut_aos[state0 * 2], lut_aos[state0 * 2 + 1]);
        const float candidate1 = exact_emission(
            first_x0, first_x1,
            lut_aos[state1 * 2], lut_aos[state1 * 2 + 1]);
        if (candidate0 < best0) {
          best0 = candidate0;
          q0 = static_cast<uint8_t>(q);
        }
        if (candidate1 < best1) {
          best1 = candidate1;
          q1 = static_cast<uint8_t>(q);
        }
      }
    }
    previous[j0] = best0;
    previous[j1] = best1;
    backpointer[static_cast<int64_t>(seq) * PREFIX_PAIRS + pair] =
        static_cast<uint8_t>(q0 | (q1 << 4));
  }
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    const float x0 = x[(static_cast<int64_t>(step) * 2) * batch + seq];
    const float x1 = x[(static_cast<int64_t>(step) * 2 + 1) * batch + seq];
    for (int pair = threadIdx.x; pair < PREFIX_PAIRS; pair += blockDim.x) {
      const int j0 = pair * 2;
      const int j1 = j0 + 1;
      const int residue0 = j0 >> 4;
      const int residue1 = j1 >> 4;
      float best0 = FLT_MAX;
      float best1 = FLT_MAX;
      uint8_t q0 = 0;
      uint8_t q1 = 0;
#pragma unroll
      for (int q = 0; q < BRANCHES; ++q) {
        const int state0 = q * PREFIXES + j0;
        const int state1 = q * PREFIXES + j1;
        const float candidate0 = exact_candidate(
            previous[q * 256 + residue0], x0, x1,
            lut_aos[state0 * 2], lut_aos[state0 * 2 + 1]);
        const float candidate1 = exact_candidate(
            previous[q * 256 + residue1], x0, x1,
            lut_aos[state1 * 2], lut_aos[state1 * 2 + 1]);
        if (candidate0 < best0) {
          best0 = candidate0;
          q0 = static_cast<uint8_t>(q);
        }
        if (candidate1 < best1) {
          best1 = candidate1;
          q1 = static_cast<uint8_t>(q);
        }
      }
      current[j0] = best0;
      current[j1] = best1;
      const int64_t sink =
          (static_cast<int64_t>(step) * batch + seq) * PREFIX_PAIRS + pair;
      backpointer[sink] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
    __syncthreads();
    float* temporary = previous;
    previous = current;
    current = temporary;
  }

  if (threadIdx.x == 0) {
    int prefix = required;
    if (!has_overlap) {
      float best = CUDART_INF_F;
      int best_state = 0;
      int best_prefix = 0;
      for (int candidate_prefix = 0; candidate_prefix < PREFIXES; ++candidate_prefix) {
        const uint8_t q = packed_q(
            backpointer, steps, batch, steps - 1, seq, candidate_prefix);
        const int state = static_cast<int>(q) * PREFIXES + candidate_prefix;
        const float candidate = previous[candidate_prefix];
        if (candidate < best || (candidate == best && state < best_state)) {
          best = candidate;
          best_state = state;
          best_prefix = candidate_prefix;
        }
      }
      prefix = best_prefix;
    }
    for (int step = steps - 1; step >= 0; --step) {
      const uint8_t q = packed_q(backpointer, steps, batch, step, seq, prefix);
      const int state = static_cast<int>(q) * PREFIXES + prefix;
      states[static_cast<int64_t>(step) * batch + seq] = state;
      prefix = state >> 4;
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
}  // namespace

std::vector<torch::Tensor> trellis_v2_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const c10::optional<torch::Tensor>& overlap) {
  CHECK_CUDA(x); CHECK_CONTIG(x);
  CHECK_CUDA(lut_aos); CHECK_CONTIG(lut_aos);
  TORCH_CHECK(x.device() == lut_aos.device(), "x and LUT must share one CUDA device");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && x.dim() == 2 &&
          x.size(0) >= 8 && x.size(0) % 2 == 0,
      "full-row exact QTIP2 requires contiguous CUDA float32 x [2*T,B]");
  TORCH_CHECK(
      lut_aos.scalar_type() == torch::kFloat32 && lut_aos.dim() == 2 &&
          lut_aos.size(0) == STATES && lut_aos.size(1) == 2,
      "full-row exact QTIP2 requires contiguous CUDA float32 LUT [65536,2]");
  const int steps = static_cast<int>(x.size(0)) / 2;
  const int batch = static_cast<int>(x.size(1));
  TORCH_CHECK(batch >= 1 && batch <= 8192, "full-row exact QTIP2 batch must be in 1..8192");
  const bool has_overlap = overlap.has_value();
  torch::Tensor overlap_tensor;
  if (has_overlap) {
    overlap_tensor = overlap.value();
    CHECK_CUDA(overlap_tensor); CHECK_CONTIG(overlap_tensor);
    TORCH_CHECK(overlap_tensor.device() == x.device(), "overlap and x must share one device");
    TORCH_CHECK(
        overlap_tensor.scalar_type() == torch::kInt32 &&
            overlap_tensor.dim() == 1 && overlap_tensor.numel() == batch,
        "overlap must be contiguous CUDA int32 [B]");
    const auto overlap_min = overlap_tensor.min().item<int32_t>();
    const auto overlap_max = overlap_tensor.max().item<int32_t>();
    TORCH_CHECK(
        overlap_min >= 0 && overlap_max < PREFIXES,
        "overlap prefixes must be in [0, 4096)");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  auto backpointer = torch::empty(
      {steps, batch, PREFIX_PAIRS}, x.options().dtype(torch::kUInt8));
  auto states = torch::empty({steps, batch}, x.options().dtype(torch::kInt32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  full_row_k2_viterbi<<<batch, THREADS, SHARED_BYTES, stream>>>(
      x.data_ptr<float>(), lut_aos.data_ptr<float>(),
      has_overlap ? overlap_tensor.data_ptr<int32_t>() : nullptr,
      backpointer.data_ptr<uint8_t>(), states.data_ptr<int32_t>(),
      steps, batch, has_overlap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {states};
}
