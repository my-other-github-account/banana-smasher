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
constexpr int THREADS = 256;
constexpr int ROWS_PER_CTA = 2;
constexpr int COST_BANKS_PER_ROW = 1;
constexpr size_t SHARED_BYTES = ROWS_PER_CTA * COST_BANKS_PER_ROW * PREFIXES * sizeof(float);

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

// One CTA owns two source rows. The pair shares every LUT load and snapshots
// the 16 predecessor costs for each residue in registers before overwriting one
// in-place FP32 cost bank. This removes the second cost bank while preserving the
// exact state order, arithmetic, tie order, and packed traceback bytes.
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
  float* costs0 = storage;
  float* costs1 = storage + PREFIXES;
  const int residue = threadIdx.x;
  const int seq0 = blockIdx.x * ROWS_PER_CTA;
  const int seq1 = seq0 + 1;
  const bool has_seq1 = seq1 < batch;
  const float first_x00 = x[seq0];
  const float first_x01 = x[batch + seq0];
  const float first_x10 = has_seq1 ? x[seq1] : 0.0f;
  const float first_x11 = has_seq1 ? x[batch + seq1] : 0.0f;
  const int required0 = has_overlap ? overlap[seq0] : -1;
  const int required1 = has_overlap && has_seq1 ? overlap[seq1] : -1;
  const int required_q0 = has_overlap ? required0 >> 8 : 0;
  const int required_q1 = has_overlap && has_seq1 ? required1 >> 8 : 0;
  const int required_residue0 = has_overlap ? required0 & 255 : 0;
  const int required_residue1 = has_overlap && has_seq1 ? required1 & 255 : 0;

#pragma unroll
  for (int low = 0; low < 16; ++low) {
    const int prefix = residue * 16 + low;
    float best0 = CUDART_INF_F, best1 = CUDART_INF_F;
    uint8_t best_q0 = 0, best_q1 = 0;
#pragma unroll
    for (int q = 0; q < BRANCHES; ++q) {
      const int state = q * PREFIXES + prefix;
      const float l0 = lut_aos[state * 2], l1 = lut_aos[state * 2 + 1];
      if (!has_overlap || (residue == required_residue0 && q == required_q0)) {
        const float candidate0 = exact_emission(first_x00, first_x01, l0, l1);
        if (candidate0 < best0) { best0 = candidate0; best_q0 = static_cast<uint8_t>(q); }
      }
      if (has_seq1 && (!has_overlap || (residue == required_residue1 && q == required_q1))) {
        const float candidate1 = exact_emission(first_x10, first_x11, l0, l1);
        if (candidate1 < best1) { best1 = candidate1; best_q1 = static_cast<uint8_t>(q); }
      }
    }
    costs0[prefix] = best0;
    const int64_t pair0 = static_cast<int64_t>(seq0) * PREFIX_PAIRS + (prefix >> 1);
    if ((low & 1) == 0) backpointer[pair0] = best_q0;
    else backpointer[pair0] = static_cast<uint8_t>(backpointer[pair0] | (best_q0 << 4));
    if (has_seq1) {
      costs1[prefix] = best1;
      const int64_t pair1 = static_cast<int64_t>(seq1) * PREFIX_PAIRS + (prefix >> 1);
      if ((low & 1) == 0) backpointer[pair1] = best_q1;
      else backpointer[pair1] = static_cast<uint8_t>(backpointer[pair1] | (best_q1 << 4));
    }
  }
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    float predecessor0[BRANCHES];
    float predecessor1[BRANCHES];
#pragma unroll
    for (int q = 0; q < BRANCHES; ++q) {
      predecessor0[q] = costs0[q * 256 + residue];
      predecessor1[q] = has_seq1 ? costs1[q * 256 + residue] : 0.0f;
    }
    __syncthreads();
    const float x00 = x[(static_cast<int64_t>(step) * 2) * batch + seq0];
    const float x01 = x[(static_cast<int64_t>(step) * 2 + 1) * batch + seq0];
    const float x10 = has_seq1 ? x[(static_cast<int64_t>(step) * 2) * batch + seq1] : 0.0f;
    const float x11 = has_seq1 ? x[(static_cast<int64_t>(step) * 2 + 1) * batch + seq1] : 0.0f;
#pragma unroll
    for (int low = 0; low < 16; ++low) {
      const int prefix = residue * 16 + low;
      float best0 = FLT_MAX, best1 = FLT_MAX;
      uint8_t best_q0 = 0, best_q1 = 0;
#pragma unroll
      for (int q = 0; q < BRANCHES; ++q) {
        const int state = q * PREFIXES + prefix;
        const float l0 = lut_aos[state * 2], l1 = lut_aos[state * 2 + 1];
        const float candidate0 = exact_candidate(predecessor0[q], x00, x01, l0, l1);
        if (candidate0 < best0) { best0 = candidate0; best_q0 = static_cast<uint8_t>(q); }
        if (has_seq1) {
          const float candidate1 = exact_candidate(predecessor1[q], x10, x11, l0, l1);
          if (candidate1 < best1) { best1 = candidate1; best_q1 = static_cast<uint8_t>(q); }
        }
      }
      costs0[prefix] = best0;
      const int64_t pair0 = (static_cast<int64_t>(step) * batch + seq0) * PREFIX_PAIRS + (prefix >> 1);
      if ((low & 1) == 0) backpointer[pair0] = best_q0;
      else backpointer[pair0] = static_cast<uint8_t>(backpointer[pair0] | (best_q0 << 4));
      if (has_seq1) {
        costs1[prefix] = best1;
        const int64_t pair1 = (static_cast<int64_t>(step) * batch + seq1) * PREFIX_PAIRS + (prefix >> 1);
        if ((low & 1) == 0) backpointer[pair1] = best_q1;
        else backpointer[pair1] = static_cast<uint8_t>(backpointer[pair1] | (best_q1 << 4));
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int lane = 0; lane < ROWS_PER_CTA; ++lane) {
      const int seq = seq0 + lane;
      if (seq >= batch) break;
      float* costs = lane == 0 ? costs0 : costs1;
      int prefix = has_overlap ? overlap[seq] : -1;
      if (!has_overlap) {
        float best = CUDART_INF_F;
        int best_state = 0;
        int best_prefix = 0;
        for (int candidate_prefix = 0; candidate_prefix < PREFIXES; ++candidate_prefix) {
          const uint8_t q = packed_q(backpointer, steps, batch, steps - 1, seq, candidate_prefix);
          const int state = static_cast<int>(q) * PREFIXES + candidate_prefix;
          const float candidate = costs[candidate_prefix];
          if (candidate < best || (candidate == best && state < best_state)) {
            best = candidate; best_state = state; best_prefix = candidate_prefix;
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
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      full_row_k2_viterbi, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(SHARED_BYTES)));
  const int blocks = (batch + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
  full_row_k2_viterbi<<<blocks, THREADS, SHARED_BYTES, stream>>>(
      x.data_ptr<float>(), lut_aos.data_ptr<float>(),
      has_overlap ? overlap_tensor.data_ptr<int32_t>() : nullptr,
      backpointer.data_ptr<uint8_t>(), states.data_ptr<int32_t>(),
      steps, batch, has_overlap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {states};
}
