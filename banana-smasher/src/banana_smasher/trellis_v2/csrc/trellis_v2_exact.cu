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
constexpr int ROWS_PER_CTA = 2;
constexpr size_t SHARED_BYTES = ROWS_PER_CTA * 2 * PREFIXES * sizeof(float);

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

// One CTA owns two source rows. The pair shares every LUT load while retaining
// independent FP32 cost banks and exact four-bit winners. This changes only the
// source-backed rows-per-CTA work-amortization variable; state order is unchanged.
__global__ __launch_bounds__(THREADS, 1) void full_row_k2_viterbi(
    const float* __restrict__ x,
    const float* __restrict__ lut_aos,
    const int32_t* __restrict__ overlap,
    uint8_t* __restrict__ backpointer,
    int32_t* __restrict__ states,
    int steps,
    int batch,
    bool has_overlap) {
  extern __shared__ float storage[];
  float* previous0 = storage;
  float* current0 = storage + PREFIXES;
  float* previous1 = storage + 2 * PREFIXES;
  float* current1 = storage + 3 * PREFIXES;
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

  for (int pair = threadIdx.x; pair < PREFIX_PAIRS; pair += blockDim.x) {
    const int j0 = pair * 2;
    const int j1 = j0 + 1;
    float best00 = CUDART_INF_F;
    float best01 = CUDART_INF_F;
    float best10 = CUDART_INF_F;
    float best11 = CUDART_INF_F;
    uint8_t q00 = 0, q01 = 0, q10 = 0, q11 = 0;
    if (has_overlap) {
      if ((j0 >> 4) == required_residue0) {
        const int state = required_q0 * PREFIXES + j0;
        best00 = exact_emission(first_x00, first_x01, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      if ((j1 >> 4) == required_residue0) {
        const int state = required_q0 * PREFIXES + j1;
        best01 = exact_emission(first_x00, first_x01, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      if (has_seq1 && (j0 >> 4) == required_residue1) {
        const int state = required_q1 * PREFIXES + j0;
        best10 = exact_emission(first_x10, first_x11, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      if (has_seq1 && (j1 >> 4) == required_residue1) {
        const int state = required_q1 * PREFIXES + j1;
        best11 = exact_emission(first_x10, first_x11, lut_aos[state * 2], lut_aos[state * 2 + 1]);
      }
      q00 = q01 = static_cast<uint8_t>(required_q0);
      q10 = q11 = static_cast<uint8_t>(required_q1);
    } else {
#pragma unroll
      for (int q = 0; q < BRANCHES; ++q) {
        const int state0 = q * PREFIXES + j0;
        const int state1 = q * PREFIXES + j1;
        const float l00 = lut_aos[state0 * 2], l01 = lut_aos[state0 * 2 + 1];
        const float l10 = lut_aos[state1 * 2], l11 = lut_aos[state1 * 2 + 1];
        const float c00 = exact_emission(first_x00, first_x01, l00, l01);
        const float c01 = exact_emission(first_x00, first_x01, l10, l11);
        if (c00 < best00) { best00 = c00; q00 = static_cast<uint8_t>(q); }
        if (c01 < best01) { best01 = c01; q01 = static_cast<uint8_t>(q); }
        if (has_seq1) {
          const float c10 = exact_emission(first_x10, first_x11, l00, l01);
          const float c11 = exact_emission(first_x10, first_x11, l10, l11);
          if (c10 < best10) { best10 = c10; q10 = static_cast<uint8_t>(q); }
          if (c11 < best11) { best11 = c11; q11 = static_cast<uint8_t>(q); }
        }
      }
    }
    previous0[j0] = best00; previous0[j1] = best01;
    backpointer[static_cast<int64_t>(seq0) * PREFIX_PAIRS + pair] = static_cast<uint8_t>(q00 | (q01 << 4));
    if (has_seq1) {
      previous1[j0] = best10; previous1[j1] = best11;
      backpointer[static_cast<int64_t>(seq1) * PREFIX_PAIRS + pair] = static_cast<uint8_t>(q10 | (q11 << 4));
    }
  }
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    const float x00 = x[(static_cast<int64_t>(step) * 2) * batch + seq0];
    const float x01 = x[(static_cast<int64_t>(step) * 2 + 1) * batch + seq0];
    const float x10 = has_seq1 ? x[(static_cast<int64_t>(step) * 2) * batch + seq1] : 0.0f;
    const float x11 = has_seq1 ? x[(static_cast<int64_t>(step) * 2 + 1) * batch + seq1] : 0.0f;
    for (int pair = threadIdx.x; pair < PREFIX_PAIRS; pair += blockDim.x) {
      const int j0 = pair * 2;
      const int j1 = j0 + 1;
      // j0 is even and j1 == j0 + 1, so a prefix pair never crosses a
      // 16-prefix residue boundary. Load each row's predecessor once per
      // branch and reuse it for both exact candidates in the pair.
      const int residue = j0 >> 4;
      float best00 = FLT_MAX, best01 = FLT_MAX;
      float best10 = FLT_MAX, best11 = FLT_MAX;
      uint8_t q00 = 0, q01 = 0, q10 = 0, q11 = 0;
#pragma unroll
      for (int q = 0; q < BRANCHES; ++q) {
        const int state0 = q * PREFIXES + j0;
        const int state1 = q * PREFIXES + j1;
        const float l00 = lut_aos[state0 * 2], l01 = lut_aos[state0 * 2 + 1];
        const float l10 = lut_aos[state1 * 2], l11 = lut_aos[state1 * 2 + 1];
        const float predecessor0 = previous0[q * 256 + residue];
        const float c00 = exact_candidate(predecessor0, x00, x01, l00, l01);
        const float c01 = exact_candidate(predecessor0, x00, x01, l10, l11);
        if (c00 < best00) { best00 = c00; q00 = static_cast<uint8_t>(q); }
        if (c01 < best01) { best01 = c01; q01 = static_cast<uint8_t>(q); }
        if (has_seq1) {
          const float predecessor1 = previous1[q * 256 + residue];
          const float c10 = exact_candidate(predecessor1, x10, x11, l00, l01);
          const float c11 = exact_candidate(predecessor1, x10, x11, l10, l11);
          if (c10 < best10) { best10 = c10; q10 = static_cast<uint8_t>(q); }
          if (c11 < best11) { best11 = c11; q11 = static_cast<uint8_t>(q); }
        }
      }
      current0[j0] = best00; current0[j1] = best01;
      const int64_t sink0 = (static_cast<int64_t>(step) * batch + seq0) * PREFIX_PAIRS + pair;
      backpointer[sink0] = static_cast<uint8_t>(q00 | (q01 << 4));
      if (has_seq1) {
        current1[j0] = best10; current1[j1] = best11;
        const int64_t sink1 = (static_cast<int64_t>(step) * batch + seq1) * PREFIX_PAIRS + pair;
        backpointer[sink1] = static_cast<uint8_t>(q10 | (q11 << 4));
      }
    }
    __syncthreads();
    float* temporary0 = previous0; previous0 = current0; current0 = temporary0;
    float* temporary1 = previous1; previous1 = current1; current1 = temporary1;
  }

  if (threadIdx.x == 0) {
    for (int lane = 0; lane < ROWS_PER_CTA; ++lane) {
      const int seq = seq0 + lane;
      if (seq >= batch) break;
      float* previous = lane == 0 ? previous0 : previous1;
      int prefix = has_overlap ? overlap[seq] : -1;
      if (!has_overlap) {
        float best = CUDART_INF_F;
        int best_state = 0;
        int best_prefix = 0;
        for (int candidate_prefix = 0; candidate_prefix < PREFIXES; ++candidate_prefix) {
          const uint8_t q = packed_q(backpointer, steps, batch, steps - 1, seq, candidate_prefix);
          const int state = static_cast<int>(q) * PREFIXES + candidate_prefix;
          const float candidate = previous[candidate_prefix];
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
