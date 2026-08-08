#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {
constexpr int STEPS = 256;
constexpr int PAIRS = 128;
constexpr int PREFIXES = 8192;
constexpr int BRANCHES = 8;
constexpr int STATES = 65536;
constexpr int BATCH = 256;
constexpr int RESIDUES = 1024;
constexpr int RESIDUE_TILE = 128;
constexpr int RESIDUE_TILES = RESIDUES / RESIDUE_TILE;
constexpr int B_TILE = 8;
constexpr int THREADS = 256;
constexpr int EVEN_VALUES = B_TILE * BRANCHES * RESIDUE_TILE;
constexpr int ODD_VALUES = B_TILE * RESIDUE_TILE * BRANCHES;
constexpr int PREVIOUS_GROUPS = RESIDUE_TILE / BRANCHES;
constexpr int PREVIOUS_VALUES =
    B_TILE * BRANCHES * BRANCHES * PREVIOUS_GROUPS;
constexpr size_t SHARED_BYTES =
    (EVEN_VALUES + PREVIOUS_VALUES) * sizeof(float)
    + EVEN_VALUES * sizeof(uint8_t);

__device__ __forceinline__ float emission(float target, float level) {
  const float delta = __fsub_rn(level, target);
  return __fmul_rn(delta, delta);
}

__device__ __forceinline__ float add_cost(float previous, float error) {
  return __fadd_rn(previous, error);
}

__device__ __forceinline__ void update_best_strict(
    float value, uint32_t q, float& best, uint32_t& best_q) {
  asm volatile(
      "{ .reg .pred better;\n\t"
      "setp.lt.f32 better, %2, %0;\n\t"
      "selp.f32 %0, %2, %0, better;\n\t"
      "selp.u32 %1, %3, %1, better;\n\t"
      "}"
      : "+f"(best), "+r"(best_q)
      : "f"(value), "r"(q));
}

template <bool FIRST_PAIR, bool HAS_OVERLAP>
__global__ void paired_step_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scalar_lut,
    const float* __restrict__ cost_in,
    float* __restrict__ cost_out,
    const int32_t* __restrict__ overlap,
    uint8_t* __restrict__ packed_backpointer) {
  extern __shared__ unsigned char shared[];
  float* even_cost = reinterpret_cast<float*>(shared);
  float* previous_cost_cache = even_cost + EVEN_VALUES;
  uint8_t* even_q_plane =
      reinterpret_cast<uint8_t*>(previous_cost_cache + PREVIOUS_VALUES);

  const int seq_base = blockIdx.x * B_TILE;
  const int residue_base = blockIdx.y * RESIDUE_TILE;

  if constexpr (!FIRST_PAIR) {
    for (int index = threadIdx.x; index < PREVIOUS_VALUES; index += blockDim.x) {
      const int row = index / (BRANCHES * BRANCHES * PREVIOUS_GROUPS);
      const int rem = index % (BRANCHES * BRANCHES * PREVIOUS_GROUPS);
      const int odd_q = rem / (BRANCHES * PREVIOUS_GROUPS);
      const int rem2 = rem % (BRANCHES * PREVIOUS_GROUPS);
      const int branch = rem2 / PREVIOUS_GROUPS;
      const int local_group = rem2 % PREVIOUS_GROUPS;
      const int prefix =
          odd_q * RESIDUES + residue_base + local_group * BRANCHES;
      const int predecessor_prefix = branch * RESIDUES + (prefix >> 3);
      previous_cost_cache[index] =
          cost_in[(seq_base + row) * PREFIXES + predecessor_prefix];
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < EVEN_VALUES; index += blockDim.x) {
    const int row = index / (BRANCHES * RESIDUE_TILE);
    const int rem = index % (BRANCHES * RESIDUE_TILE);
    const int odd_q = rem / RESIDUE_TILE;
    const int local_residue = rem % RESIDUE_TILE;
    const int residue = residue_base + local_residue;
    const int prefix = odd_q * RESIDUES + residue;
    const int seq = seq_base + row;
    const float target = x[seq];
    float best = INFINITY;
    uint32_t even_q = 0;
#pragma unroll
    for (uint32_t branch = 0; branch < BRANCHES; ++branch) {
      const int state = static_cast<int>(branch) * PREFIXES + prefix;
      float candidate = emission(target, scalar_lut[state]);
      if constexpr (FIRST_PAIR) {
        if constexpr (HAS_OVERLAP) {
          if ((state >> 3) != overlap[seq]) candidate = INFINITY;
        }
      } else {
        const int previous_index =
            ((row * BRANCHES + odd_q) * BRANCHES
             + static_cast<int>(branch)) * PREVIOUS_GROUPS
            + local_residue / BRANCHES;
        candidate = add_cost(
            previous_cost_cache[previous_index], candidate);
      }
      update_best_strict(candidate, branch, best, even_q);
    }
    even_cost[index] = best;
    even_q_plane[index] = static_cast<uint8_t>(even_q);
  }
  __syncthreads();

  for (int index = threadIdx.x; index < ODD_VALUES; index += blockDim.x) {
    const int row = index / (RESIDUE_TILE * BRANCHES);
    const int rem = index % (RESIDUE_TILE * BRANCHES);
    const int local_residue = rem / BRANCHES;
    const int suffix = rem % BRANCHES;
    const int residue = residue_base + local_residue;
    const int prefix = residue * BRANCHES + suffix;
    const int seq = seq_base + row;
    const float target = x[BATCH + seq];
    float best = INFINITY;
    uint32_t odd_q = 0;
    uint32_t even_q = 0;
#pragma unroll
    for (uint32_t q = 0; q < BRANCHES; ++q) {
      const int even_index =
          (row * BRANCHES + static_cast<int>(q)) * RESIDUE_TILE
          + local_residue;
      const int state = static_cast<int>(q) * PREFIXES + prefix;
      const float candidate = add_cost(
          even_cost[even_index], emission(target, scalar_lut[state]));
      const float previous_best = best;
      update_best_strict(candidate, q, best, odd_q);
      if (best != previous_best) {
        even_q = even_q_plane[even_index];
      }
    }
    const int64_t sink = static_cast<int64_t>(seq) * PREFIXES + prefix;
    cost_out[sink] = best;
    packed_backpointer[sink] = static_cast<uint8_t>(
        (odd_q << 3) | (even_q & 7));
  }
}

template <bool HAS_OVERLAP>
__global__ void backtrack_kernel(
    const float* __restrict__ final_cost,
    const uint8_t* __restrict__ packed_backpointer,
    const int32_t* __restrict__ overlap,
    int32_t* __restrict__ states) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int seq = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  int prefix = 0;
  if constexpr (HAS_OVERLAP) {
    prefix = overlap[seq];
  } else {
    float best = INFINITY;
    int best_prefix = 0;
    int best_full_state = STATES;
    for (int candidate = lane; candidate < PREFIXES; candidate += 32) {
      const float value = final_cost[seq * PREFIXES + candidate];
      const uint8_t packed = packed_backpointer[
          (static_cast<int64_t>(PAIRS - 1) * BATCH + seq) * PREFIXES
          + candidate];
      const int candidate_full_state = (packed >> 3) * PREFIXES + candidate;
      if (value < best
          || (value == best && candidate_full_state < best_full_state)) {
        best = value;
        best_prefix = candidate;
        best_full_state = candidate_full_state;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, best, offset);
      const int other_prefix =
          __shfl_down_sync(0xffffffffu, best_prefix, offset);
      const int other_full_state =
          __shfl_down_sync(0xffffffffu, best_full_state, offset);
      if (other < best
          || (other == best && other_full_state < best_full_state)) {
        best = other;
        best_prefix = other_prefix;
        best_full_state = other_full_state;
      }
    }
    prefix = __shfl_sync(0xffffffffu, best_prefix, 0);
  }
  if (lane != 0) return;
  for (int pair = PAIRS - 1; pair >= 0; --pair) {
    const uint8_t packed = packed_backpointer[
        (static_cast<int64_t>(pair) * BATCH + seq) * PREFIXES + prefix];
    const int odd_q = packed >> 3;
    const int even_q = packed & 7;
    const int odd_state = odd_q * PREFIXES + prefix;
    states[(pair * 2 + 1) * BATCH + seq] = odd_state;
    const int even_prefix = odd_state >> 3;
    const int even_state = even_q * PREFIXES + even_prefix;
    states[(pair * 2) * BATCH + seq] = even_state;
    prefix = even_state >> 3;
  }
}

void configure_dynamic_shared_memory() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      paired_step_kernel<true, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      paired_step_kernel<true, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      paired_step_kernel<false, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
}

struct GraphState {
  torch::Tensor x;
  torch::Tensor scalar_lut;
  torch::Tensor overlap;
  torch::Tensor states;
  torch::Tensor packed_backpointer;
  torch::Tensor cost0;
  torch::Tensor cost1;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};

using GraphKey = std::tuple<int, bool, uintptr_t>;
std::mutex graph_cache_mutex;
std::map<GraphKey, GraphState*> graph_cache;
constexpr size_t MAX_GRAPH_CACHE_ENTRIES = 4;

void build_graph(GraphState* state, bool has_overlap) {
  configure_dynamic_shared_memory();
  C10_CUDA_CHECK(cudaGraphCreate(&state->graph, 0));
  float* in_ptr = state->cost0.data_ptr<float>();
  float* out_ptr = state->cost1.data_ptr<float>();
  cudaGraphNode_t previous = nullptr;
  const dim3 grid(BATCH / B_TILE, RESIDUE_TILES);
  for (int pair = 0; pair < PAIRS; ++pair) {
    const float* x_ptr = state->x.data_ptr<float>() + pair * 2 * BATCH;
    const float* lut_ptr = state->scalar_lut.data_ptr<float>();
    const float* cost_in_ptr = in_ptr;
    float* cost_out_ptr = out_ptr;
    const int32_t* overlap_ptr =
        has_overlap ? state->overlap.data_ptr<int32_t>() : nullptr;
    uint8_t* packed_ptr = state->packed_backpointer.data_ptr<uint8_t>()
        + static_cast<int64_t>(pair) * BATCH * PREFIXES;
    void* arguments[] = {
        &x_ptr, &lut_ptr, &cost_in_ptr, &cost_out_ptr, &overlap_ptr,
        &packed_ptr};
    cudaKernelNodeParams parameters{};
    if (pair == 0) {
      parameters.func = has_overlap
          ? reinterpret_cast<void*>(paired_step_kernel<true, true>)
          : reinterpret_cast<void*>(paired_step_kernel<true, false>);
    } else {
      parameters.func = reinterpret_cast<void*>(
          paired_step_kernel<false, false>);
    }
    parameters.gridDim = grid;
    parameters.blockDim = dim3(THREADS);
    parameters.sharedMemBytes = SHARED_BYTES;
    parameters.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    C10_CUDA_CHECK(cudaGraphAddKernelNode(
        &node, state->graph, previous == nullptr ? nullptr : &previous,
        previous == nullptr ? 0 : 1, &parameters));
    previous = node;
    float* temporary = in_ptr;
    in_ptr = out_ptr;
    out_ptr = temporary;
  }
  const float* final_cost_ptr = in_ptr;
  const uint8_t* packed_ptr = state->packed_backpointer.data_ptr<uint8_t>();
  const int32_t* overlap_ptr =
      has_overlap ? state->overlap.data_ptr<int32_t>() : nullptr;
  int32_t* states_ptr = state->states.data_ptr<int32_t>();
  void* arguments[] = {
      &final_cost_ptr, &packed_ptr, &overlap_ptr, &states_ptr};
  cudaKernelNodeParams parameters{};
  parameters.func = has_overlap
      ? reinterpret_cast<void*>(backtrack_kernel<true>)
      : reinterpret_cast<void*>(backtrack_kernel<false>);
  parameters.gridDim = dim3(BATCH / 8);
  parameters.blockDim = dim3(256);
  parameters.sharedMemBytes = 0;
  parameters.kernelParams = arguments;
  cudaGraphNode_t backtrack = nullptr;
  C10_CUDA_CHECK(cudaGraphAddKernelNode(
      &backtrack, state->graph, &previous, 1, &parameters));
}

GraphState* graph_state_for(
    const torch::Tensor& x,
    const torch::Tensor& scalar_lut,
    const torch::Tensor& overlap,
    bool has_overlap,
    cudaStream_t stream) {
  const GraphKey key(
      x.get_device(), has_overlap, reinterpret_cast<uintptr_t>(stream));
  std::lock_guard<std::mutex> lock(graph_cache_mutex);
  const auto existing = graph_cache.find(key);
  if (existing != graph_cache.end()) return existing->second;
  TORCH_CHECK(
      graph_cache.size() < MAX_GRAPH_CACHE_ENTRIES,
      "Periodic QTIP3 CUDA graph cache exhausted its bounded capacity");
  auto* state = new GraphState();
  state->x = torch::empty_like(x);
  state->scalar_lut = torch::empty_like(scalar_lut);
  if (has_overlap) state->overlap = torch::empty_like(overlap);
  state->states = torch::empty(
      {STEPS, BATCH}, x.options().dtype(torch::kInt32));
  state->packed_backpointer = torch::empty(
      {PAIRS, BATCH, PREFIXES}, x.options().dtype(torch::kUInt8));
  state->cost0 = torch::empty(
      {BATCH, PREFIXES}, x.options().dtype(torch::kFloat32));
  state->cost1 = torch::empty_like(state->cost0);
  build_graph(state, has_overlap);
  C10_CUDA_CHECK(cudaGraphInstantiate(
      &state->exec, state->graph, nullptr, nullptr, 0));
  TORCH_CHECK(
      state->exec != nullptr,
      "Periodic QTIP3 CUDA graph instantiate returned null");
  graph_cache.emplace(key, state);
  return state;
}

#define CHECK_CUDA(value) TORCH_CHECK((value).is_cuda(), #value " must be CUDA")
#define CHECK_CONTIG(value) TORCH_CHECK((value).is_contiguous(), #value " must be contiguous")
}  // namespace

std::vector<torch::Tensor> periodic_qtip3_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& scalar_lut,
    const c10::optional<torch::Tensor>& overlap) {
  CHECK_CUDA(x); CHECK_CONTIG(x);
  CHECK_CUDA(scalar_lut); CHECK_CONTIG(scalar_lut);
  TORCH_CHECK(
      x.device() == scalar_lut.device(),
      "x and scalar_lut must share one CUDA device");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && x.dim() == 2
          && x.size(0) == STEPS && x.size(1) == BATCH,
      "Periodic QTIP3 exact input must be contiguous CUDA float32 [256,256]");
  TORCH_CHECK(
      scalar_lut.scalar_type() == torch::kFloat32
          && scalar_lut.dim() == 1 && scalar_lut.size(0) == STATES,
      "Periodic QTIP3 exact LUT must be contiguous CUDA float32 [65536]");
  const bool has_overlap = overlap.has_value();
  torch::Tensor overlap_tensor;
  if (has_overlap) {
    overlap_tensor = overlap.value();
    CHECK_CUDA(overlap_tensor); CHECK_CONTIG(overlap_tensor);
    TORCH_CHECK(
        overlap_tensor.device() == x.device(),
        "overlap and x must share one CUDA device");
    TORCH_CHECK(
        overlap_tensor.scalar_type() == torch::kInt32
            && overlap_tensor.numel() == BATCH,
        "Periodic QTIP3 overlap must be contiguous CUDA int32 [256]");
    const auto overlap_min = overlap_tensor.min().item<int32_t>();
    const auto overlap_max = overlap_tensor.max().item<int32_t>();
    TORCH_CHECK(
        overlap_min >= 0 && overlap_max < PREFIXES,
        "Periodic QTIP3 overlap prefixes must be in [0,8192)");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  GraphState* state = graph_state_for(
      x, scalar_lut, overlap_tensor, has_overlap, stream);
  auto states = torch::empty(
      {STEPS, BATCH}, x.options().dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      state->x.data_ptr(), x.data_ptr(), x.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      state->scalar_lut.data_ptr(), scalar_lut.data_ptr(), scalar_lut.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  if (has_overlap) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        state->overlap.data_ptr(), overlap_tensor.data_ptr(),
        overlap_tensor.nbytes(), cudaMemcpyDeviceToDevice, stream));
  }
  C10_CUDA_CHECK(cudaGraphLaunch(state->exec, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      states.data_ptr(), state->states.data_ptr(), states.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  return {states};
}
