#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kEdges = 16384;
constexpr int kPositions = 256;
constexpr uint32_t kMul1 = 0x83DCD12Du;

__device__ __forceinline__ int parent_index(uint16_t state) {
    const uint32_t product = static_cast<uint32_t>(state) * kMul1;
    return static_cast<int>((product & 0xffu) + ((product >> 8) & 0xffu) +
                            ((product >> 16) & 0xffu) + ((product >> 24) & 0xffu));
}

__device__ half* forward_pass(
    const half* input,
    const half* parent_lut,
    uint16_t* edges,
    half* costs_a,
    half* costs_b,
    int roll,
    int required_pre_state
) {
    half* previous = costs_a;
    half* current = costs_b;
    const int thread = threadIdx.x;

    for (int step = 0; step < kPositions; ++step) {
        const int position = (step + roll) & 255;
        const half target = input[position];
        for (int out_edge = thread; out_edge < kEdges; out_edge += kThreads) {
            int best_predecessor = out_edge >> 2;
            const uint16_t first_state = static_cast<uint16_t>(out_edge);
            half delta = __hsub(parent_lut[parent_index(first_state)], target);
            half best = step == 0
                ? __hmul(delta, delta)
                : __hfma(delta, delta, previous[best_predecessor]);
            if (step == 0 && required_pre_state >= 0 && best_predecessor != required_pre_state) {
                best = __ushort_as_half(0x7c00);
            }

            #pragma unroll
            for (int branch = 1; branch < 4; ++branch) {
                const uint16_t state = static_cast<uint16_t>((branch << 14) | out_edge);
                const int predecessor = state >> 2;
                delta = __hsub(parent_lut[parent_index(state)], target);
                half candidate = step == 0
                    ? __hmul(delta, delta)
                    : __hfma(delta, delta, previous[predecessor]);
                if (step == 0 && required_pre_state >= 0 && predecessor != required_pre_state) {
                    candidate = __ushort_as_half(0x7c00);
                }
                if (__hlt(candidate, best)) {
                    best = candidate;
                    best_predecessor = predecessor;
                }
            }
            current[out_edge] = best;
            edges[position * kEdges + out_edge] = static_cast<uint16_t>(best_predecessor);
        }
        __syncthreads();
        half* swap = previous;
        previous = current;
        current = swap;
    }
    // Match the historical kernel's endpoint convention: after 256 updates,
    // argmin is taken from the penultimate-position bank.  The final bank is
    // still used to populate the edge table, but selecting from it changes the
    // cyclic seed and therefore the strict-tie path.
    return current;
}

__device__ int historical_argmin(const half* costs, half* warp_min, int* warp_index) {
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    half local_min0 = __ushort_as_half(0x7c00);
    half local_min1 = __ushort_as_half(0x7c00);
    int local_index0 = -1;
    int local_index1 = -1;
    for (int edge = thread; edge < kEdges; edge += 1024) {
        const half value = costs[edge];
        if (__hlt(value, local_min0)) {
            local_min0 = value;
            local_index0 = edge;
        }
    }
    for (int edge = thread + 512; edge < kEdges; edge += 1024) {
        const half value = costs[edge];
        if (__hlt(value, local_min1)) {
            local_min1 = value;
            local_index1 = edge;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const half other_min0 = __shfl_down_sync(0xffffffff, local_min0, offset);
        const int other_index0 = __shfl_down_sync(0xffffffff, local_index0, offset);
        if (__hlt(other_min0, local_min0)) {
            local_min0 = other_min0;
            local_index0 = other_index0;
        }
        const half other_min1 = __shfl_down_sync(0xffffffff, local_min1, offset);
        const int other_index1 = __shfl_down_sync(0xffffffff, local_index1, offset);
        if (__hlt(other_min1, local_min1)) {
            local_min1 = other_min1;
            local_index1 = other_index1;
        }
    }
    // Preserve the historical kernel's unconditional same-warp shared stores
    // rather than normalizing this reduction to lane-zero-only writes.
    warp_min[warp] = local_min0;
    warp_index[warp] = local_index0;
    warp_min[16 + warp] = local_min1;
    warp_index[16 + warp] = local_index1;
    __syncthreads();

    if (warp == 0) {
        half local_min = warp_min[lane];
        int local_index = warp_index[lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const half other_min = __shfl_down_sync(0xffffffff, local_min, offset);
            const int other_index = __shfl_down_sync(0xffffffff, local_index, offset);
            if (__hlt(other_min, local_min)) {
                local_min = other_min;
                local_index = other_index;
            }
        }
        if (lane == 0) warp_index[0] = local_index;
    }
    __syncthreads();
    return warp_index[0];
}

__global__ __launch_bounds__(kThreads, 2) void quantize_k2_kernel(
    const float* input,
    const half* parent_lut,
    float* output,
    uint16_t* output_indices,
    uint16_t* all_edges
) {
    extern __shared__ half shared[];
    half* input_half = shared;
    half* costs_a = input_half + kPositions;
    half* costs_b = costs_a + kEdges;
    __shared__ int shared_state;
    __shared__ half warp_min[32];
    __shared__ int warp_index[32];

    const int tile = blockIdx.x;
    const int thread = threadIdx.x;
    const float* tile_input = input + tile * kPositions;
    float* tile_output = output + tile * kPositions;
    uint16_t* tile_indices = output_indices + tile * kPositions;
    uint16_t* tile_edges = all_edges + static_cast<int64_t>(tile) * kPositions * kEdges;

    if (thread < kPositions) input_half[thread] = __float2half_rn(tile_input[thread]);
    __syncthreads();

    half* terminal = forward_pass(
        input_half, parent_lut, tile_edges, costs_a, costs_b, 128, -1
    );
    int terminal_edge = historical_argmin(terminal, warp_min, warp_index);
    if (thread == 0) {
        int edge = terminal_edge;
        for (int step = 255; step >= 0; --step) {
            const int position = (step + 128) & 255;
            edge = static_cast<int>(tile_edges[position * kEdges + edge]);
            if (position == 0) break;
        }
        shared_state = edge;
    }
    __syncthreads();

    const int cyclic_state = shared_state;
    forward_pass(
        input_half, parent_lut, tile_edges, costs_a, costs_b, 0, cyclic_state
    );
    if (thread == 0) {
        int edge = cyclic_state;
        for (int position = 255; position >= 0; --position) {
            const int predecessor = static_cast<int>(tile_edges[position * kEdges + edge]);
            const uint16_t encoded = static_cast<uint16_t>((predecessor << 2) | edge);
            edge = predecessor;
            tile_indices[position] = encoded;
            tile_output[position] = __half2float(parent_lut[parent_index(encoded)]);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> quantize_k2_mul1_cuda(
    torch::Tensor input,
    torch::Tensor parent_lut
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(parent_lut.is_cuda(), "parent_lut must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(parent_lut.scalar_type() == torch::kFloat16, "parent_lut must be float16");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == kPositions, "input must be [tiles, 256]");
    TORCH_CHECK(parent_lut.numel() == 1024, "parent_lut must contain 1024 entries");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(parent_lut.is_contiguous(), "parent_lut must be contiguous");

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    auto indices = torch::empty(input.sizes(), input.options().dtype(torch::kInt16));
    auto edges = torch::empty(
        {input.size(0), kPositions, kEdges}, input.options().dtype(torch::kInt16)
    );
    const size_t shared_bytes = (kPositions + 2 * kEdges) * sizeof(half);
    cudaFuncSetAttribute(
        quantize_k2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_k2_kernel<<<input.size(0), kThreads, shared_bytes, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const half*>(parent_lut.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(indices.data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(edges.data_ptr<int16_t>())
    );
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "quantize_k2_kernel launch failed: ", cudaGetErrorString(error));
    return {output, indices, edges};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("quantize", &quantize_k2_mul1_cuda, "Banana K2 mul1 tile quantizer (CUDA)");
}
