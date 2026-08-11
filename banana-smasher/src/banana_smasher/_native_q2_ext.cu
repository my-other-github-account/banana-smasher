#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 512;
constexpr int kRate = 2;
constexpr int kEdges = 1 << (16 - kRate);
constexpr int kBranches = 1 << kRate;
constexpr int kRetainedBits = 16 - kRate;
constexpr uint32_t kMultiplier = 0x83DCD12Du;

__device__ __forceinline__ half decode_state(uint16_t state, const half* lut) {
    const uint32_t product = static_cast<uint32_t>(state) * kMultiplier;
    const uint32_t level =
        (product & 0xffu) +
        ((product >> 8) & 0xffu) +
        ((product >> 16) & 0xffu) +
        ((product >> 24) & 0xffu);
    return lut[level];
}

__device__ __forceinline__ half2 decode_pair(
    uint16_t state0,
    uint16_t state1,
    const half* lut) {
    return __halves2half2(decode_state(state0, lut), decode_state(state1, lut));
}

__global__ __launch_bounds__(kThreads, 2) void native_q2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint16_t* __restrict__ indices,
    half* __restrict__ scratch_costs,
    uint16_t* __restrict__ scratch_edges,
    const half* __restrict__ lut) {
    extern __shared__ unsigned char shared_bytes[];
    unsigned char* cursor = shared_bytes;
    half* shared_input = reinterpret_cast<half*>(cursor);
    cursor += 256 * sizeof(half);
    half* shared_minimum = reinterpret_cast<half*>(cursor);
    cursor += 32 * sizeof(half);
    int* shared_index = reinterpret_cast<int*>(cursor);
    cursor += 32 * sizeof(int);
    half* shared_costs = reinterpret_cast<half*>(cursor);

    const int tile = blockIdx.x;
    const int thread = threadIdx.x;
    const float* tile_input = input + tile * 256;
    float* tile_output = output + tile * 256;
    uint16_t* tile_indices = indices + tile * 256;
    uint16_t* tile_edges = scratch_edges + tile * 256 * kEdges;
    half* costs = shared_costs;
    half* next_costs = shared_costs + kEdges;

    if (thread < 256) {
        shared_input[thread] = __float2half_rn(tile_input[thread]);
    }
    __syncthreads();

    auto forward = [&](int roll, int required_predecessor) {
        int sequence_index = roll & 255;
        half* swap = costs;
        costs = next_costs;
        next_costs = swap;

        for (int edge = 2 * thread; edge < kEdges; edge += 2 * kThreads) {
            const half2 target = __half2half2(shared_input[sequence_index]);
            int predecessor = edge >> kRate;
            half2 decoded = decode_pair(edge, edge + 1, lut);
            half2 delta = __hsub2(decoded, target);
            half2 best = __hmul2(delta, delta);
            if (required_predecessor >= 0 && predecessor != required_predecessor) {
                best = __half2half2(__ushort_as_half(0x7c00));
            }
            int best0 = predecessor;
            int best1 = predecessor;

            #pragma unroll
            for (int branch = 1; branch < kBranches; ++branch) {
                const int state0 = (branch << kRetainedBits) | edge;
                predecessor = state0 >> kRate;
                decoded = decode_pair(state0, state0 + 1, lut);
                delta = __hsub2(decoded, target);
                half2 error = __hmul2(delta, delta);
                if (required_predecessor >= 0 && predecessor != required_predecessor) {
                    error = __half2half2(__ushort_as_half(0x7c00));
                }
                if (__hlt(__low2half(error), __low2half(best))) {
                    best = __halves2half2(__low2half(error), __high2half(best));
                    best0 = predecessor;
                }
                if (__hlt(__high2half(error), __high2half(best))) {
                    best = __halves2half2(__low2half(best), __high2half(error));
                    best1 = predecessor;
                }
            }
            reinterpret_cast<half2*>(costs)[edge >> 1] = best;
            tile_edges[kEdges * sequence_index + edge] = static_cast<uint16_t>(best0);
            tile_edges[kEdges * sequence_index + edge + 1] = static_cast<uint16_t>(best1);
        }
        __syncthreads();

        for (int step = 1; step < 256; ++step) {
            sequence_index = (step + roll) & 255;
            swap = costs;
            costs = next_costs;
            next_costs = swap;

            for (int edge = 2 * thread; edge < kEdges; edge += 2 * kThreads) {
                const half2 target = __half2half2(shared_input[sequence_index]);
                int predecessor = edge >> kRate;
                half2 decoded = decode_pair(edge, edge + 1, lut);
                half2 delta = __hsub2(decoded, target);
                half2 best = __hfma2(
                    delta,
                    delta,
                    __half2half2(next_costs[predecessor]));
                int best0 = predecessor;
                int best1 = predecessor;

                #pragma unroll
                for (int branch = 1; branch < kBranches; ++branch) {
                    const int state0 = (branch << kRetainedBits) | edge;
                    predecessor = state0 >> kRate;
                    decoded = decode_pair(state0, state0 + 1, lut);
                    delta = __hsub2(decoded, target);
                    half2 error = __hfma2(
                        delta,
                        delta,
                        __half2half2(next_costs[predecessor]));
                    if (__hlt(__low2half(error), __low2half(best))) {
                        best = __halves2half2(__low2half(error), __high2half(best));
                        best0 = predecessor;
                    }
                    if (__hlt(__high2half(error), __high2half(best))) {
                        best = __halves2half2(__low2half(best), __high2half(error));
                        best1 = predecessor;
                    }
                }
                reinterpret_cast<half2*>(costs)[edge >> 1] = best;
                tile_edges[kEdges * sequence_index + edge] = static_cast<uint16_t>(best0);
                tile_edges[kEdges * sequence_index + edge + 1] = static_cast<uint16_t>(best1);
            }
            __syncthreads();
        }
    };

    auto minimum_cost_index = [&]() {
        half local_minimum0 = __ushort_as_half(0x7c00);
        half local_minimum1 = __ushort_as_half(0x7c00);
        int local_index0 = -1;
        int local_index1 = -1;
        for (int edge = thread; edge < kEdges; edge += 2 * kThreads) {
            const half value = next_costs[edge];
            if (__hlt(value, local_minimum0)) {
                local_minimum0 = value;
                local_index0 = edge;
            }
        }
        for (int edge = thread + kThreads; edge < kEdges; edge += 2 * kThreads) {
            const half value = next_costs[edge];
            if (__hlt(value, local_minimum1)) {
                local_minimum1 = value;
                local_index1 = edge;
            }
        }
        const int lane = thread & 31;
        const int warp = thread >> 5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const half other_minimum0 = __shfl_down_sync(0xffffffff, local_minimum0, offset);
            const int other_index0 = __shfl_down_sync(0xffffffff, local_index0, offset);
            if (__hlt(other_minimum0, local_minimum0)) {
                local_minimum0 = other_minimum0;
                local_index0 = other_index0;
            }
            const half other_minimum1 = __shfl_down_sync(0xffffffff, local_minimum1, offset);
            const int other_index1 = __shfl_down_sync(0xffffffff, local_index1, offset);
            if (__hlt(other_minimum1, local_minimum1)) {
                local_minimum1 = other_minimum1;
                local_index1 = other_index1;
            }
        }
        if (lane == 0) {
            shared_minimum[warp] = local_minimum0;
            shared_index[warp] = local_index0;
            shared_minimum[16 + warp] = local_minimum1;
            shared_index[16 + warp] = local_index1;
        }
        __syncthreads();

        int result = 0;
        if (warp == 0) {
            half local_minimum = shared_minimum[lane];
            result = shared_index[lane];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const half other_minimum = __shfl_down_sync(0xffffffff, local_minimum, offset);
                const int other_index = __shfl_down_sync(0xffffffff, result, offset);
                if (__hlt(other_minimum, local_minimum)) {
                    local_minimum = other_minimum;
                    result = other_index;
                }
            }
        }
        return result;
    };

    auto backward = [&](int roll, bool write, int edge) {
        if (thread == 0) {
            for (int step = 255; step >= 0; --step) {
                const int sequence_index = (step + roll) & 255;
                const int predecessor = tile_edges[kEdges * sequence_index + edge];
                const int encoded = (predecessor << kRate) | edge;
                edge = predecessor;
                if (write) {
                    tile_indices[sequence_index] = static_cast<uint16_t>(encoded);
                    tile_output[sequence_index] = __half2float(decode_state(encoded, lut));
                } else if (sequence_index == 0) {
                    break;
                }
            }
        }
        if (thread == 0) {
            shared_index[0] = edge;
        }
        __syncthreads();
        return shared_index[0];
    };

    forward(128, -1);
    const int end_state = backward(128, false, minimum_cost_index());
    forward(0, end_state);
    backward(0, true, end_state);
}

}  // namespace

void quantize_tiles_q2_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indices,
    torch::Tensor temp_costs,
    torch::Tensor temp_edges,
    torch::Tensor lut) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt16, "indices must be int16");
    TORCH_CHECK(temp_costs.scalar_type() == torch::kFloat16, "temp_costs must be float16");
    TORCH_CHECK(temp_edges.scalar_type() == torch::kInt16, "temp_edges must be int16");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat16, "lut must be float16");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && indices.is_contiguous(), "tile tensors must be contiguous");
    TORCH_CHECK(temp_costs.is_contiguous() && temp_edges.is_contiguous() && lut.is_contiguous(), "scratch and LUT must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 256, "input must have shape [B,256]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape mismatch");
    TORCH_CHECK(indices.sizes() == input.sizes(), "indices shape mismatch");
    TORCH_CHECK(temp_costs.dim() == 3 && temp_costs.size(0) == input.size(0) && temp_costs.size(1) == 2 && temp_costs.size(2) == kEdges, "temp_costs shape mismatch");
    TORCH_CHECK(temp_edges.dim() == 3 && temp_edges.size(0) == input.size(0) && temp_edges.size(1) == 256 && temp_edges.size(2) == kEdges, "temp_edges shape mismatch");
    TORCH_CHECK(lut.numel() == 1024, "lut must have 1024 entries");
    TORCH_CHECK(input.device() == output.device() && input.device() == indices.device() && input.device() == temp_costs.device() && input.device() == temp_edges.device() && input.device() == lut.device(), "all tensors must share one CUDA device");

    const c10::cuda::CUDAGuard guard(input.device());
    const size_t shared_bytes =
        256 * sizeof(half) + 32 * sizeof(half) + 32 * sizeof(int) +
        2 * kEdges * sizeof(half);
    static bool shared_attribute_set = false;
    if (!shared_attribute_set) {
        const cudaError_t attribute_error = cudaFuncSetAttribute(
            native_q2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(attribute_error == cudaSuccess, "cannot admit native Q2 shared memory: ", cudaGetErrorString(attribute_error));
        shared_attribute_set = true;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    native_q2_kernel<<<input.size(0), kThreads, shared_bytes, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(indices.data_ptr<int16_t>()),
        reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(temp_edges.data_ptr<int16_t>()),
        reinterpret_cast<const half*>(lut.data_ptr<at::Half>()));
    const cudaError_t launch_error = cudaPeekAtLastError();
    TORCH_CHECK(launch_error == cudaSuccess, "native Q2 CUDA launch failed: ", cudaGetErrorString(launch_error));
}
