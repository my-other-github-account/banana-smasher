#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kTile = 16;
constexpr int kRowsPerWork = 64;
constexpr int kRowsPerThread = kRowsPerWork / kTile;
constexpr uint32_t kMul1 = 0x83DCD12Du;

__device__ __constant__ int kInversePermutation[256] = {
    0,32,64,96,128,160,192,224,4,36,68,100,132,164,196,228,
    1,33,65,97,129,161,193,225,5,37,69,101,133,165,197,229,
    8,40,72,104,136,168,200,232,12,44,76,108,140,172,204,236,
    9,41,73,105,137,169,201,233,13,45,77,109,141,173,205,237,
    16,48,80,112,144,176,208,240,20,52,84,116,148,180,212,244,
    17,49,81,113,145,177,209,241,21,53,85,117,149,181,213,245,
    24,56,88,120,152,184,216,248,28,60,92,124,156,188,220,252,
    25,57,89,121,153,185,217,249,29,61,93,125,157,189,221,253,
    2,34,66,98,130,162,194,226,6,38,70,102,134,166,198,230,
    3,35,67,99,131,163,195,227,7,39,71,103,135,167,199,231,
    10,42,74,106,138,170,202,234,14,46,78,110,142,174,206,238,
    11,43,75,107,139,171,203,235,15,47,79,111,143,175,207,239,
    18,50,82,114,146,178,210,242,22,54,86,118,150,182,214,246,
    19,51,83,115,147,179,211,243,23,55,87,119,151,183,215,247,
    26,58,90,122,154,186,218,250,30,62,94,126,158,190,222,254,
    27,59,91,123,155,187,219,251,31,63,95,127,159,191,223,255
};

__device__ __forceinline__ uint32_t branch_code(const int16_t* tile_words, int position) {
    const int pair = position >> 4;
    const int within = position & 15;
    const int word_index = pair * 2 + (within < 8 ? 1 : 0);
    const int shift = 14 - 2 * (within & 7);
    const uint16_t word = static_cast<uint16_t>(tile_words[word_index]);
    return static_cast<uint32_t>((word >> shift) & 3u);
}

__device__ __forceinline__ int decode_parent_index(const int16_t* tile_words, int logical_index) {
    const int position = kInversePermutation[logical_index];
    const int code_group = position >> 3;
    const uint32_t previous = static_cast<uint16_t>(
        tile_words[((code_group + 31) & 31) ^ 1]
    );
    const uint32_t current = static_cast<uint16_t>(tile_words[code_group ^ 1]);
    const uint32_t combined = (previous << 16) | current;
    const int shift = 2 * (7 - (position & 7));
    const uint16_t state = static_cast<uint16_t>((combined >> shift) & 0xFFFFu);
    const uint32_t product = static_cast<uint32_t>(state) * kMul1;
    return static_cast<int>(
        (product & 0xFFu) + ((product >> 8) & 0xFFu) +
        ((product >> 16) & 0xFFu) + ((product >> 24) & 0xFFu)
    );
}

__device__ __forceinline__ float q_value(
    const int16_t* tile_words,
    int logical_index,
    const float* lut
) {
    const int parent = decode_parent_index(tile_words, logical_index);
    return __half2float(__float2half_rn(lut[parent]));
}

__global__ void grouped_forward_kernel(
    const float* x,
    const int32_t* offsets,
    const int32_t* work_experts,
    const int32_t* work_starts,
    const int16_t* packed,
    const float* lut,
    float* output,
    int rows,
    int experts,
    int tiles_k,
    int tiles_m
) {
    __shared__ float q[kThreads];
    __shared__ float x_tile[kRowsPerWork * kTile];
    const int work = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int thread = threadIdx.x;
    const int expert = work_experts[work];
    if (expert < 0 || expert >= experts) return;
    const int start = work_starts[work];
    const int end = offsets[expert + 1];
    const int output_column = tile_m * kTile + (thread & (kTile - 1));
    float total[kRowsPerThread] = {0.0f};
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
        const int64_t tile_offset =
            (((static_cast<int64_t>(expert) * tiles_k + tile_k) * tiles_m + tile_m) * 32);
        q[thread] = q_value(packed + tile_offset, thread, lut);
        #pragma unroll
        for (int ordinal = 0; ordinal < 4; ++ordinal) {
            const int linear = thread + ordinal * kThreads;
            const int local_row = linear / kTile;
            const int column = linear & (kTile - 1);
            const int row = start + local_row;
            x_tile[linear] = row < end && row < rows
                ? x[row * (tiles_k * kTile) + tile_k * kTile + column]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int row_ordinal = 0; row_ordinal < kRowsPerThread; ++row_ordinal) {
            const int local_row = row_ordinal * kTile + (thread >> 4);
            const int row = start + local_row;
            if (row < end && row < rows) {
                const int output_lane = thread & (kTile - 1);
                #pragma unroll
                for (int k = 0; k < kTile; ++k) {
                    total[row_ordinal] = fmaf(
                        x_tile[local_row * kTile + k],
                        q[k * kTile + output_lane],
                        total[row_ordinal]
                    );
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int row_ordinal = 0; row_ordinal < kRowsPerThread; ++row_ordinal) {
        const int row = start + row_ordinal * kTile + (thread >> 4);
        if (row < end && row < rows) {
            output[row * (tiles_m * kTile) + output_column] = total[row_ordinal];
        }
    }
}

__global__ void grouped_grad_input_kernel(
    const float* grad_output,
    const int32_t* offsets,
    const int32_t* work_experts,
    const int32_t* work_starts,
    const int16_t* packed,
    const float* lut,
    float* grad_input,
    int rows,
    int experts,
    int tiles_k,
    int tiles_m
) {
    __shared__ float q[kThreads];
    const int work = blockIdx.x;
    const int tile_k = blockIdx.y;
    const int thread = threadIdx.x;
    const int expert = work_experts[work];
    if (expert < 0 || expert >= experts) return;
    const int row = work_starts[work] + (thread >> 4);
    const int end = offsets[expert + 1];
    const int input_column = tile_k * kTile + (thread & 15);
    float total = 0.0f;
    for (int tile_m = 0; tile_m < tiles_m; ++tile_m) {
        const int64_t tile_offset =
            (((static_cast<int64_t>(expert) * tiles_k + tile_k) * tiles_m + tile_m) * 32);
        q[thread] = q_value(packed + tile_offset, thread, lut);
        __syncthreads();
        if (row < end && row < rows) {
            const int output_base = row * (tiles_m * kTile) + tile_m * kTile;
            const int input_lane = thread & 15;
            #pragma unroll
            for (int m = 0; m < kTile; ++m) {
                total = fmaf(grad_output[output_base + m], q[input_lane * kTile + m], total);
            }
        }
        __syncthreads();
    }
    if (row < end && row < rows) {
        grad_input[row * (tiles_k * kTile) + input_column] = total;
    }
}

__global__ void grouped_grad_lut_kernel(
    const float* grad_output,
    const float* x,
    const int32_t* offsets,
    const int32_t* active_experts,
    const int16_t* packed,
    float* grad_lut,
    int active_count,
    int tiles_k,
    int tiles_m
) {
    const int active_index = blockIdx.x;
    const int tile_k = blockIdx.y;
    const int thread = threadIdx.x;
    if (active_index >= active_count) return;
    const int expert = active_experts[active_index];
    const int start = offsets[expert];
    const int end = offsets[expert + 1];
    const int input_column = tile_k * kTile + (thread >> 4);
    const int output_lane = thread & 15;
    for (int tile_m = 0; tile_m < tiles_m; ++tile_m) {
        const int output_column = tile_m * kTile + output_lane;
        const int64_t tile_offset =
            (((static_cast<int64_t>(expert) * tiles_k + tile_k) * tiles_m + tile_m) * 32);
        const int parent = decode_parent_index(packed + tile_offset, thread);
        float total = 0.0f;
        for (int row = start; row < end; ++row) {
            total = fmaf(
                x[row * (tiles_k * kTile) + input_column],
                grad_output[row * (tiles_m * kTile) + output_column],
                total
            );
        }
        atomicAdd(grad_lut + parent, total);
    }
}

}  // namespace

torch::Tensor grouped_inner_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor packed,
    torch::Tensor lut
) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int rows = static_cast<int>(x.size(0));
    const int experts = static_cast<int>(packed.size(0));
    const int tiles_k = static_cast<int>(packed.size(1));
    const int tiles_m = static_cast<int>(packed.size(2));
    auto output = torch::empty({rows, tiles_m * kTile}, x.options());
    if (rows == 0 || work_experts.numel() == 0) return output;
    const dim3 grid(static_cast<unsigned>(work_experts.numel()), static_cast<unsigned>(tiles_m));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    grouped_forward_kernel<<<grid, kThreads, 0, stream>>>(
        x.data_ptr<float>(), offsets.data_ptr<int32_t>(), work_experts.data_ptr<int32_t>(),
        work_starts.data_ptr<int32_t>(), packed.data_ptr<int16_t>(), lut.data_ptr<float>(),
        output.data_ptr<float>(), rows, experts, tiles_k, tiles_m
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> grouped_inner_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor work_experts,
    torch::Tensor work_starts,
    torch::Tensor active_experts,
    torch::Tensor packed,
    torch::Tensor lut
) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int rows = static_cast<int>(x.size(0));
    const int experts = static_cast<int>(packed.size(0));
    const int tiles_k = static_cast<int>(packed.size(1));
    const int tiles_m = static_cast<int>(packed.size(2));
    auto grad_input = torch::empty_like(x);
    auto grad_lut = torch::zeros_like(lut);
    if (rows == 0 || work_experts.numel() == 0) return {grad_input, grad_lut};
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const dim3 input_grid(static_cast<unsigned>(work_experts.numel()), static_cast<unsigned>(tiles_k));
    grouped_grad_input_kernel<<<input_grid, kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(), offsets.data_ptr<int32_t>(),
        work_experts.data_ptr<int32_t>(), work_starts.data_ptr<int32_t>(),
        packed.data_ptr<int16_t>(), lut.data_ptr<float>(), grad_input.data_ptr<float>(),
        rows, experts, tiles_k, tiles_m
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (active_experts.numel() > 0) {
        const dim3 lut_grid(static_cast<unsigned>(active_experts.numel()), static_cast<unsigned>(tiles_k));
        grouped_grad_lut_kernel<<<lut_grid, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), x.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            active_experts.data_ptr<int32_t>(), packed.data_ptr<int16_t>(), grad_lut.data_ptr<float>(),
            static_cast<int>(active_experts.numel()), tiles_k, tiles_m
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_input, grad_lut};
}
