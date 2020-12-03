#pragma once

namespace aoc {
namespace y2020 {
namespace cudautils {

// Useful for allocating number of threads.
uint nearestPower2Above(uint x) {
    int power = 1;
    while (power < x)
        power *= 2;
    return power;
}

namespace device {

uint getMaxThreadsPerBlock() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

// Generate a sensible number of threads per 1D block given the input size and
// the device capacity
uint getThreadsPerBlock(uint x) {
    return std::min(getMaxThreadsPerBlock(), nearestPower2Above(x));
}

// Simple naive way to quickly figure how may blocks required to fully run x
// parellel ops.
uint getNumberOfBlocks(uint x, uint threadCount) {
    return (x / threadCount) + 1;
}

} // namespace device

// Experimenting with and learning different parellel reduce implementations.
namespace reduce {

namespace count {

// Reduce like operation (just using single block)
// Inpired by:
// https://www.eximiaco.tech/en/2019/06/10/implementing-parallel-reduction-in-cuda/
__global__ void count(uint n, uint* counts) {
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0) {
        if (tid < number_of_threads) {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            if (snd < n) { // Safety first.
                counts[fst] += counts[snd];
            }
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    };
}

} // namespace count

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
namespace mark {

template <class T> __global__ void reduce(T* in, T* out, uint n) {
    extern __shared__ T sdata[];

    uint tid = threadIdx.x;
    uint i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    // TEST
    // out[i] = 43;

    if (i < n) {
        sdata[tid] = in[i] + in[i + blockDim.x];
    }

    __syncthreads();

    // do reduction in shared mem
    // this loop now starts with s = 512 / 2 = 256
    for (uint s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        // out[blockIdx.x] = 42;
        out[blockIdx.x] = sdata[0];
}

} // namespace mark

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
namespace nvidia {

template <class T, uint blockSize>
__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <class T, uint blockSize>
__global__ void reduce6(T* g_idata, T* g_odata, uint n) {
    extern __shared__ T sdata[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * (blockSize * 2) + tid;
    uint gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<T, blockSize>(sdata, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

} // namespace nvidia

} // namespace reduce

namespace semaphore {

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int* lock) {
    while (atomicCAS((int*)lock, 0, 1) != 0)
        ;
}

__device__ void release_semaphore(volatile int* lock) {
    *lock = 0;
    __threadfence();
}
} // namespace semaphore

} // namespace cudautils
} // namespace y2020
} // namespace aoc