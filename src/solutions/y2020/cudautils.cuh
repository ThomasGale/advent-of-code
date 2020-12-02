#pragma once

namespace aoc {
namespace y2020 {
namespace cudautils {

int getMaxThreadsPerBlock() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
namespace reducenv {

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