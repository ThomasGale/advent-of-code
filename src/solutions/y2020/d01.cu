#pragma once

#include "Common.hpp"
#include "cudautils.cuh"

namespace aoc {
namespace y2020 {

namespace d01impl {
__global__ void sumMatchMul(int n, uint* in, int target, uint* ans) {
    using namespace cudautils::semaphore;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        for (int j = i; j < n; ++j) {
            for (int k = j; k < n; ++k) {
                if (in[i] + in[j] + in[k] == target) {
                    __syncthreads();
                    if (threadIdx.x == 0)
                        acquire_semaphore(&sem);
                    __syncthreads();

                    // Single Thread here
                    *ans = in[i] * in[j] * in[k];

                    __threadfence();
                    __syncthreads();
                    if (threadIdx.x == 0)
                        release_semaphore(&sem);
                    __syncthreads();
                }
            }
        }
    }
}
} // namespace d01impl

class d01 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d01>();
    }

    void Calculate(std::istream& input) override {
        using namespace d01impl;
        int target = 2020;
        std::vector<std::string> inputStrs = utils::reader::read_input(input);

        // Init data.
        uint n = inputStrs.size();
        uint* in;
        cudaMallocManaged(&in, n * sizeof(uint));

        for (int i = 0; i < n; ++i) {
            in[i] = uint(std::stoi(inputStrs[i]));
        }

        // Get cuda properies for the current device.
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        uint blockSize = prop.maxThreadsPerBlock;
        uint numBlocks = (n + blockSize - 1) / blockSize;

        // Run
        uint* result;
        cudaMallocManaged(&result, sizeof(uint));
        *result = 0;
        sumMatchMul<<<numBlocks, blockSize>>>(n, in, target, result);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Part 1
        std::cout << *result << std::endl;

        cudaFree(in);
        cudaFree(result);
    }
};

} // namespace y2020
} // namespace aoc
