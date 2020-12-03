#pragma once

#include <cstring>

#include "Common.hpp"
#include "cudautils.cuh"

namespace aoc {
namespace y2020 {
namespace d03impl {

__global__ void computeTreeCount(uint n, uint lineLen, uint** geology,
                                 uint* treeCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        treeCount[i] += geology[i][(i * 3) % lineLen];
    }
}

} // namespace d03impl

class d03 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d03>();
    }

    void Calculate(std::istream& input) override {
        using namespace d03impl;
        using namespace cudautils::device;
        using namespace cudautils::reduce::count;

        std::vector<std::string> inputStrs = utils::reader::read_input(input);
        uint n = inputStrs.size();
        uint lineLen = inputStrs[0].length();
        uint threads = getThreadsPerBlock(n);
        uint blocks = getNumberOfBlocks(n, threads);

        // Create data structure ready to be passed to cuda.
        uint** geology;
        cudaMallocManaged(&geology, n * sizeof(uint*));
        for (size_t i = 0; i < n; ++i) {
            cudaMallocManaged(&geology[i], lineLen * sizeof(uint));
            for (size_t j = 0; j < lineLen; ++j) {
                geology[i][j] = (inputStrs[i][j] == '#' ? 1 : 0);
            }
        }

        // Compute tree count. (map).
        uint* treeCount;
        cudaMallocManaged(
            &treeCount,
            threads * sizeof(uint)); // result is padded up to thread size
                                      // (this is to enable the reduce method
                                      // lated to not exceed bounds)
        for (size_t i = 0; i < threads; ++i)
            treeCount[i] = 0;
        computeTreeCount<<<blocks, threads>>>(n, lineLen, geology, treeCount);

        cudaDeviceSynchronize();

        // Sum tree count (reduce).
        // Current reduce appoach runs in a single block.
        if (threads < n)
            throw std::runtime_error("Need a different cuda approach.");
        count<<<1, threads / 2>>>(threads, treeCount);

        cudaDeviceSynchronize();

        // Result
        std::cout << treeCount[0] << std::endl;

        // Clean up
        cudaFree(treeCount);
        for (size_t i = 0; i < n; ++i) {
            cudaFree(geology[i]);
        }
        cudaFree(geology);
    }
};

} // namespace y2020
} // namespace aoc
