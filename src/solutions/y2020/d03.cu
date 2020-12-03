#pragma once

#include <cstring>

#include "Common.hpp"
#include "cudautils.cuh"

namespace aoc {
namespace y2020 {
namespace d03impl {

using ull = unsigned long long;

class Slope {
  public:
    uint right;
    uint down;
};

__global__ void computeTreeCount(uint n, uint lineLen, uint** geology,
                                 uint numSlopes, Slope* slopes,
                                 uint** treeCounts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        for (size_t s = 0; s < numSlopes; ++s) {
            if ((i % slopes[s].down) == 0) {
                treeCounts[s][i] +=
                    geology[i]
                           [((i / slopes[s].down) * slopes[s].right) % lineLen];
            }
        }
    }
}

__global__ void sumSlopes(uint threads, uint numSlopes, uint** treeCounts) {
    using namespace cudautils::reduce::count;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int s = index; s < numSlopes; s += stride) {
        sumAdd<<<1, threads / 2>>>(threads, treeCounts[s]);
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
        using namespace cudautils::sanity;
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

        Slope* slopes;
        int numSlopes = 5;
        cudaMallocManaged(&slopes, numSlopes * sizeof(Slope));
        slopes[0].right = 1;
        slopes[0].down = 1;
        slopes[1].right = 3;
        slopes[1].down = 1;
        slopes[2].right = 5;
        slopes[2].down = 1;
        slopes[3].right = 7;
        slopes[3].down = 1;
        slopes[4].right = 1;
        slopes[4].down = 2;

        // Compute tree counts. (map).
        uint** treeCounts;
        cudaMallocManaged(
            &treeCounts,
            numSlopes * sizeof(uint*)); // result is padded up to thread size
                                        // (this is to enable the reduce method
                                        // lated to not exceed bounds)
        for (size_t s = 0; s < numSlopes; ++s) {
            cudaMallocManaged(&treeCounts[s], threads * sizeof(uint));
            for (size_t i = 0; i < threads; ++i) {
                treeCounts[s][i] = 0;
            }
        }

        computeTreeCount<<<blocks, threads>>>(n, lineLen, geology, numSlopes, slopes,
                                         treeCounts);

        cudaDeviceSynchronize();

        // Sum tree count (reduce).
        // Current reduce appoach runs in a single block.
        if (threads < n)
            throw std::runtime_error("Need a different cuda approach.");

        sumSlopes<<<1, numSlopes>>>(threads, numSlopes, treeCounts);

        cudaDeviceSynchronize();

        // Sum mul of final counts (more reduce)
        ull* muls;
        cudaMallocManaged(&muls, numSlopes * sizeof(ull));
        for (size_t i = 0; i < numSlopes; ++i) {
            muls[i] = treeCounts[i][0];
        }
        sumMul<<<1, getThreadsPerBlock(numSlopes) / 2>>>(numSlopes, muls);

        cudaDeviceSynchronize();

        // Result
        std::cout << muls[0] << std::endl;

        // Clean up
        cudaFree(muls);
        for (size_t s = 0; s < numSlopes; ++s) {
            cudaFree(treeCounts[s]);
        }
        cudaFree(treeCounts);
        for (size_t i = 0; i < n; ++i) {
            cudaFree(geology[i]);
        }
        cudaFree(geology);

        // Final check.
        checkCuda();
    }
};

} // namespace y2020
} // namespace aoc
