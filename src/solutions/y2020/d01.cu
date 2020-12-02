#pragma once

#include "Common.hpp"

namespace aoc {
namespace y2020 {

__global__ void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

class d01 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d01>();
    }

    void Calculate(std::istream& input) override {
        std::vector<std::string> inputStrs = utils::reader::read_input(input);
        std::vector<int> inputNums(inputStrs.size());

        int N = 1'000'000;
        float *x, *y;

        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&x, N * sizeof(float));
        cudaMallocManaged(&y, N * sizeof(float));

        // initialize x and y arrays on the host
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        dim3 blockSize(256, 1, 1);
        dim3 gridSize(8, 1, 1);

        // int numBlocks = (N + blockSize - 1) / blockSize;

        // Run kernel on 1M elements on the GPU
        add<<<gridSize, blockSize>>>(N, x, y);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Check for errors (all values should be 3.0f)
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(y[i] - 3.0f));
        std::cout << "Max error: " << maxError << std::endl;

        // Free memory
        cudaFree(x);
        cudaFree(y);
    }
};

} // namespace y2020
} // namespace aoc
