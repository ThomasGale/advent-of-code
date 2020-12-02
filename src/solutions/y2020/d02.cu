#pragma once

#include <cstring>

#include "Common.hpp"
#include "cudautils.cuh"

namespace aoc {
namespace y2020 {
namespace d02impl {

class Password {
  public:
    uint min;
    uint max;
    char check;
    char* pswd;
    uint pswdLen;
    bool valid;
};

// Map like operation
__global__ void checkPasswords(uint n, Password* passwords) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        // Part 1
        /*
        int c = 0;
        for (int j = 0; j < passwords[i].pswdLen; ++j) {
            if (passwords[i].pswd[j] == passwords[i].check) {
                ++c;
            }
        }
        if (c >= passwords[i].min && c <= passwords[i].max) {
            passwords[i].valid = true;
        }*/

        // Part 2
        bool first = passwords[i].pswd[passwords[i].min-1] == passwords[i].check;
        bool second = passwords[i].pswd[passwords[i].max-1] == passwords[i].check;
        passwords[i].valid = (!first) != (!second);
    }
}

// Reduce like operation (just using single block)
// Inpired by:
// https://www.eximiaco.tech/en/2019/06/10/implementing-parallel-reduction-in-cuda/
__global__ void countValid(uint n, uint* counts) {
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0) {
        if (tid < number_of_threads) // still alive?
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            if (snd < n) {
                counts[fst] += counts[snd];
            }
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    };
}

} // namespace d02impl

class d02 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d02>();
    }

    void Calculate(std::istream& input) override {
        using namespace d02impl;
        using namespace cudautils::device;
        using namespace cudautils::reduce::mark;

        std::vector<std::string> inputStrs = utils::reader::read_input(input);
        uint n = inputStrs.size();
        int strMax = std::max_element(
                         inputStrs.cbegin(), inputStrs.cend(),
                         [](auto& a, auto& b) { return a.size() < b.size(); })
                         ->length();

        // Create data structure ready to be passed to cuda.
        Password* passwords;
        cudaMallocManaged(&passwords, n * sizeof(Password));
        for (size_t i = 0; i < n; ++i) {
            std::regex re("(.+)-(.+) (.+): (.+)");
            std::smatch match;
            std::regex_search(inputStrs[i], match, re);
            passwords[i].min = std::stoi(match[1]);
            passwords[i].max = std::stoi(match[2]);
            passwords[i].check = char(match[3].str()[0]);
            cudaMallocManaged(&passwords[i].pswd, strMax * sizeof(char));
            strcpy(passwords[i].pswd, match[4].str().c_str());
            passwords[i].pswdLen = match[4].str().length();
            passwords[i].valid = false;
        }

        // Current appoach runs in a single block.
        uint threads = getThreadsPerBlock(n);
        if (threads < n)
            throw std::runtime_error("Need a different cuda approach.");

        // Check Passwords (map).
        checkPasswords<<<1, threads>>>(n, passwords);

        cudaDeviceSynchronize();

        uint* checked;
        cudaMallocManaged(&checked, threads * sizeof(uint));
        for (size_t i = 0; i < threads; ++i) {
            checked[i] = passwords[i].valid;
        }

        // Count valid (reduce).
        countValid<<<1, threads / 2>>>(threads, checked);

        cudaDeviceSynchronize();

        std::cout << checked[0] << std::endl;

        // Clean up
        cudaFree(checked);
        for (size_t i = 0; i < n; ++i) {
            cudaFree(passwords[i].pswd);
        }
        cudaFree(passwords);
    }
};

} // namespace y2020
} // namespace aoc
