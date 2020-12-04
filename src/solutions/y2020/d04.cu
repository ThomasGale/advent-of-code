#pragma once

#include <cstring>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "Common.hpp"
#include "cudautils.cuh"

namespace aoc {
namespace y2020 {
namespace d04impl {

class Passport {
  public:
    uint byr = 0;
    uint iyr = 0;
    uint eyr = 0;
    uint hgt = 0;
    uint hcl = 0;
    uint ecl = 0;
    uint pid = 0;
    uint cid = 0;
};

class PassportValid : public thrust::unary_function<Passport, uint> {
  public:
    __host__ __device__ uint operator()(const Passport& passport) {
        return passport.byr && passport.iyr && passport.eyr && passport.hgt &&
               passport.hcl && passport.ecl && passport.pid;
    }
};

} // namespace d04impl

class d04 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d04>();
    }

    void Calculate(std::istream& input) override {
        using namespace d04impl;
        using namespace cudautils::sanity;
        using namespace cudautils::device;
        using namespace cudautils::reduce::count;

        // Experimenting with the thrust library today (encapsulates some of the
        // cuda malloc).
        std::vector<std::string> inputStrs = utils::reader::read_input(input);
        thrust::host_vector<Passport> passports;

        // Boring shit.
        int i = 0;
        std::regex re("(.+:.+)+");
        while (i < inputStrs.size()) {
            Passport p;
            while (!inputStrs[i].empty()) {
                std::vector<std::string> attributes =
                    utils::reader::split(inputStrs[i], ' ');
                for (auto& attribute : attributes) {
                    std::vector<std::string> keyVal =
                        utils::reader::split(attribute, ':');
                    if (keyVal[0] == "byr")
                        p.byr = 1;
                    if (keyVal[0] == "iyr")
                        p.iyr = 1;
                    if (keyVal[0] == "eyr")
                        p.eyr = 1;
                    if (keyVal[0] == "hgt")
                        p.hgt = 1;
                    if (keyVal[0] == "hcl")
                        p.hcl = 1;
                    if (keyVal[0] == "ecl")
                        p.ecl = 1;
                    if (keyVal[0] == "pid")
                        p.pid = 1;
                    if (keyVal[0] == "cid")
                        p.cid = 1;
                }
                ++i;
            }
            passports.push_back(p);
            ++i;
        }

        // Transfer data to cuda device
        thrust::device_vector<Passport> cudaPassports = passports;

        // Compute valid vector.
        PassportValid isValidFunc;
        thrust::device_vector<uint> valid(cudaPassports.size());
        thrust::transform(cudaPassports.cbegin(), cudaPassports.cend(),
                          valid.begin(), isValidFunc);

        // Sum num valid.
        thrust::plus<uint> binOp;
        uint numValid = thrust::reduce(valid.cbegin(), valid.cend(), 0, binOp);

        // Result
        std::cout << numValid << std::endl;

        // Final check.
        checkCuda();
    }
}; // namespace y2020

} // namespace y2020
} // namespace aoc
