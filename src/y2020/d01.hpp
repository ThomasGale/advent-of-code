#pragma once

#include "Solution.hpp"

#include <iostream>

namespace aoc {

class d01 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d01>();
    }
    void Calculate(std::istream& input) override {


        
        std::cout << "Running 2020 Day 1 Calculate" << std::endl;

    }
};

} // namespace aoc
