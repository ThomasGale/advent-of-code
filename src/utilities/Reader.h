// Credit:
// https://github.com/watmough/Advent-of-Code-2018/blob/master/reader.hpp
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace aoc {
namespace utils {
namespace reader {

class line {
    std::string data;

  public:
    friend std::istream& operator>>(std::istream& is, line& l) {
        std::getline(is, l.data);
        return is;
    }
    operator std::string() const { return data; }
};

std::vector<std::string> read_input(std::istream& ifs);

std::vector<int64_t> read_ints(std::istream& ifs);

std::vector<std::string> read_input(const std::string& f);

std::vector<int64_t> read_ints(const std::string& f);

// Split a line into std::string tokens
// See: https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
std::vector<std::string> split(const std::string& s, char delimiter);

} // namespace reader
} // namespace utils
} // namespace aoc
