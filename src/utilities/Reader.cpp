// Credit:
// https://github.com/watmough/Advent-of-Code-2018/blob/master/reader.hpp
#include "Reader.h"

#include <sstream>
#include <string>

namespace aoc {
namespace utils {
namespace reader {

std::vector<std::string> read_input(std::istream& ifs) {
    std::vector<std::string> lines;
    std::copy(std::istream_iterator<line>(ifs), std::istream_iterator<line>(),
              std::back_inserter(lines));
    return lines;
}

std::vector<int64_t> read_ints(std::istream& ifs) {
    std::vector<int64_t> lines;
    std::transform(std::istream_iterator<line>(ifs),
                   std::istream_iterator<line>(), std::back_inserter(lines),
                   [&](const line& l) { return std::stol(l); });
    return lines;
}

std::vector<std::string> read_input(const std::string& f) {
    auto ifs = std::ifstream(f, std::ifstream::in);
    if (!ifs) {
        std::cerr << "Unable to open file: " << f << "\n";
        exit(-1);
    }
    return read_input(ifs);
}

std::vector<int64_t> read_ints(const std::string& f) {
    auto ifs = std::ifstream(f, std::ifstream::in);
    if (!ifs) {
        std::cerr << "Unable to open file: " << f << "\n";
        exit(-1);
    }
    return read_ints(ifs);
}

// Split a line into std::string tokens
// See: https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace reader
} // namespace utils
} // namespace aoc
