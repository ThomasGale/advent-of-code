#include "Solution.hpp"

namespace aoc {

std::map<std::string, Solution::SolPtr>& Solution::registry() {
    static std::map<std::string, Solution::SolPtr> impl;
    return impl;
}

} // namespace aoc
