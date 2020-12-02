#pragma once

#include "Solution.hpp"
#include "d01.cu"
#include "d02.cu"

namespace aoc {
namespace y2020 {

Solution::Registrar<y2020::d01> r01(2020, 01, "Report Repair");
Solution::Registrar<y2020::d02> r02(2020, 02, "Password Philosophy");

} // namespace y2020
} // namespace aoc
