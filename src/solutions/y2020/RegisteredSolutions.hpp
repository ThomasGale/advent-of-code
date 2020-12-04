#pragma once

#include "Solution.hpp"
#include "d01.cu"
#include "d02.cu"
#include "d03.cu"
#include "d04.cu"

namespace aoc {
namespace y2020 {

Solution::Registrar<y2020::d01> r01(2020, 01, "Report Repair");
Solution::Registrar<y2020::d02> r02(2020, 02, "Password Philosophy");
Solution::Registrar<y2020::d03> r03(2020, 03, "Toboggan Trajectory");
Solution::Registrar<y2020::d04> r04(2020, 04, "Passport Processing");

} // namespace y2020
} // namespace aoc
