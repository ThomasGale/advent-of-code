#pragma once

#include "Solution.hpp"
#include "d01.hpp"
#include "d02.hpp"
#include "d03.hpp"

namespace aoc {
namespace y2018 {

Solution::Registrar<d01> r01(2018, 01, "Chronal Calibration");
Solution::Registrar<d02> r02(2018, 02, "Inventory Management System");
Solution::Registrar<d03> r03(2018, 03, "No Matter How You Slice It");

} // namespace y2018
} // namespace aoc
