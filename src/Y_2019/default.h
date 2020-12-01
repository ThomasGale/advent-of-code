#pragma once

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../utilities/reader.h"
#include "../utilities/utils.hpp"
#include "../utilities/basic_window.h"
#include "../utilities/vec2.h"
#include "../utilities/mat2x2.h"

namespace aoc::y2019 {

	using clock = std::chrono::high_resolution_clock;

	template<typename T>
	inline double Duration(const T& start, const T& stop) {
		return double(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
	}

	template<typename T>
	inline void PrintDuration(const T& start, const T& stop) {
		std::cout << "Calculation duration: " << Duration(start, stop) / 1000 << " ms\n";
	}

	using bigint = long long;
}

