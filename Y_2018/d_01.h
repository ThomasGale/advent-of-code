#pragma once
#include "default.h"

namespace aoc::y2018::d01 {	
	void calculate(std::istream& input) {
		std::cout << "Chronal Calibration\n";
		std::vector<std::string> input_strings = aoc::read_input(input);

		int freq_sum = std::accumulate(input_strings.begin(), input_strings.end(), 0, [](const int& freq_sum, const std::string& freq_change) {
			return freq_sum + std::stoi(freq_change);
			});

		std::cout << "Calculated:\n";
		std::cout << freq_sum << "\n";
	}
}