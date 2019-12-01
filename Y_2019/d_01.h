#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d01 {	
	void calculate(std::istream& input) {
		std::cout << "The Tyranny of the Rocket Equation\n";
		std::vector<std::string> input_strings = aoc::read_input(input);

		int total_fuel = std::accumulate(input_strings.begin(), input_strings.end(), 0, [](const int& total_fuel, const std::string& mass_str) {
			int mass = std::stoi(mass_str);
			return total_fuel + ((mass / 3) - 2);
			}
		);

		std::cout << "1. Fuel requirements:\n";
		std::cout << total_fuel << "\n";
	}
}