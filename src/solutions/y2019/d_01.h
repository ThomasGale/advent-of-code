#pragma once
#include "default.h"

namespace aoc::y2019::d01 {	

	int computeFuelToMoveMass(int mass)
	{
		return std::max(0, (mass / 3) - 2);
	}

	int computeFuelToMoveFuel(int fuel) {
		if (fuel <= 0) return 0;
		return fuel + computeFuelToMoveFuel(computeFuelToMoveMass(fuel));
	}

	void calculate(std::istream& input) {
		std::cout << "The Tyranny of the Rocket Equation\n";
		std::vector<std::string> input_strings = aoc::utils::reader::read_input(input);
		
		// Part 1.
		int total_fuel = std::accumulate(input_strings.begin(), input_strings.end(), 0, [](const int& total_fuel, const std::string& mass_str) {
			int mass = std::stoi(mass_str);
			return total_fuel + computeFuelToMoveMass(mass);
			}
		);

		std::cout << "1. Basic fuel requirements:\n";
		std::cout << total_fuel << "\n";

		// Part 2.
		total_fuel = std::accumulate(input_strings.begin(), input_strings.end(), 0, [](const int& total_fuel, const std::string& mass_str) {
			int mass = std::stoi(mass_str);
			return total_fuel + computeFuelToMoveFuel(computeFuelToMoveMass(mass));
			}
		);

		std::cout << "2. Sum of fuel requirements:\n";
		std::cout << total_fuel << "\n";
	}
}