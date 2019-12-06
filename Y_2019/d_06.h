#pragma once
#include "default.h"

namespace aoc::y2019::d06 {

	void calculate(std::istream& input) {
		std::cout << "--- Day 6: Universal Orbit Map ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		std::map<std::string, std::string> orbitMap; // Key orbits value
		for (auto& inputStr : inputStrs) {
			std::vector<std::string> orbitsStr = aoc::utils::split(inputStr, ')');
			orbitMap[orbitsStr[1]] = orbitsStr[0];
		}

		int totalOrbits = 0;
		for (const auto& orbit : orbitMap) {
			++totalOrbits;
			std::string currentParent = orbit.second;
			while (currentParent != "COM") {
				currentParent = orbitMap[currentParent];
				++totalOrbits;
			}
		}

		std::cout << "1. Total number of direct and indirect orbits :\n";
		std::cout << totalOrbits << "\n";
	}
}