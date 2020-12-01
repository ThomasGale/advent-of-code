#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d04 {

	void calculate(std::istream& input) {
		std::cout << "--- Day 4: Secure Container ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);
		std::vector<std::string> passwordRangeStr = aoc::utils::split(input_strings[0], '-');
		int lowerRange = std::stoi(passwordRangeStr[0]);
		int upperRange = std::stoi(passwordRangeStr[1]);

		int validCount = 0;
		for (int pw = lowerRange; pw <= upperRange; ++pw) {
			auto pwStr = std::to_string(pw);

			bool incrOk = true;
			for (auto i = 0; i < pwStr.size() - 1; ++i) {
				if (pwStr[i + 1] < pwStr[i]) incrOk = false;
			}

			std::map<int, char> pairLocations;
			for (auto i = 0; i < pwStr.size() - 1; ++i) {
				if (pwStr[i] == pwStr[i + 1]) {
					pairLocations[i] = pwStr[i];
				}
			}

			bool isolatedPair = false;
			for (auto pairLoc : pairLocations) {
				if ((pairLocations.find(pairLoc.first - 1) == pairLocations.end()) && (pairLocations.find(pairLoc.first + 1) == pairLocations.end())) {
					isolatedPair = true;
				}
			}

			if (incrOk && isolatedPair) {
				++validCount;
			}
		}

		std::cout << "2. Passwords that meet criteria:\n";
		std::cout << validCount << "\n";
	}
}