#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d04 {

	void calculate(std::istream& input) {
		std::cout << "--- Day 4: Secure Container ---\n";

		std::vector<int> validPasswords;

		int validCount = 0;
		for (int pw = 124075; pw <= 580769; ++pw) {
			//pw = 144555;
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
				validPasswords.push_back(pw);
			}
		}

		std::cout << "2. Passwords that meet criteria:\n";
		std::cout << validCount << "\n";
	}
}