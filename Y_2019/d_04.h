#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d04 {	
	void calculate(std::istream& input) {
		std::cout << "--- Day 4: Secure Container ---\n";

		std::vector<int> validPasswords;

		int validCount = 0;
		for (int pw = 124075; pw <= 580769l; ++pw) {
			auto pwStr = std::to_string(pw);

			bool incrOk = true;
			for (auto i = 0; i < pwStr.size() - 1; ++i) {
				if (pwStr[i + 1] < pwStr[i]) incrOk = false;
			}

			bool dup = false;
			for (auto i = 0; i < pwStr.size() - 1; ++i) {
				if (pwStr[i] == pwStr[i + 1]) {
					dup = true;
				}
			}
			if (incrOk && dup) {
				++validCount;
				validPasswords.push_back(pw);
			}
		}

		std::cout << "1. Passwords that meet criteria:\n";
		std::cout << validCount << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}