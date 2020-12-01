#pragma once
#include "default.h"

namespace aoc::y2018::d02 {	

	int findNumDifferentChars(const std::string& str1, const std::string& str2) {
		int numDifferntChars = 0;
		for (auto i = 0; i < str1.size(); ++i) {
			if (str1[i] != str2[i]) ++numDifferntChars;
		}
		return numDifferntChars;
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 2: Inventory Management System ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);

		// Part 1.
		std::map<int, int> dupl_count;
		for(const std::string& line : input_strings) {
			std::map<char, int> char_count;
			std::set<int> dupls;
			std::for_each(line.begin(), line.end(), [&char_count](char letter) { char_count[letter]++; });
			std::for_each(char_count.begin(), char_count.end(), [&dupls](auto& char_count_pair) { dupls.insert(char_count_pair.second); });
			std::for_each(dupls.begin(), dupls.end(), [&dupl_count](int dupl) { dupl_count[dupl]++; });
		}

		int checksum = dupl_count[2] * dupl_count[3];

		std::cout << "1. Checksum of box ids:\n";
		std::cout << checksum << "\n";

		// Part 2.
		std::string commonLetters;
		for (const std::string& line1 : input_strings) {
			for (const std::string& line2 : input_strings) {
				if (findNumDifferentChars(line1, line2) == 1) {
					std::stringstream ss;
					for (auto i = 0; i < line1.size(); ++i) {
						if (line1[i] == line2[i]) ss << line1[i];
					}
					commonLetters = ss.str();
				}
			}
		}

		std::cout << "2. Common letters between two corect box ids:\n";
		std::cout << commonLetters << "\n";
	}
}