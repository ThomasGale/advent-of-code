// Credit: https://github.com/watmough/Advent-of-Code-2018/blob/master/reader.hpp
#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>

namespace aoc::utils {

	class line {
		std::string data;
	public:
		friend std::istream& operator>>(std::istream& is, line& l) {
			std::getline(is, l.data);
			return is;
		}
		operator std::string() const { return data; }
	};

	std::vector<std::string> read_input(std::istream& ifs);

	std::vector<int64_t> read_ints(std::istream& ifs);

	std::vector<std::string> read_input(const std::string& f);

	std::vector<int64_t> read_ints(const std::string& f);
}
