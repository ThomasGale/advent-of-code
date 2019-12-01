#pragma once
#include "default.h"

namespace aoc::y2018::d03 {

	struct Claim {
		Claim(const std::string& claim_str) {
		};
		Claim(int id=0, int left=0, int top=0, int width=0, int height=0) : Id(id), Left(left), Top(top), Width(width), Height(height) {};
		int Id;
	    int Left, Top;
		int Width, Height;
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 3: No Matter How You Slice It ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);

		// Part 1.
		std::vector<Claim> claims(input_strings.size());
		for (auto& input : input_strings) {

		}


		std::cout << "...\n";
		std::cout << "" << "\n";

		// Part 2.
	}
}