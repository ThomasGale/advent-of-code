#pragma once
#include "default.h"
#include <regex>

namespace aoc::y2018::d03 {

	struct Claim {

		int Id;
		int Left, Top;
		int Width, Height;

		Claim(const std::string& claim_str) {
			std::regex re("#(\\d+) @ (\\d+),(\\d+): (\\d+)x(\\d+)");
			std::smatch match;
			std::regex_search(claim_str, match, re);
			Id = std::stoi(match.str(1));
			Left = std::stoi(match.str(2)); Top = std::stoi(match.str(3));
			Width = std::stoi(match.str(4)); Height = std::stoi(match.str(5));
		};
		Claim(int id=0, int left=0, int top=0, int width=0, int height=0) : Id(id), Left(left), Top(top), Width(width), Height(height) {};

		int Area() const {
			return Width * Height;
		}

		int OverlapArea(const Claim& otherClaim) {
			return (std::min(Left + Width, otherClaim.Left + Width) - std::max(Left, otherClaim.Left))
				* (std::min(Top + Width, otherClaim.Top + Width) - std::max(Top, otherClaim.Top));
		}
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 3: No Matter How You Slice It ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);

		// Part 1.
		std::vector<Claim> claims;
		for (auto& input : input_strings) {
			Claim claim(input);
			claims.push_back(std::move(claim));
		}
	
		long totalDoubleOverlappingArea = 0;
		for (auto i = 0; i < claims.size(); ++i) {
			for (auto j = 0; j < claims.size(); ++j) {
				if (i != j)
					totalDoubleOverlappingArea += claims[i].OverlapArea(claims[j]);
			}
		}

		std::cout << "Total overlapping area:\n";
		std::cout << totalDoubleOverlappingArea / 2 << "\n";

		// Part 2.
	}
}