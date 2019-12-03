#pragma once
#include "default.h"

namespace aoc::y2019::d03 {

	struct Location {
		Location(int x, int y) : X(x), Y(y) {};
		int X, Y;
	};

	struct Wire {
		std::vector<Location> segments;
		Wire(std::vector<std::string> wire_str) {

			// Track current location
			Location current(0, 0);

			// Build up segments
			segments = std::vector<Location>{ current };
		
			// Parse wire_str
			for (auto& wire_seg_str : wire_str) {

				switch (wire_seg_str[0])
				{
				case 'U':
					current.Y += std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'R':
					current.X += std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'D':
					current.Y -= std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'L':
					current.X -= std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				default:
					throw std::runtime_error("Bad command");
				}

				segments.push_back(current);
			}




		}
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 3: Crossed Wires ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);

		std::vector<std::string> wire1_str = aoc::utils::split(input_strings[0], ',');
		std::vector<std::string> wire2_str = aoc::utils::split(input_strings[1], ',');

		Wire wire1(wire1_str);
		Wire wire2(wire2_str);

		std::cout << "1. ... :\n";
		std::cout << "" << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}