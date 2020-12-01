#pragma once
#include "default.h"
#include <map>

namespace aoc::y2019::d03 {

	// Map of Coordinates to length of wire.
	std::map<std::tuple<int, int>, int> ComputeCrawledWire(const std::vector<std::string>& wireStr) {
		std::map<std::tuple<int, int>, int> path;
		int x = 0;
		int y = 0;
		int d = 0;

		for (auto& wire_seg_str : wireStr) { // Parse wire_str
			switch (wire_seg_str[0])
			{
			case 'U':
				for (auto step = 0; step < std::stoi(wire_seg_str.substr(1, wireStr.size() - 1)); step++) {
					path[{x, ++y}] = d++;
				}
				break;
			case 'R':
				for (auto step = 0; step < std::stoi(wire_seg_str.substr(1, wireStr.size() - 1)); step++) {
					path[{++x, y}] = d++;
				}
				break;
			case 'D':
				for (auto step = 0; step < std::stoi(wire_seg_str.substr(1, wireStr.size() - 1)); step++) {
					path[{x, --y}] = d++;
				}
				break;
			case 'L':
				for (auto step = 0; step < std::stoi(wire_seg_str.substr(1, wireStr.size() - 1)); step++) {
					path[{--x, y}] = d++;
				}
				break;
			default:
				throw std::runtime_error("Bad command");
			}
		}
		return path;
	}

	int Manhat(int x, int y) {
		return std::abs(x) + std::abs(y);
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 3: Crossed Wires ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);
		std::vector<std::string> wire1_str = aoc::utils::split(input_strings[0], ',');
		std::vector<std::string> wire2_str = aoc::utils::split(input_strings[1], ',');

		auto wire1 = ComputeCrawledWire(wire1_str);
		auto wire2 = ComputeCrawledWire(wire2_str);

		std::set<std::tuple<int, int>> wire1Keys;
		std::set<std::tuple<int, int>> wire2Keys;

		for (auto it = wire1.begin(); it != wire1.end(); ++it) {
			wire1Keys.insert(it->first);
		}
		for (auto it = wire2.begin(); it != wire2.end(); ++it) {
			wire2Keys.insert(it->first);
		}

		std::set<std::tuple<int, int>> intersections;
		std::set_intersection(wire1Keys.begin(), wire1Keys.end(), wire2Keys.begin(), wire2Keys.end(), std::inserter(intersections, intersections.begin()));

		// Part 1.
		std::vector<int> manhatDistances(intersections.size());
		std::transform(intersections.begin(), intersections.end(), manhatDistances.begin(), [](const std::tuple<int, int>& loc) { return Manhat(std::get<0>(loc), std::get<1>(loc)); });
		int closestCrossingDist = *std::min_element(manhatDistances.begin(), manhatDistances.end());

		std::cout << "1. Manhattan distance from the central port to the closest intersection:\n";
		std::cout << closestCrossingDist << "\n";

		// Part 2.
		std::vector<int> crossingDistances(intersections.size());
		std::transform(intersections.begin(), intersections.end(), crossingDistances.begin(), [&wire1, &wire2](const std::tuple<int, int>& loc) { return wire1[loc] + wire2[loc] + 2; }); // Add 2 because of first step from 0 not counted.
		int closestCrawledCrossingDist = *std::min_element(crossingDistances.begin(), crossingDistances.end());

		std::cout << "2. Fewest combined steps the wires must take to reach an intersection:\n";
		std::cout << closestCrawledCrossingDist << "\n";
	}
}