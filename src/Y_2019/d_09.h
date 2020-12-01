#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d09 {
	using namespace aoc::y2019::intcc;

	void calculate(std::istream& input) {
		std::cout << "--- Day 9: Sensor Boost ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});

		IntCodeComputer comp1(inputStr);
		auto output = comp1.RunProgram(1);
		std::cout << "1. BOOST keycode :\n";
		std::cout << output.front() << "\n";

		IntCodeComputer comp2(inputStr);
		auto start = clock::now();
		output = comp2.RunProgram(2);
		auto stop = clock::now();

		std::cout << "2. Coordinates of the distress signal:\n";
		std::cout << output.front() << "\n";
		PrintDuration(start, stop);
	}
}