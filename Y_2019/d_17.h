#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d17 {	
	using namespace aoc::y2019::intcc;
	using namespace aoc::utils;

	void calculate(std::istream& input) {
		std::cout << "--- Day 17: Set and Forget ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});

		// Part 1.
		aoc::utils::BasicWindow window(71, 51);
		IntCodeComputer asciiProg1(inputStr);
		auto currentView = asciiProg1.RunProgram();

		int x = 0, y = 0;
		std::set<Vec2, Vec2LessComp> scaffold;
		for (auto code : currentView) {
			if (code == 35) {
				window.SetChar(y, x, '#');
				scaffold.insert(Vec2(x, y));
			}
			else if (code == 46) {
				window.SetChar(y, x, '.');
			}

			if (code == 10) {
				x = 0;
				++y;
			}
			else {
				++x;
			}
		}
		window.Update();

		int alignementParametersSum = 0;
		for (auto loc : scaffold) {
			auto left = scaffold.find(Vec2(loc.X - 1, loc.Y));
			auto right = scaffold.find(Vec2(loc.X + 1, loc.Y));
			auto up = scaffold.find(Vec2(loc.X, loc.Y - 1));
			auto down = scaffold.find(Vec2(loc.X, loc.Y + 1));

			if (left != scaffold.end() && right != scaffold.end() &&
				up != scaffold.end() && down != scaffold.end()) {
				alignementParametersSum += loc.X * loc.Y;
			}
		}

		std::cout << "1. Sum of the alignment parameters:\n";
		std::cout << alignementParametersSum << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}