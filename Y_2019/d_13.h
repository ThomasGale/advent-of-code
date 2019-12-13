#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d13 {	
	using bigint = long long;
	using namespace aoc::y2019::intcc;
	using Loc = std::tuple<int, int>;

	void calculate(std::istream& input) {
		std::cout << " Day 13 \n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<bigint> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoll(input); });

		IntCodeComputer gameTest1(inputProgram);
		std::vector<bigint> rawGameOutput;
		rawGameOutput = gameTest1.RunProgram();

		std::map<Loc, int> gameOutput;
		int numBlocks = 0;
		for (auto i = 0; i < rawGameOutput.size()-2; i+=3) {
			gameOutput[{rawGameOutput[i], rawGameOutput[i + 1]}] = rawGameOutput[i + 2];
			if (rawGameOutput[i + 2] == 2) ++numBlocks;
		}
		std::cout << "1. Blocks on screen when game exits :\n";
		std::cout << numBlocks << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}