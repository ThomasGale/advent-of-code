#pragma once
#include "default.h"

#include <sstream>

namespace aoc::y2019::d02 {

	void calculate(std::istream& input) {
		std::cout << "--- Day 2: 1202 Program Alarm ---\n";
		std::string input_str(std::istreambuf_iterator<char>(input), {});
		std::stringstream ss(input_str);
		std::vector<int> inputProgram;
		while (ss.good()) {
			std::string command;
			std::getline(ss, command, ',');
			inputProgram.push_back(std::stoi(command));
		}

		// Part 1.
		int pProg = 0;
		int opCode = 0;
		int pI1 = 0;
		int pI2 = 0;
		int pOut = 0;

		// Setup - part of puzzle.
		inputProgram[1] = 12;
		inputProgram[2] = 2;

		while (true) {
			opCode = inputProgram[pProg++];
			if (opCode == 99) break;
			pI1 = inputProgram[pProg++];
			pI2 = inputProgram[pProg++];
			pOut = inputProgram[pProg++];

			switch (opCode)
			{
			case 1:
				inputProgram[pOut] = inputProgram[pI1] + inputProgram[pI2];
				break;
			case 2:
				inputProgram[pOut] = inputProgram[pI1] * inputProgram[pI2];
				break;
			default:
				throw std::runtime_error("Unrecognised opcode");
			}
		}

		std::cout << "1. Program Output at Position 0:\n";
		std::cout << inputProgram[0] << "\n";
	}
}