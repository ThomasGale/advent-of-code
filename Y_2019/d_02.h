#pragma once
#include "default.h"

#include <sstream>

namespace aoc::y2019::d02 {

	int RunProgram(std::vector<int>& programState)
	{
		int pProg = 0;
		int opCode = 0;
		int pI1 = 0;
		int pI2 = 0;
		int pOut = 0;

		while (true) {
			opCode = programState[pProg++];
			if (opCode == 99) break;
			pI1 = programState[pProg++];
			pI2 = programState[pProg++];
			pOut = programState[pProg++];

			switch (opCode)
			{
			case 1:
				programState[pOut] = programState[pI1] + programState[pI2];
				break;
			case 2:
				programState[pOut] = programState[pI1] * programState[pI2];
				break;
			default:
				throw std::runtime_error("Unrecognised opcode");
			}
		}
		return programState[0];
	}

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

		// Setup - part of puzzle.
		std::vector<int> part1ProgramState(inputProgram);
		part1ProgramState[1] = 12;
		part1ProgramState[2] = 2;

		int output = RunProgram(part1ProgramState);

		std::cout << "1. Program Output at Position 0:\n";
		std::cout << output << "\n";

		// Part 2.
		// Brute search 
		int noun = 0;
		int verb = 0;
		for (noun = 0; noun < 100; ++noun) {
			for (verb = 0; verb < 100; ++verb) {
				std::vector<int> part2ProgramState(inputProgram);
				part2ProgramState[1] = noun;
				part2ProgramState[2] = verb;
				int output = RunProgram(part2ProgramState);
				if (output == 19690720)
				{
					std::cout << "2. Noun and Verb Found:\n";
					std::cout << noun << " and " << verb << "\n";
					std::cout << "Ouput: " << 100 * noun + verb << "\n";
				}
			}
		}



	}
}