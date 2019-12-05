#pragma once
#include "default.h"
#include <bitset>
#include <set>

namespace aoc::y2019::d05 {

	inline int getValue(bool valMode, int instruction, const std::vector<int>& program) {
		if (valMode) return instruction;
		else return program[instruction];
	}

	void RunProgram(std::vector<int>& programState)
	{
		int pProg = 0;
		std::bitset<3> modes;
		int opCode = 0;

		while (true) {
			if (programState[pProg] == 99) break; // Terminate

			std::string opCodeStr(std::to_string(programState[pProg])); // Read instruction as string
			if (opCodeStr.size() == 1) {
				opCode = std::stoi(opCodeStr);
				modes = 0b000;
			}
			else {
				opCode = std::stoi(opCodeStr.substr(opCodeStr.size() - 2));
				modes = std::bitset<3>(opCodeStr.substr(0, opCodeStr.size() - 2));
			}

			std::string input;
			switch (opCode)
			{
			case 1:
				programState[programState[pProg + 3]] = getValue(modes[0], programState[pProg + 1], programState) + getValue(modes[1], programState[pProg + 2], programState);
				pProg += 4;
				break;
			case 2:
				programState[programState[pProg + 3]] = getValue(modes[0], programState[pProg + 1], programState) * getValue(modes[1], programState[pProg + 2], programState);
				pProg += 4;
				break;
			case 3:
				std::cout << "Enter int: ";
				std::cin >> input;
				programState[programState[pProg + 1]] = std::stoi(input);
				pProg += 2;
				break;
			case 4:
				std::cout << getValue(modes[0], programState[pProg + 1], programState);
				pProg += 2;
				break;
			default:
				throw std::runtime_error("Unrecognised opcode");
			}
		}
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 5: Sunny with a Chance of Asteroids ---\n";
		std::string input_str(std::istreambuf_iterator<char>(input), {});
		std::stringstream ss(input_str);
		std::vector<int> inputProgram;
		while (ss.good()) {
			std::string command;
			std::getline(ss, command, ',');
			inputProgram.push_back(std::stoi(command));
		}

		std::cout << "1. Diagnostic Code:\n";
		RunProgram(inputProgram);
	}
}