#pragma once
#include "default.h"

namespace aoc::y2019::d07 {

	inline int getValue(bool valMode, int instruction, const std::vector<int>& program) {
		if (valMode) return instruction;
		else return program[instruction];
	}

	// Input is fed to any input requests in the program, return the last output (all outputs are printed)
	int RunProgram(std::vector<int> pState, int firstInput, int secondInput)
	{
		int pProg = 0;
		std::bitset<3> modes;
		int opCode = 0;

		// IO Hacking
		bool firstInputRead = false;
		bool secondInputRead = false;
		int output = 0;
		while (true) {
			if (pState[pProg] == 99) break; // Terminate

			std::string opCodeStr(std::to_string(pState[pProg])); // Read instruction as string
			if (opCodeStr.size() == 1) {
				opCode = std::stoi(opCodeStr);
				modes = 0b000;
			}
			else {
				opCode = std::stoi(opCodeStr.substr(opCodeStr.size() - 2));
				modes = std::bitset<3>(opCodeStr.substr(0, opCodeStr.size() - 2));
			}

			std::string inputStr;
			switch (opCode)
			{
			case 1:
				pState[pState[pProg + 3]] = getValue(modes[0], pState[pProg + 1], pState) + getValue(modes[1], pState[pProg + 2], pState);
				pProg += 4;
				break;
			case 2:
				pState[pState[pProg + 3]] = getValue(modes[0], pState[pProg + 1], pState) * getValue(modes[1], pState[pProg + 2], pState);
				pProg += 4;
				break;
			case 3:
				if (!firstInputRead) {
					pState[pState[pProg + 1]] = firstInput; firstInputRead = true;
				}
				else if (!secondInputRead) {
					pState[pState[pProg + 1]] = secondInput; secondInputRead = true;
				}
				else {
					std::cout << "Enter input: ";
					std::cin >> inputStr;
					pState[pState[pProg + 1]] = std::stoi(inputStr);
				}
				pProg += 2;
				break;
			case 4:
				output = getValue(modes[0], pState[pProg + 1], pState);
				//std::cout << output;
				pProg += 2;
				break;
			case 5:
				pProg = ((getValue(modes[0], pState[pProg + 1], pState) != 0) ? getValue(modes[1], pState[pProg + 2], pState) : pProg + 3);
				break;
			case 6:
				pProg = ((getValue(modes[0], pState[pProg + 1], pState) == 0) ? getValue(modes[1], pState[pProg + 2], pState) : pProg + 3);
				break;
			case 7:
				pState[pState[pProg + 3]] = getValue(modes[0], pState[pProg + 1], pState) < getValue(modes[1], pState[pProg + 2], pState);
				pProg += 4;
				break;
			case 8:
				pState[pState[pProg + 3]] = getValue(modes[0], pState[pProg + 1], pState) == getValue(modes[1], pState[pProg + 2], pState);
				pProg += 4;
				break;
			default:
				throw std::runtime_error("Unrecognised opcode");
			}
		}
		return output; // Return the last output.
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 7: Amplification Circuit ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<int> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoi(input); });

		// Phase sequence tets
		std::vector<int> phaseSeqTest{ 0,1,2,3,4 };

		int bestOutput = 0;
		do {
			int currentOutput = 0;
			for (auto phase : phaseSeqTest) {
				currentOutput = RunProgram(inputProgram, phase, currentOutput);
			}
			if (currentOutput > bestOutput) bestOutput = currentOutput;
		} while (std::next_permutation(phaseSeqTest.begin(), phaseSeqTest.end()));

		std::cout << "1. Highest signal that can be sent to the thrusters:\n";
		std::cout << bestOutput << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}