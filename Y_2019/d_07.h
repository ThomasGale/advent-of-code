#pragma once
#include "default.h"

namespace aoc::y2019::d07 {

	class IntCodeComputer {
	public:
		IntCodeComputer(std::vector<int> startState) : pState(startState), pProg(0) {};

		int RunProgram(std::vector<int> inputs) {
			int output = 0;
			while (inputs.size() > 0) {
				output = RunProgram(inputs.back());
				inputs.pop_back();
			}
			return output;
		}

		bool IsHalted() {
			return pState[pProg] == 99;
		}

		int RunProgram(int input)
		{
			std::bitset<3> modes;
			int opCode = 0;

			bool inputRead = false;
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
					if (!inputRead) {
						pState[pState[pProg + 1]] = input; inputRead = true;
					} else {
						//std::cout << "Enter input: ";
						//std::cin >> inputStr;
						//pState[pState[pProg + 1]] = std::stoi(inputStr);

						// Pause execution untill new intput is ready.
						return output;
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
	private:
		std::vector<int> pState;
		int pProg = 0;

		inline int getValue(bool valMode, int instruction, const std::vector<int>& program) {
			if (valMode) return instruction;
			else return program[instruction];
		}
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 7: Amplification Circuit ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<int> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoi(input); });

		// Part 1. 
		std::vector<int> phaseSeq{ 0,1,2,3,4 };
		int bestOutput = 0;
		do {
			int currentOutput = 0;
			for (auto phase : phaseSeq) {
				IntCodeComputer comp(inputProgram);
				currentOutput = comp.RunProgram({ currentOutput, phase });
			}
			if (currentOutput > bestOutput) bestOutput = currentOutput;
		} while (std::next_permutation(phaseSeq.begin(), phaseSeq.end()));

		std::cout << "1. Highest signal that can be sent to the thrusters:\n";
		std::cout << bestOutput << "\n";

		// Part 2.
		phaseSeq = { 5, 6, 7, 8, 9 };
		bestOutput = 0;
		do {
			std::vector<IntCodeComputer> amplifiers;
			for (auto _ : phaseSeq) {
				amplifiers.push_back(IntCodeComputer(inputProgram));
			}

			bool initRun = true;
			std::bitset<5> haltState = 0b00000;
			int currentOutput = 0;
			while (haltState != 0b11111) {
				for (auto i = 0; i < phaseSeq.size(); ++i) {
					if (!amplifiers[i].IsHalted()) {
						if (initRun) {
							currentOutput = amplifiers[i].RunProgram({ currentOutput, phaseSeq[i] });
						}
						else {
							currentOutput = amplifiers[i].RunProgram(currentOutput);
						}
					}
					else {
						haltState[i] = 1;
					}
				}
				initRun = false;
			}
			if (currentOutput > bestOutput) bestOutput = currentOutput;
		} while (std::next_permutation(phaseSeq.begin(), phaseSeq.end()));

		std::cout << "2. Highest signal that can be sent to the thrusters:\n";
		std::cout << bestOutput << "\n";
	}
}