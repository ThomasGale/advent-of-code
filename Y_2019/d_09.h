#pragma once
#include "default.h"

namespace aoc::y2019::d09 {	

	using bigint = long long;

	class IntCodeComputer {
	public:
		IntCodeComputer(const std::vector<bigint>& startState) : pProg(0), relBase(0) {
			pState = std::vector<bigint>(startState.size() * 100);
			std::copy(startState.begin(), startState.end(), pState.begin());
		};

		std::vector<bigint> RunProgram(std::vector<bigint> inputs) {
			std::vector<bigint> totalOutputs;
			for (auto input : inputs) {
				auto outputs = RunProgram(input);
				totalOutputs.insert(totalOutputs.begin(), outputs.begin(), outputs.end());
			}
			return totalOutputs;
		}

		bool IsHalted() {
			return pState[pProg] == 99;
		}

		std::vector<bigint> RunProgram(bigint input = 0) // Run program optionally inserting single input. Will return if program needs more arguments or is complete.
		{
			std::vector<int> modes(3, 0);
			int opCode = 0;

			bool inputRead = false;
			std::vector<bigint> outputs;
			while (true) {
				if (pState[pProg] == 99) break; // Terminate

				std::string opCodeStr(std::to_string(pState[pProg])); // Read instruction as string
				modes = { 0,0,0 }; // Reset modes.
				if (opCodeStr.size() == 1) {
					opCode = std::stoi(opCodeStr);
				}
				else {
					opCode = std::stoi(opCodeStr.substr(opCodeStr.size() - 2));
					std::string modeStr = opCodeStr.substr(0, opCodeStr.size() - 2);
					std::transform(modeStr.rbegin(), modeStr.rend(), modes.begin(), [](char modeC) { return modeC - '0'; }); // It's reversed.
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
					}
					else {
						return outputs; // The single input has already been processed, return until execution is resumed with new argument.
					}
					pProg += 2;
					break;
				case 4:
					outputs.push_back(getValue(modes[0], pState[pProg + 1], pState));
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
				case 9:
					relBase += getValue(modes[0], pState[pProg + 1], pState);
					pProg += 2;
					break;
				default:
					throw std::runtime_error("Unrecognised opcode");
				}
			}
			return outputs;
		}
	private:
		std::vector<bigint> pState;
		int pProg = 0;
		int relBase = 0;

		inline bigint getValue(int mode, bigint instruction, const std::vector<bigint>& program) {
			switch (mode) {
			case 0: // Absolute Mode
				return program[instruction];
			case 1: // Value Mode
				return instruction;
			case 2: // Relative Mode
				return program[relBase + instruction];
			default:
				throw new std::runtime_error("Unrecognised Mode");
			}
		}
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 9: Sensor Boost ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<bigint> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoll(input); });

		IntCodeComputer test(inputProgram);

		auto output = test.RunProgram(1);
		bool halted = test.IsHalted();
		std::cout << "1. BOOST keycode :\n";
		std::cout << output.front() << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}