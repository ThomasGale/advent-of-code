#include "IntCodeComputer.h"

namespace aoc::y2019::intcc {

	IntCodeComputer::IntCodeComputer(const std::string& programString) : pProg(0), relBase(0) {
		std::vector<std::string> inputStrs = aoc::utils::split(programString, ',');
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(pState), [](auto& input) { return std::stoll(input); });
	}

	std::vector<bigint> IntCodeComputer::RunProgram(bigint input)
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
				std::transform(modeStr.rbegin(), modeStr.rend(), modes.begin(), [](char modeC) { return modeC - '0'; }); // It's reversed (mode for first arg is last val in string).
			}

			std::string inputStr;
			switch (opCode)
			{
			case 1: // ADD
				setState(modes[2], getValue(modes[0], pState[pProg + 1]) + getValue(modes[1], pState[pProg + 2]));
				pProg += 4; break;
			case 2: // MUL
				setState(modes[2], getValue(modes[0], pState[pProg + 1]) * getValue(modes[1], pState[pProg + 2]));
				pProg += 4; break;
			case 3: // IN
				if (!inputRead) {
					setState(modes[0], input, 1); // Store Input.
					inputRead = true;
				}
				else {
					return outputs; // The single input has already been processed, return until execution is resumed with new argument.
				}
				pProg += 2; break;
			case 4: // OUT
				outputs.push_back(getValue(modes[0], pState[pProg + 1]));
				pProg += 2; break;
			case 5: // JMP_NE
				pProg = ((getValue(modes[0], pState[pProg + 1]) != 0) ? getValue(modes[1], pState[pProg + 2]) : pProg + 3); break;
			case 6: // JMP_E
				pProg = ((getValue(modes[0], pState[pProg + 1]) == 0) ? getValue(modes[1], pState[pProg + 2]) : pProg + 3); break;
			case 7: // LESS_THAN
				setState(modes[2], getValue(modes[0], pState[pProg + 1]) < getValue(modes[1], pState[pProg + 2]));
				pProg += 4; break;
			case 8: // EQUAL
				setState(modes[2], getValue(modes[0], pState[pProg + 1]) == getValue(modes[1], pState[pProg + 2]));
				pProg += 4; break;
			case 9: // BASE_OFFSET
				relBase += getValue(modes[0], pState[pProg + 1]);
				pProg += 2; break;
			default:
				throw std::runtime_error("Unrecognised opcode");
			}
		}
		return outputs;
	}

	inline bigint IntCodeComputer::getValue(int mode, bigint instruction) {
		switch (mode) {
		case 0: // Absolute Mode
			if (instruction >= bigint(pState.size())) pState.resize(instruction + 1, 0); // Expand Memory if required
			return pState[instruction];
		case 1: // Value Mode
			return instruction;
		case 2: // Relative Mode
			if (relBase + instruction >= bigint(pState.size())) pState.resize(relBase + instruction + 1, 0); // Expand Memory if required
			return pState[relBase + instruction];
		default:
			throw new std::runtime_error("Unsupported Mode");
		}
	}

	inline void IntCodeComputer::setState(int mode, bigint value, int pPOffset) {
		switch (mode) {
		case 0: // Absolute Mode
			if (pState[pProg + pPOffset] >= bigint(pState.size())) pState.resize(pState[pProg + pPOffset] + 1, 0); // Expand Memory if required
			pState[pState[pProg + pPOffset]] = value; break;
		case 2: // Relative Mode
			if (pState[pProg + pPOffset] + relBase >= bigint(pState.size())) pState.resize(pState[pProg + pPOffset] + relBase + 1, 0); // Expand Memory if required
			pState[pState[pProg + pPOffset] + relBase] = value; break;
		default:
			throw new std::runtime_error("Unsupported Mode");
		}
	}
}