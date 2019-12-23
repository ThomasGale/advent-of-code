#pragma once
#include "default.h"

namespace aoc::y2019::intcc {
	using bigint = long long;

	class IntCodeComputer {
	public:
		IntCodeComputer(const std::string& programString);
		IntCodeComputer(const std::vector<bigint>& startState) : pState(startState), pProg(0), relBase(0) {};

		bigint StepProgram(bigint input = 0);

		std::vector<bigint> RunProgram(std::vector<bigint> inputs);

		std::vector<bigint> RunProgram(bigint input = 0);

		bool IsHalted() { return pState[pProg] == 99; };

		void HackState(int address, bigint value) { pState.at(address) = value; };

		void SetProgPoint(bigint pos) { pProg = pos; };

	private:
		std::vector<bigint> pState;
		bigint pProg = 0;
		bigint relBase = 0;

		inline bigint getValue(int mode, bigint instruction);

		inline void setState(int mode, bigint value, int pPOffset = 3);
	};
}