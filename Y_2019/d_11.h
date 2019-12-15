#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d11 {
	using namespace aoc::y2019::intcc;

	using Loc = std::tuple<int, int>;
	inline int getLX(const Loc& loc) { return std::get<0>(loc); };
	inline int getLY(const Loc& loc) { return std::get<1>(loc); };
	inline void setLX(Loc& loc, int value) { std::get<0>(loc) = value; };
	inline void setLY(Loc& loc, int value) { std::get<1>(loc) = value; }

	using Ori = std::pair<Loc, Loc>;
	inline Ori rotateLeft(const Ori& ori) { return Ori({ -std::get<0>(ori.second), -std::get<1>(ori.second) }, { std::get<0>(ori.first), std::get<1>(ori.first) }); };
	inline Ori rotateRight(const Ori& ori) { return Ori({ std::get<0>(ori.second), std::get<1>(ori.second) }, { -std::get<0>(ori.first), -std::get<1>(ori.first) }); };

	void calculate(std::istream& input) {
		std::cout << "--- Day 11: Space Police ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});

		IntCodeComputer painterProgram1(inputStr);

		std::map<Loc, int> hullState; // Empty or 0 is unpainted, 1 is white painted areas.
		hullState[{0, 0}] = 1; // Part 1 = 0, Part 2 = 1.
		Loc currentRobotPosition(0, 0);
		std::pair<Loc, Loc> currentRobotOri({ 0, -1 }, { 1, 0 }); // Left handed cs. robot facing upwards.

		while (!painterProgram1.IsHalted()) { // Assuming program will not halt after one iteration.
			// Get current state on the hull.
			auto currentHullPiece = hullState.find(currentRobotPosition);
			std::vector<bigint> painterOutput;
			if (currentHullPiece == hullState.end() || std::get<1>(*currentHullPiece) == 0) {
				painterOutput = painterProgram1.RunProgram(0);
			}
			else {
				painterOutput = painterProgram1.RunProgram(1);
			}

			// Paint the hull.
			if (int(painterOutput.at(0)) == 0) hullState[currentRobotPosition] = 0;
			else if (int(painterOutput.at(0)) == 1) hullState[currentRobotPosition] = 1;
			else throw std::runtime_error("Unrecognised painter colour command.");

			// Turn the robot.
			if (int(painterOutput.at(1)) == 0) currentRobotOri = rotateLeft(currentRobotOri);
			else if (int(painterOutput.at(1)) == 1) currentRobotOri = rotateRight(currentRobotOri);

			// Move the robot forward (the first basis vector of orientation is forward).
			setLX(currentRobotPosition, getLX(currentRobotPosition) + getLX(currentRobotOri.first));
			setLY(currentRobotPosition, getLY(currentRobotPosition) + getLY(currentRobotOri.first));
		}

		std::cout << "Hull state plot:\n";
		for (auto y = 0; y < 10; ++y) {
			for (auto x = 0; x < 50; ++x) {
				auto currentHullPiece = hullState.find({ x, y });
				std::vector<bigint> painterOutput;
				if (currentHullPiece == hullState.end() || std::get<1>(*currentHullPiece) == 0) {
					std::cout << " ";
				}
				else {
					std::cout << "#";
				}
			}
			std::cout << "\n";
		}
	}
}