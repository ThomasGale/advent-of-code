#pragma once
#include "default.h"
#include "IntCodeComputer.h"
#include <random>

namespace aoc::y2019::d15 {
	using namespace aoc::y2019::intcc;

	using Loc = std::tuple<int, int>;
	inline int getLX(const Loc& loc) { return std::get<0>(loc); };
	inline int getLY(const Loc& loc) { return std::get<1>(loc); };
	inline void setLX(Loc& loc, int value) { std::get<0>(loc) = value; };
	inline void setLY(Loc& loc, int value) { std::get<1>(loc) = value; }

	using Ori = std::pair<Loc, Loc>;
	inline Ori rotateRight(const Ori& ori) {
		return Ori({ std::get<0>(ori.second), std::get<1>(ori.second) }, { -std::get<0>(ori.first), -std::get<1>(ori.first) });
	}

	struct MapState {
		MapState(int type, int distance) : Type(type), Distance(distance) {};
		int Type; // 0 clear, 1 wall, 2 target.
		int Distance; // From start.
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 15: Oxygen System ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<bigint> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoll(input); });

		// Part 1.
		IntCodeComputer droidProg1(inputProgram);
		int windowX = 51, windowY = 51;
		Loc droidPos(windowX / 2, windowY / 2);
		Ori droidOri({ 0, -1 }, { 1, 0 }); // Coordinate system left-handed, top left (like in previous examples).
		std::map<Loc, MapState> map{ {droidPos, MapState(0, 0)} }; // 0 clear, 1 wall, 2 goal.
		aoc::utils::BasicWindow window(windowX, windowY);
		Loc target;
		int currStatus = 0;
		int currDistance = 0;

		// Search for tank 
		while (true) {

			// Move droid in current direction
			if (getLX(droidOri.first) == 1) currStatus = droidProg1.RunProgram(4).front(); // Move East (right)
			else if (getLX(droidOri.first) == -1) currStatus = droidProg1.RunProgram(3).front(); // Move West (left)
			else if (getLY(droidOri.first) == 1) currStatus = droidProg1.RunProgram(2).front(); // Move South (down)
			else if (getLY(droidOri.first) == -1) currStatus = droidProg1.RunProgram(1).front(); // Move North (up)

			// Movement status update.
			if (currStatus == 0) { // Wall hit
				map.insert({ {getLX(droidPos) + getLX(droidOri.first), getLY(droidPos) + getLY(droidOri.first)}, MapState(1, currDistance) }); // Record wall (in front of droid)
			}
			else { // Droid could move forward in direction it was facing
				if (currStatus == 1) {
					setLX(droidPos, getLX(droidPos) + getLX(droidOri.first));
					setLY(droidPos, getLY(droidPos) + getLY(droidOri.first));
					auto currMap = map.find(droidPos);
					if (currMap == map.end()) {
						++currDistance;
						map.insert({ droidPos, MapState(0, currDistance) }); // Record free space
					}
					else {
						currDistance = currMap->second.Distance;
					}
				}
				else if (currStatus == 2) { // Oxygen found!
					setLX(droidPos, getLX(droidPos) + getLX(droidOri.first));
					setLY(droidPos, getLY(droidPos) + getLY(droidOri.first));
					++currDistance;
					map.insert({ droidPos, MapState(2, currDistance) }); // Record oxygen location.
					target = droidPos;
					break;
				} else { throw std::runtime_error("Unexpected status."); };

			}

			auto leftDroid = map.find({ getLX(droidPos) - getLX(droidOri.second), getLY(droidPos) - getLY(droidOri.second) });
			auto rightDroid = map.find({ getLX(droidPos) + getLX(droidOri.second), getLY(droidPos) + getLY(droidOri.second) });
			auto frontDroid = map.find({ getLX(droidPos) + getLX(droidOri.first), getLY(droidPos) + getLY(droidOri.first) });

			// Rather Hacky turn left algorithm.
			if (frontDroid == map.end()) { // Explore (no rotate, move forward)
			} else if (leftDroid == map.end()) { // Explore
				droidOri = rotateLeft(droidOri);
			} else if (rightDroid == map.end()) {
				droidOri = rotateRight(droidOri); // Explore
			} else { // We know all 3 sides.
				if (rightDroid->second.Type == 0 && frontDroid->second.Type == 1 && leftDroid->second.Type == 1) { // Only turn right at bends in passage to the right.
					droidOri = rotateRight(droidOri);
				} else if (leftDroid->second.Type == 0 || frontDroid->second.Type == 1) { // Turn left otherwise at junctions (or dead ends).
					droidOri = rotateLeft(droidOri);
				}
			}

			// Draw map
			for (auto& tile : map) {
				char c = ' ';
				switch (tile.second.Type) {
				case 0: c = '.'; break;
				case 1: c = '#'; break;
				case 2: c = 'X'; break;
				}
				window.SetChar(getLX(tile.first), getLY(tile.first), c);
			}
			// Draw start
			window.SetChar(windowX / 2, windowY / 2, 'S');
			// Draw droid
			window.SetChar(getLX(droidPos), getLY(droidPos), 'D');
			window.Update();
		}

		std::cout << "1. What is the fewest number of movement commands to oxygen system:\n";
		std::cout << currDistance << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}