#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d15 {
	using namespace aoc::y2019::intcc;

	using Loc = std::tuple<int, int>;
	inline int getLX(const Loc& loc) { return std::get<0>(loc); };
	inline int getLY(const Loc& loc) { return std::get<1>(loc); };
	inline void setLX(Loc& loc, int value) { std::get<0>(loc) = value; };
	inline void setLY(Loc& loc, int value) { std::get<1>(loc) = value; }

	using Ori = std::pair<Loc, Loc>;
	inline Ori rotateLeft(const Ori& ori) { return Ori({ -std::get<0>(ori.second), -std::get<1>(ori.second) }, { std::get<0>(ori.first), std::get<1>(ori.first) }); };
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

		// Part 1.
		IntCodeComputer droidProg1(inputStr);
		int windowX = 51, windowY = 51;
		Loc droidPos(windowX / 2, windowY / 2);
		Ori droidOri({ 0, -1 }, { 1, 0 }); // Coordinate system left-handed, top left (like in previous examples).
		std::map<Loc, MapState> map{ {droidPos, MapState(0, 0)} }; // 0 clear, 1 wall, 2 goal.
		std::set<Loc> unexplored;
		aoc::utils::BasicWindow window(windowX, windowY);
		Loc target;
		int distanceToTarget = 0; // Part 1.
		int currStatus = 0;
		int currDist = 0;

		// Exhausive Exploration.
		while (true) {

			// Move droid in current direction
			if (getLX(droidOri.first) == 1) currStatus = (int)droidProg1.RunProgram(4).front(); // Move East (right)
			else if (getLX(droidOri.first) == -1) currStatus =(int)droidProg1.RunProgram(3).front(); // Move West (left)
			else if (getLY(droidOri.first) == 1) currStatus = (int)droidProg1.RunProgram(2).front(); // Move South (down)
			else if (getLY(droidOri.first) == -1) currStatus = (int)droidProg1.RunProgram(1).front(); // Move North (up)

			// Movement status update.
			if (currStatus == 0) { // Wall hit
				map.insert({ {getLX(droidPos) + getLX(droidOri.first), getLY(droidPos) + getLY(droidOri.first)}, MapState(1, currDist) }); // Record wall (in front of droid)
			}
			else { // Droid could move forward in direction it was facing
				if (currStatus == 1) {
					setLX(droidPos, getLX(droidPos) + getLX(droidOri.first));
					setLY(droidPos, getLY(droidPos) + getLY(droidOri.first));
					auto currMap = map.find(droidPos);
					if (currMap == map.end()) {
						++currDist;
						map.insert({ droidPos, MapState(0, currDist) }); // Record free space
					}
					else {
						currDist = currMap->second.Distance;
					}
				}
				else if (currStatus == 2) { // Oxygen found!
					setLX(droidPos, getLX(droidPos) + getLX(droidOri.first));
					setLY(droidPos, getLY(droidPos) + getLY(droidOri.first));
					++currDist;
					map.insert({ droidPos, MapState(2, currDist) }); // Record oxygen location.
					target = droidPos;
					distanceToTarget = currDist;
				} else { throw std::runtime_error("Unexpected status."); };
			}

			// Get information about surroundings.
			Loc left = { getLX(droidPos) - getLX(droidOri.second), getLY(droidPos) - getLY(droidOri.second) };
			Loc right = { getLX(droidPos) + getLX(droidOri.second), getLY(droidPos) + getLY(droidOri.second) };
			Loc front = { getLX(droidPos) + getLX(droidOri.first), getLY(droidPos) + getLY(droidOri.first) };

			auto leftDroid = map.find(left);
			auto rightDroid = map.find(right);
			auto frontDroid = map.find(front);

			// Update unexplored set;
			unexplored.erase(droidPos);
			if (leftDroid != map.end()) unexplored.erase(leftDroid->first);
			else unexplored.insert(left);
			if (rightDroid != map.end()) unexplored.erase(rightDroid->first);
			else unexplored.insert(right);
			if (frontDroid != map.end()) unexplored.erase(frontDroid->first);
			else unexplored.insert(front);

			// Rather hacky turn left maze walk algorithm.
			if (frontDroid == map.end()) { // Explore (no rotate, move forward)
			} else if (leftDroid == map.end()) {
				droidOri = rotateLeft(droidOri); // Explore unknown
			} else if (rightDroid == map.end()) {
				droidOri = rotateRight(droidOri); // Explore unknown
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

			// Check if fully explored
			if (unexplored.size() == 0) break;
		}

		std::cout << "1. What is the fewest number of movement commands to oxygen system:\n";
		std::cout << distanceToTarget << "\n";
		
		// Part 2. Starting from the oxygen location. Walk out the edges of the graph of connections in the maze corridors.
		std::map<Loc, int> stepsFromOxygen;
		std::queue<std::pair<Loc, int>> unvisitedWithDist;
		unvisitedWithDist.push({ target, 0 });
		Loc currLoc;
		currDist = 0;

		while (true) { // Not a pretty loop :| 
			auto currentLocationAndDist = unvisitedWithDist.front();
			unvisitedWithDist.pop();
			currLoc = currentLocationAndDist.first;
			currDist = currentLocationAndDist.second;

			Loc left = { getLX(currLoc) - 1, getLY(currLoc) };
			Loc right = { getLX(currLoc) + 1, getLY(currLoc) };
			Loc up = { getLX(currLoc), getLY(currLoc) - 1 };
			Loc down = { getLX(currLoc), getLY(currLoc) + 1 };

			auto leftOnMap = map.find(left);
			auto rightOnMap = map.find(right);
			auto upOnMap = map.find(up);
			auto downOnMap = map.find(down);

			auto leftOxy = stepsFromOxygen.find(left);
			auto rightOxy = stepsFromOxygen.find(right);
			auto upOxy = stepsFromOxygen.find(up);
			auto downOxy = stepsFromOxygen.find(down);

			if (leftOnMap != map.end() && leftOnMap->second.Type == 0 && leftOxy == stepsFromOxygen.end()) {
				stepsFromOxygen.insert({ left, currDist + 1 });
				unvisitedWithDist.push({ left, currDist + 1 });
			}
			if (rightOnMap != map.end() && rightOnMap->second.Type == 0 && rightOxy == stepsFromOxygen.end()) {
				stepsFromOxygen.insert({ right, currDist + 1 });
				unvisitedWithDist.push({ right, currDist + 1 });
			}
			if (upOnMap != map.end() && upOnMap->second.Type == 0 && upOxy == stepsFromOxygen.end()) {
				stepsFromOxygen.insert({ up, currDist + 1 });
				unvisitedWithDist.push({ up, currDist + 1 });
			}
			if (downOnMap != map.end() && downOnMap->second.Type == 0 && downOxy == stepsFromOxygen.end()) {
				stepsFromOxygen.insert({ down, currDist + 1 });
				unvisitedWithDist.push({ down, currDist + 1 });
			}

			if (unvisitedWithDist.empty()) break; // Onces all unvisited nodes are processed, that's the end.
		}

		// Find length of deepest chain.
		int maxSteps = 0;
		for (auto& step : stepsFromOxygen) {
			if (step.second > maxSteps) maxSteps = step.second;
		}

		std::cout << "2. How many minutes will it take to fill with oxygen:\n";
		std::cout << maxSteps  << "\n";
	}
}