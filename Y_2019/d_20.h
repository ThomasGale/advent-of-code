#pragma once
#include "default.h"

namespace aoc::y2019::d20 {
	using namespace aoc::utils;

	void RenderMap(BasicWindow& window, const std::map<Vec2, int, Vec2LessComp>& map) {
		for (auto& point : map) {
			window.SetChar(point.first.X, point.first.Y, point.second);
		}
		window.Update();
	}

	bool IsCapLetter(char letter) {
		return letter >= 'A' && letter <= 'Z';
	}

	bool IsAbove(const Vec2& loc, const Vec2& other) {
		return (loc.X == other.X && loc.Y == other.Y - 1);
	}

	bool IsBelow(const Vec2& loc, const Vec2& other) {
		return (loc.X == other.X && loc.Y == other.Y + 1);
	}

	bool IsLeft(const Vec2& loc, const Vec2& other) {
		return (loc.Y == other.Y && loc.X == other.X - 1);
	}

	bool IsRight(const Vec2& loc, const Vec2& other) {
		return (loc.Y == other.Y && loc.X == other.X + 1);
	}

	// BFS vertex.
	struct Vertex {
		Vertex(Vec2 pos) : Pos(pos), Colour(0), Dist(-1), Pre(-1), LayerCh(0) {};
		Vec2 Pos;
		int Colour; // 0 white, 1 grey, 2 black.
		int Dist; // Inf / null val.
		int Pre; // -1 null. 
		int LayerCh; // +1 on inner portal, 0 on nothing, -1 on outer
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 20: Donut Maze ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		std::map<Vec2, int, Vec2LessComp> maze;
		std::vector<Vertex> corridors;
		int height = inputStrs.size();
		int width = inputStrs.at(0).size();
		for (auto y = 0; y < inputStrs.size(); ++y) {
			for (auto x = 0; x < inputStrs[y].size(); ++x) {
				if (inputStrs[y][x] != ' ') maze.insert({ Vec2(x, y), inputStrs[y][x] });
				if (inputStrs[y][x] == '.') corridors.push_back(Vertex(Vec2(x, y)));
			}
		}
		BasicWindow window(width, height);
		RenderMap(window, maze);

		std::map<Vec2, std::string, Vec2LessComp> portalInIds;
		std::map<Vec2, std::string, Vec2LessComp> portalOutIds;
		for (auto& point : maze) { // Find Portals.
			if (IsCapLetter(point.second)) { // Portal marker
				auto up = maze.find(Vec2(point.first.X, point.first.Y - 1));
				auto down = maze.find(Vec2(point.first.X, point.first.Y + 1));
				auto left = maze.find(Vec2(point.first.X - 1, point.first.Y));
				auto right = maze.find(Vec2(point.first.X + 1, point.first.Y));

				// Top portal lower letter
				if (up != maze.end() && IsCapLetter(up->second) && down != maze.end() && down->second == '.') {
					if (down->first.Y < height / 2) {
						portalOutIds.insert({ down->first, std::string{ (char)up->second, (char)point.second } });
					} else {
						portalInIds.insert({ down->first, std::string{ (char)up->second, (char)point.second } });
					}
				}
				// Bottom portal upper letter
				if (up != maze.end() && up->second == '.' && down != maze.end() && IsCapLetter(down->second)) {
					if (down->first.Y < height / 2) {
						portalInIds.insert({ up->first, std::string{ (char)point.second, (char)down->second } });
					} else {
						portalOutIds.insert({ up->first, std::string{ (char)point.second, (char)down->second } });
					}
				}
				// Left portal right letter
				if (left != maze.end() && IsCapLetter(left->second) && right != maze.end() && right->second == '.') {
					if (right->first.X < width / 2) {
						portalOutIds.insert({ right->first, std::string{ (char)left->second, (char)point.second } });
					} else {
						portalInIds.insert({ right->first, std::string{ (char)left->second, (char)point.second } });
					}
				}
				// Right portal left letter
				if (left != maze.end() && left->second == '.' && right != maze.end() && IsCapLetter(right->second)) {
					if (right->first.X < width / 2) {
						portalInIds.insert({ left->first, std::string{ (char)point.second, (char)right->second } });
					} else {
						portalOutIds.insert({ left->first, std::string{ (char)point.second, (char)right->second } });
					}
				}
			}
		}
		
		// Create portal mapping.
		Vec2 start, end;
		std::map<Vec2, Vec2, Vec2LessComp> portalOutToInMap;
		for (const auto& outerPortal : portalOutIds) {
			if (outerPortal.second == "AA") start = outerPortal.first;
			if (outerPortal.second == "ZZ") end = outerPortal.first;
			for (const auto& innerPortal : portalInIds) {
				if (/*outerPortal.first != innerPortal.first && */outerPortal.second == innerPortal.second) {
					portalOutToInMap.insert({ outerPortal.first, innerPortal.first });
				}
			}
		}

		std::map<Vec2, Vec2, Vec2LessComp> portalInToOutMap;
		for (const auto& innerPortal : portalInIds) {
			for (const auto& outerPortal : portalOutIds) {
				if (/*innerPortal.first != outerPortal.first && */innerPortal.second == outerPortal.second) {
					portalInToOutMap.insert({ innerPortal.first, outerPortal.first });
				}
			}
		}

		// Test
		for (const auto& portal : portalOutToInMap) {
			window.SetChar(portal.first.X, portal.first.Y, 'O', 1);
		}
		for (const auto& portal : portalInToOutMap) {
			window.SetChar(portal.first.X, portal.first.Y, 'I', 1);
		}
		window.SetChar(start.X, start.Y, 'S', 1);
		window.SetChar(end.X, end.Y, 'E', 1);
		window.Update();

		// Build graph
		int startI = -1, endI = -1;
		std::vector<std::vector<int>> adjCorr(corridors.size()); // Referencing index of corridors.
		for (auto i = 0; i < corridors.size(); ++i) {
			adjCorr[i] = std::vector<int>{};
			for (auto j = 0; j < corridors.size(); ++j) { // Check other normal corridors.
				if (i == j) continue;				
				if (IsAbove(corridors[i].Pos, corridors[j].Pos)) adjCorr[i].push_back(j); 
				if (IsBelow(corridors[i].Pos, corridors[j].Pos)) adjCorr[i].push_back(j);
				if (IsLeft(corridors[i].Pos, corridors[j].Pos)) adjCorr[i].push_back(j);
				if (IsRight(corridors[i].Pos, corridors[j].Pos)) adjCorr[i].push_back(j);
			}
			auto outerPortal = portalOutToInMap.find(corridors[i].Pos); // Check portal map.
			if (outerPortal != portalOutToInMap.end()) {
				corridors[i].LayerCh = -1;
				for (auto j = 0; j < corridors.size(); ++j) { // Find the corridor index
					if (corridors[j].Pos == outerPortal->second) adjCorr[i].push_back(j);
				}
			}
			auto innerPortal = portalInToOutMap.find(corridors[i].Pos); // Check portal map.
			if (innerPortal != portalInToOutMap.end()) {
				corridors[i].LayerCh = +1;
				for (auto j = 0; j < corridors.size(); ++j) { // Find the corridor index
					if (corridors[j].Pos == innerPortal->second) adjCorr[i].push_back(j);
				}
			}
			if (corridors[i].Pos == start) startI = i;
			if (corridors[i].Pos == end) endI = i;
		}

		// TEST.
		int testCorrI = 253;
		window.SetChar(corridors[testCorrI].Pos.X, corridors[testCorrI].Pos.Y, 'X', 3);
		for (auto adj : adjCorr[testCorrI]) {
	    	window.SetChar(corridors[adj].Pos.X, corridors[adj].Pos.Y, 'A', 2);
		}
		window.Update();

		// BFS
		corridors[startI].Colour = 1;
		corridors[startI].Dist = 0;
		
		std::queue<std::pair<int, int>> tips; // level and index.
		tips.push({ 0, startI });
		int currI;
		int currLevel; // Part 2.
		std::vector<std::vector<Vertex>> corridorLevels{ corridors }; // Track all levels of corridors init with level 0.
		int depthLimit = 30;

		while (true && !tips.empty()) {
			std::tie<int, int>(currLevel, currI) = tips.front();
			tips.pop();
			if (currLevel >= corridorLevels.size()) corridorLevels.push_back(corridors); // Add a new level.

			if (currLevel < 10) {
				window.SetChar(corridorLevels[currLevel][currI].Pos.X, corridorLevels[currLevel][currI].Pos.Y, '0' + currLevel, 3);
				window.Update();
			}

			for (auto adj :adjCorr[currI]) {
				if (corridorLevels[currLevel][adj].Colour == 0) { // If unvisited
					corridorLevels[currLevel][adj].Colour = 1; // Mark as visited.
					corridorLevels[currLevel][adj].Dist = corridorLevels[currLevel][currI].Dist + 1;
					corridorLevels[currLevel][adj].Pre = currI;
					// Check for level change
					if (corridorLevels[currLevel][currI].LayerCh == +1 && corridorLevels[currLevel][adj].LayerCh == -1) { // We are at inside ring.
						if (currLevel + 1 <= depthLimit) tips.push({ currLevel + 1,  adj }); // Adding a depth limit.
					} else if (corridorLevels[currLevel][currI].LayerCh == -1 && corridorLevels[currLevel][adj].LayerCh == +1) {  // We are at outside ring
						if (currLevel >= 1)	tips.push({currLevel - 1,  adj }); // There is no layer outer than 0.
					} else { // Not at ring portal
						tips.push({ currLevel,  adj });
					}
					
				}
			}

			if (currLevel == 0 && corridorLevels[currLevel][currI].Pos == end) { // Break condition, we have reached the end.
				break;
			}

			corridorLevels[currLevel][currI].Colour = 2; // Completed.
			if (currLevel < 10) {
				window.SetChar(corridorLevels[currLevel][currI].Pos.X, corridorLevels[currLevel][currI].Pos.Y, '0' + currLevel, 2);
				window.Update();
			}
		}

		// More Testing
		for (auto& corridor : corridorLevels) {
			window.Clear(width, height);
			for (auto& vert : corridor) {
				if (vert.Dist != -1) window.SetChar(vert.Pos.X, vert.Pos.Y,  (vert.Dist % 10) + '0', 1);
			}
			window.Update();
		}
		


		std::cout << "2. Steps it takes to get from the open tile marked AA to the open tile marked ZZ on outer layer. :\n";
		std::cout << corridorLevels[0][endI].Dist << "\n";
	}
}