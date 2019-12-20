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
		Vertex(Vec2 pos) : Pos(pos), Colour(0), Dist(INT_MAX), Pre(-1) {};
		Vec2 Pos;
		int Colour; // 0 white, 1 grey, 2 black.
		int Dist; // Inf / null val.
		int Pre; // -1 null. 
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

		std::map<Vec2, std::string, Vec2LessComp> portalIds;
		for (auto& point : maze) { // Find Portals.
			if (IsCapLetter(point.second)) { // Portal marker
				auto up = maze.find(Vec2(point.first.X, point.first.Y - 1));
				auto down = maze.find(Vec2(point.first.X, point.first.Y + 1));
				auto left = maze.find(Vec2(point.first.X - 1, point.first.Y));
				auto right = maze.find(Vec2(point.first.X + 1, point.first.Y));

				// Top portal lower letter
				if (up != maze.end() && IsCapLetter(up->second) && down != maze.end() && down->second == '.') {
					portalIds.insert({ down->first, std::string{ (char)up->second, (char)point.second } });
				}
				// Bottom portal upper letter
				if (up != maze.end() && up->second == '.' && down != maze.end() && IsCapLetter(down->second)) {
					portalIds.insert({ up->first, std::string{ (char)point.second, (char)down->second } });
				}
				// Left portal right letter
				if (left != maze.end() && IsCapLetter(left->second) && right != maze.end() && right->second == '.') {
					portalIds.insert({ right->first, std::string{ (char)left->second, (char)point.second } });
				}
				// Right portal left letter
				if (left != maze.end() && left->second == '.' && right != maze.end() && IsCapLetter(right->second)) {
					portalIds.insert({ left->first, std::string{ (char)point.second, (char)right->second } });
				}
			}
		}
		
		// Create portal mapping.
		Vec2 start, end;
		std::map<Vec2, Vec2, Vec2LessComp> portalMap;
		for (const auto& portal : portalIds) {
			if (portal.second == "AA") start = portal.first;
			if (portal.second == "ZZ") end = portal.first;
			for (const auto& otherPortal : portalIds) {
				if (portal.first != otherPortal.first && portal.second == otherPortal.second) {
					portalMap.insert({ portal.first, otherPortal.first });
				}
			}
		}

		// Test
		for (const auto& portal : portalMap) {
			window.SetChar(portal.first.X, portal.first.Y, 'P');
		}
		window.SetChar(start.X, start.Y, 'S');
		window.SetChar(end.X, end.Y, 'E');
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
			auto portal = portalMap.find(corridors[i].Pos); // Check portal map.
			if (portal != portalMap.end()) {
				for (auto j = 0; j < corridors.size(); ++j) { // Find the corridor index
					if (corridors[j].Pos == portal->second) adjCorr[i].push_back(j);
				}
			}
			if (corridors[i].Pos == start) startI = i;
			if (corridors[i].Pos == end) endI = i;
		}

		// TEST.
		int testCorrI = 17;
		window.SetChar(corridors[testCorrI].Pos.X, corridors[testCorrI].Pos.Y, 'T');
		for (auto adj : adjCorr[testCorrI]) {
			window.SetChar(corridors[adj].Pos.X, corridors[adj].Pos.Y, '|');
		}
		window.Update();

		// BFS
		corridors[startI].Colour = 1;
		corridors[startI].Dist = 0;
		
		std::queue<int> tips;
		tips.push(startI);
		int currI;

		while (!tips.empty()) {
			currI = tips.front();
			tips.pop();
			for (auto adj :adjCorr[currI]) {
				if (corridors[adj].Colour == 0) { // If unvisited
					corridors[adj].Colour = 1;
					corridors[adj].Dist = corridors[currI].Dist + 1;
					corridors[adj].Pre = currI;
					tips.push(adj);
				}
			}
			corridors[currI].Colour = 2; // Completed.
		}

		std::cout << "1. Steps it takes to get from the open tile marked AA to the open tile marked ZZ:\n";
		std::cout << corridors[endI].Dist << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}