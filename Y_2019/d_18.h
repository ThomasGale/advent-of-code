#pragma once
#include "default.h"

namespace aoc::y2019::d18 {
	using namespace aoc::utils;

	struct Node {
		Node(Vec2 pos, int val) : Pos(pos), Val(val) {};
		Vec2 Pos;
		int Val;
		std::vector<int> Adj;
	};

	struct NodeComparer {
		Vec2LessComp vec2Comp;
		bool operator() (const Node& lhs, const Node& rhs) const {
			return vec2Comp(lhs.Pos, rhs.Pos);
		}
	};

	void renderMap(BasicWindow& window, const std::map<Vec2, int, Vec2LessComp>& map) {
		for (auto el : map) {
			window.SetChar(el.first.Y, el.first.X, el.second);
		}
		window.Update();
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 18: Many-Worlds Interpretation ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		BasicWindow window(41, 41);

		// Process Input.
		std::map<Vec2, int, Vec2LessComp> map;
		Vec2 currPos;
		std::map<int, Vec2> doors;
		std::map<int, Vec2> keys;
		for (auto y = 0; y < inputStrs.size(); ++y) {
			for (auto x = 0; x < inputStrs[y].size(); ++x) {
				map.insert({Vec2(x, y), inputStrs[y][x]});
				if (inputStrs[y][x] == '@') currPos = Vec2(x, y);
				if (inputStrs[y][x] >= 'A' && inputStrs[y][x] <= 'Z') doors.insert({ inputStrs[y][x], Vec2(x, y) });
				if (inputStrs[y][x] >= 'a' && inputStrs[y][x] <= 'z') keys.insert({ inputStrs[y][x], Vec2(x, y) });
			}
		}
		renderMap(window, map);

		// Find Passage Graph
		std::vector<Node> passages;
		for (auto el : map) {
			if (el.second != '#') {
				passages.push_back(Node(el.first, el.second));
			}
		}
		for (auto& passage : passages) { // Simple O(n^2) adjanceny finder.
			Vec2 up(passage.Pos.X, passage.Pos.Y - 1);
			Vec2 down(passage.Pos.X, passage.Pos.Y + 1);
			Vec2 left(passage.Pos.X - 1, passage.Pos.Y);
			Vec2 right(passage.Pos.X + 1, passage.Pos.Y);

			for (auto i = 0; i < passages.size(); ++i) {
				if (passages[i].Pos == up) passage.Adj.push_back(i);
				if (passages[i].Pos == down) passage.Adj.push_back(i);
				if (passages[i].Pos == left) passage.Adj.push_back(i);
				if (passages[i].Pos == right) passage.Adj.push_back(i);
			}
		}

		// Traversal of graph.
		// Find current open corredor from current position. Mark on the tips of the graph the current Doors
		std::map<Vec2, int, Vec2LessComp> currMap(map);
		std::vector<Node> openPassages;
		std::set<std::pair<int, int>> closedDoorDistances;
		std::set<std::pair<int, int>> unCollectedKeyDistances;
		int totalDistance = 0;
		// Vec2 currPos;
		// std::map<int, Vec2> doors;
		// std::map<int, Vec2> keys;

		while (true) {
			// From currPos compute open passages. We stop at doors. Update closedDoorDistances.

			// Using openPassages travel to compute unCollectedKeyDistances.

			// Choose the shortest distance which picks up key and unlocks door (the shortest traveral along graph)
			
			// Move to key and pick up (update totalDistance and remove unCollectedKeyDistances element)
			
			// Check if all keys are picked up (unCollectedKeyDistances empty) break if so.

			// Move to door and unlock door (update totalDistance and remove closed door from closedDoorDistances)
		}

		std::cout << "1. Steps in shortest path that collects all of the keys:\n";
		std::cout << totalDistance << "\n";
	}
}