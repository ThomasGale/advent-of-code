#pragma once
#include "default.h"

namespace aoc::y2019::d06 {

	int computeDistance(std::map<std::string, std::string>& orbitMap, const std::string& start, const std::string& end) {
		int distance = 0;
		std::string currentEntity = start;
		while ((currentEntity != "COM") && (currentEntity != end)) {
			++distance;
			currentEntity = orbitMap[currentEntity];
		}
		return distance;
	}

	std::set<std::string> computeParents(std::map<std::string, std::string>& orbitMap, const std::string& entity) {
		std::set<std::string> parents;
		std::string currentEntity = entity;
		while (currentEntity != "COM") {
			parents.insert(orbitMap[currentEntity]);
			currentEntity = orbitMap[currentEntity];
		}
		return parents;
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 6: Universal Orbit Map ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		std::map<std::string, std::string> orbitMap; // Key orbits value
		for (auto& inputStr : inputStrs) {
			std::vector<std::string> orbitsStr = aoc::utils::split(inputStr, ')');
			orbitMap[orbitsStr[1]] = orbitsStr[0];
		}

		// Part 1.
		int totalOrbits = 0;
		for (const auto& orbit : orbitMap) {
			totalOrbits += computeDistance(orbitMap, orbit.first, "COM");
		}

		std::cout << "1. Total number of direct and indirect orbits :\n";
		std::cout << totalOrbits << "\n";

		// Part 2.
		auto youParents = computeParents(orbitMap, "YOU");
		auto sanParents = computeParents(orbitMap, "SAN");
		
		std::set<std::string> commonParents;
		std::set_intersection(youParents.begin(), youParents.end(), sanParents.begin(), sanParents.end(), std::inserter(commonParents, commonParents.begin()));
		
		std::vector<int> distances;
		for (auto& parent : commonParents) {
			int distance = computeDistance(orbitMap, "YOU", parent) - 1; // Transfer doesn't consider distance to current parent
			distance += computeDistance(orbitMap, "SAN", parent) - 1; // ^^
			distances.push_back(distance);
		}

		std::cout << "2. Min distance between YOU and SAN:\n";
		std::cout << *std::min_element(distances.begin(), distances.end()) << "\n";
	}
}