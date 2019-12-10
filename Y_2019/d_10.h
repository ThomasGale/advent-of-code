#pragma once
#include "default.h"

namespace aoc::y2019::d10 {
	void calculate(std::istream& input) {
		std::cout << "--- Day 10: Monitoring Station ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);

		int width = inputStrs.front().size();
		int height = inputStrs.size();
		std::set<std::tuple<int, int>> asteroids;
		for (auto y = 0; y < inputStrs.size(); ++y) {
			for (auto x = 0; x < inputStrs[y].size(); ++x) {
				if (inputStrs[y][x] == '#') asteroids.insert({ x, y });
			}
		}

		// Part 1.
		std::map<std::tuple<int, int>, int> asteroidsLOS;
		for (const auto& asteroid : asteroids) {

			std::set<std::tuple<int, int>> distanceVecs; // Compute vectors
			std::transform(asteroids.begin(), asteroids.end(), std::inserter(distanceVecs, distanceVecs.begin()), [&asteroid](const auto& currAst) {
				return std::tuple<int, int>(std::get<0>(currAst) - std::get<0>(asteroid), std::get<1>(currAst) - std::get<1>(asteroid));
				});

			distanceVecs.erase(std::tuple<int, int>(0, 0)); // Remove self.

			std::set<std::tuple<int, int>> occulsionFilteredAsteroids(asteroids);
			occulsionFilteredAsteroids.erase(asteroid); // Remove self.

			std::set<std::tuple<int, int>> occludedValuesToRemove;
			// Loop over each remaining distanceVec in set to compute all occluded values.
			for (const auto& distanceVecToTest : distanceVecs) {
				int vecX = std::get<0>(distanceVecToTest); // Vector X
				int vecY = std::get<1>(distanceVecToTest); // Vector Y

				int vecXStep = vecX / std::gcd(vecX, vecY); // Compute Occlusion steps 
				int vecYStep = vecY / std::gcd(vecX, vecY); // ^^ 

				std::set<std::tuple<int, int>> asteroidOcclusionSet;
				int currX = std::get<0>(asteroid) + vecX + vecXStep; // Consider everything behind the asteroid
				int currY = std::get<1>(asteroid) + vecY + vecYStep; // ^^
				while (true) {
					if (currX >= width || currX < 0 || currY >= height || currY < 0) break;
					asteroidOcclusionSet.insert({ currX, currY });
					currX += vecXStep;
					currY += vecYStep;
				}
				std::set_intersection(occulsionFilteredAsteroids.begin(), occulsionFilteredAsteroids.end(), asteroidOcclusionSet.begin(), asteroidOcclusionSet.end(),
					std::inserter(occludedValuesToRemove, occludedValuesToRemove.begin()));
			}
			
			// Remove occluded values.
			std::set<std::tuple<int, int>> removedResult;
			std::set_difference(occulsionFilteredAsteroids.begin(), occulsionFilteredAsteroids.end(),
				occludedValuesToRemove.begin(), occludedValuesToRemove.end(),
				std::inserter(removedResult, removedResult.begin()));
			occulsionFilteredAsteroids = std::move(removedResult);

			asteroidsLOS[asteroid] = occulsionFilteredAsteroids.size();
		}

		auto bestAsteroid = *std::max_element(asteroidsLOS.begin(), asteroidsLOS.end(), [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });

		std::cout << "1. How many asteroids detectable from best location?:\n";
		std::cout << std::get<1>(bestAsteroid) << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}