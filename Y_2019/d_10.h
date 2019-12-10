#pragma once
#include "default.h"

namespace aoc::y2019::d10 {

	using Loc = std::tuple<int, int>;
	
	inline int getLX(const Loc& loc) { return std::get<0>(loc); };
	inline int getLY(const Loc& loc) { return std::get<1>(loc); };
	inline void setLX(Loc& loc, int value) { std::get<0>(loc) = value; };
	inline void setLY(Loc& loc, int value) { std::get<1>(loc) = value; };

	inline int Mag(const Loc& loc) {
		return std::abs(getLX(loc)) + std::abs(getLY(loc));
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 10: Monitoring Station ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);

		int width = int(inputStrs.front().size());
		int height = int(inputStrs.size());
		std::set<Loc> asteroids;
		for (auto y = 0; y < inputStrs.size(); ++y) {
			for (auto x = 0; x < inputStrs[y].size(); ++x) {
				if (inputStrs[y][x] == '#') asteroids.insert({ x, y });
			}
		}

		// Part 1.
		auto p1Start = clock::now();
		std::map<Loc, int> asteroidsLOS;
		for (const auto& asteroid : asteroids) {

			std::set<Loc> distanceVecs; // Compute vectors
			std::transform(asteroids.begin(), asteroids.end(), std::inserter(distanceVecs, distanceVecs.begin()), [&asteroid](const auto& currAst) {
				return Loc(getLX(currAst) - getLX(asteroid), getLY(currAst) - getLY(asteroid));
				});

			distanceVecs.erase(Loc(0, 0)); // Remove self.
			std::set<Loc> occulsionFilteredAsteroids(asteroids); // Create copy of asteroids for filtering.
			occulsionFilteredAsteroids.erase(asteroid); // Remove self.

			std::set<Loc> occludedValuesToRemove; // Record what values are to be removed.
			// Loop over each remaining distanceVec in set to compute all occluded values.
			for (const auto& distanceVecToTest : distanceVecs) {
				int vecX = getLX(distanceVecToTest); // Vector X
				int vecY = getLY(distanceVecToTest); // Vector Y

				int vecXStep = vecX / std::gcd(vecX, vecY); // Compute Occlusion steps 
				int vecYStep = vecY / std::gcd(vecX, vecY); // ^^ 

				std::set<Loc> asteroidOcclusionSet;
				int currX = getLX(asteroid) + vecX + vecXStep; // Consider everything behind the asteroid
				int currY = getLY(asteroid) + vecY + vecYStep; // ^^
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
			std::set<Loc> removedResult;
			std::set_difference(occulsionFilteredAsteroids.begin(), occulsionFilteredAsteroids.end(),
				occludedValuesToRemove.begin(), occludedValuesToRemove.end(),
				std::inserter(removedResult, removedResult.begin()));

			asteroidsLOS[asteroid] = int(removedResult.size());
		}

		auto bestAsteroid = *std::max_element(asteroidsLOS.begin(), asteroidsLOS.end(), [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });
		auto p1End = clock::now();

		std::cout << "1. How many asteroids detectable from best location?:\n";
		std::cout << std::get<1>(bestAsteroid) << "\n";
		PrintDuration(p1Start, p1End);

		// Part 2.
		auto p2Start = clock::now();
		Loc LaserLoc = std::get<0>(bestAsteroid);
		std::set<Loc> otherAsteroidVecs; // Compute vectors to other asteroids.
		std::transform(asteroids.begin(), asteroids.end(), std::inserter(otherAsteroidVecs, otherAsteroidVecs.begin()), [&LaserLoc](const auto& currAst) {
			return std::tuple<int, int>(getLX(currAst) - getLX(LaserLoc), getLY(currAst) - getLY(LaserLoc));
			});
		otherAsteroidVecs.erase(Loc(0, 0)); // Remove self.

		// Create a map of GCD divided vectors to an ordered.
		std::map<Loc, std::set<Loc>> sortedAsteroids;
		for (auto& otherAsteroidVec : otherAsteroidVecs) {
			int vecX = getLX(otherAsteroidVec); // Vector X.
			int vecY = getLY(otherAsteroidVec); // Vector Y.
			sortedAsteroids[{vecX / std::gcd(vecX, vecY), vecY / std::gcd(vecX, vecY)}].insert(otherAsteroidVec);
		}

		// Ordered map of increasing angles to normalised direction vector.
		std::map<double, Loc> asteroidNormVecAngles; // This output should be sorted clockwise.
		for (auto& asteroid : sortedAsteroids) { 	
			int astX = getLX(std::get<0>(asteroid)); // Vector X.
			int astY = getLY(std::get<0>(asteroid)); // Vector Y.
			double theta = std::acos(-astY / std::sqrt(std::pow(astX, 2.0) + std::pow(astY, 2.0))); // Compute Angle.
			if (-astX < 0) theta = (std::atan(1) * 8) - theta; // Add PI if det is less than 0 (as this is a reflex angle).
			theta = (std::atan(1) * 8) - theta; // Flip to clockswise.
			theta = theta > (std::atan(1) * 8) - 0.0001 ? 0.0 : theta; // Small Hack to get first vector at 0, -1.
			asteroidNormVecAngles[theta] = std::get<0>(asteroid);
		}
		
		int shotsCount = 0;
		int maxShots = 200;
		while (true) { // Loop untill shot cout is reached
			// Rotate Gun Until we find an asteroid
			for (const auto& currentNormVecAngle : asteroidNormVecAngles) {
				Loc currentNormVector = std::get<1>(currentNormVecAngle);
				Loc removedAsteroid(width*height,width*height); // Large default distance.
				if (sortedAsteroids[currentNormVector].size() != 0) { // If there are asteroids to blast
					// Get the smalls abs vector asteroid from the current laser vector.
					for (auto& asteroidToRemove : sortedAsteroids[currentNormVector]) {
						if (Mag(asteroidToRemove) < Mag(removedAsteroid)) {
							removedAsteroid = asteroidToRemove;
						}
					}
					// Remove it
					sortedAsteroids[currentNormVector].erase(removedAsteroid);
					++shotsCount;
				}
				// Part 2 End condition.
				if (shotsCount >= maxShots) {
					std::cout << "2. Asteroid blasting results\n";
					std::cout << maxShots << " th asteroid: X" << getLX(LaserLoc) + getLX(removedAsteroid) << " Y" << getLY(LaserLoc) + getLY(removedAsteroid) << "\n";
					std::cout << "Puzzle output: " << (getLX(LaserLoc) + getLX(removedAsteroid)) * 100 + getLY(LaserLoc) + getLY(removedAsteroid) << "\n";
					auto p2End = clock::now();
					PrintDuration(p2Start, p2End);
					return;
				}
			}
		}
	}
}