#pragma once
#include "default.h"

namespace aoc::y2019::d12 {	

	struct Vec3 {
		Vec3(int x = 0, int y = 0, int z = 0) : X(x), Y(y), Z(z) {};
		int X, Y, Z;
		bool Vec3::operator==(const Vec3& rhs) const { return X == rhs.X && Y == rhs.Y && Z == rhs.Z; };
		Vec3 Vec3::operator+(const Vec3& rhs) const { return Vec3(X + rhs.X, Y + rhs.Y, Z + rhs.Z); };
		Vec3 Vec3::operator-(const Vec3& rhs) const { return Vec3(X - rhs.X, Y - rhs.Y, Z - rhs.Z); };
		Vec3 Inverse() const { return Vec3(-X, -Y, -Z); };
		int AbsSum() const { return std::abs(X) + std::abs(Y) + std::abs(Z); };
	};

	struct Moon {
		Moon(int id, Vec3 pos, Vec3 vel) : Id(id), Pos(pos), Vel(vel) {};
		int Id;
		Vec3 Pos, Vel;
		bool Moon::operator==(const Moon& rhs) const { return Id == rhs.Id && Pos == rhs.Pos && Vel == rhs.Vel; };
		bool Moon::operator<(const Moon& rhs) const { return Id < rhs.Id; };
	};

	long long DimensionSearchForRepeat(std::vector<std::pair<int, int>> initalPosVels) {
		auto currentPosVels(initalPosVels);
		long long rep = 0;
		while (true) {
			++rep;
			for (auto& currentPosVel : currentPosVels) {
				for (const auto& otherPosVel : currentPosVels) {
					if (currentPosVel == otherPosVel) continue;
					currentPosVel.second += (currentPosVel.first > otherPosVel.first) ? -1 : (currentPosVel.first < otherPosVel.first);
				}
			}
			for (auto& currentPosVel : currentPosVels) {
				currentPosVel.first += currentPosVel.second;
			}
			if (currentPosVels == initalPosVels) break;
		}
		return rep;
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 12: The N-Body Problem ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);

		// Get Initial Moon Positions
		std::vector<Moon> initialState;
		int id = 0;
		for (auto& inputStr : inputStrs) {
			std::regex re("<x=(.+), y=(.+), z=(.+)>");
			std::smatch match;
			std::regex_search(inputStr, match, re);
			initialState.push_back(Moon(id, Vec3{std::stoi(match[1]), std::stoi(match[2]), std::stoi(match[3]) }, Vec3{}));
			++id;
		}

		// Part 1. Compute simulation.
		auto part1State(initialState);
		for (auto step = 0; step < 10; ++step) {
			// Compute Velocity from gravity
			for (auto& moon : part1State) {
				// Compute distance and velocity to ever
				for (const auto& otherMoon : part1State) {
					if (moon.Id == otherMoon.Id) continue; // Don't consider self.
					// For each component
					moon.Vel.X += (moon.Pos.X > otherMoon.Pos.X) ? -1 : (moon.Pos.X < otherMoon.Pos.X);
					moon.Vel.Y += (moon.Pos.Y > otherMoon.Pos.Y) ? -1 : (moon.Pos.Y < otherMoon.Pos.Y);
					moon.Vel.Z += (moon.Pos.Z > otherMoon.Pos.Z) ? -1 : (moon.Pos.Z < otherMoon.Pos.Z);
				}
			}

			// Update Positions
			for (auto& moon : part1State) {
				moon.Pos = moon.Pos + moon.Vel;
			}
		}

		// Compute total energy.
		int totalEnergy = 0;
		for (auto& moon : part1State) {
			totalEnergy += moon.Pos.AbsSum() * moon.Vel.AbsSum();
		}

		std::cout << "1. Total energy:\n";
		std::cout << totalEnergy << "\n";

		// Part 2.
		// Simulate each dimension separately - as they are independent.
		std::vector<std::pair<int, int>> initX, initY, initZ;
		for (auto& moon : initialState) {
			initX.push_back({ moon.Pos.X, moon.Vel.X });
			initY.push_back({ moon.Pos.Y, moon.Vel.Y} );
			initZ.push_back({ moon.Pos.Z, moon.Vel.Z });
		}

		auto xRep = DimensionSearchForRepeat(initX);
		auto yRep = DimensionSearchForRepeat(initY);
		auto zRep = DimensionSearchForRepeat(initZ);

		// Find the GCD of the step counts for each dimension when it loops back to initial state (this will be the only state in which subsequent states can occur that are duplicate).
		auto numberStepsBeforeSame = std::lcm(xRep, std::lcm(yRep, zRep));

		std::cout << "2. Number of steps before same state reached again. :\n";
		std::cout << numberStepsBeforeSame << "\n";
	}
}