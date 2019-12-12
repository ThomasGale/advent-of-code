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
		bool Moon::operator==(const Moon& rhs) const { return Id == rhs.Id; };
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 12: The N-Body Problem ---\n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);

		// Get Initial Moon Positions
		std::vector<Moon> currentState;
		int id = 0;
		for (auto& inputStr : inputStrs) {
			std::regex re("<x=(.+), y=(.+), z=(.+)>");
			std::smatch match;
			std::regex_search(inputStr, match, re);
			currentState.push_back(Moon(id, Vec3{std::stoi(match[1]), std::stoi(match[2]), std::stoi(match[3]) }, Vec3{}));
			++id;
		}

		// Compute simulation.
		for (auto step = 0; step < 1000; ++step) {
			// Compute Velocity from gravity
			for (auto& moon : currentState) {
				// Compute distance and velocity to ever
				for (const auto& otherMoon : currentState) {
					if (moon == otherMoon) continue; // Don't consider self.
					// For each component
					moon.Vel.X += (moon.Pos.X > otherMoon.Pos.X) ? -1 : (moon.Pos.X < otherMoon.Pos.X);
					moon.Vel.Y += (moon.Pos.Y > otherMoon.Pos.Y) ? -1 : (moon.Pos.Y < otherMoon.Pos.Y);
					moon.Vel.Z += (moon.Pos.Z > otherMoon.Pos.Z) ? -1 : (moon.Pos.Z < otherMoon.Pos.Z);
				}
			}

			// Update Positions
			for (auto& moon : currentState) {
				moon.Pos = moon.Pos + moon.Vel;
			}
		}

		// Compute total energy.
		int totalEnergy = 0;
		for (auto& moon : currentState) {
			totalEnergy += moon.Pos.AbsSum() * moon.Vel.AbsSum();
		}


		std::cout << "1. Total energy:\n";
		std::cout << totalEnergy << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}