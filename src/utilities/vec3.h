#pragma once

namespace aoc::utils {


	class Vec3 {
	public:
		Vec3(int x = 0, int y = 0, int z = 0) : X(x), Y(y), Z(z) {};
		int X, Y, Z;
		bool operator==(const Vec3& rhs) const { return X == rhs.X && Y == rhs.Y && Z == rhs.Z; };
		Vec3 operator+(const Vec3& rhs) const { return Vec3(X + rhs.X, Y + rhs.Y, Z + rhs.Z); };
		Vec3 operator-(const Vec3& rhs) const { return Vec3(X - rhs.X, Y - rhs.Y, Z - rhs.Z); };
		Vec3 Inverse() const { return Vec3(-X, -Y, -Z); };
		int AbsSum() const { return std::abs(X) + std::abs(Y) + std::abs(Z); };
	};
}