#pragma once

namespace aoc::utils {
	class Vec2 {
	public:
		Vec2(int x = 0, int y = 0) : X(x), Y(y) {};
		int X, Y;
		bool operator==(const Vec2& rhs) const { return X == rhs.X && Y == rhs.Y; };
		Vec2 operator+(const Vec2& rhs) const { return Vec2(X + rhs.X, Y + rhs.Y); };
		Vec2 operator-(const Vec2& rhs) const { return Vec2(X - rhs.X, Y - rhs.Y); };
		Vec2 Inverse() const { return Vec2(-X, -Y); };
		int AbsSum() const { return std::abs(X) + std::abs(Y); };
	};

	struct Vec2LessComp {
		bool operator() (const Vec2& lhs, const Vec2& rhs) const {
			return lhs.X < rhs.X || (!(lhs.X > rhs.X) && lhs.Y < rhs.Y); // Lexicographical compare.
		}
	};
}