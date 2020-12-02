#pragma once

#include "Vec2.h"

namespace aoc {
namespace utils {

class Mat2x2 {
  public:
    Mat2x2(Vec2 i = 0, Vec2 j = 0) : I(i), J(j){};
    Vec2 I, J;
    bool operator==(const Mat2x2& rhs) const {
        return I == rhs.I && J == rhs.J;
    };
    // bool operator<(const Mat2x2& rhs) const { return I < rhs.I || J < rhs.J;
    // }; // Experimental just for map.
    Mat2x2 operator+(const Mat2x2& rhs) const {
        return Mat2x2(I + rhs.I, J + rhs.J);
    };
    Mat2x2 operator-(const Mat2x2& rhs) const {
        return Mat2x2(I - rhs.I, J - rhs.J);
    };
    // Mat2x2 Inverse() const { return Vec2(-X, -Y); };
    // int AbsSum() const { return std::abs(X) + std::abs(Y) };
    Mat2x2 rotateLeft() { return Mat2x2(Vec2(-J.X, -J.Y), Vec2(I.X, I.Y)); };
    Mat2x2 rotateRight() { return Mat2x2(Vec2(J.X, J.Y), Vec2(-I.X, -I.Y)); };
};

} // namespace utils
} // namespace aoc