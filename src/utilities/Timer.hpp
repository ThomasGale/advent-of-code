#include <chrono>
#include <iostream>

namespace aoc {
namespace utils {

using clock = std::chrono::high_resolution_clock;

template <typename T> inline double Duration(const T& start, const T& stop) {
    return double(
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count());
}

template <typename T> inline void PrintDuration(const T& start, const T& stop) {
    std::cout << "Calculation duration: " << Duration(start, stop) / 1000
              << " ms\n";
}


} // namespace utils
} // namespace aoc