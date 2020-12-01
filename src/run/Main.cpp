#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../Y_2018/y_2018.h"
#include "../Y_2019/y_2019.h"

#include "Solution.hpp"
#include "RegisteredSolutions.hpp"

int main(int argc, char** argv) {
    // Setup.
    if (argc != 3)
        throw std::invalid_argument(
            "Expected two arguments year and day. Eg. 2018 1");
    int year = atoi(argv[1]);
    int day = atoi(argv[2]);
    std::cout << "Running Avent of Code " << year << " day " << day << "\n";

    // Find data.
    std::stringstream ss;
    ss << "y" << std::setfill('0') << std::setw(2) << year << "d" << std::setfill('0') << std::setw(2) << day
       << ".in";
    std::fstream fs;
    std::cout << "Opening data file " << ss.str() << "...\n";
    fs.open(ss.str());
    if (!fs.is_open())
        throw std::runtime_error("Unable to open data file");

    // Run algorithm
    std::unique_ptr<aoc::Solution> sol;
    switch (year) {
    case 2018:
        aoc::y2018::calculate(day, fs);
        break;
    case 2019:
        aoc::y2019::calculate(day, fs);
        break;
    case 2020:
        // New thingy
        sol = aoc::Solution::instantiate(year, day);
        std::cout << sol->description << std::endl;
        sol->Calculate(fs);
        break;
    default:
        throw std::runtime_error("Unrecognised year");
    }

    return 0;
}
