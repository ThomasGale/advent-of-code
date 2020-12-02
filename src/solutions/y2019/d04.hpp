#pragma once

#include "Common.hpp"

namespace aoc {
namespace y2019 {

class d04 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d04>();
    }

    void Calculate(std::istream& input) override {
        std::vector<std::string> input_strings =
            utils::reader::read_input(input);
        std::vector<std::string> passwordRangeStr =
            utils::reader::split(input_strings[0], '-');
        int lowerRange = std::stoi(passwordRangeStr[0]);
        int upperRange = std::stoi(passwordRangeStr[1]);

        int validCount = 0;
        for (int pw = lowerRange; pw <= upperRange; ++pw) {
            auto pwStr = std::to_string(pw);

            bool incrOk = true;
            for (auto i = 0; i < pwStr.size() - 1; ++i) {
                if (pwStr[i + 1] < pwStr[i])
                    incrOk = false;
            }

            std::map<int, char> pairLocations;
            for (auto i = 0; i < pwStr.size() - 1; ++i) {
                if (pwStr[i] == pwStr[i + 1]) {
                    pairLocations[i] = pwStr[i];
                }
            }

            bool isolatedPair = false;
            for (auto pairLoc : pairLocations) {
                if ((pairLocations.find(pairLoc.first - 1) ==
                     pairLocations.end()) &&
                    (pairLocations.find(pairLoc.first + 1) ==
                     pairLocations.end())) {
                    isolatedPair = true;
                }
            }

            if (incrOk && isolatedPair) {
                ++validCount;
            }
        }

        std::cout << "2. Passwords that meet criteria:\n";
        std::cout << validCount << "\n";
    }
};

} // namespace y2019
} // namespace aoc