#pragma once

#include "Common.hpp"

namespace aoc {
namespace y2018 {

class d03 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d03>();
    }

    struct Claim {

        int Id;
        std::set<std::tuple<int, int>> Region;

        Claim(const std::string& claim_str) {
            std::regex re("#(\\d+) @ (\\d+),(\\d+): (\\d+)x(\\d+)");
            std::smatch match;
            std::regex_search(claim_str, match, re);
            Id = std::stoi(match.str(1));
            int Left = std::stoi(match.str(2));
            int Top = std::stoi(match.str(3));
            int Width = std::stoi(match.str(4));
            int Height = std::stoi(match.str(5));

            for (auto i = 0; i < Width; ++i) {
                for (auto j = 0; j < Height; ++j) {
                    Region.insert({Left + i, Top + j});
                }
            }
        };
    };

    std::set<std::tuple<int, int>>
    GetIntersectionRegion(const std::set<std::tuple<int, int>>& region1,
                          const std::set<std::tuple<int, int>>& region2) {
        std::set<std::tuple<int, int>> intersectionRegion;
        std::set_intersection(
            region1.begin(), region1.end(), region2.begin(), region2.end(),
            std::inserter(intersectionRegion, intersectionRegion.begin()));
        return intersectionRegion;
    }

    void Calculate(std::istream& input) override {
        std::cout << "--- Day 3: No Matter How You Slice It ---\n";
        std::vector<std::string> input_strings =
            aoc::utils::reader::read_input(input);

        // Part 1.
        std::vector<Claim> claims;
        std::set<std::tuple<int, int>> claimedRegions;
        std::set<std::tuple<int, int>> contestedRegions;
        for (auto& input : input_strings) {
            Claim claim(input); // Create
            auto intersectionWithClaimed = GetIntersectionRegion(
                claim.Region, claimedRegions); // Check intersection with
                                               // currently claimed regions
            contestedRegions.insert(
                intersectionWithClaimed.begin(),
                intersectionWithClaimed.end()); // Build the contested /
                                                // intersection regions set
            claimedRegions.insert(
                claim.Region.begin(),
                claim.Region.end()); // Also store the whole region inside
                                     // the claimed region set.
            claims.push_back(std::move(claim)); // Store claim
        }

        std::cout << "Total overlapping area:\n";
        std::cout << contestedRegions.size() << "\n";

        // Part 2.
        for (auto& claim : claims) {
            auto intersectionWithContested =
                GetIntersectionRegion(claim.Region, contestedRegions);
            if (intersectionWithContested.size() == 0) {
                std::cout << "ID of the only claim that doesn't overlap:\n";
                std::cout << claim.Id << "\n";
                break;
            }
        }
    }
};

} // namespace y2018
} // namespace aoc
