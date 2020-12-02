#pragma once

#include "Common.hpp"

namespace aoc {
namespace y2019 {

class d05 : public Solution {
  public:
    static std::unique_ptr<Solution> create() {
        return std::make_unique<d05>();
    }

    inline int getValue(bool valMode, int instruction,
                        const std::vector<int>& program) {
        if (valMode)
            return instruction;
        else
            return program[instruction];
    }

    void RunProgram(std::vector<int>& pState) {
        int pProg = 0;
        std::bitset<3> modes;
        int opCode = 0;

        while (true) {
            if (pState[pProg] == 99)
                break; // Terminate

            std::string opCodeStr(
                std::to_string(pState[pProg])); // Read instruction as string
            if (opCodeStr.size() == 1) {
                opCode = std::stoi(opCodeStr);
                modes = 0b000;
            } else {
                opCode = std::stoi(opCodeStr.substr(opCodeStr.size() - 2));
                modes =
                    std::bitset<3>(opCodeStr.substr(0, opCodeStr.size() - 2));
            }

            std::string input;
            switch (opCode) {
            case 1:
                pState[pState[pProg + 3]] =
                    getValue(modes[0], pState[pProg + 1], pState) +
                    getValue(modes[1], pState[pProg + 2], pState);
                pProg += 4;
                break;
            case 2:
                pState[pState[pProg + 3]] =
                    getValue(modes[0], pState[pProg + 1], pState) *
                    getValue(modes[1], pState[pProg + 2], pState);
                pProg += 4;
                break;
            case 3:
                std::cout << "Enter input: ";
                std::cin >> input;
                pState[pState[pProg + 1]] = std::stoi(input);
                pProg += 2;
                break;
            case 4:
                std::cout << getValue(modes[0], pState[pProg + 1], pState);
                pProg += 2;
                break;
            case 5:
                pProg = ((getValue(modes[0], pState[pProg + 1], pState) != 0)
                             ? getValue(modes[1], pState[pProg + 2], pState)
                             : pProg + 3);
                break;
            case 6:
                pProg = ((getValue(modes[0], pState[pProg + 1], pState) == 0)
                             ? getValue(modes[1], pState[pProg + 2], pState)
                             : pProg + 3);
                break;
            case 7:
                pState[pState[pProg + 3]] =
                    getValue(modes[0], pState[pProg + 1], pState) <
                    getValue(modes[1], pState[pProg + 2], pState);
                pProg += 4;
                break;
            case 8:
                pState[pState[pProg + 3]] =
                    getValue(modes[0], pState[pProg + 1], pState) ==
                    getValue(modes[1], pState[pProg + 2], pState);
                pProg += 4;
                break;
            default:
                throw std::runtime_error("Unrecognised opcode");
            }
        }
    }

    void Calculate(std::istream& input) override {
        std::cout << "--- Day 5: Sunny with a Chance of Asteroids ---\n";
        std::string inputStr(std::istreambuf_iterator<char>(input), {});
        std::vector<std::string> inputStrs =
            aoc::utils::reader::split(inputStr, ',');
        std::vector<int> inputProgram;
        std::transform(inputStrs.begin(), inputStrs.end(),
                       std::back_inserter(inputProgram),
                       [](auto& input) { return std::stoi(input); });

        RunProgram(inputProgram);
    }
};

} // namespace y2019
} // namespace aoc