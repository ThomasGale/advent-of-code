#pragma once
#include "default.h"

namespace aoc::y2019::d16 {	

	std::vector<int> stretch(int scale, std::vector<int> input) {
		std::vector<int> result;
		for (auto val : input) {
			for (auto num = 0; num < scale; ++num) {
				result.push_back(val);
			}
		}
		return result;
	}



	void calculate(std::istream& input) {
		std::cout << " Day 16 \n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<int> signal;
		std::transform(inputStr.begin(), inputStr.end(), std::back_inserter(signal), [](char c) { return c - '0'; });

		std::vector<int> filter{ 0, 1, 0, -1 };
		auto test = stretch(1, filter);

		std::vector<int> currentOutput(signal);
		for (auto phase = 0; phase < 100; ++phase) {
			std::vector<int> currentPhase;
			for (auto scale = 0; scale < currentOutput.size(); ++scale) { // Loop over each element
				auto currentFilter = stretch(scale + 1, filter);
				int sum = 0;
				for (auto el = 0; el < signal.size(); ++el) {
					sum += currentOutput[el] * currentFilter[(el + 1) % (4 * (scale + 1))];
				}
				std::string sumStr = std::to_string(sum);
				currentPhase.push_back(sumStr.back() - '0'); // Add 
			}
			currentOutput = currentPhase;
		}


		std::stringstream ss;
		for (auto i = 0; i < 8; ++i) {
			ss << currentOutput[i];
		}

		std::cout << "1. First eight digits in the final output list:\n";
		std::cout << ss.str() << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}