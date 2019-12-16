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

	std::vector<int> SlowFFT(std::vector<int> inputSignal) { // Naive O(n^2)
		std::vector<int> filter{ 0, 1, 0, -1 };
		for (auto phase = 0; phase < 100; ++phase) {
			std::vector<int> currentPhase;
			for (auto scale = 0; scale < inputSignal.size(); ++scale) {
				auto currentFilter = stretch(scale + 1, filter);
				int sum = 0;
				for (auto el = 0; el < inputSignal.size(); ++el) {
					sum += inputSignal[el] * currentFilter[(el + 1) % (4 * (scale + 1))];
				}
				currentPhase.push_back(std::abs(sum) % 10);
			}
			inputSignal = currentPhase;
		}
		return inputSignal;
	}

	std::vector<int> CumulativeSumTrick(int location, int repeats, std::vector<int> inputSignal) {
		std::vector<int> repeatedSignal;
		for (auto i = 0; i < repeats; ++i) {
			std::copy(inputSignal.begin(), inputSignal.end(), std::back_inserter(repeatedSignal));
		}

		assert(location > repeatedSignal.size() / 2); // This trick will only work if location is in the latter half.

		std::vector<int> currentSignal; // We only need consider the range from location to end.
		std::copy(repeatedSignal.begin() + location, repeatedSignal.end(), std::back_inserter(currentSignal));

		for (auto phase = 0; phase < 100; ++phase) {
			std::vector<int> newPattern;
			std::partial_sum(currentSignal.rbegin(), currentSignal.rend(), std::back_inserter(newPattern)); // Cumlative sum (due to high offset)
			std::vector<int> newPatternClipped;
			std::transform(newPattern.rbegin(), newPattern.rend(), std::back_inserter(newPatternClipped), [](int val) { return val % 10; }); // Store only clipped final digit
			currentSignal = newPatternClipped;
		}

		return currentSignal;
	}

	void calculate(std::istream& input) {
		std::cout << " Day 16 \n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<int> signal;
		std::transform(inputStr.begin(), inputStr.end(), std::back_inserter(signal), [](char c) { return c - '0'; });

		auto startP1 = clock::now();
		auto outputP1 = SlowFFT(signal);
		std::stringstream ss1; // Stringify output.
		for (auto i = 0; i < 8; ++i) { ss1 << outputP1[i]; }
		auto endP1 = clock::now();

		std::cout << "1. First eight digits in the final output list:\n";
		std::cout << ss1.str() << "\n";
		PrintDuration(startP1, endP1);

		// Part 2.
		auto startP2 = clock::now();
		int messageOffset = std::stoi(inputStr.substr(0, 7));
		auto outputP2 = CumulativeSumTrick(messageOffset, 10'000, signal);
		std::stringstream ss2; // Stringify output.
		for (auto i = 0; i < 8; ++i) { ss2 << outputP2[i]; }
		auto endP2 = clock::now();

		std::cout << "2. Offset eight digits message in the final output list:\n";
		std::cout << ss2.str() << "\n";
		PrintDuration(startP2, endP2);
	}
}