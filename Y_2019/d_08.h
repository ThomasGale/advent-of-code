#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d08 {
	void calculate(std::istream& input) {
		std::cout << " Day xx \n";

		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<int> image;
		std::transform(inputStr.begin(), inputStr.end(), std::back_inserter(image), [](auto& input) { return input - '0'; });
		int width = 25; // 25
		int height = 6; // 6
		int imSize = width * height;

		// Find best layer.
		int bestLayer = -1;
		int bestLayerNumZeros = INT_MAX;
		int bestLayerNum1s = 0;
		int bestLayerNum2s = 0;
		for (auto layer = 0; layer < (image.size() / imSize); ++layer) {
			int numZeros = 0;
			int num1s = 0;
			int num2s = 0;
			for (auto pix = 0; pix < imSize; ++pix) {
				if (image[layer * imSize + pix] == 0) ++numZeros;
				if (image[layer * imSize + pix] == 1) ++num1s;
				if (image[layer * imSize + pix] == 2) ++num2s;
			}
			if (numZeros < bestLayerNumZeros) {
				bestLayer = layer;
				bestLayerNumZeros = numZeros;
				bestLayerNum1s = num1s;
				bestLayerNum2s = num2s;
			}
		}

		std::cout << "1. On fewest 0 digit layer, 1 * 2 count checksum:\n";
		std::cout << "Layer " <<  bestLayer << " checksum: " << bestLayerNum1s * bestLayerNum2s << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}