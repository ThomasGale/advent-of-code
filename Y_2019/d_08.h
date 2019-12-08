#pragma once
#include "default.h"
#include <set>

namespace aoc::y2019::d08 {

	class Image {
	public:
		Image(std::vector<int> data, int width, int height) : _data(data), _width(width), _height(height) {
			_size = _width * _height;

			for (auto layerStartIt = _data.begin(); layerStartIt != _data.end(); layerStartIt += _size) {
				_layers.push_back(layerStartIt);
			}
		};

		Image Flatten() {
			std::vector<int> _flatData(_size, -1);
			for (auto layer : _layers) {
				for (auto i = 0; i < _size; ++i) {
					if ((*(layer + i) != 2) && (_flatData[i] == -1)) {
						_flatData[i] = *(layer + i);
					}
				}
			}
			return Image(_flatData, _width, _height);
		}

		void DrawLayer(int layerIndex) {
			for (auto layerIt = _layers[layerIndex]; layerIt != _layers[layerIndex] + _size; ++layerIt) {
				if (((layerIt - _layers[layerIndex]) % _width) == 0) std::cout << "\n";
				std::cout << ((*layerIt == 1) ? "0" : " "); // White is a 0, black is left empty.
			}
			std::cout << "\n";
		}

	private:
		std::vector<int> _data;
		std::vector<std::vector<int>::iterator> _layers; // Iterator pointing to the start of a layer in the _data.
		int _width, _height, _size;
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 8: Space Image Format ---\n";

		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<int> imageData;
		std::transform(inputStr.begin(), inputStr.end(), std::back_inserter(imageData), [](auto& input) { return input - '0'; });
		int width = 25;
		int height = 6;
		int imSize = width * height;

		// Part 1. Find best layer.
		int bestLayer = -1;
		int bestLayerNumZeros = INT_MAX;
		int bestLayerNum1s = 0;
		int bestLayerNum2s = 0;
		for (auto layer = 0; layer < (imageData.size() / imSize); ++layer) {
			int numZeros = 0;
			int num1s = 0;
			int num2s = 0;
			for (auto pix = 0; pix < imSize; ++pix) {
				if (imageData[layer * imSize + pix] == 0) ++numZeros;
				if (imageData[layer * imSize + pix] == 1) ++num1s;
				if (imageData[layer * imSize + pix] == 2) ++num2s;
			}
			if (numZeros < bestLayerNumZeros) {
				bestLayer = layer;
				bestLayerNumZeros = numZeros;
				bestLayerNum1s = num1s;
				bestLayerNum2s = num2s;
			}
		}

		std::cout << "1. On fewest 0 digit layer, 1 * 2 count checksum:\n";
		std::cout << "Layer " << bestLayer << " checksum: " << bestLayerNum1s * bestLayerNum2s << "\n";

		// Part 2. Draw the flattened image.
		Image image(imageData, width, height);
		auto flatImage = image.Flatten();

		std::cout << "2. Decoded Image:\n";
		flatImage.DrawLayer(0);
	}
}