#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d19 {
	using namespace aoc::y2019::intcc;
	using namespace aoc::utils;

	void calculate(std::istream& input) {
		std::cout << "--- Day 19: Tractor Beam ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		IntCodeComputer tractorProg1(inputStr);
		BasicWindow window(50, 50);

		int xb1 = 0, yb1 = 0; // Upper beam edge grad
		int xb2 = 0, yb2 = 0; // Lower beam edge grad
		int tractorBeamAffectedPoints = 0;
		for (auto x = 0; x < 50; ++x) {
			for (auto y = 0; y < 50; ++y) {
				IntCodeComputer test(tractorProg1); // Hacky reset of program.
				test.RunProgram(x);
				auto output = test.RunProgram(y).front();
				window.SetChar(x, y, (output == 0 ? '.' : '#'));
				tractorBeamAffectedPoints += (output == 1);

				if (y == 49) { // This assumptiont that the beam hits the right edge may only be true for my example.
					if (xb1 == 0 && output == 1) { // hit top edge of beam
						xb1 = x;
						yb1 = y;
					}
					else if (xb1 != 0 && xb2 == 0 && output == 0) {// hit bottom edge of beam
						xb2 = x;
						yb2 = y;
					}
				}
			}
		}
		window.SetChar(25, 0, 'X'); // Sanity check.
		window.SetChar(0, 25, 'Y');

		window.Update();

		std::cout << "1. Points that are affected by the tractor beam:\n";
		std::cout << tractorBeamAffectedPoints << "\n";

		// Part 2. 
		double beamGrowthWidthPerY = ((double)xb2 - (double)xb1) / 50.0; // Approx growth of width of beam.
		double beamX1MovePerY = ((double)xb1) / 50.0; // Approx grad of upper beam
		double beamX2MovePerY = ((double)xb2) / 50.0; // Approx grad of lower beam
		double beamAvCentrePerY = (beamX1MovePerY + beamX2MovePerY) / 2.0; // Approx grad of beam centre.
		
		int squareSize = 100;
		int ySqrTop = int(double(squareSize) / beamGrowthWidthPerY); // Starting position of Y..
		int xSqrR;
		int xSqrL;
		int ySqrBot;
		while (true) {
			xSqrR = beamAvCentrePerY * ySqrTop; // Starting position
			int sqrTopR = 1;
			while (true) { // Search for top right of square in beam.
				IntCodeComputer test(tractorProg1);
				test.RunProgram(xSqrR);
				sqrTopR = test.RunProgram(ySqrTop).front();
				if (sqrTopR == 0) break;
				++xSqrR;
			}
			xSqrL = xSqrR - squareSize;
			ySqrBot = ySqrTop; // Starting position
			int sqrBotL = 1;
			while (true) { // Search for bottom left square in beam.
				IntCodeComputer test(tractorProg1);
				test.RunProgram(xSqrL);
				sqrBotL = test.RunProgram(ySqrBot).front();
				if (sqrBotL == 0) break;
				++ySqrBot;
			}
			if (ySqrBot - ySqrTop >= squareSize) break;
			++ySqrTop; // Step down search in Y.
		}
	
		// Test with tractor beam.
		window.Clear(50, 50);

		// Test plot (for small squares).
		auto XTest = xSqrL + squareSize;
		auto YTest = ySqrTop;
		for (auto x = 0; x < 50; ++x) {
			for (auto y = 0; y < 50; ++y) {
				auto xTest = x + XTest - 25;
				auto yTest = y + YTest - 25;
				IntCodeComputer test(tractorProg1);
				test.RunProgram(xTest);
				auto output = test.RunProgram(yTest).front();
				window.SetChar(x, y, (output == 0 ? '.' : '#'));
				if (xTest >= xSqrL && xTest < xSqrL + squareSize && yTest >= ySqrTop && yTest < ySqrTop + squareSize) {
					window.SetChar(x, y, 'O');
				}
			}
		};
		window.Update();

		std::cout << "2. Coordinates of square location:\n";
		std::cout << "X: " << xSqrL << " Y: 9" << ySqrTop << " multiple sum: " << (xSqrL * 10000) + ySqrTop << "\n";
		// 10771728 Wrong.
	}
}