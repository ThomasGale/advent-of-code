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
		double beamGrowthWidthPerY = ((double)xb2 - (double)xb1) / 50.0;
		double beamX1MovePerY = ((double)xb1) / 50.0;
		double beamX2MovePerY = ((double)xb2) / 50.0;
		
		int yTest = int(100.0 / beamGrowthWidthPerY); // Starting position at least 100.
		int x1PosAvail = 0;
		int x1PosNeeded = -1;
		while (x1PosNeeded < x1PosAvail) {
			//int x1PosAtYTest = beamX1MovePerY * yTest;
			int x2PosAtYTest = beamX2MovePerY * yTest;

			x1PosAvail = beamX1MovePerY * (yTest + 100);
			x1PosNeeded = (x2PosAtYTest - 100);
			++yTest;
		}
		int finalX = (beamX2MovePerY * yTest) - 100;

		// Test with tractor beam.
		window.Clear(50, 50);

		// TEST
		auto XTest = finalX + 2;
		auto YTest = yTest + 100 + 1;
		//auto XTest = 25;
		//auto YTest = 55;
		for (auto x = 0; x < 50; ++x) {
			for (auto y = 0; y < 50; ++y) {
				auto xTest = x + XTest - 25;
				auto yTest = y + YTest - 25;
				IntCodeComputer test(tractorProg1);
				test.RunProgram(xTest);
				auto output = test.RunProgram(yTest).front();
				window.SetChar(x, y, (output == 0 ? '.' : '#'));
			}
		}
		window.SetChar(25, 25, 'X');
		window.Update();

		std::cout << "2. Coordinates of square location:\n";
		std::cout << "X: " << finalX << " Y: " << yTest << " multiple sum: " << finalX * 10000 + yTest + 1 << "\n";
	}
}