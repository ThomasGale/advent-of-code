#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d17 {
	using namespace aoc::y2019::intcc;
	using namespace aoc::utils;

	struct Segment {
		std::set<Vec2, Vec2LessComp> Start;
		std::set<Vec2, Vec2LessComp> End;
	};

	struct Scaffold {
		Scaffold(std::set<Vec2, Vec2LessComp> scaffold) : Data(scaffold) {};
		std::set<Vec2, Vec2LessComp> Data;

		bool IsVecLeft(const Vec2& loc) const { return Data.find(Vec2(loc.X - 1, loc.Y)) != Data.end(); };
		bool IsVecRight(const Vec2& loc) const { return Data.find(Vec2(loc.X + 1, loc.Y)) != Data.end(); };
		bool IsVecUp(const Vec2& loc) const { return Data.find(Vec2(loc.X, loc.Y - 1)) != Data.end(); };
		bool IsVecDown(const Vec2& loc) const { return Data.find(Vec2(loc.X, loc.Y + 1)) != Data.end(); };

		std::set<Vec2, Vec2LessComp>::iterator GetVecLeft(const Vec2& loc) const { return Data.find(Vec2(loc.X - 1, loc.Y)); };
		std::set<Vec2, Vec2LessComp>::iterator GetVecRight(const Vec2& loc) const { return Data.find(Vec2(loc.X + 1, loc.Y)); };
		std::set<Vec2, Vec2LessComp>::iterator GetVecUp(const Vec2& loc) const { return Data.find(Vec2(loc.X, loc.Y - 1)); };
		std::set<Vec2, Vec2LessComp>::iterator GetVecDown(const Vec2& loc) const { return Data.find(Vec2(loc.X, loc.Y + 1)); };

		std::set<Vec2, Vec2LessComp> GetCorners() const {
			std::set<Vec2, Vec2LessComp> corners;
			for (auto loc : Data) {
				auto isLeft = IsVecLeft(loc);
				auto isRight = IsVecRight(loc);
				auto isUp = IsVecUp(loc);
				auto isDown = IsVecDown(loc);

				// If mid point either not horizonal or vertical.
				if ((isLeft && isRight) && !(isUp && isDown)) { // Horizontal
				}
				else if (!(isLeft && isRight) && (isUp && isDown)) { // Vertical
				}
				else {
					corners.insert(loc);
				}

			}
			return corners;
		}

		std::vector<Segment> GetSegments() {
			std::set<Vec2, Vec2LessComp> unVisited(Data);
			std::vector<Segment> segments;
			Vec2 currPoint;



			//int i = 0;

			return segments;
		}
	};

	void RenderScaffold(BasicWindow& window, const std::vector<bigint>& currentView) {
		int x = 0, y = 0;
		for (auto code : currentView) {
			window.SetChar(y, x, code);

			if (code == 10) {
				x = 0;
				++y;
			}
			else {
				++x;
			}
		}
		window.Update();
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 17: Set and Forget ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});

		// Part 1.
		aoc::utils::BasicWindow window(51, 41);
		IntCodeComputer asciiProg1(inputStr);
		auto currentView = asciiProg1.RunProgram();

		int x = 0, y = 0;
		Vec2 robotPos;
		Mat2x2 roboOri;
		std::set<Vec2, Vec2LessComp> scaffoldVecs;
		for (auto code : currentView) {
			if (code == '#') {
				scaffoldVecs.insert(Vec2(x, y));
			}
			else if (code == '^') {
				robotPos = Vec2(x, y);
				roboOri = Vec2(0, -1), Vec2(1, 0);
			}
			else if (code == 'v') {
				robotPos = Vec2(x, y);
				roboOri = Vec2(0, 1), Vec2(-1, 0);
			}
			else if (code == '>') {
				robotPos = Vec2(x, y);
				roboOri = Vec2(1, 0), Vec2(0, 1);
			}
			else if (code == '<') {
				robotPos = Vec2(x, y);
				roboOri = Vec2(-1, 0), Vec2(0, -1);
			}

			if (code == 10) {
				x = 0;
				++y;
			}
			else {
				++x;
			}
		}
		
		RenderScaffold(window, currentView);

		// Part 1.
		int alignementParametersSum = 0;
		for (auto loc : scaffoldVecs) {
			auto left = scaffoldVecs.find(Vec2(loc.X - 1, loc.Y));
			auto right = scaffoldVecs.find(Vec2(loc.X + 1, loc.Y));
			auto up = scaffoldVecs.find(Vec2(loc.X, loc.Y - 1));
			auto down = scaffoldVecs.find(Vec2(loc.X, loc.Y + 1));

			if (left != scaffoldVecs.end() && right != scaffoldVecs.end() &&
				up != scaffoldVecs.end() && down != scaffoldVecs.end()) {
				alignementParametersSum += loc.X * loc.Y;
			}
		}

		std::cout << "1. Sum of the alignment parameters:\n";
		std::cout << alignementParametersSum << "\n";


		// I will continue with this at some point..

		// Create two way connected graph of segments and vertices for the scaffold.
		// Assemble the 3 proposed sets of segments
		// Search for a while check each time that the resultant path covers the entire path of the scaffold?

		// Or.. if there is no repeated path coverage, first solve the shortest path to reduce the problem space.

		Scaffold scaffold(scaffoldVecs);
		auto corners = scaffold.GetCorners();

		// Render Corners 
		for (auto corner : corners) {
			window.SetChar(corner.Y, corner.X, 'C');
		}
		window.Update();

		IntCodeComputer asciiProg2(inputStr);
		asciiProg2.HackState(0, 2);

		//std::cout << "2. ... :\n";
		//std::cout << "" << "\n";
	}
}