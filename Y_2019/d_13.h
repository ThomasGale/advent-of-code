#pragma once
#include "default.h"
#include "IntCodeComputer.h"

namespace aoc::y2019::d13 {	
	using bigint = long long;
	using namespace aoc::y2019::intcc;
	using Loc = std::tuple<int, int>;

	class BreakoutArcadeTerminal {
	public:
		BreakoutArcadeTerminal(int screenWidth, int screenHeight, IntCodeComputer game) : ScreenWidth(screenWidth), ScreenHeight(screenHeight), Game(game) {};

		void Start() {
			Game.HackState(0, 2); // Insert Coin.
			std::string input;
			while (true) {
				std::cin >> input;
				if (!std::cin) break;

				switch (std::stoi(input)) { // Hacky Key Remap
				case 1: Step(-1); break;
				case 2: Step(1); break;
				default: Step(0);
				}

				if (Game.IsHalted()) break; // Game has ended.
			}
		}

		void BotPlay(bool render) {
			Game.HackState(0, 2); // Insert Coin.
			int input = 0;
			while (true) {
				Step(input, render);
				input = (ballX < paddleX) ? -1 : (ballX > paddleX);
				if (Game.IsHalted()) break; // Game has ended.
			}

		}

		int GetScore() const { return currentScore; };

	private:

		void Step(int input, bool render = true) {
			auto rawOutput = Game.RunProgram(input);
			if (rawOutput.size() > 0) {
				ProcessOutput(rawOutput);
				if (render) Render();
			}
		}

		void ProcessOutput(const std::vector<bigint>& rawOutput) {
			int numBlocks = 0;
			for (auto i = 0; i < rawOutput.size() - 2; i += 3) {
				if (rawOutput[i] == -1) currentScore = int(rawOutput[i + 2]);
				else {
					outputState[{int(rawOutput[i]), int(rawOutput[i + 1])}] = int(rawOutput[i + 2]);
					// Update paddle and ball
					if (rawOutput[i + 2] == 3) paddleX = int(rawOutput[i]);
					if (rawOutput[i + 2] == 4) ballX = int(rawOutput[i]);
				}
			}
		}

		void Render() {
			for (auto y = 0; y < ScreenWidth; ++y) { // TODO: Replace with PDCurses.
				for (auto x = 0; x < ScreenHeight; ++x) {
					auto tile = outputState.find({ x, y });
					if (tile == outputState.end()) std::cout << " ";
					else {
						switch (tile->second) {
						case 0: std::cout << " "; break;
						case 1: std::cout << "#"; break;
						case 2: std::cout << "B"; break;
						case 3: std::cout << "_"; break;
						case 4: std::cout << "0"; break;
						}
					}
				}
				std::cout << "\n";
			}
			std::cout << "Score: " << currentScore << "\n\n";
		}

		int ScreenWidth, ScreenHeight;
		IntCodeComputer Game;

		std::map<Loc, int> outputState;
		int ballX;
		int paddleX;
		int currentScore = 0;
	};

	void calculate(std::istream& input) {
		std::cout << " Day 13 \n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<bigint> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoll(input); });

		IntCodeComputer gameTest1(inputProgram);
		std::vector<bigint> rawGameOutput;
		rawGameOutput = gameTest1.RunProgram();

		std::map<Loc, int> gameOutput;
		int numBlocks = 0;
		for (auto i = 0; i < rawGameOutput.size()-2; i+=3) {
			gameOutput[{int(rawGameOutput[i]), int(rawGameOutput[i + 1])}] = int(rawGameOutput[i + 2]);
			if (int(rawGameOutput[i + 2]) == 2) ++numBlocks;
		}
		std::cout << "1. Blocks on screen when game exits :\n";
		std::cout << numBlocks << "\n";

		// Start arcade Game 
		auto p2Start = clock::now();
		BreakoutArcadeTerminal breakout(23, 41, IntCodeComputer(inputProgram));
		breakout.BotPlay(false);
		auto p2End = clock::now();
		std::cout << "2. Final Score: " << breakout.GetScore() << "\n";
		PrintDuration(p2Start, p2End);
	}
}