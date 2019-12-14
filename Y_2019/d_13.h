#pragma once
#include "default.h"
#include "IntCodeComputer.h"
#include "curses.h"

namespace aoc::y2019::d13 {
	using namespace aoc::y2019::intcc;
	using Loc = std::tuple<int, int>;

	class BreakoutArcadeTerminal {
	public:
		BreakoutArcadeTerminal(int screenWidth, int screenHeight, IntCodeComputer game, bool visible = true) :
			Game(game), Visible(visible), window(screenWidth, screenHeight) {};

		void Start() {
			Game.HackState(0, 2); // Insert Coin.
			std::string input;
			while (true) {
				switch (window.GetCh()) { // Hacky Key Remap
				case KEY_LEFT: Step(-1); break;
				case KEY_RIGHT: Step(1); break;
				default: Step(0);
				}
				if (Game.IsHalted()) break; // Game has ended.
				std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Make playable
			}
		}

		void BotPlay() {
			Game.HackState(0, 2); // Insert Coin.
			int input = 0;
			while (true) {
				Step(input); // Game step.
				input = (ballX < paddleX) ? -1 : (ballX > paddleX); // "AI"
				if (Game.IsHalted()) break; // Game has ended.
			}

		}

		int GetScore() const { return currentScore; };

	private:
		void Step(int input) {
			auto rawOutput = Game.RunProgram(input);
			if (rawOutput.size() > 0) {
				ProcessOutput(rawOutput);
				if (Visible) Render();
			}
		}

		void ProcessOutput(const std::vector<bigint>& rawOutput) {
			for (auto i = 0; i < rawOutput.size() - 2; i += 3) {
				if (rawOutput[i] == -1) currentScore = int(rawOutput[i + 2]);
				else {
					outputState[{int(rawOutput[i]), int(rawOutput[i + 1])}] = int(rawOutput[i + 2]);
					if (rawOutput[i + 2] == 3) paddleX = int(rawOutput[i]); // Update paddle and ball
					if (rawOutput[i + 2] == 4) ballX = int(rawOutput[i]);
				}
			}
		}

		void Render() {
			for (auto& tile : outputState) {
				char c = ' ';
				switch (tile.second) {
				case 1: c = '#'; break;
				case 2: c = 'B'; break;
				case 3: c = '_'; break;
				case 4: c = '0'; break;
				}
				window.SetChar(std::get<0>(tile.first), std::get<1>(tile.first), c);
			}
			window.Update();
		}

		IntCodeComputer Game;
		bool Visible;

		aoc::utils::BasicWindow window;
		std::map<Loc, int> outputState;
		int ballX;
		int paddleX;
		int currentScore = 0;
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 13: Care Package ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		std::vector<std::string> inputStrs = aoc::utils::split(inputStr, ',');
		std::vector<bigint> inputProgram;
		std::transform(inputStrs.begin(), inputStrs.end(), std::back_inserter(inputProgram), [](auto& input) { return std::stoll(input); });

		IntCodeComputer gameTest1(inputProgram);
		std::vector<bigint> rawGameOutput;
		rawGameOutput = gameTest1.RunProgram();

		std::map<Loc, int> gameOutput;
		int numBlocks = 0;
		for (auto i = 0; i < rawGameOutput.size() - 2; i += 3) {
			gameOutput[{int(rawGameOutput[i]), int(rawGameOutput[i + 1])}] = int(rawGameOutput[i + 2]);
			if (int(rawGameOutput[i + 2]) == 2) ++numBlocks;
		}
		std::cout << "1. Blocks on screen when game exits :\n";
		std::cout << numBlocks << "\n";

		auto p2Start = clock::now();
		BreakoutArcadeTerminal breakout(23, 41, IntCodeComputer(inputProgram), false);
		breakout.BotPlay(); // breakout.Start();
		auto p2End = clock::now();
		std::cout << "2. Final Score: " << breakout.GetScore() << "\n";
		PrintDuration(p2Start, p2End);
	}
}