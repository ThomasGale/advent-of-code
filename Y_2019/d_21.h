#pragma once
#include "default.h"

namespace aoc::y2019::d21 {
	using namespace aoc::y2019::intcc;
	using namespace aoc::utils;

	std::vector<bigint> ConvertSpringScriptToRaw(const std::vector<std::string>& script) {
		std::vector<bigint> testInputRaw;
		for (const auto& line : script) {
			for (auto ch : line) {
				testInputRaw.push_back(ch);
			}
			testInputRaw.push_back('\n');
		}
		return testInputRaw;
	}

	void RenderSpringOut(BasicWindow& window, const std::vector<bigint>& output) {
		window.Clear(50, 50);
		int x = 0, y = 0;
		for (auto val : output) {
			if (int(val) == '\n') {
				++y;
				x = 0;
			}
			window.SetChar(x, y, int(val));
			++x;
		}
		window.Update();
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 21: Springdroid Adventure ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});
		BasicWindow window(50, 50);

		/*  Notes on spring script in WALK mode.
		- Jump moves 4 spaces
		- 15 instruction limit
		- T w/r register (init 0)
		- J w/r regiester (J true will jump at end of program) (init 0)
		- A, B, C, D  r only register (ground at 1, 2, 3, 4 tiles away.)
		- AND X Y (output in Y)
		- OR X Y (output in Y)
		- NOT X Y (output in Y)
		*/

		// Part 1.
		std::vector<std::string> part1Input{
			"NOT T T", // Sense up coming gap.
			"AND A T", // ..
			"AND B T", // ..
			"AND C T", // ..
			"NOT T T", // If T = 1, there is a gap coming up.
			"NOT J J", // Only jump if landing spot.
			"AND D J", // ..
			"AND T J", // Combine gap detection and landing spot.
			"WALK"
		};

		IntCodeComputer sprBotP1(inputStr);
		auto part1InputRaw = ConvertSpringScriptToRaw(part1Input);
		auto part1 = sprBotP1.RunProgram(part1InputRaw);
		RenderSpringOut(window, part1);
		bigint damage = 0;
		std::for_each(part1.begin(), part1.end(), [&damage](auto val) { if (val > 128) { damage = val; } }); // Output damage is value larger than ASCII range.

		std::cout << "1. Hull damage :\n";
		std::cout << damage << "\n";

		// Part 2. 

	    /*  Notes on spring script in RUN mode.
		- Jump moves 4 spaces and slides 1 space.
		- 15 instruction limit
		- T w/r register (init 0)
		- J w/r regiester (J true will jump at end of program) (init 0)
		- A, B, C, D, E, F, G, H, I  r only register (ground at 1, 2, 3, 4, 5, 6, 7, 8, 9 tiles away.)
		- AND X Y (output in Y)
		- OR X Y (output in Y)
		- NOT X Y (output in Y)
		*/

		std::vector<std::string> part2Input{
			"NOT T T", // Sense up coming gap.
			"AND A T", // ..
			"AND B T", // ..
			"AND C T", // ..
			"NOT T T", // If T = 1, there is a gap coming up.
			"NOT J J", // Only jump if landing spot.
			"AND D J", // ..
			"AND T J", // Combine gap detection and landing spot.
			"RUN"
		};

		// Notes, we may need to now build a method for generating inputs. There is a combinatorial set of gap pairs that will need to be detected by the spring bot script to deduce when is the
		// best time to jump.

		IntCodeComputer sprBotP2(inputStr);
		auto part2InputRaw = ConvertSpringScriptToRaw(part2Input);
		auto part2 = sprBotP2.RunProgram(part2InputRaw);
		RenderSpringOut(window, part2);
		damage = 0;
		std::for_each(part2.begin(), part2.end(), [&damage](auto val) { if (val > 128) { damage = val; } }); // Output damage is value larger than ASCII range.

		std::cout << "2. Rest of hull damage:\n";
		std::cout << damage << "\n";
	}
}