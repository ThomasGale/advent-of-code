#pragma once
#include "default.h"
#include "curses.h"

namespace aoc::y2019::d14 {
	void calculate(std::istream& input) {
		//std::cout << " Day 14 \n";

		//std::string inputStr(std::istreambuf_iterator<char>(input), {});
		//std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		//std::vector<std::string> split = aoc::utils::split(inputStr, ',');


		// Testing PDCurses.
		initscr();

		printw("Hello World from PDCurses!");
		refresh();

		keypad(stdscr, true);
		erase();

		int y, x;
		for (y = 0; y < LINES - 1; y++)
			for (x = 0; x < COLS; x++)
				printw("%d", (y + x) % 10);

		
		napms(5000);
		endwin();

		//std::cout << "1. ... :\n";
		//std::cout << "" << "\n";

		//std::cout << "2. ... :\n";
		//std::cout << "" << "\n";
	}
}