#pragma once
#include "default.h"
#include "curses.h"
#include "panel.h"

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

		//int y, x;
		//for (y = 0; y < LINES - 1; y++)
		//	for (x = 0; x < COLS; x++)
		//		printw("%d", (y + x) % 10);

		WINDOW* subWin = newwin(15, 15, 0, 0);
		PANEL* p = new_panel(subWin);
		set_panel_userptr(p, "p");

		char num = *((char*)p->user + 1);
		int y, x, maxy, maxx;

		box(subWin, 0, 0);
		mvwprintw(subWin, 1, 1, "-pan%c-", num);
		getmaxyx(subWin, maxy, maxx);

		for (y = 2; y < maxy - 1; y++)
			for (x = 1; x < maxx - 1; x++)
				mvwaddch(subWin, y, x, num);
	
		update_panels();
		doupdate();

		napms(5000);
		endwin();

		//std::cout << "1. ... :\n";
		//std::cout << "" << "\n";

		//std::cout << "2. ... :\n";
		//std::cout << "" << "\n";
	}
}