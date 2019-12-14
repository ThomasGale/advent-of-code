#include "curses.h"
#include "panel.h"

namespace aoc::utils {

	void TestVisuliser() {

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

	}

}