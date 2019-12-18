#include "basic_window.h"
#include "curses.h"
#include "panel.h"

namespace aoc::utils {

	BasicWindow::BasicWindow(int width, int height) {
		initscr();
		resize_term(height, width);
		keypad(stdscr, TRUE);
		printw("PDCurses basic window!");
	}

	BasicWindow::~BasicWindow() {
		endwin();
	}

	void BasicWindow::SetChar(int col, int line, char cVal) {
		mvaddch(line, col, cVal);
	}

	int BasicWindow::GetCh() {
		return getch();
	}

	void BasicWindow::Update() {
		refresh();
	}
}