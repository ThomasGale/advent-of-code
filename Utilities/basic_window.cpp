#include "basic_window.h"
#include "curses.h"
#include "panel.h"

namespace aoc::utils {

	BasicWindow::BasicWindow(int width, int height) {
		initscr();
		resize_term(height, width);
		keypad(stdscr, TRUE);
	}

	BasicWindow::~BasicWindow() {
		endwin();
	}

	void BasicWindow::SetChar(int x, int y, char cVal) {
		mvaddch(x, y, cVal);
	}

	int BasicWindow::GetCh() {
		return getch();
	}

	void BasicWindow::Update() {
		refresh();
	}
}