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

	// Lhs cs, top left.
	void BasicWindow::SetChar(int x, int y, char cVal) {
		mvaddch(y, x, cVal);
	}

	int BasicWindow::GetCh() {
		return getch();
	}

	void BasicWindow::Clear(int width, int height) {
		for (auto x = 0; x < width; ++x) {
			for (auto y = 0; y < height; ++y) {
				SetChar(x, y, ' ');
			}
		}
		Update();
	}

	void BasicWindow::Update() {
		refresh();
	}
}