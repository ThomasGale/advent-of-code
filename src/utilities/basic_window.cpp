#include "basic_window.h"
#include "ncurses.h"

namespace aoc::utils {

	BasicWindow::BasicWindow(int width, int height) {
		initscr();
		start_color();
		init_pair(0, COLOR_WHITE, COLOR_BLACK); // Normal
		init_pair(1, COLOR_GREEN, COLOR_BLACK); // Accent
		init_pair(2, COLOR_WHITE, COLOR_CYAN); // Active
		init_pair(3, COLOR_WHITE, COLOR_YELLOW); // Highlight
		keypad(stdscr, TRUE);
	}

	BasicWindow::~BasicWindow() {
		endwin();
	}

	// Lhs cs, top left.
	void BasicWindow::SetChar(int x, int y, char cVal, int colour) {
		attron(COLOR_PAIR(colour));
		mvaddch(y, x, cVal);
		attroff(COLOR_PAIR(colour));
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