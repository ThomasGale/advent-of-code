#pragma once

namespace aoc::utils {

	class BasicWindow {
	public:
		BasicWindow(int width, int height);
		~BasicWindow();

		BasicWindow(const BasicWindow&) = delete;
		BasicWindow& operator=(const BasicWindow&) = delete;

		// Lhs cs, top left.
		void SetChar(int x, int y, char cVal, int colour = 0);
		int GetCh();
		void Clear(int width, int height);
		void Update();
	private:
	};
}