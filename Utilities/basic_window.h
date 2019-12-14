#pragma once

namespace aoc::utils {

	class BasicWindow {
	public:
		BasicWindow(int width, int height);
		~BasicWindow();

		BasicWindow(const BasicWindow&) = delete;
		BasicWindow& operator=(const BasicWindow&) = delete;

		void SetChar(int col, int line, char cVal);
		int GetCh();
		void Update();
	private:
	};
}