#pragma once

#include <fstream>
#include <stdexcept>

#include "day_01.h"

namespace aoc::y2018 {
	void run(int day, const std::istream& input) {
		switch (day)
		{
		case 1:
			d01::calculate(input); break;
		default:
			throw std::runtime_error("Unrecognised day");
		}
	}
}