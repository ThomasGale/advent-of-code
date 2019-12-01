#include "y_2019.h"

#include "d_01.h"

namespace aoc::y2019 {
	void calculate(int day, std::istream& input) {
		switch (day)
		{
		case 1:
			aoc::y2019::d01::calculate(input); break;
		default:
			throw std::runtime_error("Unrecognised day");
		}
	}
}