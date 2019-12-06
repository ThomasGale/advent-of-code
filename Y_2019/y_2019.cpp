#include "y_2019.h"

#include "d_01.h"
#include "d_02.h"
#include "d_03.h"
#include "d_04.h"
#include "d_05.h"
#include "d_06.h"
#include "d_07.h"

namespace aoc::y2019 {
	void calculate(int day, std::istream& input) {
		switch (day)
		{
		case 1:
			aoc::y2019::d01::calculate(input); break;
		case 2:
			aoc::y2019::d02::calculate(input); break;
		case 3:
			aoc::y2019::d03::calculate(input); break;
		case 4:
			aoc::y2019::d04::calculate(input); break;
		case 5:
			aoc::y2019::d05::calculate(input); break;
		case 6:
			aoc::y2019::d06::calculate(input); break;
		case 7:
			aoc::y2019::d07::calculate(input); break;
		default:
			throw std::runtime_error("Unrecognised day");
		}
	}
}