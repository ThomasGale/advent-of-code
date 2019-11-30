#include "run.h"


int main(int argc, char** argv)
{
	// Setup.
	if (argc != 3) throw std::invalid_argument("Expected two arguments year and day. Eg. 2018 1");
	int year = atoi(argv[1]);
	int day = atoi(argv[2]);
	std::cout << "Running Avent of Code " << year << " day " << day << "\n";

	// Find data.
	std::stringstream ss;
	ss << "" << year << "/day_" << std::setfill('0') << std::setw(2) << day << ".txt";
	std::fstream fs;
	std::cout << "Opening data file " << ss.str() << "...\n";
	fs.open(ss.str());
	if (!fs.is_open()) throw std::runtime_error("Unable to open data file");

	// Run algorithm
	switch (year)
	{
	case 2018:
		aoc::y2018::calculate(day, fs); break;
	case 2019:
	default:
		throw std::runtime_error("Unrecognised year and day");;
	}

	return 0;
}
