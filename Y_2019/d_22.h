#pragma once
#include "default.h"

namespace aoc::y2019::d22 {

	enum class Technique
	{
		DealNewStack, // Reverse
		Cut, // Take top N cards of top and move as single unit (or bottom to top for -ve)
		DealWithIncrement, // Deal out cards in N interval spacing, cycle around. 
	};

	std::vector<std::pair<Technique, int>> ParseShuffleInput(const std::vector<std::string>& input) {
		std::vector<std::pair<Technique, int>> shuffleProcess;
		std::regex cutPat("cut (-?\\d+)");
		std::smatch cutMatch;
		std::regex dealIncPat("deal with increment (-?\\d+)");
		std::smatch dealIncMatch;
		for (const auto& line : input) {
			std::regex_match(line, cutMatch, cutPat);
			std::regex_match(line, dealIncMatch, dealIncPat);
			if (line == "deal into new stack") {
				shuffleProcess.push_back({ Technique::DealNewStack, 0 });
			}
			else if (cutMatch.size() == 2) {
				shuffleProcess.push_back({ Technique::Cut, std::stoi(cutMatch[1]) });
			}
			else if (dealIncMatch.size() == 2) {
				shuffleProcess.push_back({ Technique::DealWithIncrement, std::stoi(dealIncMatch[1]) });
			}
		}
		return shuffleProcess;
	}

	void calculate(std::istream& input) {
		std::cout << " Day 22 \n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		auto shuffleProcess = ParseShuffleInput(inputStrs);

		std::vector<int> deck(10007);
		std::iota(deck.begin(), deck.end(), 0);

		// Part 1. Perform shuffling ops.
		for (const auto& process : shuffleProcess) {
			std::vector<int> table; // Used for Deal with Increment.
			switch (process.first)
			{
			case Technique::DealNewStack:
				std::reverse(deck.begin(), deck.end()); break;
			case Technique::Cut:
				if (process.second >= 0) {
					std::rotate(deck.begin(), deck.begin() + process.second, deck.end()); break;
				}
				else {
					std::rotate(deck.begin(), deck.end() + process.second, deck.end()); break;
				}
			case Technique::DealWithIncrement:
				table = std::vector<int>(deck.size());
				for (auto i = 0; i < deck.size(); ++i) {
					table[(i * process.second) % deck.size()] = deck[i];
				}
				deck = table; break;
			default:
				break;
			}
			//std::for_each(deck.begin(), deck.end(), [](int el) { std::cout << el;  });
			//std::cout << "\n";
		}

		// Find 2019
		auto card = std::find(deck.begin(), deck.end(), 2019);

		std::cout << "1. Position of card 2019:\n";
		std::cout << card - deck.begin() << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}