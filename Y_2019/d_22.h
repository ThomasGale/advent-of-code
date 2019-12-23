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

	// Ripped from https://www.geeksforgeeks.org/how-to-avoid-overflow-in-modular-multiplication/
	// To compute (a * b) % mod 
	bigint mulmod(bigint a, bigint b, bigint mod)
	{
		bigint res = 0; // Initialize result 
		a = a % mod;
		while (b > 0)
		{
			// If b is odd, add 'a' to result 
			if (b % 2 == 1)
				res = (res + a) % mod;

			// Multiply 'a' with 2 
			a = (a * 2) % mod;

			// Divide b by 2 
			b /= 2;
		}

		// Return result 
		return res % mod;
	}

	// Ripped from: https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
	// To compute x^y under modulo m 
	bigint power(bigint x, bigint y, bigint m)
	{
		if (y == 0)
			return 1;
		bigint p = power(x, y / 2, m) % m;
		p = (p * p) % m;

		return (y % 2 == 0) ? p : (x * p) % m;
	}

	// Ripped from: https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
	// Function to find modular inverse of a under modulo m 
	// Assumption: m is prime 
	bigint modInverse(bigint a, bigint m)
	{
		bigint g = std::gcd(a, m);
		if (g != 1) {
			throw std::runtime_error("Inverse doesn't exist");
		}
		else {
			return power(a, m - 2, m);
		}
	}


	std::vector<bigint> computeRange(bigint rangeStart, bigint rangeEnd, bigint offset, bigint increment, bigint deckSize) {
		std::vector<bigint> deckIndices(rangeEnd - rangeStart);
		std::iota(deckIndices.begin(), deckIndices.end(), rangeStart);
		std::vector<bigint> computedDeck(deckIndices.size());
		//std::transform(deckIndices.begin(), deckIndices.end(), computedDeck.begin(), [&offset, &increment, &deckSize](bigint index) { return (((index % deckSize) * increment) % deckSize + (offset % deckSize) + deckSize) % deckSize; }); //
		std::transform(deckIndices.begin(), deckIndices.end(), computedDeck.begin(), [&offset, &increment, &deckSize](bigint index) { return (index * increment +  offset + deckSize) % deckSize; }); //
		return computedDeck;
	}

	void calculate(std::istream& input) {
		std::cout << " Day 22 \n";
		std::vector<std::string> inputStrs = aoc::utils::read_input(input);
		auto shuffleProcess = ParseShuffleInput(inputStrs);

		std::vector<int> deck(11);
		std::iota(deck.begin(), deck.end(), 0);

		// Part 1. Naive full array shuffling ops.
		for (const auto& process : shuffleProcess) {
			std::vector<int> table;
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
				throw std::runtime_error("Not recognised");
			}
			//std::for_each(deck.begin(), deck.end(), [](int el) { std::cout << std::setfill('0') << std::setw(2) << el << " ";  });
			//std::cout << "\n";
		}

		auto card = std::find(deck.begin(), deck.end(), 2019); // Find 2019
		std::cout << "1. Position of card 2019:\n";
		std::cout << card - deck.begin() << "\n";

		// Part 2. Requires knowledge of Modular Arithmetic / Multiplicative Inverse, 31.3 from CLRS.
		// Because size is prime - mod inverse can be solve with Fermats' little theorem?

		// TESTING.
		bigint deckSize = 11;// 119'315'717'514'047;
		bigint shuffleRepeat = 3;// 101'741'582'076'661;

		// Rather than computing the deck fully, encode the deck state into a minimal form:
		// All computations are MOD deck size (to keep cycling around).
		bigint increment = 1; // Spacing of adjacent numbers
		bigint offset = 0; // Offset of first number in deck from beginning
		for (const auto& process : shuffleProcess) {
			std::vector<int> table;
			switch (process.first)
			{
			case Technique::DealNewStack:
				increment *= -1; // Reverse the list.
				offset += increment; // The offset is shifted to correctly cycle around when reversed.
				break;
			case Technique::Cut:
				increment += process.second; // Simply move the start.
				break;
			case Technique::DealWithIncrement:
				// Hard bit..
				// The incrememnt deal stretches the increment
				break;
			default:
				throw std::runtime_error("Not recognised");
			}
			increment %= deckSize; // MOD result.
			offset %= deckSize; // MOD result.
		}

		// TESTING.
		for (auto i = 0; i < 3; ++i) {
			auto testRange = computeRange(0, 11, -1, -1, 11);
			std::for_each(testRange.begin(), testRange.end(), [](int el) { std::cout << std::setfill('0') << std::setw(2) << el << " ";  });
			std::cout << "\n";
		}

		// Observations
		// No cut is larger than 10,000
		// No increment is larger than 100
		// Probably only need to store first and last 10,000 cards in memory. 
		// Position 2020 is near the beginnning!
	
		// For the repeats, we need to cache all results.
		// Find repeating patterns that are a multiple / prime factor of the shuffle repeat?
		
		//std::vector<std::vector<bigint>> shuffledStartDecks;
		//std::vector<std::vector<bigint>> shuffledEndDecks;

		//std::vector<bigint> startDeck(10'000);
		//std::vector<bigint> endDeck(10'000);

		//// Part 2. Perform shuffling ops.
		//for (auto shuffle = 0; shuffle < 5; ++shuffle) {
		//	for (const auto& process : shuffleProcess) {
		//		std::vector<bigint> copy; // Used for Deal with Increment.
		//		switch (process.first)
		//		{
		//		case Technique::DealNewStack:
		//			std::reverse(startDeck.begin(), startDeck.end()); // Reverse start and end and toggle position.
		//			std::reverse(endDeck.begin(), endDeck.end());
		//			copy = startDeck;
		//			startDeck = endDeck;
		//			endDeck = copy;
		//			break;
		//		case Technique::Cut:
		//			if (process.second >= 0) {
		//				//std::rotate(startDeck.begin(), startDeck.begin() + process.second, deck.end()); break;
		//			}
		//			else {
		//				//std::rotate(deck.begin(), deck.end() + process.second, deck.end()); break;
		//			}
		//			break;
		//		case Technique::DealWithIncrement:
		//			/*table = std::vector<int>(deck.size());
		//			for (auto i = 0; i < deck.size(); ++i) {
		//				table[(i * process.second) % deck.size()] = deck[i];
		//			}
		//			deck = table; break;*/


		//			break;
		//		default:
		//			break;
		//		}
		//	}
		//	shuffledStartDecks.push_back(startDeck);
		//	shuffledStartDecks.push_back(endDeck);
		//}

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}