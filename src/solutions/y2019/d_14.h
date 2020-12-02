#pragma once
#include "default.h"

namespace aoc::y2019::d14 {

	struct Chemical {
		Chemical() = default;
		Chemical(std::string name, bigint quantity) : Name(name), Quantity(quantity) {};
		std::string Name;
		bigint Quantity;
	};

	struct ChemicalComparer {
		bool operator() (const Chemical& c1, const Chemical& c2) const {
			return c1.Name < c2.Name;
		}
	};

	struct ChemicalReaction {
		ChemicalReaction() = default;
		ChemicalReaction(Chemical output, std::vector<Chemical> reagents) : Output(output), Reagents(reagents) {};
		Chemical Output;
		std::vector<Chemical> Reagents;
	};

	using Reactions = std::map<std::string, ChemicalReaction>;

	bigint ComputeRequiredOre(const Chemical& chemical, const Reactions& reactions) {
		std::set<Chemical, ChemicalComparer> spareChems;
		std::queue<Chemical> requiredChems;
		Chemical currReqChem;
		bigint oreCount = 0, currSpareChemCount = 0, currNumReactionsNeeded = 0;

		requiredChems.push(chemical); // Resolve Required chemicals
		while (!requiredChems.empty()) {
			currReqChem = requiredChems.front(); // Consider the first required chemical in queue.
			requiredChems.pop();

			auto spareChemical = spareChems.find(currReqChem); // Do we have any spare chemical already?
			if (spareChemical != spareChems.end()) {
				if (currReqChem.Quantity >= spareChemical->Quantity) { // We require more than we have spare.
					currReqChem.Quantity -= spareChemical->Quantity; // We have fully used up the spare chemical.
					spareChems.erase(spareChemical);
				}
				else { // All required chemical satisfied, we have partially used the spare chemical.
					currSpareChemCount = spareChemical->Quantity;
					spareChems.erase(spareChemical);
					spareChems.insert(Chemical(currReqChem.Name, currSpareChemCount - currReqChem.Quantity));
					continue; // No need to futher resolve the chemicals (we have used spare to satisfy requirements.
				}
			}

			auto currChemReaction = reactions.find(currReqChem.Name)->second; // Get the associated reaction for this required chemical.
			currNumReactionsNeeded = (currReqChem.Quantity + currChemReaction.Output.Quantity - 1) / currChemReaction.Output.Quantity; // Fast ceiling division

			if (currChemReaction.Output.Quantity * currNumReactionsNeeded > currReqChem.Quantity) { // Store excess spare chemicals. 
				spareChemical = spareChems.find(currReqChem);
				if (spareChemical != spareChems.end()) { // Update spare chemical counts (for existing chemical)
					currSpareChemCount = spareChemical->Quantity;
					spareChems.erase(spareChemical);
					spareChems.insert(Chemical(currReqChem.Name,
						currSpareChemCount + currChemReaction.Output.Quantity * currNumReactionsNeeded - currReqChem.Quantity));
				}
				else { // Update spare chemical counts (for new chemicals).
					spareChems.insert(Chemical(currReqChem.Name,
						currChemReaction.Output.Quantity * currNumReactionsNeeded - currReqChem.Quantity));
				}
			}

			for (auto reagent : currChemReaction.Reagents) { // Finally update the required chemicals map with the new reagents.
				if (reagent.Name == "ORE") { // Count number of ores.
					oreCount += (bigint)currNumReactionsNeeded * reagent.Quantity;
				}
				else { // Any other chemical, add to the required chemicals queue for resolving.
					requiredChems.push(Chemical(reagent.Name, currNumReactionsNeeded * reagent.Quantity));
				}
			}
		}
		return oreCount;
	}

	bigint BinaryChemicalCountSearch(const std::string& chemical, bigint targetOre, const Reactions& reactions) {
		bigint chemOutLower = 0;
		bigint chemOutUpper = targetOre; // There are no fractions so target ore input will be higher than chem output.
		bigint chemOutMiddle = chemOutLower + (chemOutUpper - chemOutLower) / 2;

		while (chemOutLower < chemOutUpper) {
			if (ComputeRequiredOre(Chemical(chemical, chemOutMiddle), reactions) < targetOre) chemOutLower = chemOutMiddle + 1;
			else chemOutUpper = chemOutMiddle - 1;
			chemOutMiddle = chemOutLower + (chemOutUpper - chemOutLower) / 2;
		}
		return chemOutMiddle - 1;
	}

	void calculate(std::istream& input) {
		std::cout << "--- Day 14: Space Stoichiometry ---\n";
		std::vector<std::string> inputStrs = aoc::utils::reader::read_input(input);

		// Parse the input.
		Reactions reactions;
		for (auto& inputStr : inputStrs) {
			std::regex re("(\\d+) ([\\w]+)");
			std::smatch match;
			std::vector<std::pair<std::string, std::string>> components;
			std::string::const_iterator searchStart(inputStr.cbegin());
			while (std::regex_search(searchStart, inputStr.cend(), match, re))
			{
				components.push_back({ match[1], match[2] });
				searchStart = match.suffix().first;
			}

			std::vector<Chemical> reagents;
			std::transform(components.begin(), components.end() - 1, std::back_inserter(reagents), [](const auto& rawComp) {
				return Chemical(rawComp.second, std::stoi(rawComp.first));
				});

			reactions.try_emplace(components.back().second, Chemical(components.back().second, std::stoi(components.back().first)), reagents);
		}

		// Part 1. 
		auto oreCount = ComputeRequiredOre(Chemical("FUEL", 1), reactions);
		std::cout << "1. Minimum amount of ORE required to produce exactly 1 FUEL :\n";
		std::cout << oreCount << "\n";

		// Part 2.
		auto part2Start = clock::now();
		auto maxFuel = BinaryChemicalCountSearch("FUEL", 1'000'000'000'000, reactions);
		auto part2End = clock::now();
		std::cout << "2. Maximum amount of FUEL produced :\n";
		std::cout << maxFuel << "\n";
		PrintDuration(part2Start, part2End);
	}
}