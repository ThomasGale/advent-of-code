#pragma once
#include "default.h"

namespace aoc::y2019::d23 {
	using namespace aoc::y2019::intcc;
	using namespace aoc::utils;

	void calculate(std::istream& input) {
		std::cout << "--- Day 23: Category Six ---\n";
		std::string inputStr(std::istreambuf_iterator<char>(input), {});

		int networkSize = 50;
		BasicWindow debugWindow(networkSize, 30);
		std::vector<int> addresses(networkSize); // 50 addressess available
		std::iota(addresses.begin(), addresses.end(), 0);
		std::vector<IntCodeComputer> nICs;

		// Create computers.
		for (int address : addresses) {
			nICs.emplace_back(inputStr);
		}

		// Packets are stored in a map of queues. Initial mesage is assign each machine with address.
		std::map<int, std::tuple<int, int, int>> packetConstruction; // Key: Source address, Value: Target, X, Y data.
		for (int address : addresses) {
			packetConstruction.insert({ address, {-1, -1, -1} });
		}

		std::map<int, std::queue<int>> packets; // Key: Target address, Value: X, Y data.
		for (int address : addresses) {
			std::queue<int> addressPacketQueue;
			addressPacketQueue.push(address);
			packets.insert({ address, addressPacketQueue });
		}

		// Run the network
		while (true) {
			for (auto address = 0; address < networkSize; ++address) {

				bigint output;
				if (packets[address].size() >= 1) { // Process single input and step.
					output = nICs[address].StepProgram(packets[address].front());
					packets[address].pop();
				}
				else {
					output = nICs[address].StepProgram(-1);
				}

				if (output != -1) { // If we have output.
					auto& pc = packetConstruction[address];
					if (std::get<0>(pc) == -1) { // Set address
						std::get<0>(pc) = output;
						if (packets.find(output) == packets.end()) // Create slot in packets.
							packets.insert({ output, std::queue<int>{} });
					}
					else if (std::get<1>(pc) == -1) { // Set first value and send.
						std::get<1>(pc) = output;
						packets[std::get<0>(pc)].push(output);
					}
					else if (std::get<2>(pc) == -1) { // Set second value and send and clear.
						std::get<2>(pc) = output;
						packets[std::get<0>(pc)].push(output);
						std::get<0>(pc) = -1;
						std::get<1>(pc) = -1;
						std::get<2>(pc) = -1;
					}
				}

				//if (packetConstruction[address].size() == 3) { // Broadcast packet once ready.
				//	auto target = packetConstruction[address].front();
				//	packetConstruction[address].pop();
				//	auto X = packetConstruction[address].front();
				//	packetConstruction[address].pop();
				//	auto Y = packetConstruction[address].front();
				//	packetConstruction[address].pop();



				//	packets[target].push(X);
				//	packets[target].push(Y);
				//}

				//if (output.size() >= 2) {
				//	//if (output[outIndex + 1] != -1)
				//	packets[output[0]].push(output[1]);
				//	if (outIndex <= output.size() - 3) {
				//		//if (output[outIndex + 2] != -1)
				//		packets[output[outIndex]].push(output[outIndex + 2]);
				//	}
				//}

				//for (auto outIndex = 0; outIndex < output.size(); outIndex += 3) { // Process Output.
				//	if (output.size() % 3 != 0) {
				//		throw std::runtime_error("Not designed for this yet");
				//	}
				//	// Try to find the key or create
				//	if (packets.find(output[outIndex]) == packets.end())
				//		packets.insert({ output[outIndex], std::queue<int>{} });

				//	if (outIndex <= output.size() - 2) {
				//		//if (output[outIndex + 1] != -1)
				//		packets[output[outIndex]].push(output[outIndex + 1]);
				//		if (outIndex <= output.size() - 3) {
				//			//if (output[outIndex + 2] != -1)
				//			packets[output[outIndex]].push(output[outIndex + 2]);
				//		}
				//	}
				//}
			}

			if (packets.find(255) != packets.end()) {
				break;
			}

			//if (packets.find(255) != packets.end() && packets[255].size() >= 2) {
			//	//break;
			//	auto queue255 = packets.find(255)->second;
			//	auto queue255X = queue255.front();
			//	queue255.pop();
			//	auto queue255Y = queue255.front();
			//	queue255.pop();
			//	if (queue255Y != -1)
			//		break;
			//	packets[255].push(queue255X);
			//	packets[255].push(queue255Y);
			//}
		}

		std::cout << "1. The Y value of the first packet sent to address 255:\n";
		auto queue255 = packets.find(255)->second;
		auto queue255X = queue255.front();
		queue255.pop();
		auto queue255Y = queue255.front();
		std::cout << queue255Y << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}