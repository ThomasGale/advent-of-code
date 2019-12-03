#pragma once
#include "default.h"

namespace aoc::y2019::d03 {

	struct Location {
		int X, Y;
		Location(int x, int y) : X(x), Y(y) {};
	};

	struct Segment {
		Location Start, End;
		Segment(Location start, Location end) : Start(start), End(end) {}

		bool IsHorizontal() const { return Start.Y == End.Y; }
		bool IsVertical() const { return Start.X == End.X; }

		bool IsHorizontalOverlap(const Location& location) const {
			return (std::min(Start.X, End.X) < location.X) && (std::max(Start.X, End.X) > location.X);
		}

		bool IsVerticalOverlap(const Location& location) const {
			return (std::min(Start.Y, End.Y) < location.Y) && (std::max(Start.Y, End.Y) > location.Y);
		}

		std::optional<Location> GetIntersection(const Segment& otherSeg) {
			if (IsHorizontal() && otherSeg.IsVertical() && (IsHorizontalOverlap(otherSeg.Start)) && (otherSeg.IsVerticalOverlap(Start))) {
				return Location(otherSeg.Start.X, Start.Y);
			}
			else if (IsVertical() && otherSeg.IsHorizontal() && (IsVerticalOverlap(otherSeg.Start)) && (otherSeg.IsHorizontalOverlap(Start))) {
				return Location(Start.X, otherSeg.Start.Y);
			}
			else {
				return std::nullopt;
			}
		}
	};

	struct Wire {
		std::vector<Segment> Segments;

		Wire(std::vector<std::string> wire_str) {
			Location previous(0, 0); // Track previous location
			Location current(0, 0); // Track current location

			Segments = std::vector<Segment>(); // Build up segments			
			for (auto& wire_seg_str : wire_str) { // Parse wire_str
				switch (wire_seg_str[0])
				{
				case 'U':
					current.Y += std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'R':
					current.X += std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'D':
					current.Y -= std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				case 'L':
					current.X -= std::stoi(wire_seg_str.substr(1, wire_seg_str.size() - 1));
					break;
				default:
					throw std::runtime_error("Bad command");
				}
				Segments.push_back(Segment(previous, current));
				previous = current;
			}
		}

		std::vector<Location> FindIntersections(const Wire& otherWire) {
			std::vector<Location> intersections;
			// Loop over the current segments
			for (auto i = 0; i < Segments.size(); ++i) {
				// Loop over the other wire segments
				for (auto j = 0; j < otherWire.Segments.size(); ++j) {
					// Find intersection
					auto intersection = Segments[i].GetIntersection(otherWire.Segments[j]);
					if (intersection != std::nullopt)
						intersections.push_back(intersection.value());
				}
			}
			return intersections;
		}
	};

	void calculate(std::istream& input) {
		std::cout << "--- Day 3: Crossed Wires ---\n";
		std::vector<std::string> input_strings = aoc::utils::read_input(input);
		std::vector<std::string> wire1_str = aoc::utils::split(input_strings[0], ',');
		std::vector<std::string> wire2_str = aoc::utils::split(input_strings[1], ',');

		Wire wire1(wire1_str);
		Wire wire2(wire2_str);

		// Part 1
		std::vector<Location> intersections = wire1.FindIntersections(wire2);
		std::vector<int> manhatDistances(intersections.size());
		std::transform(intersections.begin(), intersections.end(), manhatDistances.begin(), [](const Location& loc) { return std::abs(loc.X) + std::abs(loc.Y); });
		int closedCrossingDist = *std::min_element(manhatDistances.begin(), manhatDistances.end());

		std::cout << "1. Manhattan distance from the central port to the closest intersection:\n";
		std::cout << closedCrossingDist << "\n";

		std::cout << "2. ... :\n";
		std::cout << "" << "\n";
	}
}