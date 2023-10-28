#pragma once

#include "common_defines.cuh"

using Coords = std::pair<int, int>;

/// Class that contains cords (x, y) of samples
class Claster
{
public:
	/// Returns average (R, G, B, module ^2) vector of samples
	AVERAGE_TYPE Init(const std::vector<uchar4>&image, int width, int height);
	friend std::istream& operator>>(std::istream& is, Claster& claster);
private:
	Coords left_upper, right_down;
};

