#pragma once

#include "Math/matrix.cuh"

class engTEXTURE {
public:
	engTEXTURE() = default;
	static engTEXTURE ImportData(std::string filePath);

	int Width() const { return width; }
	int Height() const { return height; }

	uchar4* Data() { return data.data(); }
	const uchar4* Data() const { return data.data(); }

	void ChangeColor(const vec3& color);
	uchar4 GetColor(const vec3& pos) const;

private:
	std::vector<uchar4> data;
	int width = 0;
	int height = 0;
};