#include <fstream>

#include "texture.h"
#include "Common/common_structures.cuh"

engTEXTURE engTEXTURE::ImportData(std::string filePath)
{
	std::ifstream ifs(filePath, std::ios::binary);
	ASSERT_MSG(ifs, "Fail to open file %s", filePath.c_str());
	engTEXTURE res;
	// read size
	ifs.read(reinterpret_cast<char*>(&res.width), sizeof(int));
	ifs.read(reinterpret_cast<char*>(&res.height), sizeof(int));

	// read data
	res.data.resize(res.width * res.height);
	res.data.shrink_to_fit(); // set capacity to size
	ifs.read(reinterpret_cast<char*>(res.data.data()), res.width * res.height * sizeof(uchar4));
	ifs.close();
	return res;
}

void engTEXTURE::ChangeColor(const vec3& color)
{
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			auto& cur = data[i * width + j];
			cur.x = static_cast<unsigned char>(color.x * cur.x);
			cur.y = static_cast<unsigned char>(color.y * cur.y);
			cur.z = static_cast<unsigned char>(color.z * cur.z);
		}
	}
}

uchar4 engTEXTURE::GetColor(const vec3& pos) const
{
	uchar4 res = make_uchar4(0, 0, 0, 0);
	if (width != 0 && height != 0)
	{
		int x = static_cast<int>(pos.x * width);
		int y = static_cast<int>(pos.y * height);

		res = data[y * width + x];
	}
	return res;
}