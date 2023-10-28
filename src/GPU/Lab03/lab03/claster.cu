#include <cmath>

#include "claster.cuh"
#include "common_structures.cuh"

std::istream& operator>>(std::istream& is, Claster& claster)
{
    std::size_t size;
    is >> size;
    claster.samples.resize(size);
    claster.samples.shrink_to_fit();
    for (int i = 0; i < static_cast<int>(size); ++i) {
        is >> claster.samples[i].first >> claster.samples[i].second;
    }
    return is;
}

AVERAGE_TYPE Claster::Init(const std::vector<uchar4>& image, int width, int height)
{
    AVERAGE_TYPE result = { 0, 0, 0, 0};
    for (const auto& pair : samples) {
        int cordX = pair.first, cordY = pair.second;
		ASSERT_MSG(cordX < width, "Index (%d) out of range %d", cordX, width);
        ASSERT_MSG(cordY < height, "Index (%d) out of range %d", cordY, height);
        const auto& toAdd = image[cordY * width + cordX];
        result.x += toAdd.x;
        result.y += toAdd.y;
        result.z += toAdd.z;
    }
    result.x /= samples.size();
    result.y /= samples.size();
    result.z /= samples.size();
    result.w = result.x * result.x + result.y * result.y + result.z * result.z;

    return result;
}
