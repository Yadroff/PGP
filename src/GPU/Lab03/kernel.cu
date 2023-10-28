#include <float.h>
#include <texture_indirect_functions.h>

#include "os.cuh"
#include "common_structures.cuh"
#include "common_defines.cuh"
#include "claster.cuh"

constexpr int MAX_CLASSES_NUM = 32;

__constant__ AVERAGE_TYPE CLASTERS[MAX_CLASSES_NUM];

/// Returns (pixel, average) / |average|
__device__ float scalarProduct(const uchar4 &pixel, AVERAGE_TYPE average)
{
	return 1.0 * (pixel.x * average.x + pixel.y * average.y + pixel.z * average.z) * rsqrtf(average.w);
}

__global__ void kernel(uchar4* img, int width, int height, int clastersSize) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	uchar4 pixel;
	for (; idx < width * height; idx += offset) {
		pixel = img[idx];
		float maxScalar = -FLT_MAX;
		int argMax = 0;
		for (int claster = 0; claster < clastersSize; ++claster) {
			float scalar = scalarProduct(pixel, CLASTERS[claster]);
			if (scalar > maxScalar) {
				maxScalar = scalar;
				argMax = claster;
			}
		}
#ifndef _RELEASE
		img[idx].x = CLASTERS[argMax].x;
		img[idx].y = CLASTERS[argMax].y;
		img[idx].z = CLASTERS[argMax].z;
#else
		img[idx].w = argMax;
#endif
	}
}

int main()
{
    INIT_IO();
    // Image read
	int width, height;
	std::vector<uchar4> img;
    std::string inputFilename, outputFilename;
    std::cin >> inputFilename >> outputFilename;
    osFileRead(inputFilename, img, width, height);

	// Samples read
	int clastersCount;
	std::cin >> clastersCount;
	std::vector<Claster> clasters(clastersCount);
	for (auto& claster : clasters) {
		std::cin >> claster;
	}
	// Calculate average of each sample
	std::vector<AVERAGE_TYPE> clasterAverage(clastersCount);
	for (int i = 0; i < clastersCount; ++i) {
		clasterAverage[i] = clasters[i].Init(img, width, height);
	}
//#ifdef _DEBUG
	for (const auto& average : clasterAverage) {
		StringBuilder builder;
		builder.AppendFmt("%5f %5f %5f %5f", average.x, average.y, average.z, average.w);
		std::cout << builder.Str() << std::endl;
	}
//#endif
	CALL_CUDA_FUNC(cudaMemcpyToSymbol, CLASTERS, clasterAverage.data(), sizeof(uint4) * clasterAverage.size());
	// Move image to device
	CudaArray<uchar4> image(width * height);
	image.MoveToDevice(img.data());

	CALL_KERNEL(image.Data(), width, height, clastersCount);
	std::vector<uchar4> result(width * height);
	image.MoveToHost(result.data());
	osFileWrite(outputFilename, result, width, height);
    return 0;
}
