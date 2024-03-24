#include "ssaa.cuh"
#include <future>

__host__ __device__ vec3 rendAvgKernel(vec3* img, int i, int j, int width, int height, int kernelWidth, int kernelHeight)
{
	vec3 sum(0.0f);
	for (int y = i; y < i + kernelWidth; ++y) {
		for (int x = j; x < j + kernelWidth; ++x) {
			sum += img[y * width + x];
		}
	}

	int pixelsNum = kernelWidth * kernelHeight;
	float coef = 1.0f / pixelsNum;

	return coef * sum;
}

__global__ void rendCoreSSAA_GPU(vec3* ref, vec3* res, int width, int height, int newWidth, int newHeight)
{
	int idX = threadIdx.x + blockIdx.x * blockDim.x;
	int idY = threadIdx.y + blockIdx.y * blockDim.y;

	int offsetX = blockDim.x * gridDim.x;
	int offsetY = blockDim.y * gridDim.y;

	int kernelW = width / newWidth;
	int kernelH = height / newHeight;

	for (int i = idY; i < newHeight; i += offsetY) {
		for (int j = idX; j < newWidth; j += offsetX) {
			int pixI = i * kernelH;
			int pixJ = j * kernelW;

			res[i * newWidth + j] = rendAvgKernel(ref, pixI, pixJ, width, height, kernelW, kernelH);
		}
	}
}

void rendCoreSSAA_CPU(vec3* ref, vec3* res, int width, int height, int newWidth, int newHeight, int idX, int idY,
	int offsetX, int offsetY)
{
	int kernelW = width / newWidth;
	int kernelH = height / newHeight;

	for (int i = idY; i < newHeight; i += offsetY) {
		for (int j = idX; j < newWidth; j += offsetX) {
			int pixI = i * kernelH;
			int pixJ = j * kernelW;

			res[i * newWidth + j] = rendAvgKernel(ref, pixI, pixJ, width, height, kernelW, kernelH);
		}
	}
}

void rendSSAA_GPU(vec3* ref, vec3* res, int w, int h, int newWidth, int newHeight)
{
	vec3* devRes, * devRef;
	resGetCollector().Alloc(&devRef, w * h * sizeof(vec3));
	resGetCollector().Alloc(&devRes, newWidth * newHeight * sizeof(vec3));

	CALL_CUDA_FUNC(cudaMemcpy, devRef, ref, w * h * sizeof(vec3), cudaMemcpyHostToDevice);
	CALL_KERNEL(rendCoreSSAA_GPU, devRef, devRes, w, h, newWidth, newHeight);

	CALL_CUDA_FUNC(cudaMemcpy, res, devRes, newWidth * newHeight * sizeof(vec3), cudaMemcpyDeviceToHost);
}

void rendSSAA_CPU(vec3* ref, vec3* res, int w, int h, int newWidth, int newHeight)
{
	std::vector<std::future<void>> futures;
	int xCount = std::thread::hardware_concurrency();
	int yCount = std::thread::hardware_concurrency();
	for (size_t i = 0; i < xCount; ++i)
	{
		for (size_t j = 0; j < yCount; ++j)
		{
			futures.push_back(std::async(std::launch::async, rendCoreSSAA_CPU, ref, res, w, h, newWidth, newHeight, i, j, xCount, yCount));
		}
	}
	for (auto& future : futures)
	{
		future.get();
	}
}