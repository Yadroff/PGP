#include "common_defines.cuh"
#include "common_structures.hpp"


__global__ void kernel(float* first, float* second, OUTPUT<float> result, int n) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	while (ind < n) {
		result[ind] = min(first[ind], second[ind]);
		ind += offset;
	}
}


int main(int argc, const char* argv[]) {
	INIT_IO();
#ifdef BENCHMARK
	osDisplayHardwareInfo(std::cout);
	checkCommandLine(argc, argv);
#endif
	int n;
	std::cin >> n;
	std::vector<float> first(n), second(n), result(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> first[i];
	}
	for (int i = 0; i < n; ++i) {
		std::cin >> second[i];
	}

	CudaArray<float> cuda_first(n), cuda_second(n), cuda_result(n);
	cuda_first.MoveToDevice(first.data());
	cuda_second.MoveToDevice(second.data());
	cuda_result.MoveToDevice(result.data());

	CALL_KERNEL(cuda_first.Data(), cuda_second.Data(), cuda_result.Data(), n);
	cuda_result.MoveToHost(result.data());
#ifndef BENCHMARK
	for (auto &component: result) {
		std::cout << component << " ";
	}
	std::cout << std::endl;
#endif
}