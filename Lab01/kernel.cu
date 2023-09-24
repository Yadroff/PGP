#include <vector>
#include <iomanip>

#include "common_defines.cuh"

__global__ void kernel(float* first, float* second, float* result, int n) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	while (ind < n) {
		result[ind] = min(first[ind], second[ind]);
		ind += offset;
	}
}


int main() {
	std::setprecision(4);
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
#ifdef CUDA_TIME_RECORD
	CUDA_EVENT(start);
	CUDA_EVENT(stop);
	CALL_CUDA_FUNC(cudaEventRecord, event_start.event);
#endif // CUDA_TIME_RECORD

	kernel<<< 1024, 1024 >>>(cuda_first.Data(), cuda_second.Data(), cuda_result.Data(), n);
	CALL_CUDA_FUNC(cudaDeviceSynchronize);
	CALL_CUDA_FUNC(cudaGetLastError);
#ifdef CUDA_TIME_RECORD
	CALL_CUDA_FUNC(cudaEventRecord, event_stop.event);
	CALL_CUDA_FUNC(cudaEventSynchronize, event_stop.event);
	float time;
	CALL_CUDA_FUNC(cudaEventElapsedTime, &time, event_start.event, event_stop.event);
	StringBuilder timeStr;
	timeStr.AppendFmt("Elapsed time: %f", time);
	std::cout << timeStr.Str() << std::endl;
#endif // CUDA_TIME_RECORD

	cuda_result.MoveToHost(result.data());
	for (auto &component: result) {
		std::cout << component << " ";
	}
	std::cout << std::endl;
}