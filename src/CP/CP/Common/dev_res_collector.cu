#include "dev_res_collector.cuh"

#include "common_structures.cuh"

DevResourceCollector::~DevResourceCollector()
{
	for (auto& pointer : pointers)
	{
		CALL_CUDA_FUNC(cudaFree, pointer);
	}
	pointers.clear();
}

void DevResourceCollector::Alloc(void** ptr, size_t size)
{
	CALL_CUDA_FUNC(cudaMalloc, ptr, size);
	pointers.insert(*ptr);
}

void DevResourceCollector::Free(void* ptr)
{
	if (!ptr) return;
	auto it = pointers.find(ptr);
	ASSERT_MSG(it != pointers.end(), "Try to free non allocated memory %p", ptr);
	CALL_CUDA_FUNC(cudaFree, ptr);
	pointers.erase(it);
}

DevResourceCollector& resGetCollector()
{
	static DevResourceCollector collector;
	return collector;
}