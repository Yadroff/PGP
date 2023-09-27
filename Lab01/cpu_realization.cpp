#include <iostream>
#include <vector>
#include <thread>

#ifdef TIME_RECORD
#include <chrono>
#endif

void ThreadFunc(const std::vector<double> &first, const std::vector<double> &second, std::vector<double>& result, std::size_t offset, std::size_t start) {
	for (std::size_t cur = start; cur < result.size(); cur += offset) {
		result[cur] = std::min(first[cur], second[cur]);
	}
}

void CPU_Realization(const std::vector<double>& first, const std::vector<double>& second, std::vector<double>& result) {
	std::size_t threadCount = std::min(static_cast<std::size_t>(1000), first.size());
	std::vector<std::thread> threads;
#ifdef TIME_RECORD
	auto start = std::chrono::system_clock::now();
#endif
	for (std::size_t i = 0; i < threadCount; ++i) {
		threads.emplace_back(ThreadFunc, first, second, result, threadCount, i);
	}
	for (auto& thread : threads) {
		thread.join();
	}
#ifdef TIME_RECORD

#endif
	for (auto val : result) {
		std::cout << val << " " << std::endl;
	}
}