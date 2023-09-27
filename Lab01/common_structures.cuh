#pragma once

#include "common_defines.cuh"

struct StringBuilder {
public:
	template<typename T>
	StringBuilder& Append(const T& arg, const char* delimiter = " ") {
		buff << arg << delimiter;
		return *this;
	}
	template<typename ...Args>
	StringBuilder& AppendFmt(const char* fmt, Args... args) {
		static constexpr size_t maxAppendSize = 200;
		char str[maxAppendSize] = "";
		sprintf(str, fmt, args...);
		return Append(std::string(str));
	}
	operator std::string() const {
		return buff.str();
	}
	std::string Str() const {
		return buff.str();
	}
private:
	std::stringstream buff;
};


struct CudaEventWrapper {
	CudaEventWrapper() {
		ERROR_WRAPPER_CALL(cudaEventCreate, &event);
	}
	~CudaEventWrapper() {
		ERROR_WRAPPER_CALL(cudaEventDestroy, event);
	}
	cudaEvent_t event;
};

template<typename T>
class CudaArray {
public:
	CudaArray(std::size_t size) : size(size) {
		ERROR_WRAPPER_CALL(cudaMalloc, &data, sizeof(T) * size);
	}
	void MoveToDevice(T* from, std::size_t n) {
		ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
		ERROR_WRAPPER_CALL(cudaMemcpy, data, from, n * sizeof(T), cudaMemcpyHostToDevice);
	}

	void MoveToDevice(T* from) {
		MoveToDevice(from, size);
	}

	void MoveToHost(T* to, std::size_t n) {
		ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
		ERROR_WRAPPER_CALL(cudaMemcpy, to, data, n * sizeof(T), cudaMemcpyDeviceToHost);
	}

	void MoveToHost(T* to) {
		MoveToHost(to, size);
	}
	~CudaArray() {
		ERROR_WRAPPER_CALL(cudaFree, data);
	}
	T& operator[](std::size_t ind) {
		ASSERT_MSG(ind <= size, "Index out of range: size is %d, index is %d", size, ind);
		return data[ind];
	}
	T operator[](std::size_t ind) const {
		ASSERT_MSG(ind <= size, "Index out of range: size is %d, index is %d", size, ind);
		return data[ind];
	}
	std::size_t Size() const {
		return size;
	}
	T* Data() {
		return data;
	}
private:
	T* data;
	std::size_t size;
};

#ifdef OS_WIN

/// returns CPU name (e.g. "Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz") 
std::string osGetCPUName() {
	int CPUInfo[4] = { -1 };
	char CPUBrandString[0x40];
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];

	memset(CPUBrandString, 0, sizeof(CPUBrandString));

	// Get the information associated with each extended ID.
	for (int i = 0x80000000; i <= nExIds; ++i) {
		__cpuid(CPUInfo, i);
		// Interpret CPU brand string.
		if (i == 0x80000002)
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	return std::string(CPUBrandString);
}

/// returns RAM size in KB 
ULONGLONG osGetRAMSize() {
	ULONGLONG memory;
	ASSERT_NO_MSG(GetPhysicallyInstalledSystemMemory(&memory));
	return memory;
}
#else
unsigned long long osGetRAMSize() { return 0; }
std::string osGetCPUName() { return ""; }
#endif

std::vector<std::string> osGetDevicesInfo() {
	static constexpr int kb = 1024;
	static constexpr int mb = kb * kb;
	int devCount;
	ERROR_WRAPPER_CALL(cudaGetDeviceCount, &devCount);
	std::vector<std::string> result(devCount);
	for (int i = 0; i < devCount; ++i)
	{
		StringBuilder builder;
		cudaDeviceProp props;
		ERROR_WRAPPER_CALL(cudaGetDeviceProperties, &props, i);
		builder.AppendFmt("%d: %s: %d.%d\n", i, props.name, props.major, props.minor);
		builder.AppendFmt("  Global memory:   %d mb\n", static_cast<int>(props.totalGlobalMem / mb));
		builder.AppendFmt("  Shared memory:   %d kb\n", static_cast<int>(props.sharedMemPerBlock / kb));
		builder.AppendFmt("  Constant memory: %d kb\n", static_cast<int>(props.totalConstMem / kb));
		builder.AppendFmt("  Block registers: %d\n", props.regsPerBlock);

		builder.AppendFmt("  Warp size:         %d\n", props.warpSize);
		builder.AppendFmt("  Threads per block: %d\n", props.maxThreadsPerBlock);
		builder.AppendFmt("  Max block dimensions: [ %d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		builder.AppendFmt("  Max grid dimensions:  [ %d, %d, %d]", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		result[i] = builder.Str();
	}
	return result;
}

void DisplayInfo() {
	static constexpr const int kb = 1024;
	static constexpr const int mb = kb * kb;

	std::cout << "CPU: " << osGetCPUName() << '\n';
	std::cout << "RAM size: " << osGetRAMSize() / 1024 << " mb\n";

	std::cout << "CUDA version: v." << CUDART_VERSION << '\n';

	int devCount;
	cudaGetDeviceCount(&devCount);
	std::cout << "CUDA Devices: " << devCount << '\n';
	auto devicesInfo = osGetDevicesInfo();
	for (auto& deviceInfo : devicesInfo) {
		std::cout << deviceInfo << '\n';
	}
	std::cout << std::endl;
}