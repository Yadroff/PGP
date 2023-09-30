#include "os.cuh"
#include "common_structures.hpp"

void osDisplayHardwareInfo(std::ostream& os)
{
	static constexpr const int kb = 1 << 10;
	os << "CPU: " << osGetCPUName() << '\n';
	os << "RAM size: " << osGetRAMSize() / kb << " mb\n";

	os << "CUDA version: v." << CUDART_VERSION << '\n';

	int devCount;
	cudaGetDeviceCount(&devCount);
	os << "CUDA Devices: " << devCount << '\n';
	auto devicesInfo = osGetDevicesInfo();
	for (auto& deviceInfo : devicesInfo) {
		os << deviceInfo << '\n';
	}
	os << std::endl;
}

#ifdef _DEBUG
void osDebugBreak() {
	int* pointer = nullptr;
	*pointer = 1;
}
#else
void osDebugBreak() {
	exit(-1);
}
#endif

#ifdef OS_WIN
/// returns CPU name (e.g. "Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz") 
std::string osGetCPUName() {
	int CPUInfo[4] = { -1 };
	char CPUBrandString[0x40] = "\0";
	__cpuid(CPUInfo, 0x80000000);
	int nExIds = CPUInfo[0];

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
	static constexpr int kb = 1 << 10;
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

		builder.AppendFmt("  Warp size:              %d\n", props.warpSize);
		builder.AppendFmt("  Threads per block:      %d\n", props.maxThreadsPerBlock);
		builder.AppendFmt("  Multi processors count: %d\n", props.multiProcessorCount);
		builder.AppendFmt("  Max block dimensions:   [ %d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		builder.AppendFmt("  Max grid dimensions:    [ %d, %d, %d]", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		result[i] = builder.Str();
	}
	return result;
}

