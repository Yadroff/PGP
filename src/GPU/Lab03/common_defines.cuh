#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "os.cuh"

#ifdef _PROFILE
#define BENCHMARK
#endif

#define INIT_IO() \
	std::ios_base::sync_with_stdio(false);	\
	std::cin.tie(nullptr);					\
	std::cout.tie(nullptr);					\
	std::setprecision(10);					


/// Wrapper to call CUDA functions
///
/// Callcs CUDA `func` and chekcs returned error
#define ERROR_WRAPPER_CALL(func, ...)																\
	do {																							\
		cudaError_t status = func(__VA_ARGS__);														\
		ASSERT_MSG(status == cudaSuccess, "%s: CUDA ERROR: %s", #func, cudaGetErrorString(status));	\
	}																								\
	while (0)																				


/// Assert with description and file, line information 
#ifdef _DEBUG
#define ASSERT_MSG(condition, format, ...)														\
	do {																						\
		if (!(condition)){																		\
			StringBuilder report;																\
			report.Append("ASSERT FAIL at").Append(__FILE__, ", ").Append(__LINE__, "\n")		\
			.Append("Condition", ": ").Append(#condition, "\n")									\
			.Append("Description", ": ").AppendFmt(format, __VA_ARGS__);						\
			std::cerr << report.Str() << std::endl;												\
			osDebugBreak();																		\
		}																						\
	}																							\
	while (0)
#else
#define ASSERT_MSG(condition, format, ...)
#endif

#define ASSERT_FALSE(format, ...) ASSERT_MSG(false, format, __VA_ARGS__);
#define ASSERT_NO_MSG(condition) ASSERT_MSG(condition, "");

/// Create and accsess to CUDA Events
#define CUDA_EVENT_WRAPPER(name) CudaEventWrapper event_##name
#define CUDA_EVENT(name) event_##name##.event

/// CUDA func call with benchmark if defined
#ifdef BENCHMARK
extern int BLOCKS_NUM;
extern int THREADS_IN_BLOCK;

void checkCommandLine(int argc, const char** argv);

#define CALL_CUDA_FUNC(func, ...)																			\
	do{																										\
		float time;																							\
		CUDA_EVENT_WRAPPER(start);																			\
		CUDA_EVENT_WRAPPER(stop);																			\
		ERROR_WRAPPER_CALL(cudaEventRecord, CUDA_EVENT(start));												\
		ERROR_WRAPPER_CALL(func, __VA_ARGS__);																\
		ERROR_WRAPPER_CALL(cudaEventRecord, CUDA_EVENT(stop));												\
		ERROR_WRAPPER_CALL(cudaEventSynchronize, CUDA_EVENT(stop));											\
		ERROR_WRAPPER_CALL(cudaEventElapsedTime, &time, CUDA_EVENT(start), CUDA_EVENT(stop));				\
		StringBuilder timeStr;																				\
		timeStr.AppendFmt("Elapsed time for call %s: %f", #func, time);										\
		std::cout << timeStr.Str() << std::endl;															\
	} while (0)

#define CALL_KERNEL(...)																						\
	do{																											\
		float time;																								\
		StringBuilder builder;																					\
		builder.AppendFmt("CUDA Kernel function parameters: <<< %d, %d >>>", BLOCKS_NUM, THREADS_IN_BLOCK);		\
		std::cout << builder.Str() << std::endl;																\
		CUDA_EVENT_WRAPPER(start);																				\
		CUDA_EVENT_WRAPPER(stop);																				\
		ERROR_WRAPPER_CALL(cudaEventRecord, CUDA_EVENT(start));													\
		kernel<<< BLOCKS_NUM, THREADS_IN_BLOCK >>>(__VA_ARGS__);												\
		ERROR_WRAPPER_CALL(cudaDeviceSynchronize);																\
		ERROR_WRAPPER_CALL(cudaGetLastError);																	\
		ERROR_WRAPPER_CALL(cudaEventRecord, CUDA_EVENT(stop));													\
		ERROR_WRAPPER_CALL(cudaEventSynchronize, CUDA_EVENT(stop));												\
		ERROR_WRAPPER_CALL(cudaEventElapsedTime, &time, CUDA_EVENT(start), CUDA_EVENT(stop));					\
		StringBuilder timeStr;																					\
		timeStr.AppendFmt("Elapsed time for call %s: %f ms", "kernel", time);									\
		std::cout << timeStr.Str() << std::endl;																\
	} while (0)
#else
#define CALL_CUDA_FUNC(func, ...) ERROR_WRAPPER_CALL(func, __VA_ARGS__);
#define CALL_KERNEL(...)																					\
	do {																									\
		kernel<<< 256, 256 >>>(__VA_ARGS__);																\
		ERROR_WRAPPER_CALL(cudaDeviceSynchronize);															\
		ERROR_WRAPPER_CALL(cudaGetLastError);																\
	} while(0)
#endif // TIME_RECORD

using AVERAGE_TYPE = float4;