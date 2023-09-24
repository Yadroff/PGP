#pragma once

#include <iostream>
#include <sstream>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define CALL_CUDA_FUNC(func, ...) \
	do { \
		cudaError_t status = func(__VA_ARGS__); \
		if (status != cudaSuccess) { \
			StringBuilder builder;\
			builder.Append("Cuda ERROR at").Append(__FILE__, ", ").Append(__LINE__, "\n")\
			.Append("Message", ": ").Append(cudaGetErrorString(status), "");\
			std::cerr << builder.Str() << std::endl;\
			exit(0); \
		}\
	}\
	while (0)\

#define ASSERT_MSG(condition, format, ...) \
	do {\
		if (!(condition)){\
			/* Get warning if format is wrong */ \
			sizeof(sprintf(nullptr, format, __VA_ARGS__)); \
			StringBuilder report;\
			report.Append("ASSERT FAIL at").Append(__FILE__, ", ").Append(__LINE__, "\n")\
			.Append("Condition", ": ").Append(#condition, "\n")\
			.Append("Description", ": ").AppendFmt(format, __VA_ARGS__); \
			std::cerr << report.Str() << std::endl; \
		}\
	}\
	while (0)\

#define CUDA_EVENT(name) CudaEventWrapper event_##name

struct StringBuilder {
public:
	template<typename T>
	StringBuilder& Append(const T& arg, const char* delimiter = " ") {
		buff << arg << delimiter;
		return *this;
	}
	template<typename ...Args>
	StringBuilder& AppendFmt(const char* fmt, Args... args) {
		static constexpr size_t maxAppendSize = 100;
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
		CALL_CUDA_FUNC(cudaEventCreate, &event);
	}
	~CudaEventWrapper() {
		CALL_CUDA_FUNC(cudaEventDestroy, event);
	}
	cudaEvent_t event;
};

template<typename T>
class CudaArray {
public:
	CudaArray(std::size_t size): size(size) {
		CALL_CUDA_FUNC(cudaMalloc, &data, sizeof(T) * size);
	}
	void MoveToDevice(T* from, std::size_t n) {
		ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
		CALL_CUDA_FUNC(cudaMemcpy, data, from, n * sizeof(T), cudaMemcpyHostToDevice);
	}

	void MoveToDevice(T* from) {
		MoveToDevice(from, size);
	}

	void MoveToHost(T* to, std::size_t n) {
		ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
		CALL_CUDA_FUNC(cudaMemcpy, to, data, n * sizeof(T), cudaMemcpyDeviceToHost);
	}

	void MoveToHost(T* to) {
		MoveToHost(to, size);
	}
	~CudaArray() {
		CALL_CUDA_FUNC(cudaFree, data);
	}
	T& operator[](std::size_t ind) {
		ASSERT_MSG(ind <= size, "Index out of range: size is %d, index is %d", size, ind);
		return data[ind];
	}
	T operator[](std::size_t ind) const{
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