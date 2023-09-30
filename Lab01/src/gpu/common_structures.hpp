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