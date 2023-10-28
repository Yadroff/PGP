#pragma once

template<typename T>
StringBuilder& StringBuilder::Append(const T& arg, const char* delimiter) 
{
	buff << arg << delimiter;
	return *this;
}

template<typename ...Args>
StringBuilder& StringBuilder::AppendFmt(const char* fmt, Args... args) 
{
	static constexpr size_t maxAppendSize = 200;
	char str[maxAppendSize] = "";
	sprintf(str, fmt, args...);
	return Append(std::string(str));
}

template<typename T>
CudaArray<T>::CudaArray(std::size_t size) 
	: size(size) 
{
	ERROR_WRAPPER_CALL(cudaMalloc, &data, sizeof(T) * size);
}

template<typename T>
void CudaArray<T>::MoveToDevice(T* from, std::size_t n) {
	ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
	ERROR_WRAPPER_CALL(cudaMemcpy, data, from, n * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void CudaArray<T>::MoveToDevice(T* from) 
{
	MoveToDevice(from, size);
}

template<typename T>
void CudaArray<T>::MoveToHost(T* to, std::size_t n)  const
{
	ASSERT_MSG(n <= size, "Index out of range: size is %zd, index is %zd", size, n);
	ERROR_WRAPPER_CALL(cudaMemcpy, to, data, n * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
T& CudaArray<T>::operator[](std::size_t ind) 
{
	ASSERT_MSG(ind <= size, "Index out of range: size is %d, index is %d", size, ind);
	return data[ind];
}

template<typename T>
T CudaArray<T>::operator[](std::size_t ind) const {
	ASSERT_MSG(ind <= size, "Index out of range: size is %d, index is %d", size, ind);
	return data[ind];
}

template<typename T>
Cuda2DArray<T>::Cuda2DArray(std::size_t w, std::size_t h) 
	: width(w)
	, height(h) 
{
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<T>();
	ERROR_WRAPPER_CALL(cudaMallocArray,&data, &ch, w, h);
}

template<typename T>
void Cuda2DArray<T>::MoveToDevice(T* from, std::size_t w, std::size_t h) 
{
	ASSERT_MSG((w <= width) && (h <= height), "Sizes out of range: Widht is %zd (original: %zd), Heihgt is %zd (original: %zd)", w, width, h, height);
	ERROR_WRAPPER_CALL(cudaMemcpy2DToArray, data, 0, 0, from, w * sizeof(T), w * sizeof(T), h, cudaMemcpyHostToDevice);
}

template<typename T>
void Cuda2DArray<T>::MoveToHost(T* to, std::size_t w, std::size_t h) {
	ASSERT_MSG(w <= width && h <= height, "Sizes out of range: Widht is %zd (original: %zd), Heihgt is %zd (original: %zd)", w, width, h, height);
	ERROR_WRAPPER_CALL(cudaMemcpy2DToArray, to, 0, 0, data, w * sizeof(T), w * sizeof(T), h, cudaMemcpyDeviceToHost);
}