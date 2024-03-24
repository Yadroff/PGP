#pragma once

#include <unordered_map>
#include <unordered_set>
#include <functional>

#include <driver_types.h>

#include "common.cuh"

class dsSTRING_BUILDER {
public:
	template<typename T>
	dsSTRING_BUILDER& Append(const T& arg, const char* delimiter = " ");

	template<typename ...Args>
	dsSTRING_BUILDER& AppendFmt(const char* fmt, Args... args);

	operator std::string() const { return buff.str(); }
	std::string Str() const { return buff.str(); }
private:
	std::stringstream buff;
};

struct cudaEVENT_WRAPPER {
	cudaEVENT_WRAPPER() { ERROR_WRAPPER_CALL(cudaEventCreate, &event); }
	~cudaEVENT_WRAPPER() { ERROR_WRAPPER_CALL(cudaEventDestroy, event); }
	cudaEvent_t event;
};

template<typename T>
class cudaARRAY {
public:
	cudaARRAY(std::size_t size);
	void MoveToDevice(T* from, std::size_t n);
	void MoveToDevice(T* from);
	void MoveToHost(T* to, std::size_t n) const;
	void MoveToHost(T* to) const { MoveToHost(to, size); }
	T& operator[](std::size_t ind);
	T operator[](std::size_t ind) const;
	std::size_t Size() const { return size; }
	T* Data() { return data; }
	~cudaARRAY() { ERROR_WRAPPER_CALL(cudaFree, data); }
private:
	T* data;
	std::size_t size;
};

template<typename T>
class cudaARRAY2D {
public:
	cudaARRAY2D(std::size_t w, std::size_t h);
	void MoveToDevice(T* from, std::size_t widght, std::size_t height);
	void MoveToDevice(T* from) { MoveToDevice(from, width, height); }
	void MoveToHost(T* to, std::size_t widght, std::size_t height);
	void MoveToHost(T* to) { MoveToHost(to, width, height); }
	std::size_t Widght() const { return width; }
	std::size_t Height() const { return height; }
	cudaArray_t Data() { return data; }
	~cudaARRAY2D() { ERROR_WRAPPER_CALL(cudaFreeArray, data); }
private:
	cudaArray_t data;
	std::size_t width;
	std::size_t height;
};

class cudaRES_DESC {
public:
	cudaRES_DESC(cudaResourceType type, cudaArray_t array)
	{
		memset(&desc, 0, sizeof(desc));
		desc.resType = type;
		desc.res.array.array = array;
	}

	void SetResType(cudaResourceType type) { desc.resType = type; }
	cudaResourceType GetResType() const { return desc.resType; }
	void SetArray(cudaArray_t array) { desc.res.array.array = array; }
	cudaArray_t GetArray() const { return desc.res.array.array; }
	const cudaResourceDesc* Desc() const { return &desc; }
private:
	cudaResourceDesc desc;
};

class cudaTEXTURE_DESC {
public:
	cudaTEXTURE_DESC()
	{
		memset(&desc, 0, sizeof(desc));
		desc.addressMode[0] = cudaAddressModeClamp;
		desc.addressMode[1] = cudaAddressModeClamp;
		desc.filterMode = cudaFilterModePoint;
		desc.readMode = cudaReadModeElementType;
		desc.normalizedCoords = false;
	}

	void SetTextureAddressMode(cudaTextureAddressMode mode, int i)
	{
		//ASSERT_MSG(i < 3, "Wrong index of address mode");
		desc.addressMode[i] = mode;
	}
	cudaTextureAddressMode GetTextureAddressMode(int i) const
	{
		//ASSERT_MSG(i < 3, "Wrong index of address mode");
		return desc.addressMode[i];
	}
	void SetFilterMode(cudaTextureFilterMode mode) { desc.filterMode = mode; }
	cudaTextureFilterMode GetFilterMode() const { return desc.filterMode; }
	void SetReadMode(cudaTextureReadMode mode) { desc.readMode = mode; }
	cudaTextureReadMode GetReadMode() const { return desc.readMode; }
	void SetNormilazedCords(bool mode) { desc.normalizedCoords = mode; }
	bool GetSetNormilazedCords() const { return desc.normalizedCoords; }
	const cudaTextureDesc* Desc() const { return &desc; }
private:
	cudaTextureDesc desc;
};

class cudaTEXTURE_OBJ {
public:
	cudaTEXTURE_OBJ(const cudaTEXTURE_DESC& texture, const cudaRES_DESC& res)
	{
		CALL_CUDA_FUNC(cudaCreateTextureObject, &obj, res.Desc(), texture.Desc(), nullptr);
	}

	cudaTextureObject_t GetObj() const { return obj; }
	~cudaTEXTURE_OBJ() { CALL_CUDA_FUNC(cudaDestroyTextureObject, obj); }
private:
	cudaTextureObject_t obj = 0;
};

template<typename... Args>
class dsEVENT
{
public:
	using CallbackID = int;
	using CallbackType = std::function<void(Args...)>;
public:
	dsEVENT() = default;

	CallbackID Subscribe(CallbackType callback);
	void Trigger(Args... args);
	CallbackID operator+=(CallbackType callback) { return Subscribe(callback); }
	void operator()(Args... args) { Trigger(args...); }
	void Unsubscribe(CallbackID id);
private:
	std::unordered_map<CallbackID, CallbackType> callbacks;
	CallbackID newID = 0;
};

#include "common_structures.hpp"