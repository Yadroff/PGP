#pragma once

#include <unordered_set>

class DevResourceCollector
{
public:
	DevResourceCollector() = default;
	virtual ~DevResourceCollector();
	void Alloc(void** ptr, size_t size);
	void Free(void* ptr);
	template<typename T>
	void Alloc(void** ptr)
	{
		Alloc(ptr, sizeof(T));
	}
	template<typename T, typename = std::enable_if<!std::is_pointer<T>::value>>
	void Alloc(T** ptr, size_t size)
	{
		Alloc(reinterpret_cast<void**>(ptr), size);
	}
	template<typename T, typename = std::enable_if<!std::is_pointer<T>::value>>
	void Alloc(T** ptr)
	{
		Alloc<T>(reinterpret_cast<void**>(ptr));
	}
private:
	std::unordered_set<void*> pointers;
};

DevResourceCollector& resGetCollector();