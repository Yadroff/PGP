#pragma once

#include <type_traits>
#include <algorithm>

#include "vec.cuh"
#include "Common/common.cuh"

#define EPS 0.0003f

template<typename T, typename = std::enable_if<std::is_floating_point<T>::value>>
__host__ __device__ bool ApproxEqual(const T& lhs, const T& rhs, const T& eps = EPS)
{
	return std::abs(lhs - rhs) < eps;
}

template<typename T>
__host__ __device__ T Clamp(const T& val, const T& minimum = 0, const T& maximum = 1)
{
	return std::max(std::min(val, maximum), minimum);
}

__host__ __device__ vec3 CylindricalToDecart(const vec3& vec);