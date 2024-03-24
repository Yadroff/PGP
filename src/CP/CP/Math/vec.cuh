#pragma once

#include "Common/common.cuh"

template<typename T>
struct VEC3_T {
	T x, y, z;
	__host__ __device__ VEC3_T() = default;

	__host__ __device__ VEC3_T(const T& val) : x(val), y(val), z(val) {}

	__host__ __device__ VEC3_T(const T& x, const T& y, const T& z) : x(x), y(y), z(z) {}

	__host__ __device__ float Length() const;
	__host__ __device__ VEC3_T Normalized() const;

	__host__ __device__ VEC3_T& operator+=(const VEC3_T& other);
	__host__ __device__ VEC3_T operator-() const;
	__host__ __device__ VEC3_T& operator-=(const VEC3_T& other) { return *this += (-other); }
	// Component-by-component multiplication
	__host__ __device__ VEC3_T& operator*=(const VEC3_T& other);

	__host__ __device__ static VEC3_T CrossProduct(const VEC3_T& left, const VEC3_T& right);
	__host__ __device__ T DotProduct(const VEC3_T& other) const;
};

template<typename T>
struct VEC4_T {
	T x, y, z, w;
	__host__ __device__ VEC4_T() = default;
	__host__ __device__ VEC4_T(T val) : x(val), y(val), z(val), w(val) {}
	__host__ __device__ VEC4_T(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

	__host__ __device__ float Length() const;

	__host__ __device__ VEC4_T Normalized() const;

	__host__ __device__ VEC4_T& operator+=(const VEC4_T& other);
	__host__ __device__ VEC4_T& operator-=(const VEC4_T& other);
	__host__ __device__ VEC4_T operator-() const;
	// Component-by-component multiplication
	__host__ __device__ VEC4_T& operator*=(const VEC4_T& other);
};

/// vec  norm  return
///	\    |    /
///	 \   |   /
///	  \  |  /
///	   \ | /
/// ---------------
template<typename T>
__host__ __device__ VEC3_T<T> Reflect(const VEC3_T<T>& vec, const VEC3_T<T>& normal)
{
	return vec - 2.0f * vec.DotProduct(normal) * normal;
}

/// vec normal Reflect
///	\    |    /
///	 \   |   /
///	  \  |  /
///	   \ | /      n1
/// ---------------
///	     |\      n2
///		 | \
///		 |  \
///		 |   \
///			Return
///	Angle between return vector and `normal` depends on `n1` and `n2`
template<typename T>
__host__ __device__ VEC3_T<T> Refract(const VEC3_T<T>& vec, const VEC3_T<T>& normal, const float n1, const float n2)
{
	float r = n1 / n2;
	float c = -normal.DotProduct(vec);

	return r * vec + (r * c - sqrt(1.0f - r * r * (1.0f - c * c))) * normal;
}

#include "vec3.hpp"
#include "vec4.hpp"

using vec3f = VEC3_T<float>;
using vec3i = VEC3_T<int>;
using vec3d = VEC3_T<double>;
using vec3 = vec3f; // default

using vec4f = VEC4_T<float>;
using vec4i = VEC4_T<int>;
using vec4d = VEC4_T<double>;
using vec4 = vec4f; // default