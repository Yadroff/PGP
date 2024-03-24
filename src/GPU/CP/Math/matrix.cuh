#pragma once

#include "vec.cuh"

template<typename T>
struct MAT3_T {
	T Data[3][3];
	__host__ __device__ MAT3_T() = default;
	__host__ __device__ MAT3_T(float m11, float m12, float m13,
		float m21, float m22, float m23,
		float m31, float m32, float m33) {
		Data[0][0] = m11; Data[0][1] = m12; Data[0][2] = m13;
		Data[1][0] = m21; Data[1][1] = m22; Data[1][2] = m23;
		Data[2][0] = m31; Data[2][1] = m32; Data[2][2] = m33;
	}

	__host__ __device__ static MAT3_T Identity() {
		MAT3_T res;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				res.Data[i][j] = static_cast<T>(0);
			}
		}
		for (int i = 0; i < 3; ++i) {
			res.Data[i][i] = static_cast<T>(1);
		}

		return res;
	}

	__host__ __device__ T Det() const;
	__host__ __device__ MAT3_T Inv() const;
};

template<typename T>
struct MAT4_T {
	T Data[4][4];
	__host__ __device__ MAT4_T() = default;
	__host__ __device__ MAT4_T(
		float m11, float m12, float m13, float m14,
		float m21, float m22, float m23, float m24,
		float m31, float m32, float m33, float m34,
		float m41, float m42, float m43, float m44)
	{
		Data[0][0] = m11; Data[0][1] = m12; Data[0][2] = m13; Data[0][3] = m14;
		Data[1][0] = m21; Data[1][1] = m22; Data[1][2] = m23; Data[1][3] = m24;
		Data[2][0] = m31; Data[2][1] = m32; Data[2][2] = m33; Data[2][3] = m34;
		Data[3][0] = m41; Data[3][1] = m42; Data[3][2] = m43; Data[3][3] = m44;
	}
};

#include "mat3.hpp"
#include "mat4.hpp"

template<typename T>
__host__ __device__ VEC3_T<T> HomogeneousMult(const MAT4_T<T>& m, const VEC3_T<T>& v)
{
	VEC4_T<T> tmp(v.x, v.y, v.z, 1.0f);
	tmp = m * tmp;
	return { tmp.x, tmp.y, tmp.z };
}

template<typename T>
__host__ __device__ MAT3_T<T> AlignMat(const VEC3_T<T>& a, const VEC3_T<T>& b) {
	VEC3_T<T> v = VEC3_T<T>::CrossProduct(a, b);
	float c = a.DotProduct(b);

	MAT3_T<T> m(
		0.0f, -v.z, v.y,
		v.z, 0.0f, -v.x,
		-v.y, v.x, 0.0f
	);

	return MAT3_T<T>::Identity() + m + 1.0f / (1.0f + c) * m * m;
}

using mat3f = MAT3_T<float>;
using mat3d = MAT3_T<double>;
using mat3i = MAT3_T<int>;
using mat3 = mat3f; // default

using mat4f = MAT4_T<float>;
using mat4d = MAT4_T<double>;
using mat4i = MAT4_T<int>;
using mat4 = mat4f; // default