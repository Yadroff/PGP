#include "math.cuh"

__host__ __device__ vec3 CylindricalToDecart(const vec3& vec)
{
	return { vec.x * cos(vec.y), vec.x * sin(vec.y), vec.z };
}