#pragma once

#include "render_core_common.cuh"
#include "Common/common.cuh"
#include "Engine/object_gpu.cuh"
#include "Engine/scene.h"

struct rendGPU_KERNEL_OPTIONS
{
	vec3* frameBuffer = nullptr;
	int width = 0;
	int height = 0;
	int maxDepth = MAX_DEPTH;
	float aspectRatio;
	float tanFOV;
	vec3 backgroundColor;
	mat4 lookAt;
	int objectsNum;
	int lightsNum;
	engDEV_OBJECT* objects;
	scnDEV_LIGHT_SOURCE* lights;
	int* rayCounts;
};

__device__ void rendTriangleIntersectionGPU(const vec3& origin, const vec3& dir, const engDEV_OBJECT& obj, const engTRIANGLE& trig, OUTPUT<float> t, OUTPUT<float> u, OUTPUT<float> v);

__device__ bool rendShadowRayHitGPU(const vec3& origin, const vec3& dir, engDEV_OBJECT* objects, int objectsNum, OUTPUT<float> hitT);

__device__ bool rendHitGPU(const vec3& origin, const vec3& dir, engDEV_OBJECT* objects, int objectsNum,
	OUTPUT<int> objHitIdx, OUTPUT<vec3> hitPos, OUTPUT<int> trigHitIdx, OUTPUT<float> hitU, OUTPUT<float> hitV, OUTPUT<float> hitT);

struct rendGPU_CONTEXT {
	vec3 origin;
	vec3 direction;
	vec3 color;
	int stage;
	int objHitIdx;
	int trigHitIdx;
	float hitT, hitU, hitV;
	vec3 coef;

	vec3 hitPos;
	vec3 normal;
	float n1, n2;
};

__device__ vec3 rendPhongModelGPU(const vec3& pos, const vec3& direction, const engTRIANGLE& trig, float u, float v,
	const engDEV_OBJECT& obj, engDEV_OBJECT* objects, int objNum, scnDEV_LIGHT_SOURCE* lights, int lightsNum);

__device__ vec3 rendCastRayGPU(const vec3& origin, const vec3& direction, const rendGPU_KERNEL_OPTIONS& opts, OUTPUT<int> rayCount);

__global__ void rendRenderCoreGPU(rendGPU_KERNEL_OPTIONS opts);