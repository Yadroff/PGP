#pragma once

#include "Common/common.cuh"
#include "render_core_common.cuh"
#include "Engine/object.h"
#include "Engine/scene.h"

struct rendCPU_OPTIONS
{
	rendCPU_OPTIONS(std::vector<std::shared_ptr<engOBJECT>>&& objects, std::vector<std::shared_ptr<scnLIGHT_SOURCE>>&& lights, std::vector<int>& rays, OUTPUT<std::vector<vec3>> frames)
		: frameBuffer(frames)
		, objects(objects)
		, lights(lights)
		, rayCounts(rays)
	{}

	OUTPUT<std::vector<vec3>> frameBuffer;
	int width = 0;
	int height = 0;
	int maxDepth = MAX_DEPTH;
	float aspectRatio;
	float tanFOV;
	vec3 backgroundColor;
	mat4 lookAt;
	std::vector<std::shared_ptr<engOBJECT>> objects;
	std::vector< std::shared_ptr<scnLIGHT_SOURCE>> lights;
	std::vector<int>& rayCounts;
};

void rendRenderAsyncCPU(const rendCPU_OPTIONS& options);
void rendRenderCoreCPU(const rendCPU_OPTIONS& opts, int idX, int idY, int offsetX, int offsetY);
vec3 rendCastRayCPU(const vec3& origin, const vec3& direction, const rendCPU_OPTIONS& opts, int curDepth, float refractive, OUTPUT<int> rayCount);
bool rendHitCPU(const vec3& origin, const vec3& direction, const std::vector<std::shared_ptr<engOBJECT>>& objects, OUTPUT<int> hitObjIdx, OUTPUT<vec3> hitPos, OUTPUT<int> hitTriangleIdx, OUTPUT<float> hitU, OUTPUT<float> hitV, OUTPUT<float> hitT);
vec3 rendPhongModelCPU(const vec3& hitPos, const vec3& direction, const engTRIANGLE& polygon, float hitU, float hitV, const engOBJECT& obj, const std::vector<std::shared_ptr<engOBJECT>>& objects, const std::vector<std::shared_ptr<scnLIGHT_SOURCE>>& lights);
void rendTriangleIntersectionCPU(const vec3& origin, const vec3& dir, const engOBJECT& obj, const engTRIANGLE& trig, OUTPUT<float> t, OUTPUT<float> u, OUTPUT<float> v);
bool rendShadowRayHitCPU(const vec3& origin, const vec3& direction, const std::vector<std::shared_ptr<engOBJECT>>& objects, OUTPUT<float> hitT);