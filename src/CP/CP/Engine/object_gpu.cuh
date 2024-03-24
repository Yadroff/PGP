#pragma once

#include "Math/vec.cuh"
#include "Math/math.cuh"
#include "object.h"

struct engTRIANGLE;

struct engDEV_TEXTURE {
	uchar4* data = nullptr;
	int width = 0;
	int height = 0;

	__device__ uchar4 GetColor(const vec3& pos) const
	{
		if (width == 0 || height == 0) {
			return make_uchar4(0, 0, 0, 0);
		}
		int x = min(width, max((int)(pos.x * width), 0));
		int y = min(height, max((int)(pos.y * height), 0));

		return data[y * width + x];
	}
};

struct engDEV_OBJECT {
	engDEV_OBJECT(engOBJECT* obj);
	vec3* vertices;
	vec3* normals;
	vec3* textureCoords;
	engTRIANGLE* triangles;

	int verticesNum;
	int normalsNum;
	int textureCoordsNum;
	int trianglesNum;

	float refractive;

	vec3 ambientColor;
	vec3 diffuseColor;
	vec3 specularColor;

	engDEV_TEXTURE* texture = nullptr;

	float ambientCoeff = 0.2f;
	float diffuseCoeff = 0.6f;
	float specularCoeff = 0.5f;
	float shininess;

	float transparencyCoeff = 1.0f;
	float reflectionCoeff = 1.0f;

	__device__ bool HasTexture() const
	{
		return texture != nullptr;
	}
};