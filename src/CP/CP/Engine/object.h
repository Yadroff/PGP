#pragma once

#include "texture.h"

struct engTRIANGLE {
	int n1, n2, n3;
	int v1, v2, v3;
	int t1, t2, t3;

	engTRIANGLE(
		int v1, int v2, int v3,
		int n1, int n2, int n3,
		int t1, int t2, int t3
	) :
		n1(n1), n2(n2), n3(n3),
		v1(v1), v2(v2), v3(v3),
		t1(t1), t2(t2), t3(t3)
	{}
};

struct engOBJECT
{
#pragma region Values
	std::vector<vec3> vertices;
	std::vector<vec3> normals;
	std::vector<vec3> textureCords;
	std::vector<engTRIANGLE> polygons;

	float refractive;

	vec3 ambientColor = vec3(1.0f);
	vec3 diffuseColor = vec3(1.0f);
	vec3 specularColor = vec3(1.0f);

	float shininess;
	float ambientCoeff;
	float diffuseCoeff;
	float specularCoeff;

	float transparencyCoeff;
	float reflectionCoeff;

	engTEXTURE* ptrTexture = nullptr;

	std::string name;
#pragma endregion

#pragma region Methods
	engOBJECT() = default;
	engOBJECT(const std::string& name) : name(name) {}
	virtual ~engOBJECT() = default;
	void SetTexture(engTEXTURE* texture) { ptrTexture = texture; }

	bool HasTexture() const { return ptrTexture != nullptr; }

	engOBJECT& Move(const vec3& pos);
	engOBJECT& Rotate(const vec3& rot);
	engOBJECT& Rotate(const mat3 rotMat);
	engOBJECT& Scale(const vec3& scaleCoef);
	static engOBJECT ImportFromObj(const std::string& path);
	void SetName(const std::string& name) { this->name = name; }
	std::string GetName() const { return name; }
	std::vector<std::shared_ptr<engOBJECT>> GenerateLights(float r, float a, float margin, float offset, int lightsNum, float lightRadius);
#pragma endregion
};