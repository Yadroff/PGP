#include "quadrangle.h"

objQUADRANGLE::objQUADRANGLE(const vec3 vert[4])
{
	auto v1 = vert[0], v2 = vert[1], v3 = vert[2], v4 = vert[3];
	vec3 e2 = v2 - v1, e3 = v3 - v1, e4 = v4 - v1;
	vec3 norm1 = vec3::CrossProduct(e3, e2).Normalized();
	vec3 norm2 = vec3::CrossProduct(e4, e3).Normalized();

	vertices = { v1, v2, v3, v4 };
	normals = { norm1, norm2 };
	textureCords = { vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0) };
	polygons = { {
					0, 2, 1,
					0, 0, 0,
					0, 2, 1
					},
					{
					0, 3, 2,
					1, 1, 1,
					0, 3, 2
					} };
	// Set floor params
	ambientColor = vec3(0.1f, 0.0f, 0.0f);
	diffuseColor = vec3(1.0f, 0.0f, 0.0f);
	specularColor = vec3(1.0f);

	ambientCoeff = 0.2f;
	diffuseCoeff = 0.4f;
	specularCoeff = 0.8f;
	shininess = 128.0f;

	transparencyCoeff = 0.0f;
	refractive = 1.0f;
}