#include "circle.h"

objCIRCLE::objCIRCLE(float r, int segmentsNum)
{
	float angle = 2 * M_PI / segmentsNum;
	vertices.emplace_back(0.0);
	textureCords.emplace_back(0.5, 0.5, 0);
	normals.emplace_back(0.0, 1.0, 0.0);

	for (int i = 0; i < segmentsNum; ++i)
	{
		float c = cosf(angle * i), s = sinf(angle * i);

		vertices.emplace_back(r * c, 0.0, r * s);
		textureCords.emplace_back(0.5f + 0.5f * c, 0.5f + 0.5f * s, 0.0f);
		if (i != 0)
		{
			polygons.emplace_back(i + 1, i, 0, 0, 0, 0, i + 1, i, 0);
		}
	}
	polygons.emplace_back(1, segmentsNum, 0, 0, 0, 0, 1, segmentsNum, 0);

	// Light Coeffs
	reflectionCoeff = 0.0f;
	transparencyCoeff = 0.0f;
	refractive = 1.0f;

	ambientCoeff = 1.0f;
	diffuseCoeff = 0.0f;
	specularCoeff = 0.0f;
	shininess = 2.0f;

	ambientColor = vec3(10.0f);
	diffuseColor = vec3(10.0f);
	specularColor = vec3(10.0f);
}