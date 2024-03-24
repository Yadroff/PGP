#include "object_gpu.cuh"
#include "Common/common_structures.cuh"
#include "Common/dev_res_collector.cuh"

engDEV_OBJECT::engDEV_OBJECT(engOBJECT* obj)
{
	ASSERT_MSG(!obj->GetName().empty(), "Object should have a name");
	verticesNum = obj->vertices.size();
	normalsNum = obj->vertices.size();
	textureCoordsNum = obj->textureCords.size();
	trianglesNum = obj->polygons.size();

	resGetCollector().Alloc(&vertices, sizeof(vec3) * verticesNum);
	resGetCollector().Alloc(&normals, sizeof(vec3) * normalsNum);
	resGetCollector().Alloc(&textureCoords, sizeof(vec3) * textureCoordsNum);
	resGetCollector().Alloc(&triangles, sizeof(engTRIANGLE) * trianglesNum);

	// Copy coeffs
	refractive = obj->refractive;
	ambientColor = obj->ambientColor;
	diffuseColor = obj->diffuseColor;
	specularColor = obj->specularColor;
	ambientCoeff = obj->ambientCoeff;
	diffuseCoeff = obj->diffuseCoeff;
	specularCoeff = obj->specularCoeff;
	shininess = obj->shininess;
	transparencyCoeff = obj->transparencyCoeff;
	reflectionCoeff = obj->reflectionCoeff;

	// Move arrays to device
	CALL_CUDA_FUNC(cudaMemcpy, vertices, obj->vertices.data(), sizeof(vec3) * verticesNum, cudaMemcpyHostToDevice);
	CALL_CUDA_FUNC(cudaMemcpy, triangles, obj->polygons.data(), sizeof(engTRIANGLE) * trianglesNum, cudaMemcpyHostToDevice);
	CALL_CUDA_FUNC(cudaMemcpy, normals, obj->normals.data(), sizeof(vec3) * normalsNum, cudaMemcpyHostToDevice);
	CALL_CUDA_FUNC(cudaMemcpy, textureCoords, obj->textureCords.data(), sizeof(vec3) * textureCoordsNum, cudaMemcpyHostToDevice);
}