#pragma once

#include <unordered_map>

#include "renderer.cuh"
#include "Common/common.cuh"
#include "Common/common_structures.cuh"
#include "Engine/scene.h"

class engTEXTURE;
struct rendSETTINGS;
class engCAMERA;
struct engDEV_OBJECT;
struct engDEV_TEXTURE;

class rendRENDERER_GPU : public rendRENDERER
{
public:
	void PrepareGPU();
	rendRENDERER_GPU(scnSCENE* scene)
		: scene(scene)
	{}
	~rendRENDERER_GPU() override;
	void Render(rendSETTINGS* settings, engCAMERA* camera, OUTPUT<std::vector<vec3>> frameBuffer) override;
	int RayCount() override;
private:
	void PrepareTextures();
	void PrepareObjects();
	void PrepareLights();
	void ClearObjectsAndLights();
	void ClearTextures();
private:
	std::unordered_map<engTEXTURE*, engDEV_TEXTURE*> textures{ {nullptr, nullptr} };
	scnSCENE* scene = nullptr;
	int objectsNum = 0;
	int lightsNum = 0;
	engDEV_OBJECT* devObjects = nullptr;
	scnDEV_LIGHT_SOURCE* devLights = nullptr;

	vec3* devFrameBuffer = nullptr;
	int* devRays = nullptr;
	int rayCount = 0;

	int lastWidth = 0;
	int lastHeight = 0;
};