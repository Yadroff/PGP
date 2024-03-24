#include <cmath>

#include "renderer_gpu.cuh"

#include "render_core_gpu.cuh"
#include "ssaa.cuh"
#include "Engine/object_gpu.cuh"
#include "Common/common_structures.cuh"
#include "Common/dev_res_collector.cuh"
#include "Engine/camera.h"

void rendRENDERER_GPU::PrepareGPU()
{
	if (!scene->WasChanged())
	{
		return;
	}
	ClearObjectsAndLights();
	PrepareTextures();
	PrepareObjects();
	PrepareLights();
	scene->SetWasChanged(false);
}

rendRENDERER_GPU::~rendRENDERER_GPU()
{
	ClearObjectsAndLights();
	scene = nullptr;
	ClearTextures();
}

void rendRENDERER_GPU::Render(rendSETTINGS* settings, engCAMERA* camera, OUTPUT<std::vector<vec3>> frameBuffer)
{
	camera->Update();
	lastWidth = settings->width * (settings->useSSAA ? settings->ssaaKernelSize : 1);
	lastHeight = settings->height * (settings->useSSAA ? settings->ssaaKernelSize : 1);
	rayCount = 0;
	PrepareGPU();
	if (!devFrameBuffer)
	{
		resGetCollector().Alloc(&devFrameBuffer, lastWidth * lastHeight * sizeof(vec3));
		resGetCollector().Alloc(&devRays, lastWidth * lastHeight * sizeof(int));
	}

	float aspectRatio = static_cast<float>(settings->width) / settings->height;
	float tanFOV = tanf(settings->fov / 2.0f / 180.0f * static_cast<float>(M_PI));

	rendGPU_KERNEL_OPTIONS options;
	options.frameBuffer = devFrameBuffer;
	options.width = lastWidth;
	options.height = lastHeight;
	options.maxDepth = settings->maxDepth;
	options.backgroundColor = settings->backgroundColor;
	options.aspectRatio = aspectRatio;
	options.tanFOV = tanFOV;
	options.lookAt = camera->LookAt();
	options.objectsNum = objectsNum;
	options.lightsNum = lightsNum;
	options.objects = devObjects;
	options.lights = devLights;
	options.rayCounts = devRays;

	CALL_KERNEL(rendRenderCoreGPU, options);

	frameBuffer->resize(lastWidth * lastHeight);
	CALL_CUDA_FUNC(cudaMemcpy, frameBuffer->data(), devFrameBuffer, lastWidth * lastHeight * sizeof(vec3), cudaMemcpyDeviceToHost);

	if (settings->useSSAA)
	{
		int aaWidth = settings->width;
		int aaHeight = settings->height;
		std::vector<vec3> aaFrames(aaWidth * aaHeight);
		rendSSAA_GPU(frameBuffer->data(), aaFrames.data(), lastWidth, lastHeight, aaWidth, aaHeight);
		std::swap(*frameBuffer, aaFrames);
	}
}

int rendRENDERER_GPU::RayCount()
{
	if (devRays == nullptr || devFrameBuffer == nullptr)
	{
		return rayCount;
	}
	rayCount = 0;
	std::vector<int> raysHost(lastWidth * lastHeight);
	CALL_CUDA_FUNC(cudaMemcpy, raysHost.data(), devRays, lastWidth * lastHeight * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < lastHeight; ++i)
	{
		for (int j = 0; j < lastWidth; ++j)
		{
			rayCount += raysHost[i * lastWidth + j];
		}
	}
	return rayCount;
}

void rendRENDERER_GPU::PrepareTextures()
{
	for (auto pObj : scene->GetObjects())
	{
		if (pObj->HasTexture())
		{
			// if texture not in map
			if (textures.find(pObj->ptrTexture) == textures.end())
			{
				engDEV_TEXTURE textureHost;
				textureHost.width = pObj->ptrTexture->Width();
				textureHost.height = pObj->ptrTexture->Height();

				// Allocate GPU buffer
				resGetCollector().Alloc(&textureHost.data, textureHost.width * textureHost.height * sizeof(uchar4));
				// Copy texture to GPU
				CALL_CUDA_FUNC(cudaMemcpy, textureHost.data, pObj->ptrTexture->Data(), textureHost.width * textureHost.height * sizeof(uchar4), cudaMemcpyHostToDevice);

				// Allocate structure at GPU
				engDEV_TEXTURE* devTexture;
				resGetCollector().Alloc(&devTexture);

				// Move texture from CPU to GPU
				CALL_CUDA_FUNC(cudaMemcpy, devTexture, &textureHost, sizeof(*devTexture), cudaMemcpyHostToDevice);

				// Add new texture to map
				textures[pObj->ptrTexture] = devTexture;
			}
		}
	}
}

void rendRENDERER_GPU::PrepareObjects()
{
	objectsNum = scene->GetObjects().size();
	std::vector<engDEV_OBJECT> devObjectsHost;
	devObjectsHost.reserve(objectsNum);

	for (auto& obj : scene->GetObjects())
	{
		engDEV_OBJECT objectHost(obj.get());
		objectHost.texture = textures[obj->ptrTexture];
		devObjectsHost.emplace_back(objectHost);
	}

	// Alloc GPU memory for objects
	resGetCollector().Alloc(&devObjects, sizeof(engDEV_OBJECT) * objectsNum);
	// Move objects to GPU
	CALL_CUDA_FUNC(cudaMemcpy, devObjects, devObjectsHost.data(), objectsNum * sizeof(engDEV_OBJECT), cudaMemcpyHostToDevice);
}

void rendRENDERER_GPU::PrepareLights()
{
	lightsNum = scene->GetLights().size();
	std::vector<scnDEV_LIGHT_SOURCE> devLightsHost;
	devLightsHost.reserve(lightsNum);

	for (const auto& obj : scene->GetLights())
	{
		scnDEV_LIGHT_SOURCE lightHost(*obj);
		devLightsHost.emplace_back(lightHost);
	}

	// Alloc GPU memory for objects
	resGetCollector().Alloc(&devLights, sizeof(scnDEV_LIGHT_SOURCE) * lightsNum);
	// Move objects to GPU
	CALL_CUDA_FUNC(cudaMemcpy, devLights, devLightsHost.data(), lightsNum * sizeof(scnDEV_LIGHT_SOURCE), cudaMemcpyHostToDevice);
}

void rendRENDERER_GPU::ClearObjectsAndLights()
{
	resGetCollector().Free(devLights);
	resGetCollector().Free(devObjects);
	devLights = nullptr;
	devObjects = nullptr;
	objectsNum = 0;
	lightsNum = 0;
}

void rendRENDERER_GPU::ClearTextures()
{
	for (const auto it : textures)
	{
		const auto devText = it.second;
		resGetCollector().Free(devText->data);
		resGetCollector().Free(devText);
	}
	textures.clear();
}