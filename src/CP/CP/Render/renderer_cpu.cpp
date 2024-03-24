#include <numeric>
#include <fstream>
#include "renderer_cpu.h"

#include "render_core_cpu.h"
#include "ssaa.cuh"
#include "Engine/scene.h"
#include "Math/math.cuh"

static void SavePPM(const std::string& path, const std::vector<vec3>& buffer, int width, int height)
{
	std::ofstream out(path, std::ios::binary);
	out << "P6\n" << width << " " << height << "\n255\n";
	for (const auto& rgb : buffer)
	{
		unsigned char r = static_cast<unsigned char>(255.0 * Clamp(rgb.x));
		unsigned char g = static_cast<unsigned char>(255.0 * Clamp(rgb.y));
		unsigned char b = static_cast<unsigned char>(255.0 * Clamp(rgb.z));
		out << r << g << b;
	}
}

void rendRENDERER_CPU::Render(rendSETTINGS* settings, engCAMERA* camera, OUTPUT<std::vector<vec3>> frameBuffer)
{
	lastWidth = settings->width * (settings->useSSAA ? settings->ssaaKernelSize : 1);
	lastHeight = settings->height * (settings->useSSAA ? settings->ssaaKernelSize : 1);
	rendCPU_OPTIONS options(std::move(scene->GetObjects()), std::move(scene->GetLights()), rayCount, frameBuffer);
	float aspectRatio = static_cast<float>(settings->width) / settings->height;
	float tanFOV = tan(settings->fov / 2.0 / 180.0f * M_PI);
	options.width = lastWidth;
	options.height = lastHeight;
	options.maxDepth = settings->maxDepth;
	options.backgroundColor = settings->backgroundColor;
	options.aspectRatio = aspectRatio;
	options.tanFOV = tanFOV;
	options.lookAt = camera->LookAt();
	rayCount.resize(lastWidth * lastHeight);
	frameBuffer->resize(lastWidth * lastHeight);
	rendRenderAsyncCPU(options);
	// Save upscale image
	{
		SavePPM(R"(.\upscale_frame.ppm)", *frameBuffer, lastWidth, lastHeight);
	}
	if (settings->useSSAA)
	{
		int aaWidth = settings->width;
		int aaHeight = settings->height;
		std::vector<vec3> aaFrames(aaWidth * aaHeight);
		rendSSAA_CPU(frameBuffer->data(), aaFrames.data(), lastWidth, lastHeight, aaWidth, aaHeight);
		{
			SavePPM(R"(.\ssaa_frame.ppm)", aaFrames, aaWidth, aaHeight);
		}
		*frameBuffer = std::move(aaFrames);
	}
}

int rendRENDERER_CPU::RayCount()
{
	return std::accumulate(all(rayCount), 0);
}