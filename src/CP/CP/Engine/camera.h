#pragma once

#include "Math/vec.cuh"
#include "Math/matrix.cuh"

struct rendSETTINGS
{
	const char* GetDevice() const
	{
		switch (device)
		{
		case GPU:
			return "GPU";
		case CPU:
			return "CPU";
		}
	}

	enum Device
	{
		CPU,
		GPU
	};

	int width = 1920;
	int height = 1080;
	float fov = 90.0;
	int maxDepth;
	vec3 backgroundColor;
	Device device = GPU;
	bool useSSAA = true;
	int ssaaKernelSize = 4;
};

rendSETTINGS& engGetRenderSettings();

class engCAMERA
{
public:
	engCAMERA(const vec3& pos, const vec3& view)
		: position(pos)
		, eye(view)
	{
		BuildLookAtMatrix();
	}

	void SetPosition(const vec3& pos)
	{
		position = pos;
	}

	void SetView(const vec3& view)
	{
		eye = view;
	}

	void Update()
	{
		BuildLookAtMatrix();
	}

	vec3 CamToWorld(const vec3& vec) const
	{
		return HomogeneousMult(lookat, vec);
	}

	mat4 LookAt() const { return lookat; }
private:
	void BuildLookAtMatrix(const vec3& tmp = vec3(0.0, 0.0, 1.0));
private:
	vec3 position;
	vec3 eye; // the direction of view

	mat4 lookat;
};
