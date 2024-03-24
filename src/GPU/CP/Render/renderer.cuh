#pragma once

#include "Engine/camera.h"

class scnSCENE;

class rendRENDERER
{
public:
	virtual void Render(rendSETTINGS* settings, engCAMERA* camera, OUTPUT<std::vector<vec3>> frameBuffer) {}
	virtual int RayCount() { return 0; }
	virtual ~rendRENDERER() = default;
};