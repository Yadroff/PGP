#pragma once
#include "renderer.cuh"

class rendRENDERER_CPU : public rendRENDERER
{
public:
	rendRENDERER_CPU(scnSCENE* scn)
		: scene(scn)
	{}

	void Render(rendSETTINGS* settings, engCAMERA* camera, OUTPUT<std::vector<vec3>> frameBuffer) override;
	int RayCount() override;
private:
	scnSCENE* scene = nullptr;
	std::vector<int> rayCount;
	int lastWidth = 0;
	int lastHeight = 0;
};
