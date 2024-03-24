#pragma once

#include <memory>

#include "Common/common.cuh"
#include "object.h"

struct scnLIGHT_SOURCE
{
	vec3 pos;
	vec3 color;

	float intensity;

	scnLIGHT_SOURCE(const vec3& pos, const vec3& color, float intensity)
		: pos(pos)
		, color(color)
		, intensity(intensity)
	{}
};

using scnDEV_LIGHT_SOURCE = scnLIGHT_SOURCE;

class scnSCENE
{
public:
	scnSCENE() = default;
	scnSCENE(const std::string& name, const std::vector<std::shared_ptr<engOBJECT>>& objects)
		: name(name)
		, objects(objects)
	{}

	void AddObject(std::shared_ptr<engOBJECT> obj) { objects.emplace_back(obj);  wasChanged = true; }
	auto GetObjects() { return objects; }
	void AddLightSource(std::shared_ptr<scnLIGHT_SOURCE> source) { lights.emplace_back(source); wasChanged = true; }
	auto GetLights() { return lights; }
	std::string Name() const { return name; }
	bool WasChanged() const { return wasChanged; }
	void SetWasChanged(bool val) { wasChanged = val; }
private:
	bool wasChanged = false;
	std::string name;
	std::vector<std::shared_ptr<engOBJECT>> objects;
	std::vector<std::shared_ptr<scnLIGHT_SOURCE>> lights;
};
