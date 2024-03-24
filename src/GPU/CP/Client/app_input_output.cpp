#include "app_input_output.h"

#include "Engine/camera.h"

std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::CameraMovement& movement)
{
	is >> movement.r0 >> movement.z0 >> movement.phi0 >> movement.Ar >> movement.Az >> movement.wr >> movement.wz >> movement.wphi >> movement.pr >> movement.pz;
	return is;
}

std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::ObjectParams& object)
{
	is >> object.center >> object.color >> object.radius >> object.reflectionCoeff >> object.transparencyCoeff >> object.lightsNum;
	return is;
}

std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::FloorParams& floor)
{
	for (int i = 0; i < appINPUT_SETTINGS::FloorParams::POINT_NUMBER; ++i)
	{
		is >> floor.points[i];
	}
	is >> floor.texturePath >> floor.color >> floor.reflectionCoeff;
	return is;
}

std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::LightParams& light)
{
	is >> light.pos >> light.color;
	return is;
}

std::istream& operator>>(std::istream& is, appINPUT_SETTINGS& settings)
{
	is >> settings.framesNum >> settings.outputPathFormat >> settings.width >> settings.height >> settings.fov >> settings.pos >> settings.eye;
	for (int i = 0; i < settings.OBJECT_NUM; ++i)
	{
		is >> settings.objects[i];
		settings.objects[i].name = settings.objectNames[i];
	}
	is >> settings.floor;
	is >> settings.lightNum;
	settings.light.resize(settings.lightNum);
	settings.light.shrink_to_fit();
	for (auto& light : settings.light)
	{
		is >> light;
	}
	is >> settings.maxDepth >> settings.rayCount;
	return is;
}

void appINPUT_SETTINGS::ApplySettings() const
{
	auto& rendSettings = engGetRenderSettings();
	rendSettings.width = width;
	rendSettings.height = height;
	rendSettings.fov = fov;
	rendSettings.maxDepth = maxDepth;
	rendSettings.backgroundColor = vec3(0.0);
}