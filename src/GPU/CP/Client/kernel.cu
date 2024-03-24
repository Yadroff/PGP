#include <chrono>
#include <fstream>

#include "Common/common.cuh"
#include "Common/common_structures.cuh"
#include "app_input_output.h"
#include "ap_cmd_line.h"
#include "Engine/camera.h"
#include "Engine/scene.h"
#include "Engine/Primitives/quadrangle.h"
#include "Math/math.cuh"
#include "Render/renderer.cuh"
#include "Render/renderer_cpu.h"
#include "Render/renderer_gpu.cuh"

static void DataSave(const std::string& path, const std::vector<vec3>& buffer, int width, int height)
{
	std::ofstream out(path, std::ios::binary);
	ASSERT_MSG(out.is_open(), "Can not open file %s", path.c_str());
	out.write(reinterpret_cast<const char*>(&width), sizeof(width));
	out.write(reinterpret_cast<const char*>(&height), sizeof(height));
	for (const auto& rgb : buffer)
	{
		unsigned char r = static_cast<unsigned char>(255.0 * Clamp(rgb.x));
		unsigned char g = static_cast<unsigned char>(255.0 * Clamp(rgb.y));
		unsigned char b = static_cast<unsigned char>(255.0 * Clamp(rgb.z));
		unsigned char a = 0;
		out.write(reinterpret_cast<const char*>(&r), sizeof(r));
		out.write(reinterpret_cast<const char*>(&g), sizeof(g));
		out.write(reinterpret_cast<const char*>(&b), sizeof(b));
		out.write(reinterpret_cast<const char*>(&a), sizeof(a));
	}
}

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

dsEVENT<int> onFrameRendered;
dsEVENT<int> onStartFrameRender;

static constexpr const char* resourcesPath = "./resources/";

#define SetUpObject(objVar, objType)							\
	objVar.ambientColor = objType##AmbientColor;				\
	objVar.diffuseColor = objType##DiffuseColor;				\
	objVar.specularColor = objType##SpecularColor;				\
	objVar.transparencyCoeff = objType##TransparencyCoeff;		\
	objVar.reflectionCoeff = objType##ReflectionCoeff;			\
	objVar.refractive = objType##RefractiveIdx;					\
	objVar.ambientCoeff = objType##AmbientCoeff;				\
	objVar.diffuseCoeff = objType##DiffuseCoeff;				\
	objVar.specularCoeff = objType##SpecularCoeff;				\
	objVar.shininess = objType##Shininess

std::vector<std::shared_ptr<engOBJECT>> LoadObject(const std::string& path, const appINPUT_SETTINGS::ObjectParams& settings, std::function<float(float)> getSight)
{
	static vec3 FrameAmbientColor(0.101f);
	static vec3 FrameDiffuseColor(0.125f);
	static vec3 FrameSpecularColor(1.0);

	static constexpr float FrameAmbientCoeff = 0.2f;
	static constexpr float FrameDiffuseCoeff = 0.4f;
	static constexpr float FrameSpecularCoeff = 1.0f;

	static constexpr float FrameTransparencyCoeff = 0.0f;
	static constexpr float FrameReflectionCoeff = 0.0f;
	static constexpr float FrameRefractiveIdx = 1.0f;
	static constexpr float FrameShininess = 32.0f;

	static vec3 GlassSpecularColor(1.0f);
	static constexpr float GlassAmbientCoeff = 1.0f;
	static constexpr float GlassDiffuseCoeff = 0.2;
	static constexpr float GlassSpecularCoeff = 1.0f;
	static constexpr float GlassRefractiveIdx = 1.5f;
	static constexpr float GlassShininess = 64.0f;
	std::vector<std::shared_ptr<engOBJECT>> res;
	// Load glass
	{
		vec3 GlassAmbientColor = settings.color;
		vec3 GlassDiffuseColor = settings.color;
		float GlassTransparencyCoeff = settings.transparencyCoeff;
		float GlassReflectionCoeff = settings.reflectionCoeff;
		dsSTRING_BUILDER builder;
		builder.AppendFmt("%s/%s_glass.obj", path.c_str(), settings.name.c_str());
		engOBJECT obj = engOBJECT::ImportFromObj(builder.Str())
			.Scale(settings.radius)
			.Move(settings.center);
		dsSTRING_BUILDER nameBuilder;
		nameBuilder.AppendFmt("%s_%s", settings.name.c_str(), "glass");
		obj.SetName(nameBuilder.Str());
		SetUpObject(obj, Glass);
		res.emplace_back(std::make_shared<engOBJECT>(std::move(obj)));
	}
	// Load object
	{
		dsSTRING_BUILDER builder;
		builder.AppendFmt("%s/%s_frame.obj", path.c_str(), settings.name.c_str());
		engOBJECT obj = engOBJECT::ImportFromObj(builder.Str()).Scale(settings.radius).Move(settings.center);
		dsSTRING_BUILDER nameBuilder;
		nameBuilder.AppendFmt("%s_%s", settings.name.c_str(), "frame");
		obj.SetName(nameBuilder.Str());
		SetUpObject(obj, Frame);
		float sight = getSight(settings.radius);
		auto lights = obj.GenerateLights(settings.radius, sight, 0.2f * settings.radius, 0.08f * settings.radius, settings.lightsNum, 0.02f * settings.radius);
		res.emplace_back(std::make_shared<engOBJECT>(std::move(obj)));
		std::for_each(all(lights), [&res](auto light) {res.emplace_back(light); });
	}
	return res;
}
#undef SET_UP_OBJECT

static void PrintDefault()
{
	dsSTRING_BUILDER builder;
	// frames, output, resoulution, fov
	builder.AppendFmt("%d\n", 10);
	builder.AppendFmt("%s\n", R"(.\output\PIC%03d.data)");
	builder.AppendFmt("%d %d %d\n", 640, 480, 90);

	// camera position
	builder.AppendFmt("%f %f %f\t", 4.1f, 1.0f, 0.0f);
	builder.AppendFmt("%f %f\t", 2.0f, 0.3f);
	builder.AppendFmt("%f %f %f\t", 1.0f, 1.0f, 2.0f);
	builder.AppendFmt("%f %f\n", 0.0, 0.0);

	// camera eye
	builder.AppendFmt("%f %f %f\t", 1.0f, 0.0f, 0.0f);
	builder.AppendFmt("%f %f\t", 0.5f, 0.1f);
	builder.AppendFmt("%f %f %f\t", 1.0f, 1.0f, 2.0f);
	builder.AppendFmt("%f %f\n", 0.0, 0.0);

	// 1st figure
	builder.AppendFmt("%f %f %f\t", 3.0f, 3.0f, 0.0f);
	builder.AppendFmt("%f %f %f\t", 1.0f, 0.0f, 0.0f);
	builder.AppendFmt("%f\t", 2.0f);
	builder.AppendFmt("%f\t", 0.1f);
	builder.AppendFmt("%f\t", 0.9f);
	builder.AppendFmt("%d\n", 10);

	// 2nd figure
	builder.AppendFmt("%f %f %f\t", 0.0f, 0.0f, 0.0f);
	builder.AppendFmt("%f %f %f\t", 0.0f, 1.0f, 0.0f);
	builder.AppendFmt("%f\t", 1.5f);
	builder.AppendFmt("%f\t", 0.2f);
	builder.AppendFmt("%f\t", 0.8f);
	builder.AppendFmt("%d\n", 10);

	// 3rd figure
	builder.AppendFmt("%f %f %f\t", -3.0f, -3.0f, 0.0f);
	builder.AppendFmt("%f %f %f\t", 0.0f, 0.7f, 0.7f);
	builder.AppendFmt("%f\t", 1.0f);
	builder.AppendFmt("%f\t", 0.3f);
	builder.AppendFmt("%f\t", 0.7f);
	builder.AppendFmt("%d\n", 10);

	// floor
	builder.AppendFmt("%f %f %f\t", -5.0f, -5.0f, -2.1f);
	builder.AppendFmt("%f %f %f\t", -5.0f, 5.0f, -2.1f);
	builder.AppendFmt("%f %f %f\t", 5.0f, 5.0f, -2.1f);
	builder.AppendFmt("%f %f %f\t", 5.0f, -5.0f, -2.1f);
	builder.AppendFmt("%s\t", R"( .\resources\texture4.data)");
	builder.AppendFmt("%f %f %f\t", 1.0f, 1.0f, 1.0f);
	builder.AppendFmt("%f\n", 0.5);

	//lights
	builder.AppendFmt("%d\n", 4);

	builder.AppendFmt("%f %f %f\t", -5.0f, -5.0f, 5.0f);
	builder.AppendFmt("%f %f %f\n", 1.0f, 1.0f, 1.0f);

	builder.AppendFmt("%f %f %f\t", 5.0f, 5.0f, 5.0f);
	builder.AppendFmt("%f %f %f\n", 1.0f, 1.0f, 1.0f);

	builder.AppendFmt("%f %f %f\t", -5.0f, 5.0f, 5.0f);
	builder.AppendFmt("%f %f %f\n", 1.0f, 0.996f, 0.890f);

	builder.AppendFmt("%f %f %f\t", 5.0f, -5.0f, 5.0f);
	builder.AppendFmt("%f %f %f\n", 1.0f, 0.996f, 0.890f);

	//kernels
	builder.AppendFmt("%d\t%d", 7, 4);
	std::cout << builder.Str() << std::endl;
}

int main(int argc, const char** argv)
{
	auto check = Split("v  0.0000 1.0000 -0.0000");
	ASSERT_NO_MSG(check.size() == 4);
	appCMD_LINE cmdLine(argc, argv);
	if (cmdLine.isDefault())
	{
		PrintDefault();
		return 0;
	}
	appINPUT_SETTINGS input;
	std::cin >> input;
	input.ApplySettings();

	vec3 cameraPosition(0.0f), cameraEye(0.0f);
	engCAMERA camera(cameraPosition, cameraEye);
	auto updateCamera = [input, &cameraPosition, &cameraEye, &camera](int frame)
		{
			auto posMove = input.GetPositionMovement();
			auto eyeMove = input.GetEyeMovement();
			float t = 2 * M_PI / input.Frames() * frame;
			auto posR = posMove.r0 + posMove.Ar * std::sinf(posMove.wr * t + posMove.pr);
			auto posZ = posMove.z0 + posMove.Az * std::sinf(posMove.wz * t + posMove.pz);
			auto posPhi = posMove.phi0 + posMove.wphi * t;

			auto eyeR = eyeMove.r0 + eyeMove.Ar * std::sinf(eyeMove.wr * t + eyeMove.pr);
			auto eyeZ = eyeMove.z0 + eyeMove.Az * std::sinf(eyeMove.wz * t + eyeMove.pz);
			auto eyePhi = eyeMove.phi0 + eyeMove.wphi * t;

			cameraPosition = CylindricalToDecart({ posR, posPhi, posZ });
			cameraEye = CylindricalToDecart({ eyeR, eyePhi, eyeZ });
			std::cout << "Camera pos: (" << cameraPosition << ")\n";
			std::cout << "Camera eye: (" << cameraEye << ")" << std::endl;
			camera.SetPosition(cameraPosition);
			camera.SetView(cameraEye);
			camera.Update();
		};
	updateCamera(0);
	onFrameRendered += updateCamera;

#pragma region Load Objects
	auto getTetraSight = [](float radius) {return 4.0f / std::sqrtf(6.0f) * radius; };
	auto getOctaSight = [](float radius) {return 2.0f / std::sqrtf(2.0f) * radius; };
	auto getDodecSight = [](float radius) {return radius / 1.401258538; };

	std::vector<std::shared_ptr<engOBJECT>> objects;
	std::vector<std::function<float(float)>> functions = { getTetraSight, getOctaSight, getDodecSight };
	auto objectsSettings = input.GetObjects();
	ASSERT_MSG(functions.size() == objectsSettings.size(), "Mismath with num of functions for get sights and objects num");
	for (int i = 0; i < objectsSettings.size(); ++i)
	{
		auto objectsLoaded = LoadObject(resourcesPath, objectsSettings[i], functions[i]);
		std::for_each(all(objectsLoaded), [&objects](auto obj) {objects.emplace_back(obj); });
	}
	auto floorSettings = input.GetFloor();
	std::shared_ptr<engOBJECT> floor = std::make_shared<objQUADRANGLE>(floorSettings.points);
	floor->reflectionCoeff = floorSettings.reflectionCoeff;
	engTEXTURE floorText = engTEXTURE::ImportData(floorSettings.texturePath);
	floorText.ChangeColor(floorSettings.color);
	floor->SetTexture(&floorText);
	floor->SetName("Floor");
	objects.emplace_back(floor);
	constexpr float lightIntensity = 10.0f;

	scnSCENE scene("I finally do it", objects);
	auto lightSettings = input.GetLight();
	for (auto& light : lightSettings)
	{
		scene.AddLightSource(std::make_shared<scnLIGHT_SOURCE>(light.pos, light.color, lightIntensity));
	}
#pragma endregion
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	onStartFrameRender += [&start](int frame) {start = std::chrono::high_resolution_clock::now(); };
	std::vector<vec3> image;
	{
		dsSTRING_BUILDER builder;
		builder.AppendFmt("Rendering on %s", engGetRenderSettings().GetDevice());
		std::cout << builder.Str() << std::endl;
	}

	{
		dsSTRING_BUILDER builder;
		builder.AppendFmt("frame\ttime(ms)\trays count");
		std::cout << builder.Str() << std::endl;
	}
#pragma region Set renderer
	auto rendDevice = engGetRenderSettings().device;
	rendRENDERER* renderer;
	switch (rendDevice)
	{
	case rendSETTINGS::CPU:
	{
		renderer = new rendRENDERER_CPU(&scene);
		break;
	}
	case rendSETTINGS::GPU:
	{
		renderer = new rendRENDERER_GPU(&scene);
	}
	}
#pragma endregion
	//onFrameRendered += [&image, &input](int frame)
	//	{
	//		std::string format = input.GetOutputPathFormat();
	//		dsSTRING_BUILDER dataPathBuilder;
	//		dataPathBuilder.AppendFmt(format.c_str(), frame);
	//		DataSave(dataPathBuilder.Str(), image, input.Width(), input.Height());
	//		size_t extensionPos = format.find_last_of('.');
	//		format = format.substr(0, extensionPos) + ".ppm";
	//		dsSTRING_BUILDER ppmPathBuilder;
	//		ppmPathBuilder.AppendFmt(format.c_str(), frame);
	//		SavePPM(ppmPathBuilder, image, input.Width(), input.Height());
	//	};
	onFrameRendered += [&end, &start, &renderer](int frame)
		{
			end = std::chrono::high_resolution_clock::now();
			dsSTRING_BUILDER builder;
			builder.AppendFmt("%d\t%lld\t\t%d", frame, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), renderer->RayCount());
			std::cout << builder.Str() << std::endl;
		};
#pragma region Game loop
	for (int frame = 0; frame < input.Frames(); onFrameRendered(frame), ++frame)
	{
		onStartFrameRender(frame);
		//renderer->Render(&engGetRenderSettings(), &camera, &image);
	}
#pragma endregion
}