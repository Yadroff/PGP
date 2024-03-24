#include "ap_cmd_line.h"
#include "../Engine/camera.h"

appCMD_LINE_SETTINGS appCMD_LINE::Parse()
{
	appCMD_LINE_SETTINGS settings;
	if (launchOptions.size() != 1) {
		for (int i = 1; i < launchOptions.size(); ++i)
		{
			std::string option = ToLower(launchOptions[i]);
			auto it = devices.find(option);
			if (it != devices.end())
			{
				settings.device = it->second;
			}
			else if (option == "--ssaa") {
				settings.ssaa = true;
			}
			else if (option == "--default")
			{
				settings.isDefault = true;
			}
		}
	}
	SetRenderOptions(settings);
	return settings;
}

appCMD_LINE::appCMD_LINE(int argc, const char** argv)
{
	for (int i = 0; i < argc; ++i)
	{
		launchOptions.emplace_back(argv[i]);
	}
	settings = Parse();
}

void appCMD_LINE::SetRenderOptions(const appCMD_LINE_SETTINGS& settings)
{
	auto& renderSettings = engGetRenderSettings();
	renderSettings.useSSAA = settings.ssaa;
	rendSETTINGS::Device device;
	switch (settings.device)
	{
	case appCMD_LINE_SETTINGS::CPU:
	{
		device = rendSETTINGS::CPU;
		break;
	}
	case appCMD_LINE_SETTINGS::GPU:
	{
		device = rendSETTINGS::GPU;
		break;
	}
	}
	renderSettings.device = device;
}