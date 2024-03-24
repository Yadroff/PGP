#pragma once

#include <unordered_map>
#include "../Common/common.cuh"

struct appCMD_LINE_SETTINGS
{
	enum DEVICE
	{
		CPU,
		GPU,
		DEFAULT = GPU
	};

	bool ssaa = false;
	DEVICE device = DEFAULT;
	bool isDefault = false;
};

class appCMD_LINE
{
public:
	appCMD_LINE(int argc, const char** argv);
	bool isDefault() const { return settings.isDefault; }
private:
	static void SetRenderOptions(const appCMD_LINE_SETTINGS& settings);
	appCMD_LINE_SETTINGS Parse();

private:
	std::unordered_map<std::string, appCMD_LINE_SETTINGS::DEVICE> devices = {
		{"--gpu", appCMD_LINE_SETTINGS::GPU},
		{"--cpu", appCMD_LINE_SETTINGS::CPU},
		{"--default", appCMD_LINE_SETTINGS::DEFAULT}
	};
	std::vector<std::string> launchOptions;
	appCMD_LINE_SETTINGS settings;
};
