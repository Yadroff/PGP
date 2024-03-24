#pragma once

#include <vector>
#include "common.cuh"

#ifdef OS_WIN
#include <windows.h>
#include <sysinfoapi.h>
#include <intrin.h>
#endif

/// returns CPU name (e.g. "Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz")
std::string osGetCPUName();
/// returns RAM size in KB
unsigned long long osGetRAMSize();
/// returns a vector in each element of which information about the GPU is stored
std::vector<std::string> osGetDevicesInfo();
/// Prints all system information to `os`
void osDisplayHardwareInfo(std::ostream& os);
void osDebugBreak();

void osFileRead(const std::string& filename, std::vector<uchar4>& arg, int& weight, int& height);
void osFileWrite(const std::string& filename, const std::vector<uchar4>& arg, int w, int h);