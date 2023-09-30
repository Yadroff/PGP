#pragma once

#include<vector>
#include <iostream>

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
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