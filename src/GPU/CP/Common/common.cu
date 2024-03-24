#include "common.cuh"

#include <algorithm>

#ifdef BENCHMARK
int BLOCKS_NUM = 256;
int THREADS_IN_BLOCK = 256;

void checkCommandLine(int argc, const char** argv) {
	if (argc == 3) {
		BLOCKS_NUM = atoi(argv[1]);
		THREADS_IN_BLOCK = atoi(argv[2]);
	}
}
#endif

std::vector<std::string> Split(const std::string& str, char del) {
	std::vector<std::string> result;
	std::string::size_type last = 0;
	while (last < str.size()) {
		std::string::size_type next = str.find(del, last);
		if (next == std::string::npos) {
			result.push_back(str.substr(last));
			break;
		}
		std::string sub = str.substr(last, next - last);
		if (!sub.empty()) {
			result.push_back(sub);
		}
		last = next + 1;
	}
	return result;
}

std::string ToLower(std::string str)
{
	std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {return std::tolower(c); });
	return str;
}