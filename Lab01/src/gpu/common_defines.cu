#include "common_defines.cuh"

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