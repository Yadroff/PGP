CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = --std=c++11 -g -O3 -lm --Werror cross-execution-space-call
BIN = lab03

all: $(BIN)

$(BIN): kernel.cu os.o claster.o common_structures.cuh common_defines.cuh
	$(CC) $(CC_FLAGS) $< os.o claster.o -o $@
claster.o: claster.cu claster.cuh common_structures.cuh common_defines.cuh
	$(CC) $(CC_FLAGS) -c $< -o $@
os.o: os.cu os.cuh common_defines.cuh common_structures.cuh
	$(CC) $(CC_FLAGS) -c $< -o $@
