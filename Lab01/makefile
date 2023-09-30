CC = nvcc
CC_FLAGS = -g -O3 -lm --Werror cross-execution-space-call
BIN = lab01

all: $(BIN)

$(BIN): kernel.cu os.o common_structures.hpp common_defines.cuh
	$(CC) $(CC_FLAGS) kernel.cu os.o -o $@
os.o: os.cu os.cuh common_defines.cuh common_structures.hpp
	$(CC) $(CC_FLAGS) -c $< -o $@
