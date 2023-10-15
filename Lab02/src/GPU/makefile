CC = nvcc
CC_FLAGS = -g -O3 -lm --Werror cross-execution-space-call
BIN = lab02

all: $(BIN)

$(BIN): kernel.cu os.o common_structures.cuh common_defines.cuh
	$(CC) $(CC_FLAGS) $< os.o -o $@
os.o: os.cu os.cuh common_defines.cuh common_structures.cuh
	$(CC) $(CC_FLAGS) -c $< -o $@
