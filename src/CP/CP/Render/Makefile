CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = --std=c++11 -g -O3 -lm --Werror cross-execution-space-call -I../
LD_FLAGS = -static
TARGET = $PWD/../build/lib/librender.a
SOURCES=$(wildcard *.cu) $(wildcard *.cpp)
OBJECTS=$(SOURCES:.cu=.o) $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Compiling library $<"
	ar rcs $@ build/$^

%.o: %.cu
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o build/$@
	@echo "Finish compiling $<"

$.o $.cpp
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o build/$@
	@echo "Finish compiling $<"