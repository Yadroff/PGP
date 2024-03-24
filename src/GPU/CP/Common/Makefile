CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = --std=c++11 -g -O3 -lm --Werror cross-execution-space-call -I../
LD_FLAGS = -static

SOURCES=$(wildcard *.cu) $(wildcard *.cpp)
OBJECTS=$(SOURCES:.cu=.o) $(SOURCES:.cpp=.o)

all: librender.a

librender.a: $(OBJECTS)
	ar rcs $@ $^

%.o: %.cu
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o $@
	@echo "Finish compiling $<"

$.o $.cpp
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o $@
	@echo "Finish compiling $<"