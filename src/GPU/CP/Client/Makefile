CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = --std=c++11 -g -O3 -lm --Werror cross-execution-space-call -I../ -L../build/libs/
LD_FLAGS = -static
TARGET = CP
TARGET_OUT = ../CP
SOURCES=$(wildcard *.cu) $(wildcard *.cpp)
OBJECTS=$(SOURCES:.cu=.o) $(SOURCES:.cpp=.o)
LIBS = -lrender -lcommon -lengine

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $(LIBS) -o $(TARGET_OUT) build/$(OBJECTS)

%.o: %.cu
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o build/$@
	@echo "Finish compiling $<"

$.o $.cpp
	@echo "Start compiling $<"
	$(CC) $(CC_FLAGS) -c $< -o build/$@
	@echo "Finish compiling $<"
