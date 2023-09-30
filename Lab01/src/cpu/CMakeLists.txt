cmake_minimum_required(VERSION 3.23)
project(cpu)

set(CMAKE_CXX_STANDARD 11)

set(HEADERS
)
set(SOURCES

)
add_executable(cpu_realization ${HEADERS} ${SOURCES} cpu_realization.cpp)
add_executable(benchmarked_cpu_realization ${HEADERS} ${SOURCES} cpu_realization.cpp)
target_compile_definitions(benchmarked_cpu_realization PRIVATE BENCHMARK)