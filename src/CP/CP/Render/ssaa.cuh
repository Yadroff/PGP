#pragma once

#include "Common/common.cuh"
#include "Common/common_structures.cuh"
#include "Common/dev_res_collector.cuh"
#include "Math/vec.cuh"

__host__ __device__ vec3 rendAvgKernel(vec3* img, int i, int j, int width, int height, int kernelWidth, int kernelHeight);

__global__ void rendCoreSSAA_GPU(vec3* ref, vec3* res, int width, int height, int newWidth, int newHeight);

void rendCoreSSAA_CPU(vec3* ref, vec3* res, int width, int height, int newWidth, int newHeight, int idX, int idY, int offsetX, int offsetY);

void rendSSAA_GPU(vec3* ref, vec3* res, int w, int h, int newWidth, int newHeight);

void rendSSAA_CPU(vec3* ref, vec3* res, int w, int h, int newWidth, int newHeight);