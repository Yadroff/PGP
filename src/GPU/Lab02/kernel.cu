#include <texture_indirect_functions.h>

#include "os.cuh"
#include "common_structures.cuh"
#include "common_defines.cuh"


static constexpr char hostWx[3][3] = {  {-1, 0, 1},
                                        {-2, 0, 2},
                                        {-1, 0, 1} };
static constexpr char hostWy[3][3] = {  {-1, -2, -1},
                                        { 0,  0,  0},
                                        { 1,  2,  1} };

__constant__ char Wx[3][3], Wy[3][3];

__global__ void kernel(cudaTextureObject_t image, uchar4* result, int width, int height) {
	int xID = blockDim.x * blockIdx.x + threadIdx.x;
	int yID = blockDim.y * blockIdx.y + threadIdx.y;
	int xOffset = blockDim.x * gridDim.x;
	int yOffset = blockDim.y * gridDim.y;

	double gradX = 0, gradY = 0, grad, lum;
	uchar4 pixel;
	for (int y = yID; y < height; y += yOffset)
		for (int x = xID; x < width; x += xOffset, gradX = 0, gradY = 0) {
			for (int dx = -1; dx <= 1; ++dx) {
				for (int dy = -1; dy <= 1; ++dy) {
					pixel = tex2D<uchar4>(image, x + dx, y + dy);
					lum = 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
					gradX += Wx[dx + 1][dy + 1] * lum;
					gradY += Wy[dx + 1][dy + 1] * lum;
				}
			}
			grad = min(255., sqrt(gradX * gradX + gradY * gradY));
			result[y * width + x] = make_uchar4(grad, grad, grad, pixel.w);
		}
}

int main()
{
    INIT_IO();
    int width, height;
    std::vector<uchar4> img;
    std::string inputFilename, outputFilename;
    std::cin >> inputFilename >> outputFilename;
    osFileRead(inputFilename, img, width, height);
    Cuda2DArray<uchar4> devImg(width, height);
    devImg.MoveToDevice(img.data());

    CALL_CUDA_FUNC(cudaMemcpyToSymbol, Wx, hostWx, 3 * 3);
    CALL_CUDA_FUNC(cudaMemcpyToSymbol, Wy, hostWy, 3 * 3);

    CudaResDesc resDesc(cudaResourceTypeArray, devImg.Data());
    CudaTextureDesc textDesc;
	CudaTextureObj texture(textDesc, resDesc);
    CudaArray<uchar4> res(width * height);
	CALL_KERNEL(texture.GetObj(), res.Data(), width, height);

	std::vector<uchar4> result(width * height);
	res.MoveToHost(result.data());
	osFileWrite(outputFilename, result, width, height);
    return 0;
}
