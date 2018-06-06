#include<stdio.h>
#include<assert.h>
#include"../CudaCheckError.h"

__device__
float gridOp(float leftMat, float centerMat, float rightMat,
             float leftX, float centerX, float rightX,
             float rhs)
{
    return centerX + 1;
}

#include"../UpperTriangular.h"

int main()
{
    const int triangleBaseLength = 128;
    float x0Cpu[triangleBaseLength];
    for (int i = 0; i < triangleBaseLength; ++i) {
        x0Cpu[i] = 0;
    }

    float * xLeftGpu, * xRightGpu, * x0Gpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * triangleBaseLength);
    cudaMalloc(&xRightGpu, sizeof(float) * triangleBaseLength);
    x0Gpu = xLeftGpu;
    cudaMemcpy(x0Gpu, x0Cpu, sizeof(float) * triangleBaseLength,
               cudaMemcpyHostToDevice);

    float * notUsed = x0Gpu;
    int nBytesShared = sizeof(float) * 2 * triangleBaseLength;
    _jacobiGpuUpperTriangle<<<1, triangleBaseLength, nBytesShared>>>(
            xLeftGpu, xRightGpu, x0Gpu, notUsed, notUsed, notUsed, notUsed);
    cudaDeviceSynchronize();
    cudaCheckError();

    float xLeftCpu[triangleBaseLength], xRightCpu[triangleBaseLength];
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * triangleBaseLength,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * triangleBaseLength,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < triangleBaseLength; ++i) {
        assert(xLeftCpu[i] == i / 2);
        assert(xRightCpu[i] == i / 2);
    }
}
