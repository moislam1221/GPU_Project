#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <utility>

__device__ 
void __iterativeBlockUpperTriangleFromShared(
		float * xLeftBlock, float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    for (int k = 1; k < blockDim.x/2; ++k) {
        if (threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) {
            float leftX = x0[threadIdx.x - 1];
            float centerX = x0[threadIdx.x];
            float rightX = x0[threadIdx.x + 1];
	    if (iGrid == 0) {
		leftX = 0.0f;
	    }
	    if (iGrid == nGrids-1) {
		rightX = 0.0f;
	    }
            float leftMat = leftMatrixBlock[threadIdx.x];
            float centerMat = centerMatrixBlock[threadIdx.x];
            float rightMat = rightMatrixBlock[threadIdx.x];
            float rhs = rhsBlock[threadIdx.x];
	    if (k % 2 == 1) {
	        x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
	    else {
	        x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
        }
        __syncthreads();	
	float * tmp = x1; x1 = x0; x0 = tmp;
    } 
    
    float * tmp = x1; x1 = x0; x0 = tmp;

    int remainder = threadIdx.x % 4;
    xLeftBlock[threadIdx.x] = x0[(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];
    xRightBlock[threadIdx.x] = x0[blockDim.x-1-(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];

}

__global__
void _iterativeGpuUpperTriangle(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * leftMatrixGpu, const float *centerMatrixGpu,
                             const float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    sharedMemory[threadIdx.x + blockDim.x] = x0Block[threadIdx.x];

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
    		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}

__device__ 
void __iterativeBlockLowerTriangleFromShared(
		const float * xLeftBlock, const float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    int remainder = threadIdx.x % 4;

    if (threadIdx.x != blockDim.x-1) {
        x0[blockDim.x-1-((blockDim.x+threadIdx.x+1)/2) + blockDim.x*(remainder>1)] = xLeftBlock[threadIdx.x];
	x0[(blockDim.x+threadIdx.x+1)/2 + blockDim.x*(remainder>1)] = xRightBlock[threadIdx.x];
    } 

    # pragma unroll
    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) {
                float leftX = x0[threadIdx.x - 1];
                float centerX = x0[threadIdx.x];
                float rightX = x0[threadIdx.x + 1];
		if (iGrid == 0) {
		    leftX = 0.0f;
		}
		if (iGrid == nGrids-1) {
		    rightX = 0.0f;
		}
		float leftMat = leftMatrixBlock[threadIdx.x];
		float centerMat = centerMatrixBlock[threadIdx.x];
 		float rightMat = rightMatrixBlock[threadIdx.x];
		float rhs = rhsBlock[threadIdx.x];
	        if (k % 2 == 1) {	
	            x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
		else {
		    x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
	    }
 	    float * tmp = x1; x1 = x0; x0 = tmp;
        }
	__syncthreads();
    }

    float leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x0[threadIdx.x - 1];
    float centerX = x0[threadIdx.x];
    float rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x0[threadIdx.x + 1];
    if (iGrid == 0) {
       leftX = 0.0;    
    }
    if (iGrid == nGrids-1) {
        rightX = 0.0;
    }
    // The last step! - Should i just perform one of the grid operations
    // The last step of the for loop above uses k = 1 where gridOperation is used, so I'll use gridOperation2 here
    x1[threadIdx.x] = iterativeOperation2(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid, method);
    float * tmp = x1; x1 = x0; x0 = tmp; 

}

__global__
void _iterativeGpuLowerTriangle(float * x0Gpu, float *xLeftGpu,
                             float * xRightGpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    float * x0Block = x0Gpu + blockShift;
    float * rhsBlock = rhsGpu + blockShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _iterativeGpuShiftedDiamond(float * xLeftGpu, float * xRightGpu,
                              float * rhsGpu, 
			      float * leftMatrixGpu, float * centerMatrixGpu,
                              float * rightMatrixGpu, int nGrids, int method)
{

    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xRightGpu + blockShift;
    float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;
    
    extern __shared__ float sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);  

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

}

__global__
void _iterativeGpuDiamond(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;

    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];

    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
    
    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}
float * iterativeGpuSwept(const float * initX, const float * rhs,
        const float * leftMatrix, const float * centerMatrix,
        const float * rightMatrix, int nGrids, int nIters,
        const int threadsPerBlock, const int method) { 
    
    // Determine number of threads and blocks 
    const int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);

    // Allocate memory for solution and inputs
    float *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(float) * threadsPerBlock * nBlocks);
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rhsGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&leftMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&centerMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rightMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    cudaMemcpy(x0Gpu + nGrids, initX, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu + nGrids, leftMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu + nGrids, centerMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu + nGrids, rightMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;

    float residualSwept;
    float nCycles = nIters / threadsPerBlock;
    float * currentSolution = new float[nGrids];
    std::ofstream residuals;
    residuals.open("dummy.txt",std::ios_base::app);
    
    for (int i = 0; i < nCycles; i++) {
        _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (x0Gpu, xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
        cudaMemcpy(currentSolution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);
        residualSwept = Residual(currentSolution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        residuals << nGrids << "\t" << threadsPerBlock << "\t" << i*threadsPerBlock << "\t" << residualSwept << "\n";
    }
   
    residuals.close();
/*
    _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(float) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, method);
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method);

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _iterativeGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method); 
    }

    _iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
*/
    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}
