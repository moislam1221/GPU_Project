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

enum method_type { JACOBI, GS, SOR };

__device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix,
              const float leftX, const float centerX, const float rightX, const float topX, const float bottomX,
              const float centerRhs) {
    float result = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
    return result;
}

template <typename method_type>
__host__ __device__
float iterativeOperation(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix, float leftX, float centerX, float rightX, float topX, float bottomX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
        case JACOBI:
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 1) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 1) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}


template <typename method_type>
__host__ __device__
float iterativeOperation2(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix, float leftX, float centerX, float rightX, float topX, float bottomX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
	case JACOBI:	
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 0) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 0) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}

float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float topMatrix, float bottomMatrix, float leftX, float centerX, float rightX,  float topX, float bottomX, float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}

float Residual(const float * solution, const float * rhs, const float * matrixElements, int nxGrids, int nyGrids)
{
    int nDofs = nxGrids * nyGrids;
    float residual = 0.0;  

    float bottomMatrix = matrixElements[0];
    float leftMatrix = matrixElements[1];
    float centerMatrix = matrixElements[2];
    float rightMatrix = matrixElements[3];
    float topMatrix = matrixElements[4];
    
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        float leftX = ((iGrid % nxGrids) == 0) ? 0.0 : solution[iGrid-1];
        float centerX = solution[iGrid];
        float rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : solution[iGrid+1];
        float topX = (iGrid < nxGrids * (nyGrids - 1)) ? solution[iGrid + nxGrids] : 0.0;
        float bottomX = (iGrid >= nxGrids) ?  solution[iGrid-nxGrids] : 0.0;
        float residualContributionFromRow = normFromRow(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
    }
    residual = sqrt(residual);
    return residual;
}

__host__ __device__
void boundaryConditions(int IGrid, int nxGrids, int nyGrids, float &leftX, float &rightX, float&bottomX, float &topX)
{
    // Left
    if (IGrid % nxGrids == 0) {
        leftX = 0.0;
    }               

    // Right
    if (((IGrid+1) % nxGrids) == 0) {
        rightX = 0.0;
    }               
    
    // Bottom
    if (IGrid < nxGrids) {
        bottomX = 0.0;
    }

    // Top
    if (IGrid >= (nxGrids * nyGrids - nxGrids)) {
        topX = 0.0;
    }

    return;
}

float * iterativeCpu(const float * initX, const float * rhs,
                  const float * matrixElements, int nxGrids, int nyGrids,
		  int nIters, int method)
{
    int nDofs = nxGrids * nyGrids;
    float * x0 = new float[nDofs];
    float * x1 = new float[nDofs];
    memcpy(x0, initX, sizeof(float) * nDofs);
    memcpy(x1, initX, sizeof(float)* nDofs);
    float bottomMatrix = matrixElements[0];
    float leftMatrix = matrixElements[1];
    float centerMatrix = matrixElements[2];
    float rightMatrix = matrixElements[3];
    float topMatrix = matrixElements[4];
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
            float leftX = ((iGrid % nxGrids) == 0) ? 0.0f : x0[iGrid - 1];
            float centerX = x0[iGrid];
            float rightX = (((iGrid + 1) % nxGrids) == 0) ? 0.0f : x0[iGrid + 1];
	    float bottomX = (iGrid < nxGrids) ? 0.0f : x0[iGrid - nxGrids];
            float topX = (iGrid < nDofs - nxGrids) ? x0[iGrid + nxGrids] : 0.0f;
	    if (iIter % 2 == 0) {
                x1[iGrid] = iterativeOperation(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	    }
	    else { 
                x1[iGrid] = iterativeOperation2(leftMatrix, centerMatrix,
                                    rightMatrix, topMatrix, bottomMatrix,
				    leftX, centerX, rightX, topX, bottomX,
                                    rhs[iGrid], iGrid, method);
            }
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}

__global__
void _iterativeGpuClassicIteration(float * x1, const float * x0, const float * rhs,
                         const float * matrixElements, int nxGrids, int nyGrids, int iteration, int method)
{
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x; // Col
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int iGrid = iyGrid * (nxGrids) + ixGrid;
    int nDofs = nxGrids * nyGrids;

    float bottomMatrix = matrixElements[0];
    float leftMatrix = matrixElements[1];
    float centerMatrix = matrixElements[2];
    float rightMatrix = matrixElements[3];
    float topMatrix = matrixElements[4];

    if (iGrid < nDofs) {
        float leftX = (ixGrid == 0) ? 0.0f : x0[iGrid - 1] ;
        float centerX = x0[iGrid];
        float rightX = (ixGrid == nxGrids - 1) ?  0.0f : x0[iGrid + 1];
	float topX = (iyGrid == nyGrids - 1) ? 0.0f : x0[iGrid + nxGrids];
        float bottomX = (iyGrid == 0) ? 0.0f : x0[iGrid - nxGrids];
	if (iteration % 2 == 0) {
            x1[iGrid] = iterativeOperation(leftMatrix, centerMatrix,
                                    rightMatrix, topMatrix, bottomMatrix,
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	}
	else { 
            x1[iGrid] = iterativeOperation2(leftMatrix, centerMatrix,
                                    rightMatrix, topMatrix, bottomMatrix,
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	}
    }
    __syncthreads();
}

float * iterativeGpuClassic(const float * initX, const float * rhs,
                         const float * matrixElements,
			 int nxGrids, int nyGrids, int nIters, const int threadsPerBlock, int method)
{	
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for the solution
    float * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(float) * nDofs);
   
    // Allocate CPU memory for other variables
    float * rhsGpu, * matrixElementsGpu;
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    cudaMalloc(&matrixElementsGpu, sizeof(float) * 5);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixElementsGpu, matrixElements, sizeof(float) * 5,
            cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    int nxBlocks = (int)ceil(nxGrids / (float)threadsPerBlock);
    int nyBlocks = (int)ceil(nyGrids / (float)threadsPerBlock);

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU (used to be <<<nBlocks, threadsPerBlock>>>)
        _iterativeGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, matrixElementsGpu, 
		nxGrids, nyGrids, iIter, method); 
        float * tmp = x1Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(matrixElementsGpu);

    return solution;
}

//// SWEPT METHODS HERE ////

__device__
void __iterativeBlockUpdateToLeftRight(float * xLeftBlock, float * xRightBlock, const float *rhsBlock, 
                             const float leftMatrix, const float centerMatrix, const float rightMatrix, 
			     const float topMatrix, const float bottomMatrix, int nxGrids, int nyGrids, int iGrid, int method, 
                             int subdomainLength, bool diagonal, int maxSteps)
{
    // Initialize shared memory and pointers to x0, x1 arrays containing Jacobi solutions
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;

    // Define number of Jacobi steps to take, and current index and stride value
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    // Define rhs
    float centerRhs = rhsBlock[index];

    // Perform Jacobi iterations
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
           if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-(subdomainLength-1))) {
                // Define necessary constants
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+subdomainLength];
                float bottomX = x0[idx-subdomainLength];
                
                // Apply boundary conditions
                int step = idx / stride;
                int Idx = (stride % subdomainLength) + (stride/subdomainLength) * nxGrids;
                int IGrid  = iGrid + step * Idx;

                if (diagonal == true) {
                    int nDofs = nxGrids * nyGrids;
                    if ((blockIdx.y == gridDim.y-1) && idx/subdomainLength >= subdomainLength/2) {
                        IGrid = IGrid - nDofs;
                    }
                    if ((blockIdx.x == gridDim.x-1) && (idx % subdomainLength) >= (subdomainLength/2)) {
                        IGrid = IGrid - nxGrids;
                    }
                }

                boundaryConditions(IGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);

	        // Perform update
   	        // x1[idx] = increment(centerX);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
	    }
        }
        __syncthreads();
        float * tmp; 
        tmp = x0; x0 = x1; x1 = tmp;
    }

    index = threadIdx.x + threadIdx.y * blockDim.x;
    stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xLeftBlock[idx] = x0[subdomainLength * (idx % subdomainLength) + (idx/subdomainLength)];
        xRightBlock[idx] = x0[subdomainLength * (idx % subdomainLength) - (idx/subdomainLength) + (subdomainLength-1)];
    }
    
}

__device__
void __iterativeBlockUpdateToNorthSouth(float * xTopBlock, float * xBottomBlock, const float *rhsBlock, 
                             const float leftMatrix, const float centerMatrix, const float rightMatrix, 
			     const float topMatrix, const float bottomMatrix, int nxGrids, int nyGrids, int iGrid, int method, int subdomainLength, bool vertical, int maxSteps)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    
    // Define rhs
    float centerRhs = rhsBlock[index];

    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
            if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-subdomainLength-1)) {
                // Define necessary constants
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+subdomainLength];
                float bottomX = x0[idx-subdomainLength];
                
                // Apply boundary conditions
                int step = idx / stride;
                int Idx = (stride % subdomainLength) + (stride/subdomainLength) * nxGrids;
                int IGrid  = iGrid + step * Idx;

                if (vertical == true) {
                    int nDofs = nxGrids * nyGrids;
                    if ((blockIdx.y == gridDim.y-1) && idx/subdomainLength >= subdomainLength/2) {
                        IGrid = IGrid - nDofs;
                    }
                }
                else {
                    if ((blockIdx.x == gridDim.x-1) && (idx % subdomainLength) >= (subdomainLength/2)) {
                        IGrid = IGrid - nxGrids;
                    }
                }
                
                boundaryConditions(IGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);
                
                // Perform update
     	        //x1[idx] = increment(centerX);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
                
               }
	    }
            __syncthreads();
	    float * tmp; 
            tmp = x0; x0 = x1; x1 = tmp;
        }   
    

    // Return values for xTop and xBottom here
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xBottomBlock[idx] = x0[idx];
        xTopBlock[idx] = x0[subdomainLength * (subdomainLength-1-idx/subdomainLength) + (idx % subdomainLength)];
    }

}

__global__
void _iterativeGpuOriginal(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * matrixElementsGpu, int nxGrids, int nyGrids, int method, int subdomainLength, int maxSteps)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    
    const float bottomMatrix = matrixElementsGpu[0];
    const float leftMatrix = matrixElementsGpu[1];
    const float centerMatrix = matrixElementsGpu[2];
    const float rightMatrix = matrixElementsGpu[3];
    const float topMatrix = matrixElementsGpu[4];

    int numElementsPerBlock = subdomainLength * subdomainLength;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = (numElementsPerBlock*blockID)/2;
    float * xLeftBlock = xLeftGpu + arrayShift;
    float * xRightBlock = xRightGpu + arrayShift;
    
    extern __shared__ float sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        sharedMemory[idx] = x0Block[Idx];
        sharedMemory[idx + numElementsPerBlock] = x0Block[Idx];
    }
    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength;
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, false, maxSteps);
}

__global__
void _iterativeGpuHorizontalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                  const float * x0Gpu, const float * rhsGpu, 
                                  const float * matrixElementsGpu, int nxGrids, int nyGrids, int method, int subdomainLength, int maxSteps)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = subdomainLength/2;

    const float * rhsBlock = rhsGpu + blockShift;

    const float bottomMatrix = matrixElementsGpu[0];
    const float leftMatrix = matrixElementsGpu[1];
    const float centerMatrix = matrixElementsGpu[2];
    const float rightMatrix = matrixElementsGpu[3];
    const float topMatrix = matrixElementsGpu[4];

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    float * xLeftBlock =  xRightGpu + arrayShift;
    float * xRightBlock = (blockIdx.x != gridDim.x-1) ?
                           xLeftGpu + arrayShift + numElementsPerBlock :
			   xLeftGpu + (numElementsPerBlock * blockIdx.y * gridDim.x);
    float * xBottomBlock = xBottomGpu + arrayShift;
    float * xTopBlock = xTopGpu + arrayShift;

    extern __shared__ float sharedMemory[];
    
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
        }
    }

    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + horizontalShift;

    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, false, maxSteps);
}

__global__
void _iterativeGpuVerticalandHorizontalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                const float * x0Gpu, const float *rhsGpu, 
                                const float * matrixElementsGpu, int nxGrids, int nyGrids, int method, int subdomainLength, int maxSteps)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    
    int horizontalShift = subdomainLength/2;
    int verticalShift = subdomainLength/2 * nxGrids;

    const float * rhsBlock = rhsGpu + blockShift;
    
    const float bottomMatrix = matrixElementsGpu[0];
    const float leftMatrix = matrixElementsGpu[1];
    const float centerMatrix = matrixElementsGpu[2];
    const float rightMatrix = matrixElementsGpu[3];
    const float topMatrix = matrixElementsGpu[4];

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    float * xBottomBlock = xTopGpu + arrayShift;
    float * xTopBlock = (blockIdx.y != gridDim.y-1) ?
                         xBottomGpu + numElementsPerBlock * gridDim.x + arrayShift :
			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    
    float * xLeftBlock = xLeftGpu + arrayShift;
    float * xRightBlock = xRightGpu + arrayShift;

    extern __shared__ float sharedMemory[];
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx < numElementsPerBlock) {
            sharedMemory[idx] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
            sharedMemory[idx + subdomainLength * subdomainLength] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
        }
        else {
            sharedMemory[idx] = xTopBlock[idx - numElementsPerBlock];
            sharedMemory[idx + subdomainLength * subdomainLength] = xTopBlock[idx - numElementsPerBlock];
        }
    }
    
    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + horizontalShift + verticalShift;
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, true, maxSteps);
}


__global__
void _iterativeGpuVerticalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                const float * x0Gpu, const float *rhsGpu, 
                                const float * matrixElementsGpu, int nxGrids, int nyGrids, int method, int subdomainLength, int maxSteps)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = subdomainLength/2 * nxGrids;

    const float * rhsBlock = rhsGpu + blockShift;
    
    const float bottomMatrix = matrixElementsGpu[0];
    const float leftMatrix = matrixElementsGpu[1];
    const float centerMatrix = matrixElementsGpu[2];
    const float rightMatrix = matrixElementsGpu[3];
    const float topMatrix = matrixElementsGpu[4];

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    float * xRightBlock =  xLeftGpu + arrayShift;
    float * xLeftBlock = (blockIdx.x != 0) ?
                           xRightGpu + arrayShift - numElementsPerBlock :
    			   xRightGpu + numElementsPerBlock * ((gridDim.x-1) + blockIdx.y * gridDim.x);
    
    float * xBottomBlock = xBottomGpu + arrayShift;
    float * xTopBlock = xTopGpu + arrayShift;

    extern __shared__ float sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
        }
    }

    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + verticalShift;

    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, true, maxSteps);
}

__global__
void _finalSolution(float * xTopGpu, float * xBottomGpu, float * x0Gpu, int nxGrids, int subdomainLength)
{
    extern __shared__ float sharedMemory[];
    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;

    float * xTopBlock = xBottomGpu + arrayShift;
    float * xBottomBlock = (blockIdx.y != 0) ?
			    xTopGpu + (blockIdx.x + (blockIdx.y-1) * gridDim.x) * numElementsPerBlock :
			    xTopGpu + (gridDim.x * (gridDim.y-1) + blockIdx.x) * numElementsPerBlock;

    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    float * x0Block = x0Gpu + blockShift;

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[(subdomainLength/2 - 1 - idx/subdomainLength) * subdomainLength + idx % subdomainLength] = xBottomBlock[idx];
    }

    __syncthreads();
    
    for (int idx = index; idx < 2*numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        x0Block[Idx] = sharedMemory[idx];
    }
}

///////////////////////////////////////////////////

float * iterativeGpuSwept(const float * initX, const float * rhs,
        const float * matrixElements,
	int nxGrids, int nyGrids, int nIters, int maxSteps, const int threadsPerBlock, const int method, const int subdomainLength)
{     
    // Determine number of threads and blocks 
    const int nxBlocks = (int)ceil(nxGrids / (float)threadsPerBlock);
    const int nyBlocks = (int)ceil(nyGrids / (float)threadsPerBlock);
    const int nDofs = nxGrids * nyGrids;

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    
    // Allocate memory for solution and inputs
    float *xLeftGpu, *xRightGpu, *xTopGpu, *xBottomGpu;
    int numSharedElemPerBlock = subdomainLength * subdomainLength / 2;
    cudaMalloc(&xLeftGpu, sizeof(float) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xRightGpu, sizeof(float) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xTopGpu, sizeof(float) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xBottomGpu, sizeof(float) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    float * x0Gpu, * rhsGpu, * matrixElementsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    cudaMalloc(&matrixElementsGpu, sizeof(float) * 5);

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixElementsGpu, matrixElements, sizeof(float) * 5,
            cudaMemcpyHostToDevice);

    int sharedBytes = 2 * subdomainLength * subdomainLength * sizeof(float);

    for (int i = 0; i < nIters; i++) {

        // APPLY METHOD TO ADVANCE POINTS (NO SHIFT)
        _iterativeGpuOriginal <<<grid, block, sharedBytes>>>(xLeftGpu, xRightGpu, x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, method, subdomainLength, maxSteps);

        // APPLY HORIZONTAL SHIFT
        _iterativeGpuHorizontalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, method, subdomainLength, maxSteps);

        // APPLY VERTICAL SHIFT (ALONG WITH PREVIOUS HORIZONTAL SHIFT)
        _iterativeGpuVerticalandHorizontalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, method, subdomainLength, maxSteps);

        // APPLY VERTICAL SHIFT
        _iterativeGpuVerticalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, method, subdomainLength, maxSteps);

        // APPLY FINAL STEP
        _finalSolution <<<grid, block, sharedBytes>>>(xTopGpu, xBottomGpu, x0Gpu, nxGrids, subdomainLength);

    }

    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);
    cudaFree(matrixElementsGpu);

    return solution;
}

int main(int argc, char *argv[])
{
    // Ask user for inputs
    const int nxGrids = atoi(argv[1]); 
    const int nyGrids = atoi(argv[1]); 
    const int subdomainLength = atoi(argv[2]);
    const int threadsPerBlock = atoi(argv[3]); 
    const int nCycles = atoi(argv[4]);
    const int maxSteps = atoi(argv[5]);
    const int nIters = atoi(argv[6]);

    method_type method = JACOBI;

    int nDofs = nxGrids * nyGrids;
    
    // Declare arrays and population with values for Poisson equation
    float * initX = new float[nDofs];
    float * rhs = new float[nDofs];
    
    float dx = 1.0f / (nxGrids + 1);
    float dy = 1.0f / (nyGrids + 1);

    for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
        initX[iGrid] = (float)iGrid; 
        rhs[iGrid] = 1.0f;
    }

    float * matrixElements = new float[5];
    matrixElements[0] = -1.0f / (dy * dy);
    matrixElements[1] = -1.0f / (dx * dx);
    matrixElements[2] = 2.0f / (dx * dx) + 2.0f / (dy * dy);
    matrixElements[3] = -1.0f / (dx * dx);
    matrixElements[4] = -1.0f / (dy * dy);

    // Amount of shared memory to be requested
    int sharedMem = 2 * subdomainLength * subdomainLength * sizeof(float);
    
    // Run the CPU Implementation and measure the time required
    clock_t cpuStartTime = clock();
    float * solutionCpu = iterativeCpu(initX, rhs, matrixElements, nxGrids, nyGrids, nIters, method);
    clock_t cpuEndTime = clock();
    float cpuTime = (cpuEndTime - cpuStartTime) / (float) CLOCKS_PER_SEC;

    // Run the Classic GPU Implementation and measure the time required
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    float * solutionGpuClassic = iterativeGpuClassic(initX, rhs, matrixElements, nxGrids, nyGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    // Run the Swept GPU Implementation and measure the time required
    cudaEvent_t startSwept, stopSwept;
    float timeSwept;
    cudaEventCreate( &startSwept );
    cudaEventCreate( &stopSwept );
    cudaEventRecord( startSwept, 0);
    float * solutionGpuSwept = iterativeGpuSwept(initX, rhs, matrixElements, nxGrids, nyGrids, nCycles, maxSteps, threadsPerBlock, method, subdomainLength);
    cudaEventRecord(stopSwept, 0);
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);
    
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of total grid points (size of the linear system): %d\n", nDofs);
    printf("Number of grid points in x-direction: %d\n", nxGrids);
    printf("Number of grid points in y-direction: %d\n", nyGrids);
    printf("Size of each subdomain handled by a block: %d\n", subdomainLength);
    printf("Threads Per Block in each direction: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIters);
    printf("Number of Swept Cycles performed: %d\n", nCycles);
    printf("Number of Iterations performed internally at each Swept Stage: %d\n", maxSteps);
    printf("Amount of shared memory to be requested: %d B\n", sharedMem);
    
    // Print out results to the screen, notify if any GPU Classic or Swept values differ significantly
    for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
        printf("%d %f %f %f \n",iGrid, solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
	//assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	// if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-2) {
	//    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	// }
    }


    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = timeClassic / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    //float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    /* printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); */
    printf("===============TIMING INFORMATION============================\n");
    printf("Total Time needed for the CPU: %f ms\n", cpuTime * 1e3);
    printf("Total Time needed for the Classic GPU is %f ms\n", timeClassic);
    printf("Total Time needed for the Swept GPU: %f ms\n", timeSwept);
    printf("===========================================\n");
    printf("Time per iteration needed for the CPU: %f ms\n", cpuTimePerIteration);
    printf("Time per iteration needed for the Classic GPU is %f ms\n", classicTimePerIteration);
    printf("Time per iteration needed for the Swept GPU: %f ms\n", sweptTimePerIteration);
    printf("Swept takes %f the time Classic takes\n", timeSwept / timeClassic);

    // Compute the residual of the resulting solution (||b-Ax||)
    float residualCPU = Residual(solutionGpuClassic, rhs, matrixElements, nxGrids, nyGrids);
    float residualClassicGPU = Residual(solutionGpuClassic, rhs, matrixElements, nxGrids, nyGrids);
    float residualSweptGPU = Residual(solutionGpuSwept, rhs, matrixElements, nxGrids, nyGrids);
    printf("===============RESIDUAL INFORMATION============================\n");
    printf("Residual of the CPU solution is %f\n", residualCPU);
    printf("Residual of the Classic GPU solution is %f\n", residualClassicGPU);
    printf("Residual of the Swept GPU solution is %f\n", residualSweptGPU); 
    printf("The residual of Swept is %f times that of Classic\n", residualSweptGPU / residualClassicGPU); 
  
    // Save residual to a file
    /* std::ofstream residuals;
    residuals.open("residual-gs.txt",std::ios_base::app);
    residuals << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << residualSwept << "\n";
    residuals.close(); */

    // Save Results to a file "N tpb Iterations CPUTime/perstep ClassicTime/perstep SweptTime/perStep ClassicTime/SweptTime"
    std::ofstream timings;
    timings.open("time.txt",std::ios_base::app);
//    timings << nxGrids << "\t" << nyGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << cpuTimePerIteration << "\t" << classicTimePerIteration << "\t" << sweptTimePerIteration << "\t" << timeMultiplier << "\n";
    timings.close();

    // Free memory
    cudaEventDestroy(startClassic);
    cudaEventDestroy(startSwept);
    delete[] initX;
    delete[] rhs;
    delete[] matrixElements;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
    delete[] solutionGpuSwept;
}
