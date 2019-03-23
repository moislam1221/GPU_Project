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

template <typename method_type>
__host__ __device__
double iterativeOperation(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix, double leftX, double centerX, double rightX, double topX, double bottomX, const double centerRhs, int gridPoint, method_type method)
{
    double gridValue = centerX;
    switch(method)
    {
        case JACOBI:
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 1) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	    }
	case SOR:
	    double relaxation = 1.9939;
	    if (gridPoint % 2 == 1) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}


template <typename method_type>
__host__ __device__
double iterativeOperation2(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix, double leftX, double centerX, double rightX, double topX, double bottomX, const double centerRhs, int gridPoint, method_type method)
{
    double gridValue = centerX;
    switch(method)
    {
	case JACOBI:	
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 0) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
	    }
	case SOR:
	    double relaxation = 1.9939;
	    if (gridPoint % 2 == 0) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}

double normFromRow(double leftMatrix, double centerMatrix, double rightMatrix, double topMatrix, double bottomMatrix, double leftX, double centerX, double rightX,  double topX, double bottomX, double centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}

double Residual(const double * solution, const double * rhs, const double * leftMatrix, const double * centerMatrix, const double * rightMatrix, const double * topMatrix, const double * bottomMatrix, int nxGrids, int nyGrids)
{
    int nDofs = nxGrids * nyGrids;
    double residual = 0.0;  

    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : solution[iGrid-1];
        double centerX = solution[iGrid];
        double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : solution[iGrid+1];
        double topX = (iGrid < nxGrids * (nyGrids - 1)) ? solution[iGrid + nxGrids] : 0.0;
        double bottomX = (iGrid >= nxGrids) ?  solution[iGrid-nxGrids] : 0.0;
        double residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid], leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
//	printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

double * iterativeCpu(const double * initX, const double * rhs,
                  const double * leftMatrix, const double * centerMatrix,
                  const double * rightMatrix, const double * topMatrix, 
		  const double * bottomMatrix, int nxGrids, int nyGrids,
		  int nIters, int method)
{
    int nDofs = nxGrids * nyGrids;
    double * x0 = new double[nDofs];
    double * x1 = new double[nDofs];
    memcpy(x0, initX, sizeof(double) * nDofs);
    memcpy(x1, initX, sizeof(double)* nDofs);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
            double leftX = ((iGrid % nxGrids) == 0) ? 0.0f : x0[iGrid - 1];
            double centerX = x0[iGrid];
            double rightX = (((iGrid + 1) % nxGrids) == 0) ? 0.0f : x0[iGrid + 1];
	    double bottomX = (iGrid < nxGrids) ? 0.0f : x0[iGrid - nxGrids];
            double topX = (iGrid < nDofs - nxGrids) ? x0[iGrid + nxGrids] : 0.0f;
	    if (iIter % 2 == 0) {
                x1[iGrid] = iterativeOperation(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid],
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	    }
	    else { 
                x1[iGrid] = iterativeOperation2(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid],
				    leftX, centerX, rightX, topX, bottomX,
                                    rhs[iGrid], iGrid, method);
            }
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}

__global__
void _iterativeGpuClassicIteration(double * x1, const double * x0, const double * rhs,
                         const double * leftMatrix, const double * centerMatrix,
                         const double * rightMatrix, const double * topMatrix, const double * bottomMatrix,
			 int nxGrids, int nyGrids, int iteration, int method)
{
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x; // Col
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int iGrid = iyGrid * (nxGrids) + ixGrid;
    int nDofs = nxGrids * nyGrids;
    if (iGrid < nDofs) {
        double leftX = (ixGrid == 0) ? 0.0f : x0[iGrid - 1] ;
        double centerX = x0[iGrid];
        double rightX = (ixGrid == nxGrids - 1) ?  0.0f : x0[iGrid + 1];
	double topX = (iyGrid == nyGrids - 1) ? 0.0f : x0[iGrid + nxGrids];
        double bottomX = (iyGrid == 0) ? 0.0f : x0[iGrid - nxGrids];
	if (iteration % 2 == 0) {
            x1[iGrid] = iterativeOperation(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid],
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	}
	else { 
            x1[iGrid] = iterativeOperation2(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid],
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	}
    }
    __syncthreads();
}

double * iterativeGpuClassic(const double * initX, const double * rhs,
                         const double * leftMatrix, const double * centerMatrix,
                         const double * rightMatrix, const double * topMatrix, const double * bottomMatrix,
			 int nxGrids, int nyGrids, int nIters, const int threadsPerBlock, int method)
{
  	
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for the solution
    double * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
   
    // Allocate CPU memory for other variables
    double * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu, * topMatrixGpu, * bottomMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&leftMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&centerMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&rightMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&topMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&bottomMatrixGpu, sizeof(double) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(topMatrixGpu, topMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(bottomMatrixGpu, bottomMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock);
    int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock);

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU (used to be <<<nBlocks, threadsPerBlock>>>)
        _iterativeGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, topMatrixGpu, bottomMatrixGpu,  
		nxGrids, nyGrids, iIter, method); 
        double * tmp = x1Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    double * solution = new double[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}

//// SWEPT METHODS HERE ////
__device__
void __iterativeBlockUpdateToLeftRight(double * xLeftBlock, double * xRightBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = blockDim.x * blockDim.y;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 100;
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if ((threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) && (threadIdx.y >= 1 && threadIdx.y <= blockDim.y-2)) {
        for (int k = 0; k < maxSteps; k++) {
            // Define necessary constants
            double centerRhs = rhsBlock[idx];
            double leftMatrix = leftMatrixBlock[idx];
            double centerMatrix = centerMatrixBlock[idx];
            double rightMatrix = rightMatrixBlock[idx];
            double topMatrix = topMatrixBlock[idx];
            double bottomMatrix = bottomMatrixBlock[idx];
            double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
            double centerX = x0[idx];
            double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
            double topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
            double bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0;
           
            //printf("In iGrid %d, idx = %d, left %f, right %f, center %f, top %f, bottom %f\n", iGrid, idx, leftX, rightX, centerX, topX, bottomX	);
	    // Perform update
   	    //x1[idx] = increment(centerX);
            x1[idx] = iterativeOperation(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                             leftX, centerX, rightX, topX, bottomX, centerRhs, iGrid, method);
            // Synchronize
	    __syncthreads();
            printf("Updated value in idx = %d is %f\n", idx, x1[idx]);
	    double * tmp; tmp = x0; x0 = x1;
	}
    }
    
    // Save xLeft, xRight, xTop, xBottom
    if (idx < (blockDim.x * blockDim.y)/2) {
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    }
}

__device__
void __iterativeBlockUpdateToNorthSouth(double * xTopBlock, double * xBottomBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = blockDim.x * blockDim.y;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 100;
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if ((threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) && (threadIdx.y >= 1 && threadIdx.y <= blockDim.y-2)) {
        for (int k = 0; k < maxSteps; k++) {
            // Define necessary constants
            double centerRhs = rhsBlock[idx];
            double leftMatrix = leftMatrixBlock[idx];
            double centerMatrix = centerMatrixBlock[idx];
            double rightMatrix = rightMatrixBlock[idx];
            double topMatrix = topMatrixBlock[idx];
            double bottomMatrix = bottomMatrixBlock[idx];
            double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
            double centerX = x0[idx];
            double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
            double topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
            double bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0;
            // Perform update
	    //x1[idx] = increment(centerX);
            x1[idx] = iterativeOperation(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                             leftX, centerX, rightX, topX, bottomX, centerRhs, iGrid, method); 
            // Synchronize
	    __syncthreads();
            printf("In blockIdx %d, blockIdy %d, iGrid %d, Updated value in idx = %d is %f\n", blockIdx.x, blockIdx.y, iGrid, idx, x1[idx]);
	    double * tmp; tmp = x0; x0 = x1;
	}
    }

    // Return values for xTop and xBottom here
    if (idx < (blockDim.x * blockDim.y)/2) {
        xBottomBlock[idx] = x0[idx];
	xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
    }
}

__global__
void _iterativeGpuOriginal(double * xLeftGpu, double *xRightGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{

    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift;

    int numElementsPerBlock = blockDim.x * blockDim.y;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = (numElementsPerBlock*blockID)/2;
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;
    
    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;
    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x] = x0Block[threadIdx.x + threadIdx.y * nxGrids];

    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y] = x0Block[threadIdx.x + threadIdx.y * nxGrids];
   
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _iterativeGpuHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                  const double *rhsGpu, const double * leftMatrixGpu, const double *centerMatrixGpu, 
                                  const double * rightMatrixGpu, const double * topMatrixGpu, const double * bottomMatrixGpu, 
                                  int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = blockDim.x/2;

    const double * rhsBlock = rhsGpu + blockShift; //+ horizontalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ horizontalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ horizontalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ horizontalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ horizontalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ horizontalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xLeftBlock =  xRightGpu + arrayShift;
    double * xRightBlock = (blockIdx.x != gridDim.x-1) ?
                           xLeftGpu + arrayShift + numElementsPerBlock :
			   xLeftGpu + (numElementsPerBlock * blockIdx.y * gridDim.x);
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx + horizontalShift;
   
    if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
        iGrid = iGrid - nxGrids;
    }
    
    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (threadIdx.x < blockDim.x/2) {
        sharedMemory[idx] = xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y];
    }
    else {
        sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
    }
    
    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _iterativeGpuVerticalandHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double *rhsGpu, const double * leftMatrixGpu, const double *centerMatrixGpu, 
                                const double * rightMatrixGpu, const double * topMatrixGpu, const double * bottomMatrixGpu, 
                                int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = blockDim.x/2;
    int verticalShift = blockDim.y/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    double * xBottomBlock = xTopGpu + arrayShift;
    double * xTopBlock = (blockIdx.y != gridDim.y-1) ?
                         xBottomGpu + arrayShift + numElementsPerBlock * gridDim.x :
			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + verticalShift + horizontalShift + idx;

    if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
        iGrid = iGrid - nxGrids;
    }

    int nDofs = nxGrids * nyGrids;

    if ((blockIdx.y == gridDim.y-1) && threadIdx.y >= (blockDim.y/2)) {
        iGrid = iGrid - nDofs;
    } 

/*
    if ((blockIdx.x == gridDim.x-1) && (threadIdx.x >= (blockDim.x/2)) && (iGrid >= nDofs-1)) {
        iGrid = blockShift + verticalShift + horizontalShift + idx - nDofs - nxGrids; 
    }
*/   
    // printf("I am idx %d with tidx %d and tidy %d and grid point %d\n", idx, threadIdx.x, threadIdx.y, iGrid);

    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (idx < numElementsPerBlock) {
        sharedMemory[idx] = xBottomBlock[threadIdx.x + (blockDim.y/2-1-threadIdx.y)*blockDim.x];
    }
    else {
        sharedMemory[idx] = xTopBlock[threadIdx.x + (threadIdx.y-(blockDim.y/2))*blockDim.x];
    }
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}


__global__
void _iterativeGpuVerticalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double *rhsGpu, const double * leftMatrixGpu, const double *centerMatrixGpu, 
                                const double * rightMatrixGpu, const double * topMatrixGpu, const double * bottomMatrixGpu, 
                                int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = blockDim.y/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xRightBlock =  xLeftGpu + arrayShift;
    double * xLeftBlock = (blockIdx.x != 0) ?
                           xRightGpu + arrayShift - numElementsPerBlock :
    			   xRightGpu + numElementsPerBlock * ((gridDim.x-1) + blockIdx.y * gridDim.x);
    
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int nDofs = nxGrids * nyGrids;
    int iGrid = blockShift + verticalShift + threadIdx.y * nxGrids + threadIdx.x;
    iGrid = (iGrid >= nDofs) ? iGrid - nDofs : iGrid;

    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
    
    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (threadIdx.x < blockDim.x/2) {
        sharedMemory[idx] = xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y];
    }
    else {
        sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
    }

    __iterativeBlockUpdateToNorthSouth( xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _finalSolution(double * xTopGpu, double * xBottomGpu, double * x0Gpu, int nxGrids)
{

    extern __shared__ double sharedMemory[];
    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;

    double * xTopBlock = xBottomGpu + arrayShift;
    double * xBottomBlock = (blockIdx.y != 0) ?
			    xTopGpu + (blockIdx.x + (blockIdx.y-1) * gridDim.x) * numElementsPerBlock :
			    xTopGpu + (gridDim.x * (gridDim.y-1) + blockIdx.x) * numElementsPerBlock;

    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    double * x0Block = x0Gpu + blockShift;

    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (idx < (blockDim.x * blockDim.y)/2) {
//        printf("The %dth entry of xTopBlock is %f\n", idx, xTopBlock[idx]);
//        printf("xTopBlock[idx=%d] goes into sharedMemory[%d]\n", idx, idx+numElementsPerBlock);
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[threadIdx.x + (blockDim.x)*(blockDim.x/2-1-threadIdx.y)] = xBottomBlock[idx];
    }

    __syncthreads();

//    printf("sharedMemory[idx=%d] is %f \n", idx, sharedMemory[idx]);
    
    double * x0 = x0Gpu + blockShift;

    idx = threadIdx.x + threadIdx.y * nxGrids;
    x0[threadIdx.x + threadIdx.y * nxGrids] = sharedMemory[threadIdx.x + threadIdx.y * blockDim.x];    
}

///////////////////////////////////////////////////

double * iterativeGpuSwept(const double * initX, const double * rhs,
        const double * leftMatrix, const double * centerMatrix,
        const double * rightMatrix, const double * topMatrix, const double * bottomMatrix,
	int nxGrids, int nyGrids, int nIters, const int threadsPerBlock, const int method)
{     
    // Determine number of threads and blocks 
    const int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock);
    const int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock);
    const int nDofs = nxGrids * nyGrids;

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    
    // Allocate memory for solution and inputs
    double *xLeftGpu, *xRightGpu, *xTopGpu, *xBottomGpu;
    int numSharedElemPerBlock = threadsPerBlock * (threadsPerBlock / 2 + 1);
    cudaMalloc(&xLeftGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xRightGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xTopGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xBottomGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    double * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu, * topMatrixGpu, * bottomMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&leftMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&centerMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&rightMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&topMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&bottomMatrixGpu, sizeof(double) * nDofs);

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(topMatrixGpu, topMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);
    cudaMemcpy(bottomMatrixGpu, bottomMatrix, sizeof(double) * nDofs,
            cudaMemcpyHostToDevice);

    int sharedBytes = 2 * threadsPerBlock * threadsPerBlock * sizeof(double);

    for (int i = 0; i < nIters; i++) {

        // APPLY METHOD TO ADVANCE POINTS (NO SHIFT)
        _iterativeGpuOriginal <<<grid, block, sharedBytes>>>(xLeftGpu, xRightGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, method);

        // APPLY HORIZONTAL SHIFT
        _iterativeGpuHorizontalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, method);

        // APPLY VERTICAL SHIFT (ALONG WITH PREVIOUS HORIZONTAL SHIFT)
        _iterativeGpuVerticalandHorizontalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, method);

        // APPLY VERTICAL SHIFT
        _iterativeGpuVerticalShift <<<grid, block, sharedBytes>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, method);

        // APPLY FINAL STEP
        _finalSolution <<<grid, block, sharedBytes>>>(xTopGpu, xBottomGpu, x0Gpu, nxGrids);

    }

    double * solution = new double[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs,
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

int main(int argc, char *argv[])
{
    // Ask user for inputs
    const int nxGrids = atoi(argv[1]); 
    const int nyGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    const int nIters = atoi(argv[3]);
    const int nCycles = atoi(argv[4]);

    method_type method = JACOBI;

    int nDofs = nxGrids * nyGrids;
    
    // Declare arrays and population with values for Poisson equation
    double * initX = new double[nDofs];
    double * rhs = new double[nDofs];
    double * leftMatrix = new double[nDofs];
    double * centerMatrix = new double[nDofs];
    double * rightMatrix = new double[nDofs];
    double * bottomMatrix = new double[nDofs];
    double * topMatrix = new double[nDofs];
    
    double dx = 1.0f / (nxGrids + 1);
    double dy = 1.0f / (nyGrids + 1);

    for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
        initX[iGrid] = (double)iGrid; 
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx) + 2.0f / (dy * dy);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
	bottomMatrix[iGrid] = -1.0f / (dy * dy);
	topMatrix[iGrid] = -1.0f / (dy * dy);
    }

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Run the CPU Implementation and measure the time required
    clock_t cpuStartTime = clock();
    double * solutionCpu = iterativeCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids, nIters, method);
    clock_t cpuEndTime = clock();
    double cpuTime = (cpuEndTime - cpuStartTime) / (double) CLOCKS_PER_SEC;

    // Run the Classic GPU Implementation and measure the time required
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    double * solutionGpuClassic = iterativeGpuClassic(initX, rhs, leftMatrix, centerMatrix,
                                                      rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    // Run the Swept GPU Implementation and measure the time required
    cudaEvent_t startSwept, stopSwept;
    float timeSwept;
    cudaEventCreate( &startSwept );
    cudaEventCreate( &stopSwept );
    cudaEventRecord( startSwept, 0);
    double * solutionGpuSwept = iterativeGpuSwept(initX, rhs, leftMatrix, centerMatrix,
                                                  rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids, nCycles, threadsPerBlock, method);
    cudaEventRecord(stopSwept, 0);
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);
    
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of total grid points: %d\n", nDofs);
    printf("Number of grid points in x-direction: %d\n", nxGrids);
    printf("Number of grid points in y-direction: %d\n", nyGrids);
    printf("Threads Per Block in each direction: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIters);
    printf("\n");

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
    double cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    double classicTimePerIteration = timeClassic / nIters;
    double sweptTimePerIteration = timeSwept / nIters;
    double timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    /* printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); */
    printf("Total Time needed for the CPU: %f ms\n", cpuTime * 1e3);
    printf("Total Time needed for the Classic GPU is %f ms\n", timeClassic);
    printf("Total Time needed for the Swept GPU: %f ms\n", timeSwept);

    // Compute the residual of the resulting solution (||b-Ax||)
    double residualCPU = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids);
    double residualClassicGPU = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids);
    double residualSweptGPU = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids);
    printf("Residual of the CPU solution is %f\n", residualCPU);
    printf("Residual of the Classic GPU solution is %f\n", residualClassicGPU);
    printf("Residual of the Swept GPU solution is %f\n", residualSweptGPU); 
  
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
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
}
