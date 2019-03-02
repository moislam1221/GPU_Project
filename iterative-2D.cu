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

__device__ __host__
double jacobiGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix, 
		  const double topMatrix, const double bottomMatrix, 
		  const double leftX, double centerX, const double rightX, const double topX, const double bottomX,
		  const double centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX))
         / centerMatrix;
}

__device__ __host__
double RBGSGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix,
		const double topMatrix, const double bottomMatrix,
		const double leftX, double centerX, const double rightX, const double topX, const double bottomX,
		const double centerRhs, const int gridPoint)
{  
    
    // Update all points of a certain parity (i.e. update red, keep black the same)
    if (gridPoint % 2 == 1)
    {
    	return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX))
	 / centerMatrix;
    }
    else
    {
	return centerX;
    }
}

__device__ __host__
double SORGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix,
	       const double topMatrix, const double bottomMatrix, 
	       const double leftX, double centerX, const double rightX, const double topX, const double bottomX,
	       const double centerRhs, const int gridPoint)
{  
    // Similar to red-black gauss-seidel, but take weighted average of rbgs 
    // value and current centerX value based on relaxation parameter
    // printf("Relaxation is %f\n", relaxation);
    double relaxation = 1.0;
    if (gridPoint % 2 == 1)
    {
    	return relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix) + (1.0-relaxation)*centerX;
    }
    else
    {
	return centerX;
    }
}

double normFromRow(double leftMatrix, double centerMatrix, double rightMatrix, double leftX, double centerX, double rightX,  double centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

double Residual(const double * solution, const double * rhs, const double * leftMatrix, const double * centerMatrix, const double * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    double residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        double leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f; 
        double centerX = solution[iGrid];
        double rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        double residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
	// printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

/*double * readExactSolution(int nGrids)
{
    double exactSolution[nGrids];
    std::ifstream input("exactSolution.txt");
    for (int i = 0; i < nGrids; i++)
    {
        input >> exactSolution[i];
        // printf("Data is %f\n", exactSolution[i]);
    }
    return exactSolution;
}*/

double solutionError(double * solution, double * exactSolution, int nGrids)
{
    double error = 0.0;
    double difference; 
    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
         difference = solution[iGrid] - exactSolution[iGrid];
	 error = error + difference*difference;
    }
    error = sqrt(error);
    return error;
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

__device__ 
void __iterativeBlockUpperPyramidalFromShared(
		double * xLeftBlock, double *xRightBlock, double *xTopBlock, double *xBottomBlock, const double *rhsBlock,
		const double * leftMatrixBlock, const double * centerMatrixBlock,
                const double * rightMatrixBlock, const double * topMatrixBlock, const double * bottomMatrixBlock,
	       	int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x * blockDim.y; 

    int idx = threadIdx.x + blockDim.x * threadIdx.y;
    
    printf("Idx %d, initial solution %f\n", idx, x0[idx]);
    for (int k = 0; k <= blockDim.x/2-1; ++k) {
           printf("Time step %d\n", k); 
        if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1 && threadIdx.y >= k && threadIdx.y <= blockDim.y-k-1) {
        
	// Bottom 
        if (threadIdx.y == k)
        {
	    xBottomBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))] = x0[idx];
     	}
	if (threadIdx.y == k + 1)
    	{
            xBottomBlock[threadIdx.x-k+(2*k)*(blockDim.x-k) + blockDim.x] = x0[idx];
    	}

	// Top
        if (threadIdx.y == blockDim.x - 1 - k)
    	{
	    xTopBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))] = x0[idx];
    	}
	if (threadIdx.y == blockDim.x - 2 - k)
    	{
            xTopBlock[threadIdx.x-k+(2*k)*(blockDim.x-k) + blockDim.x] = x0[idx];
      	}
	
        // Left
        if (threadIdx.x == k)
        {
            xLeftBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k-1))] = x0[idx];
        }
        if (threadIdx.x == k + 1)
        {
            xLeftBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k)) + blockDim.x] = x0[idx];
        }

        // Right
        if (threadIdx.x == blockDim.x - 1 - k)
        {
            xRightBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k-1))] = x0[idx];
        }
        if (threadIdx.x == blockDim.x - 2 - k)
        {
            xRightBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k)) + blockDim.x] = x0[idx];
        }    

	}

        if (threadIdx.x > k && threadIdx.x < blockDim.x-k-1 && threadIdx.y > k && threadIdx.y < blockDim.y-k-1 && k != blockDim.x/2) {
	    
	    double leftX = ((iGrid % nxGrids) == 0) ? 0.0f : x0[idx - 1];
            double centerX = x0[idx];
            double rightX = (((iGrid + 1) % nxGrids) == 0) ? 0.0f : x0[idx + 1];
	    double bottomX = (iGrid < nxGrids) ? 0.0f : x0[idx - blockDim.x];
            double topX = (iGrid < nxGrids*(nyGrids-1)) ? x0[idx + blockDim.x] : 0.0f;
            
	    double leftMat = leftMatrixBlock[idx];
            double centerMat = centerMatrixBlock[idx];
            double rightMat = rightMatrixBlock[idx];
      	    double topMat = topMatrixBlock[idx];
            double bottomMat = bottomMatrixBlock[idx];
            double rhs = rhsBlock[idx];
	    
            if (k % 2 == 0) {
                x1[idx] = centerX; /*iterativeOperation(leftMat, centerMat, rightMat, topMat, bottomMat, 
				                     leftX, centerX, rightX, topX, bottomX, rhs, iGrid, method); */
            }
	    else {
	        x1[idx] = centerX;/* iterativeOperation2(leftMat, centerMat, rightMat, topMat, bottomMat,
				                      leftX, centerX, rightX, topX, bottomX, rhs, iGrid, method); */
	    }
        }
        
	__syncthreads();	
    	double * tmp = x1; x1 = x0; x0 = tmp;
    
    } 

    printf("Idx %d, Top %f, Bottom %f, Left %f, Right %f\n", idx, xTopBlock[idx], xBottomBlock[idx], xLeftBlock[idx], xRightBlock[idx]);    
    printf("Idx %d, SharedMemoryValue %f\n", idx, x0[idx]);
    double * tmp = x1; x1 = x0; x0 = tmp;

}

__global__
void _iterativeGpuUpperPyramidal(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;

    int blockShift = xShift + yShift * nxGrids;

    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    double * xTopBlock = xTopGpu + blockShift;
    double * xBottomBlock = xBottomGpu + blockShift;
    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift;


    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;
    
    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x] = x0Block[threadIdx.x + threadIdx.y * nxGrids];

    __iterativeBlockUpperPyramidalFromShared(xLeftBlock, xRightBlock, xTopBlock, xBottomBlock, rhsBlock,
    		                             leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
					     nxGrids, nyGrids, iGrid, method);
}


__device__
void __iterativeBlockLongitudinalBridgeFromShared(double * xLowerBlock, double * xUpperBlock, double * xEastBlock, double * xWestBlock, double * rhsBlock, double * leftMatrixBlock, double * centerMatrixBlock, double * rightMatrixBlock, double * topMatrixBlock, double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{  
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory;
    double * x1 = sharedMemory + blockDim.x * blockDim.x;
   
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    // At every step, load xLower and xUpper and fill in values
    for (int k = 0; k <= blockDim.x/2-1; --k) 
    {
	if (threadIdx.y == blockDim.y/2-1-k)
	{
	     x0[idx] = xLowerBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	}
	if (threadIdx.y == blockDim.y/2-2-k)
	{
	     x0[idx] = xLowerBlock[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	}
	if (threadIdx.y == blockDim.y/2+1-k)
	{
	     x0[idx] = xUpperBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	}
	if (threadIdx.y == blockDim.y+2-k)
	{
	     x0[idx] = xUpperBlock[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	}

	if (threadIdx.x == blockDim.x-2-k && threadIdx.x == blockDim.x-1-k && threadIdx.y >= blockDim.x/2+1-k && threadIdx.y <= blockDim.x/2+2+k) 
	{
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        xEastBlock[blockDim.x/2+2+k-threadIdx.y+shift] = x0[idx]; // Right column
	        xEastBlock[blockDim.x/2+2+k-threadIdx.x+shift+2*(k+1)] = x0[idx]; // Left column
	    }
	    else {
                int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
		xEastBlock[shift] = xUpperBlock[numSharedElemPerBlock-1];
	        xEastBlock[shift+2*(k+1)] = xUpperBlock[numSharedElemPerBlock-2];
		if (threadIdx.y != blockDim.x+2+k) {
	            xEastBlock[blockDim.x/2+2+k-threadIdx.y+shift] = x0[idx]; // Right column
	            xEastBlock[blockDim.x/2+2+k-threadIdx.x+shift+2*(k+1)] = x0[idx]; // Left column
		}
	    }	
	}
	
	if (threadIdx.x == k && threadIdx.x == k+1 && threadIdx.y >= blockDim.x/2-1-k && threadIdx.y <= blockDim.x/2+k) 
	{
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        xWestBlock[threadIdx.y-(blockDim.x/2-1-k)+shift] = x0[idx]; // Right column
	        xWestBlock[threadIdx.y-(blockDim.x/2-1-k)+shift+2*(k+1)] = x0[idx]; // Left column
	    }
	    else {
	        xWestBlock[shift] = xLowerBlock[0];
		xWestBlock[shift+2*(k+1)] = xLowerBlock[0];
	    }
	}
	
	if (k != blockDim.x/2 && threadIdx.x > k && threadIdx.x < blockDim.x-1-k && threadIdx.y >= blockDim.y/2-1-k && threadIdx.y <= blockDim.y/2 + k)
	{
	    double leftX = x0[idx];
	    double centerX = x0[idx];
	    double rightX = x0[idx];
	    double topX = x0[idx];
            double bottomX = x0[idx];

            double leftMat = leftMatrixBlock[idx];
	    double centerMat = centerMatrixBlock[idx];
	    double rightMat = rightMatrixBlock[idx];
	    double topMat = topMatrixBlock[idx];
	    double bottomMat = bottomMatrixBlock[idx];
	    
	    if (k % 2 == 0) {
                x1[idx] = centerX; /* (leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); */
            }
	    else {
                x1[idx] = centerX; /*  iterativeOperation2(leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); */
            }
	}
    }
}

__global__       
void _iterativeGpuLongitudinalBridge(double * xTopGpu, double * xBottomGpu, double * xEastGpu, double * xWestGpu, double * x0Gpu,
                                  double * rhsGpu, double * leftMatrixGpu, double * centerMatrixGpu, double * rightMatrixGpu, 
				  double * topMatrixGpu, double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
    int blockID =  blockIdx.y * gridDim.x + blockIdx.x;
    int nDofs = nxGrids * nyGrids;

    int sharedShift = numSharedElemPerBlock * blockID;
    double * xLowerBlock = xTopGpu + sharedShift;
    double * xUpperBlock = (blockIdx.y == (gridDim.y-1)) ?
                           xBottomGpu + numSharedElemPerBlock * blockIdx.x : 
                           xBottomGpu + sharedShift + gridDim.x * numSharedElemPerBlock;

    int blockShift = (blockDim.x * blockDim.y) * blockID;
    int verticalShift = blockDim.y/2 * nxGrids;
    
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    int iGrid = blockShift + (threadIdx.x + threadIdx.y * nxGrids) + verticalShift;
    iGrid = (iGrid < nDofs) ? iGrid : iGrid - nDofs; 

    double * rhsBlock = rhsGpu + blockShift + verticalShift;
    double * leftMatrixBlock = leftMatrixGpu + blockShift + verticalShift;
    double * centerMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    double * rightMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
    double * topMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    double * bottomMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
     
    double * xEastBlock = xEastGpu + blockShift + verticalShift;
    double * xWestBlock = xWestGpu + blockShift + verticalShift;

    extern __shared__ double sharedMemory[];
    
    __iterativeBlockLongitudinalBridgeFromShared(xLowerBlock, xUpperBlock, xEastBlock, xWestBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
				       nxGrids, nyGrids, iGrid, method);  

}

/*
__global__       
void _iterativeGpuLatitudinalBridge(double * xLeftGpu, double * xRightGpu, double * xNorthGpu, double * xSouthGpu,
                                  double * rhsGpu, double * leftMatrixGpu, double * centerMatrixGpu, double * rightMatrixGpu, 
				  double * topMatrixGpu, double * bottomMatrixGpu,
				  int nxGrids, int nyGrids, int method)
{
    int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
    int blockID =  blockIdx.y * gridDim.x + blockIdx.x;

    int sharedShift = numSharedElemPerBlock * blockID;
    double * xLeftBlock = xRightGpu + sharedShift;
    double * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                           xLeftGpu + blockIdx.y * (numSharedElemPerBlock * gridDim.x) : 
                           xLeftGpu + sharedShift + (1 * numSharedElemPerBlock);

    int blockShift = (blockDim.x * blockDim.y) * blockID;
    int horizontalShift = blockDim.x/2;
    
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    int iGrid = blockShift + (threadIdx.x + threadIdx.y * nxGrids) + horizontalShift;
    iGrid = (iGrid < nDofs) ? iGrid : iGrid - nDofs; 

    double * rhsBlock = rhsGpu + blockShift + horizontalShift;
    double * leftMatrixBlock = leftMatrixGpu + blockShift + horizontalShift;
    double * centerMatrixBlock = centerMatrixGpu + blockShift + horizontalShift;
    double * rightMatrixBlock = rightMatrixGpu + blockShift + horizontalShift;
    double * topMatrixBlock = centerMatrixGpu + blockShift + horizontalShift;
    double * bottomMatrixBlock = rightMatrixGpu + blockShift + horizontalShift;
    
    extern __shared__ double sharedMemory[];
    
    __iterativeBlockLongitudinalBridgeFromShared(xLeftBlock, xRightBlock, xNorthBlock, xSouthBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
				       nxGrids, nyGrids, iGrid, method);  
}
*/
/*
__device__
void __iterativeBlockLatitudinalBridgeFromShared(double * xLeftBlock, double * xRightBlock, double * xNorthBlock, double * xSouthBlock, double * rhsBlock,
		                        double * leftMatrixBlock, double * centerMatrixBlock, double * rightMatrixBlock, double * topMatrixBlock, double * bottomMatrixBlock,
                                        int nxGrids, int ny Grids, int iGrid, int method)
{
    idx = threadIdx.x + threadIdx.y * blockDim.x;
    // At every step, load xLower and xUpper and fill in values
    for (int k = 0; k <= blockDim.x/2-1; --k) 
    {
	if (threadIdx.y == blockDim.y/2-1-k)
	{
	     x0[idx] = xLower[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	}
	if (threadIdx.y == blockDim.y/2-2-k)
	{
	     x0[idx] = xLower[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	}
	if (threadIdx.y == blockDim.y/2+1-k)
	{
	     x0[idx] = xUpper[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	}
	if (threadIdx.y == blockDim.y+2-k)
	{
	     x0[idx] = xUpper[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	}

	if (threadIdx.x == blockDim.x-2-k && threadIdx.x == blockDim.x-1-k && threadIdx.y >= blockDim.x/2+1-k && threadIdx.y <= blockDim.x/2+2+k) 
	{
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        xEast[blockDim.x/2+2+k-threadIdx.y+shift] = x0[idx]; // Right column
	        xEast[blockDim.x/2+2+k-threadIdx.x+shift+2*(k+1)] = x0[idx]; // Left column
	    }
	    else {
		xEast[shift] = xUpperBlock[];
	        xEast[shift+2*(k+1)] = xUpperBlock[];
	    }	
	}
	
	if (threadIdx.x == k && threadIdx.x == k+1 && threadIdx.y >= blockDim.x/2-1-k && threadIdx.y <= blockDim.x/2+k) 
	{
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        xWest[threadIdx.y-(blockDim.x/2-1-k)+shift] = x0[idx]; // Right column
	        xWest[threadIdx.y-(blockDim.x/2-1-k)+shift+2*(k+1)] = x0[idx]; // Left column
	    else {
	        xWest[shift] = xLowerBlock[];
		xWest[shift+2*(k+1)] = xLowerBlock[];
	    }
	}
        
	if (k != blockDim.x/2 && threadIdx.x > k && threadIdx.x < blockDim.x-1-k && threadIdx.y >= blockDim.y/2-1-k && threadIdx.y <= blockDim.y/2 + k)
	{
	    double leftX = x0[idx];
	    double centerX = x0[idx];
	    double rightX = x0[idx];
	    double topX = x0[idx];
            double bottomX = x0[idx]

            double leftMat = leftMatrixBlock[idx];
	    double centerMat = centerMatrixBlock[idx];
	    double rightMat = rightMatrixBlock[idx];
	    double topMat = topMatrixBlock[idx];
	    double bottomMat = bottomMatrixBlock[idx];
	    
	    if (k % 2 == 0) {
                x1[idx] = centerX;  (leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); 
            }
	    else {
                x1[idx] = centerX; iterativeOperation2(leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); 
            }
	}
    }
}
*/

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

    double * xNorthGpu, * xSouthGpu, * xEastGpu, * xWestGpu;
    cudaMalloc(&xNorthGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xSouthGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xEastGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);
    cudaMalloc(&xWestGpu, sizeof(double) * numSharedElemPerBlock * nxBlocks * nyBlocks);

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

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    /*cudaMemcpy(x0Gpu + nGrids, initX, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu + nGrids, leftMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu + nGrids, centerMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu + nGrids, rightMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    */

    int sharedFloatsPerBlock = threadsPerBlock * threadsPerBlock * 2;

/*    double residualSwept;
    double nCycles = nIters / threadsPerBlock;
    double * currentSolution = new double[nGrids];
    std::ofstream residuals;
    residuals.open("dummy.txt",std::ios_base::app);
    
    for (int i = 0; i < nCycles; i++) {
        _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (x0Gpu, xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
        cudaMemcpy(currentSolution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);
        residualSwept = Residual(currentSolution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        residuals << nGrids << "\t" << threadsPerBlock << "\t" << i*threadsPerBlock << "\t" << residualSwept << "\n";
    }
   
    residuals.close();
*/
    _iterativeGpuUpperPyramidal <<<grid, block,
        sizeof(double) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu, xTopGpu, xBottomGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, 
		nxGrids, nyGrids, method);
    _iterativeGpuLongitudinalBridge <<<grid, block,
            sizeof(double) * sharedFloatsPerBlock>>>(
                    xTopGpu, xBottomGpu, xEastGpu, xWestGpu,
                    x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, topMatrixGpu, bottomMatrixGpu,
		    nxGrids, nyGrids, method);
  /*  _iterativeGpuLatitudinalBridge <<<grid, block,
            sizeof(double) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu, xNorthGpu, xSouthGpu,
                    x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, topMatrixGpu, bottomMatrixGpu,
		    nxGrids, nyGrids, method);

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _iterativeGpuDiamond <<<grid, block,
                sizeof(double) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
    _iterativeGpuShiftedDiamond <<<grid, block,
            sizeof(double) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method); 
    }

    _iterativeGpuLowerTriangle <<<grid, block,
                sizeof(double) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
*/
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
    const int nyGrids = atoi(argv[2]); 
    const int threadsPerBlock = atoi(argv[3]); 
    const int nIters = atoi(argv[4]);

    method_type method = GS;

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
                                                  rightMatrix, topMatrix, bottomMatrix, nxGrids, nyGrids, nIters, threadsPerBlock, method);
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
//    double cpuTimePerIteration = (cpuTime / nIters) * 1e3;
//    double classicTimePerIteration = timeClassic / nIters;
//    double sweptTimePerIteration = timeSwept / nIters;
//    double timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
//    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
//    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
//    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); 

    // Compute the residual of the resulting solution (|b-Ax|)
    //double residualClassic = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    //double residualSwept = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
//    printf("Residual of the converged solution is %f\n", residualSwept);
//    printf("Residual of Classic result is %f\n", residualClassic); 
  
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
