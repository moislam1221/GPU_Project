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

float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float topMatrix, float bottomMatrix, float leftX, float centerX, float rightX,  float topX, float bottomX, float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}

/*
float Residual(const float * solution, const float * rhs, const float * matrixElements, int nxGrids, int nyGrids)
{
    int nDofs = nxGrids * nyGrids;
    float residual = 0.0;  

    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
	float leftX = solution[iGrid-1];
	float centerX = solution[iGrid];
	float rightX = solution[iGrid+1];
	float bottomX = solution[iGrid-nxGrids];
        float topX;
        if (iGrid + nxGrids < nDofs) {
            topX = solution[iGrid+nxGrids];
        }
	boundaryConditions(iGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);
        float residualContributionFromRow = normFromRow(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
    }
    residual = sqrt(residual);
    return residual;
}
*/


float Residual(const float * solution, const float * rhs, const float * matrixElements, const int nxGrids, const int nyGrids)
{
    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];
    float leftX, rightX, centerX, bottomX, topX;
    int dof;
    float residualContributionFromDOF;
    float residual = 0.0;  

    // INITIAL SOLUTION
    for (int iGrid = 0; iGrid < nyGrids; iGrid++) {
        for (int jGrid = 0; jGrid < nxGrids; jGrid++) {
            
            dof = jGrid + iGrid * nxGrids;
            
            if (iGrid == 0 || iGrid == nxGrids - 1 || jGrid == 0 || jGrid == nyGrids - 1) {
                residualContributionFromDOF = 0.0f;
            }
            
            else {
	        leftX = solution[dof-1];
	        centerX = solution[dof];
	        rightX = solution[dof+1];
	        bottomX = solution[dof-nxGrids];
                topX = solution[dof+nxGrids];
                residualContributionFromDOF = normFromRow(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, leftX, centerX, rightX, topX, bottomX, rhs[dof]);
            }
	  
            // printf("Residual contribution from DOF %d is %f\n", dof, residualContributionFromDOF);      
            residual = residual + residualContributionFromDOF * residualContributionFromDOF;

        }
    }

    residual = sqrt(residual);
    return residual;
}

void print2DSolution(const float * solution, const int nxGrids, const int nyGrids)
{
    
    int dof;
    // INITIAL SOLUTION
    for (int iGrid = nyGrids-1; iGrid > -1; iGrid--) {
        for (int jGrid = 0; jGrid < nxGrids; jGrid++) {
            dof = jGrid + iGrid * nxGrids;
            // `printf("%d ", dof);
            printf("%f ", solution[dof]); 
        }
        printf("\n"); 
    }
}


float * iterativeCpu(const float * initX, const float * rhs,
                  const float * matrixElements, const int nxGrids, const int nyGrids,
		  const int nIters, const int method)
{
    int nDofs = nxGrids * nyGrids;
    float * x0 = new float[nDofs];
    float * x1 = new float[nDofs];
    memcpy(x0, initX, sizeof(float) * nDofs);
    memcpy(x1, initX, sizeof(float)* nDofs);
    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nDofs; ++iGrid) {
            const float leftX = ((iGrid % nxGrids) == 0) ? 0.0f : x0[iGrid - 1];
            const float centerX = x0[iGrid];
            const float rightX = (((iGrid + 1) % nxGrids) == 0) ? 0.0f : x0[iGrid + 1];
	    const float bottomX = (iGrid < nxGrids) ? 0.0f : x0[iGrid - nxGrids];
            const float topX = (iGrid < nDofs - nxGrids) ? x0[iGrid + nxGrids] : 0.0f;
	    if (iIter % 2 == 0) {
                x1[iGrid] = iterativeOperation(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
				    leftX, centerX, rightX, topX, bottomX, rhs[iGrid], iGrid, method);
	    }
	    else { 
                x1[iGrid] = iterativeOperation(leftMatrix, centerMatrix,
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
                         const float leftMatrix, const float centerMatrix,
                         const float rightMatrix, const float topMatrix, const float bottomMatrix,
			 const int nxGrids, const int nyGrids, const int iteration, const int method)
{
    const int ixGrid = blockIdx.x * blockDim.x + threadIdx.x; // Col
    const int iyGrid = blockIdx.y * blockDim.y + threadIdx.y; // Row
    const int iGrid = iyGrid * (nxGrids) + ixGrid;
    const int nDofs = nxGrids * nyGrids;
    if (iGrid < nDofs) {
        const float leftX = (ixGrid == 0) ? 0.0f : x0[iGrid - 1] ;
        const float centerX = x0[iGrid];
        const float rightX = (ixGrid == nxGrids - 1) ?  0.0f : x0[iGrid + 1];
	const float topX = (iyGrid == nyGrids - 1) ? 0.0f : x0[iGrid + nxGrids];
        const float bottomX = (iyGrid == 0) ? 0.0f : x0[iGrid - nxGrids];
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
			 const int nxGrids, const int nyGrids, const int nIters, const int threadsPerBlock, const int method)
{
  	
    const int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for the solution
    float * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(float) * nDofs);
   
    // Allocate CPU memory for other variables
    float * rhsGpu;
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    
    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    const int nxBlocks = (int)ceil(nxGrids / (float)threadsPerBlock);
    const int nyBlocks = (int)ceil(nyGrids / (float)threadsPerBlock);

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);

    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];
    
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU (used to be <<<nBlocks, threadsPerBlock>>>)
        _iterativeGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrix, centerMatrix,
                rightMatrix, topMatrix, bottomMatrix,  
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

    return solution;
}

//// SWEPT METHODS HERE ////

///////////////////////////////////////////////////

// USEFUL HELPER FUNCTIONS

// Print solution from GPU
__device__
void printSolutionGPU(const float * solution, int nxGrids, int nyGrids) {
    
    for (int index = threadIdx.x + blockDim.x * threadIdx.y; index < nxGrids * nyGrids; index += blockDim.x * blockDim.y) {
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("x0Gpu[%d] = %f\n", index, solution[index]);         
        }
    }

}

// Print shared memory values
__device__
void printSharedMemoryContents(int blockIDX, int blockIDY, int xLength, int yLength) {
    
    extern __shared__ float sharedMemory[];

    for (int index = threadIdx.x + blockDim.x * threadIdx.y; index < xLength * yLength; index += blockDim.x * blockDim.y) {
        if (blockIdx.x == blockIDX && blockIdx.y == blockIDY) {
            printf("In shared[%d] = %f\n", index, sharedMemory[index]);         
        }
    }
}

// Apply boundary conditions
__host__ __device__
void boundaryConditions(float &leftX, float &rightX, float &bottomX, float &topX, const int globalGridPoint, const int nxGrids, const int nyGrids)
{
    // Left
    if ((globalGridPoint-1) % nxGrids == 0) {
        leftX = 0.0;
    }

    // Right
    if ((globalGridPoint+2) % nxGrids == 0) {
        rightX = 0.0;
    }

    // Bottom
    if ((globalGridPoint > nxGrids) && (globalGridPoint < 2*nxGrids-2)) {
        bottomX = 0.0;
    }

    // Top
    if ((globalGridPoint < (nxGrids * nyGrids - nxGrids - 1)) && (globalGridPoint > (nxGrids * nyGrids - 2*nxGrids))) {
        topX = 0.0;
    }

    return;
}

// Method 1 - Move from global memory to shared memory
__device__
void globalToShared(const float * x0Block, const int xLength, const int yLength, const int nxGrids) {
    
    // Instantiate shared memory here
    extern __shared__ float sharedMemory[];
    
    // Move values from global memory to shared memory of block
    int ID= threadIdx.x + blockDim.x * threadIdx.y;
    int stride = blockDim.x * blockDim.y;
    int xval, yval;
    for (int index = ID; index < xLength * yLength; index += stride) {
        xval = index % xLength;
        yval = (index / xLength) * nxGrids;
        sharedMemory[index] = x0Block[xval + yval];
        sharedMemory[index + xLength * yLength] = x0Block[xval + yval];
        /*if ((blockIdx.x == 0 && blockIdx.y == 0) || (blockIdx.x == 1 && blockIdx.y == 1)) {
            printf("x0Block[%d] = %f\n", xval + yval, x0Block[xval + yval]);
        }*/
        //printf("(%d, %d): sharedMemory[%d] = %f\n", blockIdx.x, blockIdx.y, index, x0Block[xval + yval]);
        //printf("(%d, %d): sharedMemory[%d] = %f\n", blockIdx.x, blockIdx.y, index + xLength * yLength, x0Block[xval + yval]);
    }

}

// Method 2 - Update values in shared memory
__device__
void redBlackBlockUpdate(const float * rhsBlock, const float * matrixElementsGpu, const int xLength, const int yLength, const int max_JacobiIters, const int step)
{
    
    // Initialize shared memory and pointers to x0, x1 arrays 
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    float * x1 = sharedMemory + xLength * yLength;

    // Define point ID and stride for "for loop"
    int ID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    // Declare variables which will be necessary later
    float centerRhs, leftX, rightX, centerX, topX, bottomX;

    // Extract the matrix elements into register variables
    float bottomMatrix = matrixElementsGpu[0]; 
    float leftMatrix = matrixElementsGpu[1]; 
    float centerMatrix = matrixElementsGpu[2]; 
    float rightMatrix = matrixElementsGpu[3]; 
    float topMatrix = matrixElementsGpu[4]; 

    // For specified number of jacobi iterations to be performed each cycle
    for (int iter = 0; iter < max_JacobiIters; iter++) {
        // For all gridpoints in the subdomain 
        for (int index = ID; index < xLength * yLength; index += stride) {
        // If the grid point lies in the interior
            if ((index % xLength != 0) && ((index+1) % xLength != 0) && (index > xLength-1) && (index < xLength * (yLength-1))) {
                                   
                // Obtain adjacent grid point values and rhs value necessary for update
	        centerRhs = rhsBlock[index];
	        leftX = x0[index-1];
	        centerX = x0[index];
	        rightX = x0[index+1];
	        topX = x0[index+xLength];
	        bottomX = x0[index-xLength];

                // Perform jacobi updates on all inner points within appropriate block
                if ((blockIdx.x + blockIdx.y) % 2 == ((step+1) % 2)) {
                    x1[index] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                       leftX, centerX, rightX, topX, bottomX, centerRhs); 
                    // x1[index] = leftX + rightX + centerX + topX + bottomX + 1.0;
                    // printf("In blockIdx.x = %d, blockIdx.y = %d the x1[%d] = %f (leftX %f rightX %f centerX %f topX %f bottomX %f)\n", blockIdx.x, blockIdx.y, index, x1[index], x1[index-1], x1[index+1], x1[index], x1[index+xLength], x1[index-xLength]);
                }
           
            }
            
        }

        // Synchronize
	__syncthreads();

	// Exchange between new and old values
        float * tmp; tmp = x0; x0 = x1, x1 = tmp;
    }
       
}

// Method 3 - Move from shared memory to global memory
__device__
void sharedToGlobal(float * x0Block, const int subdomainLength, const int xLength, const int nxGrids) {
    
    // Initialize shared memory and pointers to x0, x1 arrays 
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    
    // Define the amount of overlap
    int overlap = subdomainLength / 2;
    
    // Define point ID and stride for "for loop"
    int ID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    
    // Obtain the x,y shifts necessary to go from larger overlapping subdomain to restricted subdomain
    int xShiftToRestrict = 0;
    int yShiftToRestrict = 0;
    
    if (blockIdx.x != 0) {
        xShiftToRestrict = overlap;
    }
    
    if (blockIdx.y != 0) {
        yShiftToRestrict = overlap;
    }
   
    int restrictShift = xShiftToRestrict + yShiftToRestrict * xLength;

    int xval, yval, innerShift;
    for (int index = ID; index < subdomainLength * subdomainLength; index += stride) {
        xval = index % subdomainLength;
        yval = (index / subdomainLength);
        innerShift = xval + yval * (subdomainLength + 2 * overlap);
        // if (blockIdx.x == 0 || blockIdx.x == (gridDim.x-1) || blockIdx.y == 0 || blockIdx.y == (gridDim.y-1)) {
        if (blockIdx.x == 0 || blockIdx.x == (gridDim.x-1)) {
            innerShift = xval + yval * (subdomainLength + overlap);
        }

      /*  if (blockIdx.x == 3 && blockIdx.y == 0) {     
            printf("x0Block[%d] = x0[%d] = %f\n", xShiftToRestrict + yShiftToRestrict * nxGrids + xval + yval * nxGrids, restrictShift + innerShift, x0[restrictShift + innerShift]);   
        }*/


        x0Block[xShiftToRestrict + yShiftToRestrict * nxGrids + xval + yval * nxGrids] = x0[restrictShift + innerShift];

    }

}


__global__
void _2D_Algorithm(float * x0Gpu, const float * rhsGpu, const float * matrixElementsGpu, const int nxGrids, const int nyGrids, const int method, const int subdomainLength, const int step, const int num_JacobiIters)
{
    // Instantiate shared memory here
    extern __shared__ float sharedMemory[];

    // Define the boundaries of the subdomain
    int xLeft = blockIdx.x * subdomainLength;
    int xRight = (blockIdx.x + 1) * subdomainLength - 1;
    int yLower = blockIdx.y * subdomainLength;
    int yUpper = (blockIdx.y + 1) * subdomainLength - 1;

    // Define amount of overlap (half of the subdomain length)
    const int overlapLength = subdomainLength / 2;

    // Adjust the boundaries of the subdomain so that there is overlap
    if (blockIdx.x != 0) {
        xLeft = xLeft - overlapLength;
    }
    if (blockIdx.x != gridDim.x - 1) {
        xRight = xRight + overlapLength;
    }
    if (blockIdx.y != 0) {
        yLower = yLower - overlapLength;
    }
    if (blockIdx.y != gridDim.y - 1) {
        yUpper = yUpper + overlapLength;
    }

    // Obtain the length in x and y of the subdomains
    const int xLength = xRight - xLeft + 1;
    const int yLength = yUpper - yLower + 1;

    // Point to the correct section of rhs
    const int blockShift = xLeft + yLower * nxGrids;
    float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;

    //for (int step = 0; step < cycles; step++) {
        // printf("In step %d\n", step);
        // Print current solution array
        // printSolutionGPU(x0Gpu, nxGrids, nyGrids);
        // Move grid point values to the shared memory of each block    
        globalToShared(x0Block, xLength, yLength, nxGrids);
        // printSharedMemoryContents(4,1,xLength,yLength);
        // Update the inner grid points of block shared memory (alternate between red blocks and black blocks, depending on step k)
        redBlackBlockUpdate(rhsBlock, matrixElementsGpu, xLength, yLength, num_JacobiIters, step);
        //__syncthreads();
        // printSharedMemoryContents(0,4,xLength,yLength);
        // Move updated values from shared memory to to appropriate restricted subdomain of global memory
        sharedToGlobal(x0Block, subdomainLength, xLength, nxGrids);
        // printSolutionGPU(x0Gpu, nxGrids, nyGrids); 
    //}
}

float * iterativeGpuSwept(const float * initX, const float * rhs,
        const float * matrixElements,
	const int nxGrids, const int nyGrids, const int cycles, const int num_JacobiIters, const int threadsPerBlock, const int method, const int subdomainLength)
{     
    // Determine number of threads and blocks 
    const int nxBlocks = (int)ceil(nxGrids / (float)subdomainLength);
    const int nyBlocks = (int)ceil(nyGrids / (float)subdomainLength);
    const int nDofs = nxGrids * nyGrids;

    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    
    // Allocate memory for solution and right hand side
    float * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);

    // STORING MATRIX IN GLOBAL MEMORY
    float * matrixElementsGpu;
    cudaMalloc(&matrixElementsGpu, sizeof(float) * 5);
    cudaMemcpy(matrixElementsGpu, matrixElements, sizeof(float) * 5, cudaMemcpyHostToDevice);

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int maxLength = 2 * subdomainLength;
    const int sharedBytes = 2 * maxLength * maxLength * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < 2*cycles; step++) {
        _2D_Algorithm <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, method, subdomainLength, step, num_JacobiIters);
    }

    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

/*    for (int i = 0; i < nDofs; i++) {
        printf("solution[%d] = %f \n", i, solution[i]);
    }
*/
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);
    cudaFree(matrixElementsGpu);

    return solution;
}



