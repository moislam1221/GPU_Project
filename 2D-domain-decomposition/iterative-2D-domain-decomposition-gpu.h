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
#include "helper-gpu.h"

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
                    x1[index] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                       leftX, centerX, rightX, topX, bottomX, centerRhs); 
            
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
void sharedToGlobal(float * x0Block, const int subdomainLength, const int overlap, const int xLength,  const int nxGrids) {
    
    // Initialize shared memory and pointers to x0, x1 arrays 
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    
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
        
        if (blockIdx.x == 0 || blockIdx.x == (gridDim.x-1)) {
            innerShift = xval + yval * (subdomainLength + overlap);
        }

        x0Block[xShiftToRestrict + yShiftToRestrict * nxGrids + xval + yval * nxGrids] = x0[restrictShift + innerShift];

    }

}

__global__
void _2D_Algorithm(float * x0Gpu, const float * rhsGpu, const float * matrixElementsGpu, const int nxGrids, const int nyGrids, const int subdomainLength, const int overlap, const int step, const int num_JacobiIters)
{

    if ((blockIdx.x + blockIdx.y) % 2 == ((step + 1) % 2)) {
     
        // Instantiate shared memory here
        extern __shared__ float sharedMemory[];

        // Define the boundaries of the subdomain
        int xLeft = blockIdx.x * subdomainLength;
        int xRight = (blockIdx.x + 1) * subdomainLength - 1;
        int yLower = blockIdx.y * subdomainLength;
        int yUpper = (blockIdx.y + 1) * subdomainLength - 1;

        // Adjust the boundaries of the subdomain so that there is overlap
        if (blockIdx.x != 0) {
            xLeft = xLeft - overlap;
        } 
        if (blockIdx.x != gridDim.x - 1) {
            xRight = xRight + overlap;
        }
        if (blockIdx.y != 0) {
            yLower = yLower - overlap;
        }
        if (blockIdx.y != gridDim.y - 1) {
            yUpper = yUpper + overlap;
        }
    
        // Obtain the length in x and y of the subdomains
        const int xLength = xRight - xLeft + 1;
        const int yLength = yUpper - yLower + 1;

        // Point to the correct section of rhs
        const int blockShift = xLeft + yLower * nxGrids;
        float * x0Block = x0Gpu + blockShift;
        const float * rhsBlock = rhsGpu + blockShift;

        // Move grid point values to the shared memory of each block    
        globalToShared(x0Block, xLength, yLength, nxGrids);
    
        // Update the inner grid points of block shared memory (alternate between red blocks and black blocks, depending on step k)
        redBlackBlockUpdate(rhsBlock, matrixElementsGpu, xLength, yLength, num_JacobiIters, step);
    
        // Move updated values from shared memory to to appropriate restricted subdomain of global memory
        sharedToGlobal(x0Block, subdomainLength, overlap, xLength, nxGrids);

    }

}

float * iterativeGpuSwept(const float * initX, const float * rhs,
        const float * matrixElements,
	const int nxGrids, const int nyGrids, const int cycles, const int num_JacobiIters, const int threadsPerBlock, const int subdomainLength, const int overlap)
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
    // const int maxLength = 2 * subdomainLength;
    const int maxLength = subdomainLength + 2 * overlap;
    const int sharedBytes = 2 * maxLength * maxLength * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < 2*cycles; step++) {
        _2D_Algorithm <<<grid, block, sharedBytes>>> (x0Gpu, rhsGpu, matrixElementsGpu, nxGrids, nyGrids, subdomainLength, overlap, step, num_JacobiIters);
    }

    float * solution = new float[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nDofs,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(rhsGpu);
    cudaFree(matrixElementsGpu);

    return solution;
}
