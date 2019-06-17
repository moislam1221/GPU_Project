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

__global__
void _iterativeGpuClassicIteration(float * x1, const float * x0, const float * rhs,
                         const float leftMatrix, const float centerMatrix,
                         const float rightMatrix, const float topMatrix, const float bottomMatrix,
			 const int nxGrids, const int nyGrids, const int iteration)
{
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x; // Col
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int dof = iyGrid * (nxGrids) + ixGrid;
    int nDofs = nxGrids * nyGrids;
    
    float leftX, rightX, centerX, topX, bottomX;    

    if (dof < nDofs) {
        if (((dof % nxGrids) != 0) && (((dof+1) % nxGrids) != 0) && (dof > nxGrids) && (dof < nDofs - nxGrids)) {

            leftX = x0[dof - 1];
            centerX = x0[dof];
            rightX = x0[dof + 1];
            bottomX = x0[dof - nxGrids];
            topX = x0[dof + nxGrids];                   

            x1[dof] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                             leftX, centerX, rightX, topX, bottomX, rhs[dof]);
        }
        
    }

}

float * iterativeGpuClassic(const float * initX, const float * rhs,
                         const float * matrixElements,
			 const int nxGrids, const int nyGrids, const int nIters, const int threadsPerBlock)
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
    cudaMemcpy(x1Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
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
		nxGrids, nyGrids, iIter); 
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

