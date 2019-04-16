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

#include "original-strided.h"

int main()
{
    // INPUTS
    int nxGrids = 8;
    int nyGrids = 8;
    int subdomainLength = 4;
    int threadsPerBlock = 2;
   
    // SETTING GRID, BLOCK, THREAD INFORMATION 
    int nxBlocks = nxGrids / subdomainLength;
    int nyBlocks = nyGrids / subdomainLength;
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    int sharedMemory = subdomainLength * subdomainLength * 2 * sizeof(float);
    
    // PANEL ARRAY SIZES
    int elemPerBlock = subdomainLength * subdomainLength;
    int numBridgeElemPerBlock = elemPerBlock / 2;
    int numBridgeElemTotal = nxBlocks * nyBlocks * numBridgeElemPerBlock;

    // OTHER PARAMETERS
    float dx = 1.0/ (nxGrids + 1);
    float dy = 1.0/ (nyGrids + 1);
    int nDofs = nxGrids * nyGrids;
    
    // INT AND POINTER FOR UNUSED PARAMETERS    
    int notUsedInt;
    float * notUsed;

    // INITIALIZATION
    float * initX = new float[nDofs];
    float * x0Cpu = new float[nDofs];
    float * xLeftCpu = new float[numBridgeElemTotal];
    float * xRightCpu = new float[numBridgeElemTotal];
    float * xTopCpu = new float[numBridgeElemTotal];
    float * xBottomCpu = new float[numBridgeElemTotal];

    // INITIAL SOLUTION
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        initX[iGrid] = iGrid;
    }

    // INITALIZE MATRIX ELEMENTS
    float * rhsCpu = new float[nDofs];
    float * leftMatrixCpu = new float[nDofs];
    float * centerMatrixCpu = new float[nDofs];
    float * rightMatrixCpu = new float[nDofs];
    float * bottomMatrixCpu = new float[nDofs];
    float * topMatrixCpu = new float[nDofs];

    // FILL IN MATRIX ELEMENTS FOR 2D POISSON
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        leftMatrixCpu[iGrid] = -1.0 / (dx * dx);
        centerMatrixCpu[iGrid] = 2.0 / (dx * dx) + 2.0 / (dy * dy);
        rightMatrixCpu[iGrid] = -1.0 / (dx * dx);
        bottomMatrixCpu[iGrid] = -1.0 / (dy * dy);
        topMatrixCpu[iGrid] = -1.0 / (dy * dy);
        rhsCpu[iGrid] = 1.0;
    }
   
    // ALLOCATE SOLUTION MEMORY - CPU AND GPU
    float * x0Gpu;
    cudaMalloc(&x0Gpu, sizeof(float) * (nDofs));
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    
    // ALLOCATE PANEL ARRAY MEMORY
    float *xLeftGpu, *xRightGpu, *xTopGpu, *xBottomGpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * numBridgeElemTotal);
    cudaMalloc(&xRightGpu, sizeof(float) * numBridgeElemTotal);
    cudaMalloc(&xTopGpu, sizeof(float) * numBridgeElemTotal);
    cudaMalloc(&xBottomGpu, sizeof(float) * numBridgeElemTotal);
   
    // ALLOCATE MATRIX MEMORY
    float * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu, * bottomMatrixGpu, * topMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(float) * nDofs);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nDofs);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nDofs);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nDofs);
    cudaMalloc(&bottomMatrixGpu, sizeof(float) * nDofs);
    cudaMalloc(&topMatrixGpu, sizeof(float) * nDofs);
    cudaMemcpy(rhsGpu, rhsCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrixCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrixCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrixCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(bottomMatrixGpu, bottomMatrixCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(topMatrixGpu, topMatrixCpu, sizeof(float) * nDofs, cudaMemcpyHostToDevice);

    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING FIRST UPDATE" << "\n" << std::endl;
    
    // APPLY METHOD TO ADVANCE POINTS (NO SHIFT)
    _iterativeGpuOriginal <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu,
		   						 leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);

/*    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xLeft " << xLeftCpu[iGrid] << " xRight " << xRightCpu[iGrid] << std::endl;
    }
*/
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING HORIZONTAL UPDATE" << "\n" << std::endl;

    // APPLY HORIZONTAL SHIFT
    _iterativeGpuHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);

/*   
    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xTop " << xTopCpu[iGrid] << " xBottom " << xBottomCpu[iGrid] << std::endl;
    }
*/

    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING VERTICAL AND HORIZONTAL SHIFT METHOD" << "\n" << std::endl;

    // APPLY VERTICAL SHIFT
    _iterativeGpuVerticalandHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);
/*   
    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xLeft " << xLeftCpu[iGrid] << " xRight " << xRightCpu[iGrid] << std::endl;
    }
*/
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING PURELY VERTICAL SHIFT METHOD" << "\n" << std::endl;

    // APPLY VERTICAL SHIFT
    _iterativeGpuVerticalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);
   
/*    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " xTop " << xTopCpu[iGrid] << " xBottom " << xBottomCpu[iGrid] << std::endl;
    }
*/    
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING FINAL METHOD" << "\n" << std::endl;

    // APPLY FINAL STEP
    _finalSolution <<<grid, block, sharedMemory>>>(xTopGpu, xBottomGpu, x0Gpu, nxGrids, subdomainLength);
/*    
    // COPY TO CPU 
    cudaMemcpy(x0Cpu, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " x0Cpu " << x0Cpu[iGrid]  << std::endl;
    }
*/
/* 
    // COMPUTE THE RESIDUAL
    double residual = Residual(x0Cpu, rhsCpu, leftMatrixCpu, centerMatrixCpu, rightMatrixCpu, topMatrixCpu, bottomMatrixCpu, nxGrids, nyGrids); 

    // PRINT TO SCREEN
    std::cout << "\n" << "THE RESIDUAL OF MY SOLUTION IS " << residual <<  "\n" << std::endl;
*/

    // CLEAN UP
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(xTopGpu);
    cudaFree(xBottomGpu);
    delete[] initX;
    delete[] xLeftCpu;
    delete[] xRightCpu;
    delete[] xTopCpu;
    delete[] xBottomCpu;    
}


