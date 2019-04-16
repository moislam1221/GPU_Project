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

int main(int argc, char *argv[])
{
    // INPUTS
    int nxGrids = 8;
    int nyGrids = 8;
    int subdomainLength = 4;
    int threadsPerBlock = 2;
    int iterations = atoi(argv[1]);
   
    // SETTING GRID, BLOCK, THREAD INFORMATION 
    int nxBlocks = nxGrids / subdomainLength;
    int nyBlocks = nyGrids / subdomainLength;
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    int sharedMemory = subdomainLength * subdomainLength * 2 * sizeof(double);
    
    // PANEL ARRAY SIZES
    int elemPerBlock = subdomainLength * subdomainLength;
    int numBridgeElemPerBlock = elemPerBlock / 2;
    int numBridgeElemTotal = nxBlocks * nyBlocks * numBridgeElemPerBlock;

    // OTHER PARAMETERS
    double dx = 1.0/ (nxGrids + 1);
    double dy = 1.0/ (nyGrids + 1);
    int nDofs = nxGrids * nyGrids;
    
    // INT AND POINTER FOR UNUSED PARAMETERS    
    int notUsedInt;
    double * notUsed;

    // INITIALIZATION
    double * initX = new double[nDofs];
    double * x0Cpu = new double[nDofs];
    double * xLeftCpu = new double[numBridgeElemTotal];
    double * xRightCpu = new double[numBridgeElemTotal];
    double * xTopCpu = new double[numBridgeElemTotal];
    double * xBottomCpu = new double[numBridgeElemTotal];

    // INITIAL SOLUTION
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        initX[iGrid] = 1.0;
    }

    // INITALIZE MATRIX ELEMENTS
    double * rhsCpu = new double[nDofs];
    double * leftMatrixCpu = new double[nDofs];
    double * centerMatrixCpu = new double[nDofs];
    double * rightMatrixCpu = new double[nDofs];
    double * bottomMatrixCpu = new double[nDofs];
    double * topMatrixCpu = new double[nDofs];

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
    double * x0Gpu;
    cudaMalloc(&x0Gpu, sizeof(double) * (nDofs));
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    
    // ALLOCATE PANEL ARRAY MEMORY
    double *xLeftGpu, *xRightGpu, *xTopGpu, *xBottomGpu;
    cudaMalloc(&xLeftGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xRightGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xTopGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xBottomGpu, sizeof(double) * numBridgeElemTotal);
   
    // ALLOCATE MATRIX MEMORY
    double * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu, * bottomMatrixGpu, * topMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&leftMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&rightMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&centerMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&bottomMatrixGpu, sizeof(double) * nDofs);
    cudaMalloc(&topMatrixGpu, sizeof(double) * nDofs);
    cudaMemcpy(rhsGpu, rhsCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrixCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrixCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrixCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(bottomMatrixGpu, bottomMatrixCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(topMatrixGpu, topMatrixCpu, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // PRINT TO SCREEN
//    std::cout << "\n" << "NOW APPLYING FIRST UPDATE" << "\n" << std::endl;
    
    for (int i = 0; i < iterations; i++) {  

    // APPLY METHOD TO ADVANCE POINTS (NO SHIFT)
    _iterativeGpuOriginal <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu,
		   						 leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);

/*    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xLeft " << xLeftCpu[iGrid] << " xRight " << xRightCpu[iGrid] << std::endl;
    }

    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING HORIZONTAL UPDATE" << "\n" << std::endl;
*/
    // APPLY HORIZONTAL SHIFT
    _iterativeGpuHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);
/*
    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xTop " << xTopCpu[iGrid] << " xBottom " << xBottomCpu[iGrid] << std::endl;
    }


    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING VERTICAL AND HORIZONTAL SHIFT METHOD" << "\n" << std::endl;
*/
    // APPLY VERTICAL SHIFT
    _iterativeGpuVerticalandHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);
/*   
    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xLeft " << xLeftCpu[iGrid] << " xRight " << xRightCpu[iGrid] << std::endl;
    }

    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING PURELY VERTICAL SHIFT METHOD" << "\n" << std::endl;
*/
    // APPLY VERTICAL SHIFT
    _iterativeGpuVerticalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, topMatrixGpu, bottomMatrixGpu, nxGrids, nyGrids, notUsedInt, subdomainLength);
/*   
    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " xTop " << xTopCpu[iGrid] << " xBottom " << xBottomCpu[iGrid] << std::endl;
    }
    
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING FINAL METHOD" << "\n" << std::endl;
*/
    // APPLY FINAL STEP
    _finalSolution <<<grid, block, sharedMemory>>>(xTopGpu, xBottomGpu, x0Gpu, nxGrids, subdomainLength);
    
    }

    // COPY TO CPU 
    cudaMemcpy(x0Cpu, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " x0Cpu " << x0Cpu[iGrid]  << std::endl;
    }

    // COMPUTE THE RESIDUAL
    double residual = Residual(x0Cpu, rhsCpu, leftMatrixCpu, centerMatrixCpu, rightMatrixCpu, topMatrixCpu, bottomMatrixCpu, nxGrids, nyGrids); 

    // PRINT TO SCREEN
    std::cout << "\n" << "THE RESIDUAL OF MY SOLUTION IS " << residual <<  "\n" << std::endl;


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


