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

#include "upperPyramidal.h"
#include "longitudinalBridge.h"

int main()
{
    // INPUTS
    int nxGrids = 4;
    int nyGrids = 4;
    int threadsPerBlock = 4;
   
    // SETTING GRID, BLOCK, THREAD INFORMATION 
    int nxBlocks = nxGrids / threadsPerBlock;
    int nyBlocks = nyGrids / threadsPerBlock;
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock, threadsPerBlock);
    int sharedMemory = threadsPerBlock * threadsPerBlock * 2 * sizeof(double);
    
    // PANEL ARRAY SIZES
    int numBridgeElemPerBlock = 2 * threadsPerBlock/2 * (threadsPerBlock/2 + 1);
    int numBridgeElemTotal = nxBlocks * nyBlocks * numBridgeElemPerBlock;

    // OTHER PARAMETERS
    int dx = 1.0/ (nxGrids + 1);
    int dy = 1.0/ (nyGrids + 1);
    int nDofs = nxGrids * nyGrids;
    
    // INT AND POINTER FOR UNUSED PARAMETERS    
    int notUsedInt;
    double * notUsed;

    // INITIALIZATION
    double * initX = new double[nDofs];
    double * xLeftCpu = new double[numBridgeElemTotal];
    double * xRightCpu = new double[numBridgeElemTotal];
    double * xTopCpu = new double[numBridgeElemTotal];
    double * xBottomCpu = new double[numBridgeElemTotal];

    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        initX[iGrid] = iGrid;
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
    
    // APPLY UPPER TRIANGULAR METHOD
    _iterativeGpuUpperPyramidal <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, notUsed,
		   						 notUsed, notUsed, notUsed, notUsed, notUsed, nxGrids, nyGrids, notUsedInt);
   
    // INITIALIZATION OF NEW BRIDGE ARRAYS
    double * xNorthCpu = new double[numBridgeElemTotal];
    double * xSouthCpu = new double[numBridgeElemTotal];
    double * xEastCpu = new double[numBridgeElemTotal];
    double * xWestCpu = new double[numBridgeElemTotal];

    // ALLOCATE PANEL ARRAY MEMORY
    double *xNorthGpu, *xSouthGpu, *xEastGpu, *xWestGpu;
    cudaMalloc(&xNorthGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xSouthGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xEastGpu, sizeof(double) * numBridgeElemTotal);
    cudaMalloc(&xWestGpu, sizeof(double) * numBridgeElemTotal);
    
    // APPLY LONGITUDINAL BRIDGE METHOD
    _iterativeGpuLongitudinalBridge <<<grid, block, sharedMemory>>> (xTopGpu, xBottomGpu, xEastGpu, xWestGpu, x0Gpu, 
		                                                     notUsed, notUsed, notUsed, notUsed, notUsed, notUsed,
				                                     nxGrids, nyGrids, notUsedInt);

    // COPY RESULTS TO CPU
    //cudaMemcpy(xNorthCpu, xNorthGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);    
    //cudaMemcpy(xSouthCpu, xSouthGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);    
    cudaMemcpy(xEastCpu, xEastGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);    
    cudaMemcpy(xWestCpu, xWestGpu, sizeof(double) * numBridgeElemTotal, cudaMemcpyDeviceToHost);    

    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xEast " << xEastCpu[iGrid] << " xWest " << xWestCpu[iGrid] << std::endl;
    }

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
