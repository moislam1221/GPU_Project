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
    int nxGrids = 128;
    int nyGrids = 128;
    int subdomainLength = 64;
    int threadsPerBlock = 64;
   
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
    int dx = 1.0/ (nxGrids + 1);
    int dy = 1.0/ (nyGrids + 1);
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
//          initX[iGrid] = 1;
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
    
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING FIRST UPDATE" << "\n" << std::endl;
    
    // APPLY METHOD TO ADVANCE POINTS (NO SHIFT)
    _iterativeGpuOriginal <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, x0Gpu, notUsed,
		   						 notUsed, notUsed, notUsed, notUsed, notUsed, nxGrids, nyGrids, notUsedInt, subdomainLength);

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
    _iterativeGpuHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, notUsed,
 		   						 notUsed, notUsed, notUsed, notUsed, notUsed, nxGrids, nyGrids, notUsedInt, subdomainLength);
   
/*    // COPY TO CPU 
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
    _iterativeGpuVerticalandHorizontalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, notUsed,
		   						 notUsed, notUsed, notUsed, notUsed, notUsed, nxGrids, nyGrids, notUsedInt, subdomainLength);
   
/*    // COPY TO CPU 
    cudaMemcpy(xLeftCpu, xLeftGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xRightCpu, xRightGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xTopCpu, xTopGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy(xBottomCpu, xBottomGpu, sizeof(float) * numBridgeElemTotal, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < numBridgeElemTotal; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid << " xLeft " << xLeftCpu[iGrid] << " xRight " << xRightCpu[iGrid] << " xTop " << xTopCpu[iGrid] << " xBottom " << xBottomCpu[iGrid] << std::endl;
    }
*/
    // PRINT TO SCREEN
    std::cout << "\n" << "NOW APPLYING PURELY VERTICAL SHIFT METHOD" << "\n" << std::endl;

    // APPLY VERTICAL SHIFT
    _iterativeGpuVerticalShift <<<grid, block, sharedMemory>>> (xLeftGpu, xRightGpu, xTopGpu, xBottomGpu, x0Gpu, notUsed,
		   						 notUsed, notUsed, notUsed, notUsed, notUsed, nxGrids, nyGrids, notUsedInt, subdomainLength);
   
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
    
    // COPY TO CPU 
    cudaMemcpy(x0Cpu, x0Gpu, sizeof(float) * nDofs, cudaMemcpyDeviceToHost);
    
    // PRINT RESULTS
    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " x0Cpu " << x0Cpu[iGrid]  << std::endl;
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


