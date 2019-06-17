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

#include "iterative-methods.h"
#include "iterative-2D-cpu.h"
#include "iterative-2D-gpu.h"
#include "iterative-2D-domain-decomposition-gpu.h"
#include "helper.h"

int main(int argc, char *argv[])
{
    // INPUTS
    int nxGrids = 128;
    int nyGrids = 128;
    int subdomainLength = 32; 
    int threadsPerBlock = 32;

    int cycles = 500;
    int num_JacobiIters = 100; 
    int nIters = 100;

    printf("Cycles: %d, Jacobi Iterations: %d\n", cycles, num_JacobiIters);

    float dx = 1.0/ (nxGrids - 1);
    float dy = 1.0/ (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    
    // INITIALIZATION
    float * initX = new float[nDofs];
    float * rhs = new float[nDofs];
    float * solutionCPU = new float[nDofs];
    float * solutionGPU = new float[nDofs];
    float * solutionDDGPU = new float[nDofs];

    // INITIAL SOLUTION
    int dof;
    for (int iGrid = 0; iGrid < nyGrids; iGrid++) { 
        for (int jGrid = 0; jGrid < nxGrids; jGrid++) {
            dof = jGrid + iGrid * nxGrids;
            if (iGrid == 0 || iGrid == nxGrids - 1 || jGrid == 0 || jGrid == nyGrids-1) {
                initX[dof] = 0.0f;
            }
            else {
                initX[dof] = 1.0f; 
            }
            rhs[dof] = 1.0f;
        }
    }

    float * matrixElements = new float[5];
    matrixElements[0] = -1.0f / (dy * dy);
    matrixElements[1] = -1.0f / (dx * dx);
    matrixElements[2] = 2.0f / (dx * dx) + 2.0f / (dy * dy);
    matrixElements[3] = -1.0f / (dx * dx);
    matrixElements[4] = -1.0f / (dy * dy);

    clock_t startCpuTime = clock();
    solutionCPU = iterativeCpu(initX, rhs, matrixElements, nxGrids, nyGrids, nIters);  
    clock_t endCpuTime = clock();
    double cpuTime = (endCpuTime - startCpuTime) / (double) CLOCKS_PER_SEC;
    cpuTime = cpuTime * (1e3);
 
    cudaEvent_t gpuStart, gpuStop;
    float gpuTime;
    cudaEventCreate( &gpuStart );
    cudaEventCreate( &gpuStop );
    cudaEventRecord(gpuStart, 0);
    solutionGPU = iterativeGpuClassic(initX, rhs, matrixElements, nxGrids, nyGrids, nIters, threadsPerBlock); 
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&gpuTime, gpuStart, gpuStop); 
    
    cudaEvent_t dd_gpuStart, dd_gpuStop;
    float dd_gpuTime;
    cudaEventCreate( &dd_gpuStart );
    cudaEventCreate( &dd_gpuStop );
    cudaEventRecord(dd_gpuStart, 0);
    solutionDDGPU = iterativeGpuSwept(initX, rhs, matrixElements, nxGrids, nyGrids, cycles, num_JacobiIters, threadsPerBlock, subdomainLength);  
    cudaEventRecord(dd_gpuStop, 0);
    cudaEventSynchronize(dd_gpuStop);
    cudaEventElapsedTime(&dd_gpuTime, dd_gpuStart, dd_gpuStop); 
    
    // PRINT RESULTS
/*    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " Before " << initX[iGrid] << " After " << x0Cpu[iGrid] << std::endl;
    }
*/
/*    printf("CPU Solution:\n");
    print2DSolution(solutionCPU, nxGrids, nyGrids);    
    printf("GPU Solution:\n");
    print2DSolution(solutionGPU, nxGrids, nyGrids);    
    printf("Domain Decomposition GPU Solution:\n");
    print2DSolution(solutionDDGPU, nxGrids, nyGrids);    
*/
    // COMPUTE RESIDUAL
    float residualCPU = Residual(solutionCPU, rhs, matrixElements, nxGrids, nyGrids);
    float residualGPU = Residual(solutionGPU, rhs, matrixElements, nxGrids, nyGrids);
    float residualDDGPU = Residual(solutionDDGPU, rhs, matrixElements, nxGrids, nyGrids);

    // PRINT RESIDUAL
    printf("The residual from CPU is %f\n", residualCPU);
    printf("The residual from GPU is %f\n", residualGPU);
    printf("The residual from Domain Decomposition GPU is  %f\n", residualDDGPU);
    
    // PRINT TIMINGS
    printf("The time required for the CPU to perform %d iterations is %f ms\n", nIters, cpuTime);
    printf("The time required for the GPU to perform %d iterations is %f ms\n", nIters, gpuTime);
    printf("The time required for the Domain Decomposition method on GPU (%d cycles, %d Jacobi iterations) is %f ms\n", cycles, num_JacobiIters, dd_gpuTime);

    // CLEAN UP
    delete[] initX;
    delete[] rhs;
    delete[] solutionCPU;
    delete[] solutionGPU;
    delete[] solutionDDGPU;

    return 0;
}



