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

// HEADER FILES
#include "helper.h"
#include "iterative-methods.h"
#include "iterative-1D-cpu.h"
#include "iterative-1D-gpu.h"
#include "iterative-1D-rectangular.h"
#include "iterative-1D-rectangular-multiple.h"

int main(int argc, char *argv[])
{
    // INPUTS
    const int nGrids = 21; //atoi(argv[1]); 
    const int threadsPerBlock = 8; //atoi(argv[2]); 
    const int cycles = 1;
    const int nIterations = 2; //atoi(argv[3]);
    const int nCpuIterations = 1000; //atoi(argv[3]);
    const int nGpuIterations = 1000; //atoi(argv[3]);
    method_type method = JACOBI;

    // INITIALIZE ARRAYS
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    float * leftMatrix = new float[nGrids];
    float * centerMatrix = new float[nGrids];
    float * rightMatrix = new float[nGrids];
    float dx = 1.0f / (nGrids + 1);
    
    // 1D POISSON MATRIX
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            initX[iGrid] = 0.0f;
        }
        else {
            initX[iGrid] = 1.0f; 
        }
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
    }

    // CPU
    clock_t cpuStartTime = clock();
    float * solutionCpu = iterativeCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nCpuIterations, method);
    clock_t cpuEndTime = clock();
    double cpuTime = (cpuEndTime - cpuStartTime) / (float) CLOCKS_PER_SEC;
    cpuTime = cpuTime * (1e3); // Convert to ms

    // GPU
    cudaEvent_t startGpu, stopGpu;
    float gpuTime;
    cudaEventCreate( &startGpu );
    cudaEventCreate( &stopGpu );
    cudaEventRecord(startGpu, 0);
    float * solutionGpu = iterativeGpuClassic(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nGpuIterations, threadsPerBlock, method);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);

    // RECTANGULAR METHOD
    cudaEvent_t startGpuRectangular, stopGpuRectangular;
    float gpuRectangularTime;
    cudaEventCreate( &startGpuRectangular );
    cudaEventCreate( &stopGpuRectangular );
    cudaEventRecord( startGpuRectangular, 0);
    float * solutionGpuRectangular = iterativeGpuRectangular(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, cycles, nIterations, method);
    cudaEventRecord(stopGpuRectangular, 0);
    cudaEventSynchronize(stopGpuRectangular);
    cudaEventElapsedTime(&gpuRectangularTime, startGpuRectangular, stopGpuRectangular);
    
    // RECTANGULAR MULTIPLE METHOD
    cudaEvent_t startGpuRectangularMultiple, stopGpuRectangularMultiple;
    float gpuRectangularMultipleTime;
    cudaEventCreate( &startGpuRectangularMultiple );
    cudaEventCreate( &stopGpuRectangularMultiple );
    cudaEventRecord( startGpuRectangularMultiple, 0);
    float * solutionGpuRectangularMultiple = iterativeGpuRectangularMultiple(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, cycles, nIterations, method, 2);
    cudaEventRecord(stopGpuRectangularMultiple, 0);
    cudaEventSynchronize(stopGpuRectangularMultiple);
    cudaEventElapsedTime(&gpuRectangularMultipleTime, startGpuRectangularMultiple, stopGpuRectangularMultiple);

    // PRINT SOLUTION
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f %f %f\n", i, solutionCpu[i], solutionGpu[i], solutionGpuRectangular[i], solutionGpuRectangularMultiple[i]);
    }
    
    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIterations);
    printf("\n");
    
    // Print out time for cpu, classic gpu, and swept gpu approaches
    printf("Time needed for the CPU: %f ms\n", cpuTime);
    printf("Time needed for the GPU: %f ms\n", gpuTime);
    printf("Time needed for the GPU Rectangular method: %f ms\n", gpuRectangularTime);
    printf("Time needed for the GPU Rectangular Multiple method: %f ms\n", gpuRectangularMultipleTime);

/*    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = gpuTime / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration);
*/

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualCpu = Residual(solutionCpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualGpu = Residual(solutionGpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualGpuRectangular = Residual(solutionGpuRectangular, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualGpuRectangularMultiple = Residual(solutionGpuRectangularMultiple, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the CPU solution is %f\n", residualCpu);
    printf("Residual of the GPU solution is %f\n", residualGpu);
    printf("Residual of the Rectangular solution is %f\n", residualGpuRectangular);
    printf("Residual of the Rectangular Multiple solution is %f\n", residualGpuRectangularMultiple);
    
    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionGpuRectangular;

    return 0;
}