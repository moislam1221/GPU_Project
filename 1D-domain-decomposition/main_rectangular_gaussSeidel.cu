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
#include "iterative-1D-rectangular-gaussSeidel.h"
#include "iterative-1D-rectangular-multiple.h"

#define PI 3.14159265358979323

int main(int argc, char *argv[])
{
    // INPUTS
    const int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    const int nInnerUpdates = atoi(argv[3]);
    // const int TOL = atoi(argv[4]);
    const int TOL = 10;
    int nGSCpuIterations;
    int nGSGpuIterations;
    int nGSCycles;

    // INITIALIZE ARRAYS
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    float * leftMatrix = new float[nGrids];
    float * centerMatrix = new float[nGrids];
    float * rightMatrix = new float[nGrids];
    float dx = 1.0f / (nGrids - 1);
    
    // 1D POISSON MATRIX
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            initX[iGrid] = 0.0f;
        }
        else {
            initX[iGrid] = (float)iGrid; 
        }
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
    }

    // OBTAIN NUMBER OF ITERATIONS NECESSARY TO ACHIEVE TOLERANCE FOR EACH METHOD
    nGSCpuIterations = gaussSeidelCpuIterationCount(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, TOL);
    nGSGpuIterations = gaussSeidelGpuClassicIterationCount(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, TOL, threadsPerBlock);
    nGSCycles = gaussSeidelGpuRectangularIterationCount(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, TOL, nInnerUpdates);

    // CPU - GAUSS-SEIDEL
    clock_t cpuGSStartTime = clock();
    float * solutionGSCpu = gaussSeidelCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nGSCpuIterations);
    clock_t cpuGSEndTime = clock();
    double cpuGSTime = (cpuGSEndTime - cpuGSStartTime) / (float) CLOCKS_PER_SEC;
    cpuGSTime = cpuGSTime * (1e3); // Convert to ms

    // GPU - GAUSS-SEIDEL
    cudaEvent_t startGSGpu, stopGSGpu;
    float gpuGSTime;
    cudaEventCreate( &startGSGpu );
    cudaEventCreate( &stopGSGpu );
    cudaEventRecord(startGSGpu, 0);
    float * solutionGSGpu = gaussSeidelGpuClassic(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nGSGpuIterations, threadsPerBlock);
    cudaEventRecord(stopGSGpu, 0);
    cudaEventSynchronize(stopGSGpu);
    cudaEventElapsedTime(&gpuGSTime, startGSGpu, stopGSGpu);

    // RECTANGULAR METHOD - GAUSS-SEIDEL
    cudaEvent_t startGSGpuRectangular, stopGSGpuRectangular;
    float gpuGSRectangularTime;
    cudaEventCreate( &startGSGpuRectangular );
    cudaEventCreate( &stopGSGpuRectangular );
    cudaEventRecord( startGSGpuRectangular, 0);
    float * solutionGSGpuRectangular = gaussSeidelGpuRectangular(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, nGSCycles, nInnerUpdates);
    cudaEventRecord(stopGSGpuRectangular, 0);
    cudaEventSynchronize(stopGSGpuRectangular);
    cudaEventElapsedTime(&gpuGSRectangularTime, startGSGpuRectangular, stopGSGpuRectangular);

    // COMPUTE TIME FACTORS   
    float cpuToGpu = cpuGSTime / gpuGSTime;
    float gpuToRectangular = gpuGSTime / gpuGSRectangularTime;
    float cpuToRectangular = cpuGSTime / gpuGSRectangularTime;

    // PRINT SOLUTION
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f %f\n", i, solutionGSCpu[i], solutionGSGpu[i], solutionGSGpuRectangular[i]);
    }
    
    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Number of Cycles of Gauss-Seidel Rectangular performed: %d\n", nGSCycles);
    printf("CPU -> GPU Speedup Factor is %f\n", cpuToGpu);
    printf("GPU -> GPU Rectangular Speedup Factor is %f\n", gpuToRectangular);
    printf("CPU -> GPU Rectangular Speedup Factor is %f\n", cpuToRectangular);
    printf("======================================================\n");
    
    // Print out number of iterations needed for each method
    printf("Number of Iterations needed for GS CPU: %d \n", nGSCpuIterations);
    printf("Number of Iterations needed for GS GPU: %d \n", nGSGpuIterations);
    printf("Number of Cycles needed for GS GPU Rectangular: %d (with %d inner updates) \n", nGSCycles, nInnerUpdates);
    printf("======================================================\n");
    
    // Print out time for cpu, classic gpu, and swept gpu approaches
    printf("Time needed for the GS CPU: %f ms\n", cpuGSTime);
    printf("Time needed for the GS GPU: %f ms\n", gpuGSTime);
    printf("Time needed for the GS GPU Rectangular method: %f ms\n", gpuGSRectangularTime);
    printf("======================================================\n");

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualGSCpu = Residual(solutionGSCpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualGSGpu = Residual(solutionGSGpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualGSGpuRectangular = Residual(solutionGSGpuRectangular, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the GS CPU solution is %f\n", residualGSCpu);
    printf("Residual of the GS GPU solution is %f\n", residualGSGpu);
    printf("Residual of the GS Rectangular solution is %f\n", residualGSGpuRectangular);

/*    for (int i = 0; i < nGrids; i++) {
        if (i == 0 || i == nGrids-1) {
            assert(solutionGpuRectangular[i] == 0.0);
        }
        else {
            assert(solutionGpuRectangular[i] == (float)(cycles * nIterations + 1.0));
        }
    }   
*/

/*    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = gpuTime / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration);
*/

    // Write results to file
    std::ofstream results;
    results.open("results-gs.txt", std::ios_base::app);
    results << nGrids << " " << threadsPerBlock << " " << TOL << " " << cpuGSTime << " " << gpuGSTime << " " << gpuGSRectangularTime << "\n";
    results.close();

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionGSCpu;
    delete[] solutionGSGpu;
    delete[] solutionGSGpuRectangular;
    
    return 0;
}
