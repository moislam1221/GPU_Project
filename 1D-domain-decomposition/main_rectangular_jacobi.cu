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

#define PI 3.14159265358979323

int main(int argc, char *argv[])
{
    // INPUTS
    const int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    const int nInnerUpdates = atoi(argv[3]);
    const int TOL = atoi(argv[4]);
    int nJacobiCpuIterations;
    int nJacobiGpuIterations;
    int nJacobiCycles;

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

    // OBTAIN NUMBER OF ITERATIONS NECESSARY TO ACHIEVE TOLERANCE FOR EACH METHOD
    nJacobiCpuIterations = jacobiCpuIterationCount(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, TOL);
    nJacobiGpuIterations = jacobiGpuClassicIterationCount(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, TOL, threadsPerBlock);
    nJacobiCycles = jacobiGpuRectangularIterationCount(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, TOL, nInnerUpdates);
    
    // CPU - JACOBI
    clock_t cpuJacobiStartTime = clock();
    float * solutionJacobiCpu = jacobiCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nJacobiCpuIterations);
    clock_t cpuJacobiEndTime = clock();
    float cpuJacobiTime = (cpuJacobiEndTime - cpuJacobiStartTime) / (float) CLOCKS_PER_SEC;
    cpuJacobiTime = cpuJacobiTime * (1e3); // Convert to ms

    // GPU - JACOBI
    cudaEvent_t startJacobiGpu, stopJacobiGpu;
    float gpuJacobiTime;
    cudaEventCreate( &startJacobiGpu );
    cudaEventCreate( &stopJacobiGpu );
    cudaEventRecord(startJacobiGpu, 0);
    float * solutionJacobiGpu = jacobiGpuClassic(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nJacobiGpuIterations, threadsPerBlock);
    cudaEventRecord(stopJacobiGpu, 0);
    cudaEventSynchronize(stopJacobiGpu);
    cudaEventElapsedTime(&gpuJacobiTime, startJacobiGpu, stopJacobiGpu);

    // RECTANGULAR METHOD - JACOBI
    cudaEvent_t startJacobiGpuRectangular, stopJacobiGpuRectangular;
    float gpuJacobiRectangularTime;
    cudaEventCreate( &startJacobiGpuRectangular );
    cudaEventCreate( &stopJacobiGpuRectangular );
    cudaEventRecord( startJacobiGpuRectangular, 0);
    float * solutionJacobiGpuRectangular = jacobiGpuRectangular(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids,  threadsPerBlock, nJacobiCycles, nInnerUpdates);
    cudaEventRecord(stopJacobiGpuRectangular, 0);
    cudaEventSynchronize(stopJacobiGpuRectangular);
    cudaEventElapsedTime(&gpuJacobiRectangularTime, startJacobiGpuRectangular, stopJacobiGpuRectangular);

    // COMPUTE TIME FACTORS   
    float cpuToGpu = cpuJacobiTime / gpuJacobiTime;
    float gpuToRectangular = gpuJacobiTime / gpuJacobiRectangularTime;
    float cpuToRectangular = cpuJacobiTime / gpuJacobiRectangularTime;
/*
    // PRINT SOLUTION
    for (int i = 0; i < nGrids; i++) {
        printf("Grid %d = %f %f %f %f %f %f\n", i, solutionJacobiCpu[i], solutionJacobiGpu[i], solutionJacobiGpuRectangular[i]);
    }
*/
    
    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Number of Cycles of Jacobi Rectangular performed: %d\n", nJacobiCycles);
    printf("CPU -> GPU Speedup Factor is %f\n", cpuToGpu);
    printf("GPU -> GPU Rectangular Speedup Factor is %f\n", gpuToRectangular);
    printf("CPU -> GPU Rectangular Speedup Factor is %f\n", cpuToRectangular);
    printf("======================================================\n");
    
    // Print out number of iterations needed for each method
    printf("Number of Iterations needed for Jacobi CPU: %d \n", nJacobiCpuIterations);
    printf("Number of Iterations needed for Jacobi GPU: %d \n", nJacobiGpuIterations);
    printf("Number of Cycles needed for Jacobi GPU Rectangular: %d (with %d inner updates) \n", nJacobiCycles, nInnerUpdates);
    printf("======================================================\n");
    
    // Print out time for cpu, classic gpu, and swept gpu approaches
    printf("Time needed for the Jacobi CPU: %f ms\n", cpuJacobiTime);
    printf("Time needed for the Jacobi GPU: %f ms\n", gpuJacobiTime);
    printf("Time needed for the Jacobi GPU Rectangular method: %f ms\n", gpuJacobiRectangularTime);
    printf("======================================================\n");

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualJacobiCpu = Residual(solutionJacobiCpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualJacobiGpu = Residual(solutionJacobiGpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualJacobiGpuRectangular = Residual(solutionJacobiGpuRectangular, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the Jacobi CPU solution is %f\n", residualJacobiCpu);
    printf("Residual of the Jacobi GPU solution is %f\n", residualJacobiGpu);
    printf("Residual of the Jacobi Rectangular solution is %f\n", residualJacobiGpuRectangular);

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
    results.open("results-jac.txt", std::ios_base::app);
    results << nGrids << " " << threadsPerBlock << " " << TOL << " " << cpuJacobiTime << " " << gpuJacobiTime << " " << gpuJacobiRectangularTime << " " << "\n";
    results.close();

    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionJacobiCpu;
    delete[] solutionJacobiGpu;
    delete[] solutionJacobiGpuRectangular;
    
    return 0;
}
