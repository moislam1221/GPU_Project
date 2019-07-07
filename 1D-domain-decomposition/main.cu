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
#include "iterative-1D-swept.h"

int main(int argc, char *argv[])
{
    // INPUTS
    const int nGrids = 128; //atoi(argv[1]); 
    const int threadsPerBlock = 32; //atoi(argv[2]); 
    const int nIters = 10000; //atoi(argv[3]);
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
        initX[iGrid] = 1.0f; 
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
    }

    // CPU
    clock_t cpuStartTime = clock();
    float * solutionCpu = iterativeCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nIters, method);
    clock_t cpuEndTime = clock();
    double cpuTime = (cpuEndTime - cpuStartTime) / (float) CLOCKS_PER_SEC;
    cpuTime = cpuTime * (1e3); // Convert to ms

    // GPU
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    float * solutionGpuClassic = iterativeGpuClassic(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    // SWEPT
    cudaEvent_t startSwept, stopSwept;
    float timeSwept;
    cudaEventCreate( &startSwept );
    cudaEventCreate( &stopSwept );
    cudaEventRecord( startSwept, 0);
    float * solutionGpuSwept = iterativeGpuSwept(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopSwept, 0);
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);

    // PRINTOUT
    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIters);
    printf("\n");

    // Print out results to the screen, notify if any GPU Classic or Swept values differ significantly
/*    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%d %f %f %f \n",iGrid, solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
	//assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-2) {
	    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	}
    }
*/
    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = timeClassic / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); 

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualCpu = Residual(solutionCpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualClassic = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualSwept = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the CPU solution is %f\n", residualCpu);
    printf("Residual of GPU Classic result is %f\n", residualClassic); 
    printf("Residual of the Swept solution is %f\n", residualSwept);
/*  
    // Save residual to a file
    std::ofstream residuals;
    residuals.open("residual-gs.txt",std::ios_base::app);
    residuals << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << residualSwept << "\n";
    residuals.close(); 

    // Save Results to a file "N tpb Iterations CPUTime/perstep ClassicTime/perstep SweptTime/perStep ClassicTime/SweptTime"
  //  std::ofstream timings;
  //  timings.open("time.txt",std::ios_base::app);
  //  timings << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << cpuTimePerIteration << "\t" << classicTimePerIteration << "\t" << sweptTimePerIteration << "\t" << timeMultiplier << "\n";
  //  timings.close();
*/
    // FREE MEMORY
    cudaEventDestroy(startClassic);
    cudaEventDestroy(startSwept);
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
    delete[] solutionGpuSwept;

    return 0;
}
