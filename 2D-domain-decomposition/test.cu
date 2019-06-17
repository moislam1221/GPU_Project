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
    int nxGrids = 4;
    int nyGrids = 4;
    int subdomainLength = 2; 
    int threadsPerBlock = 2;

    int cycles = 1;
    int num_JacobiIters = 2; 
    int nIters = 2;

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

    solutionCPU = iterativeCpu(initX, rhs, matrixElements, nxGrids, nyGrids, nIters);  
    solutionGPU = iterativeGpuClassic(initX, rhs, matrixElements, nxGrids, nyGrids, nIters, threadsPerBlock);  
    solutionDDGPU = iterativeGpuSwept(initX, rhs, matrixElements, nxGrids, nyGrids, cycles, num_JacobiIters, threadsPerBlock, subdomainLength);  
 
    // PRINT RESULTS
/*    for (int iGrid = 0; iGrid < nDofs; iGrid++) 
    {
        std::cout << "Grid Point " << iGrid <<  " Before " << initX[iGrid] << " After " << x0Cpu[iGrid] << std::endl;
    }
*/
    printf("CPU Solution:\n");
    print2DSolution(solutionCPU, nxGrids, nyGrids);    
    printf("GPU Solution:\n");
    print2DSolution(solutionGPU, nxGrids, nyGrids);    
    printf("Domain Decomposition GPU Solution:\n");
    print2DSolution(solutionDDGPU, nxGrids, nyGrids);    

    // COMPUTE RESIDUAL
    float residualCPU = Residual(solutionCPU, rhs, matrixElements, nxGrids, nyGrids);
    float residualGPU = Residual(solutionGPU, rhs, matrixElements, nxGrids, nyGrids);
    float residualDDGPU = Residual(solutionDDGPU, rhs, matrixElements, nxGrids, nyGrids);

    // PRINT RESIDUAL
    printf("The residual from CPU is %f\n", residualCPU);
    printf("The residual from GPU is %f\n", residualGPU);
    printf("The residual from Domain Decomposition GPU is  %f\n", residualDDGPU);

    // CLEAN UP
    delete[] initX;
    delete[] rhs;
    delete[] solutionCPU;
    delete[] solutionGPU;
    delete[] solutionDDGPU;

    return 0;
}



