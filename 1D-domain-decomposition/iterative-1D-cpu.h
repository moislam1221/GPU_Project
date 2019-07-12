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

#define PI 3.14159265358979323

float * jacobiCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters)
{
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    memcpy(x1, initX, sizeof(float) * nGrids);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float centerX = x0[iGrid];
            float rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid]);
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}

int jacobiCpuIterationCount(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, float TOL)
{
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    memcpy(x1, initX, sizeof(float) * nGrids);

    float residual = 100.0;
    int iIter = 0;
    while (residual > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float centerX = x0[iGrid];
            float rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid]);
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
        iIter++;
        residual = Residual(x0, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    }
    int nIters = iIter;
    delete[] x1;
    return nIters;
}

float * gaussSeidelCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters)
{
    float * x0 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    const float omega = 2.0 / (1.0 + sin(PI / (float)(nGrids - 1)));
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float centerX = x0[iGrid];
            float rightX = x0[iGrid + 1];
	    if (iIter % 2 == iGrid % 2) {
                x0[iGrid] = relaxedJacobi(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], omega);
	    }
        }
    }
    return x0;
}

int gaussSeidelCpuIterationCount(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, float TOL)
{
    float * x0 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    float omega = 2.0 / (1.0 + sin(PI / (float)(nGrids - 1)));
    float residual = 100.0;
    int iIter = 0;
    while (residual > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float centerX = x0[iGrid];
            float rightX = x0[iGrid + 1];
	    if ((iIter % 2) == (iGrid % 2)) {
                x0[iGrid] = relaxedJacobi(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], omega);
	    }
        }
        iIter++;
        residual = Residual(x0, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    }
    int nIters = iIter;
    return nIters;
}
