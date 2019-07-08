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

float * iterativeCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters, int method)
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
	    if (iIter % 2 == 0) {
                x1[iGrid] = iterativeOperation(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
	    }
	    else { 
                x1[iGrid] = iterativeOperation2(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
            }
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}

