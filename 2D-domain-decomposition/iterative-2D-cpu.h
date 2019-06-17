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
                     const float * matrixElements, const int nxGrids, const int nyGrids, const int nIters)
{
    int nDofs = nxGrids * nyGrids;
    float * x0 = new float[nDofs];
    float * x1 = new float[nDofs];
    
    memcpy(x0, initX, sizeof(float) * nDofs);
    memcpy(x1, initX, sizeof(float)* nDofs);
    
    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];
   
    float leftX, rightX, centerX, topX, bottomX;
    int dof;
    
    for (int iIter = 0; iIter < nIters; ++ iIter) {
       
        for (int iGrid = 0; iGrid < nyGrids; ++iGrid) {
            for (int jGrid = 0; jGrid < nxGrids; ++jGrid) {
       
                if ((iGrid != 0) && (iGrid != nyGrids - 1) && (jGrid != 0) && (jGrid != nxGrids-1)) {
                    
                    dof = jGrid + iGrid * nxGrids;

                    leftX = x0[dof-1];
                    centerX = x0[dof];
                    rightX = x0[dof + 1];
	            bottomX = x0[dof - nxGrids];
                    topX = x0[dof + nxGrids];
                
                    x1[dof] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
		  		       leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
                } 
            }
        }

        float * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;

}

