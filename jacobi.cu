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

__device__ __host__
float jacobiGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
                 const float leftX, float centerX, const float rightX, const float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
         / centerMatrix;
}

__device__ __host__
float RBGSGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
		      const float leftX, float centerX, const float rightX, const float centerRhs,
		      const int gridPoint)
{  
    
    // Update all points of a certain parity (i.e. update red, keep black the same)
    if (gridPoint % 2 == 1)
    {
    	return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
	 / centerMatrix;
    }
    else
    {
	return centerX;
    }
}

__device__ __host__
float SORGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
		      const float leftX, float centerX, const float rightX, const float centerRhs,
		      const int gridPoint, float relaxation)
{  
    // Similar to red-black gauss-seidel, but take weighted average of rbgs 
    // value and current centerX value based on relaxation parameter
    if (gridPoint % 2 == 1)
    {
    	return relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
    }
    else
    {
	return centerX;
    }
}

float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float leftX, float centerX, float rightX,  float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

float Residual(const float * solution, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    float residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        float leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f; 
        float centerX = solution[iGrid];
        float rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        float residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
	// printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

/*float * readExactSolution(int nGrids)
{
    float exactSolution[nGrids];
    std::ifstream input("exactSolution.txt");
    for (int i = 0; i < nGrids; i++)
    {
        input >> exactSolution[i];
        // printf("Data is %f\n", exactSolution[i]);
    }
    return exactSolution;
}*/

float solutionError(float * solution, float * exactSolution, int nGrids)
{
    float error = 0.0;
    float difference; 
    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
         difference = solution[iGrid] - exactSolution[iGrid];
	 error = error + difference*difference;
    }
    error = sqrt(error);
    return error;
}


float * jacobiCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters, int method, float relaxation)
{
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
            float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
            float centerX = x0[iGrid];
            float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
	    if (method == 0) {
                x1[iGrid] = jacobiGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                       rightMatrix[iGrid], leftX, centerX, rightX,
                                       rhs[iGrid]);
	    }
	    else if (method == 1) {
                x1[iGrid] = RBGSGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                   rightMatrix[iGrid], leftX, centerX, rightX,
                                   rhs[iGrid], iGrid);
		leftX = (iGrid > 0) ? x1[iGrid - 1] : 0.0f;
                centerX = x1[iGrid];
                rightX = (iGrid < nGrids - 1) ? x1[iGrid + 1] : 0.0f;
	        x1[iGrid] = RBGSGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                   rightMatrix[iGrid], leftX, centerX, rightX,
                                   rhs[iGrid], iGrid+1);
	    }	
	    else {	
                x1[iGrid] = SORGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                   rightMatrix[iGrid], leftX, centerX, rightX,
                                   rhs[iGrid], iGrid, relaxation);
		leftX = (iGrid > 0) ? x1[iGrid - 1] : 0.0f;
                centerX = x1[iGrid];
                rightX = (iGrid < nGrids - 1) ? x1[iGrid + 1] : 0.0f;
	        x1[iGrid] = SORGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                   rightMatrix[iGrid], leftX, centerX, rightX,
                                   rhs[iGrid], iGrid+1, relaxation);
	    }
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}


__global__
void _jacobiGpuClassicIteration(float * x1,
                         const float * x0, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int iteration, int method, float relaxation)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid < nGrids) {
        float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
        float centerX = x0[iGrid];
        float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
	if (method == 0) {
             x1[iGrid] = jacobiGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid]);
	}
	else if (method == 1) {
	    x1[iGrid] = RBGSGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], iGrid);
            __syncthreads();
	    leftX = (iGrid > 0) ? x1[iGrid - 1] : 0.0f;
	    centerX = x1[iGrid];
            rightX = (iGrid < nGrids - 1) ? x1[iGrid + 1] : 0.0f;
            x1[iGrid] = RBGSGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], iGrid+1);
	}
	else {
	    x1[iGrid] = SORGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], iGrid, relaxation);
	    leftX = (iGrid > 0) ? x1[iGrid - 1] : 0.0f;
	    centerX = x1[iGrid];
            rightX = (iGrid < nGrids - 1) ? x1[iGrid + 1] : 0.0f;
            x1[iGrid] = SORGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid], iGrid+1, relaxation);
        }    
    }
    __syncthreads();
}

float * jacobiGpuClassic(const float * initX, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int nIters,
                         const int threadsPerBlock, int method, float relaxation)
{
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(float) * nGrids);
    float * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, iIter, method, relaxation); 
        float * tmp = x1Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}


__device__ 
void __jacobiBlockUpperTriangleFromShared(
		float * xLeftBlock, float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method, float relaxation)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    printf("ENTERING UPPER TRIANGLE, x0 value at thread %d is %f\n", iGrid, x0[threadIdx.x]);
    printf("ENTERING UPPER TRIANGLE, x1 value at thread %d is %f\n", iGrid, x1[threadIdx.x]);

    #pragma unroll
    for (int k = 1; k < blockDim.x/2; ++k) {
        if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
            float leftX = x0[threadIdx.x - 1];
            float centerX = x0[threadIdx.x];
            float rightX = x0[threadIdx.x + 1];
	    if (iGrid == 0) {
		leftX = 0.0f;
	    }
	    if (iGrid == nGrids-1) {
		rightX = 0.0f;
	    }
            float leftMat = leftMatrixBlock[threadIdx.x];
            float centerMat = centerMatrixBlock[threadIdx.x];
            float rightMat = rightMatrixBlock[threadIdx.x];
            float rhs = rhsBlock[threadIdx.x];
	    // Select which kernel to use
            if (method == 0) {
	        x1[threadIdx.x] = jacobiGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs);
		printf("after iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
	    }
	    else if (method == 1) {
	        x1[threadIdx.x] = RBGSGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid); 
	        printf("upper half iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
	        __syncthreads();	
                leftX = x1[threadIdx.x - 1];
                centerX = x1[threadIdx.x];
                rightX = x1[threadIdx.x + 1];
		if (iGrid == 0) {
	 	    leftX = 0.0f;
	        }
	        if (iGrid == nGrids-1) {
		    rightX = 0.0f;
	        }
		printf("iGrid: %d, leftX: %f, centerX: %f, rightX: %f\n", iGrid, leftX, centerX, rightX); 
	        x1[threadIdx.x] = RBGSGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid+1); 
	        printf("upper full iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
	    }

	    else {
		x1[threadIdx.x] = SORGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid, relaxation); 
	        x1[threadIdx.x] = SORGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid+1, relaxation); 
            }
        }
        __syncthreads();	
	float * tmp = x1; x1 = x0; x0 = tmp;
    } 
    
    float * tmp = x1; x1 = x0; x0 = tmp;

    printf("END OF UPPER TRIANGLE, x0 value at iGrid %d is %f\n", iGrid, x0[iGrid]);
    printf("END OF UPPER TRIANGLE, x1 value at iGrid %d is %f\n", iGrid, x1[iGrid]);

    int remainder = threadIdx.x % 4;
    xLeftBlock[threadIdx.x] = x0[(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];
    xRightBlock[threadIdx.x] = x0[blockDim.x-1-(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];

}

__global__
void _jacobiGpuUpperTriangle(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * leftMatrixGpu, const float *centerMatrixGpu,
                             const float * rightMatrixGpu, int nGrids, int method, float relaxation)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    sharedMemory[threadIdx.x + blockDim.x] = x0Block[threadIdx.x];

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
    		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);
}

__device__ 
void __jacobiBlockLowerTriangleFromShared(
		const float * xLeftBlock, const float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method, float relaxation)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    int remainder = threadIdx.x % 4;

    if (threadIdx.x != blockDim.x-1) {
        x0[blockDim.x-1-((blockDim.x+threadIdx.x+1)/2) + blockDim.x*(remainder>1)] = xLeftBlock[threadIdx.x];
	x0[(blockDim.x+threadIdx.x+1)/2 + blockDim.x*(remainder>1)] = xRightBlock[threadIdx.x];
    } 

    printf("ENTERING LOWER TRIANGLE, x0 value at thread %d is %f\n", iGrid, x0[threadIdx.x]);
    printf("ENTERING LOWER TRIANGLE, x1 value at thread %d is %f\n", iGrid, x1[threadIdx.x]);

    //printf("Thread %d: In x0 I have: %f\n", threadIdx.x, x0[iGrid]);
    //printf("Thread %d: In x0 I have: %f\n", threadIdx.x+blockDim.x, x0[iGrid+blockDim.x]);
    # pragma unroll
    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                float leftX = x0[threadIdx.x - 1];
                float centerX = x0[threadIdx.x];
                float rightX = x0[threadIdx.x + 1];
		if (iGrid == 0) {
		    leftX = 0.0f;
		}
		if (iGrid == nGrids-1) {
		    rightX = 0.0f;
		}
		float leftMat = leftMatrixBlock[threadIdx.x];
		float centerMat = centerMatrixBlock[threadIdx.x];
 		float rightMat = rightMatrixBlock[threadIdx.x];
		float rhs = rhsBlock[threadIdx.x];
	    
	    
	    // Select which kernel to use
            if (method == 0) {
	        x1[threadIdx.x] = jacobiGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs);
		printf("after iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
	    }
	    else if (method == 1) {
	        x1[threadIdx.x] = RBGSGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid); 
		printf("lower half iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
		__syncthreads();
		leftX = x1[threadIdx.x - 1];
                centerX = x1[threadIdx.x];
                rightX = x1[threadIdx.x + 1];
		if (iGrid == 0) {
		    leftX = 0.0f;
		}
		if (iGrid == nGrids-1) {
		    rightX = 0.0f;
		}
	        x1[threadIdx.x] = RBGSGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid+1); 
		printf("lower full iteration %d, iGrid %d, x1 value %f\n", k-1, iGrid, x1[threadIdx.x]);
            }
	    else {
		x1[threadIdx.x] = SORGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid, relaxation); 
	        x1[threadIdx.x] = SORGrid(leftMat, centerMat, rightMat,leftX, centerX, rightX, rhs, iGrid+1, relaxation); 
            }
	    }
 	    float * tmp = x1; x1 = x0; x0 = tmp;
        }
	__syncthreads();
    }

    printf("ALMOST END OF LOWER TRIANGLE, x0 value at thread %d is %f\n", iGrid, x0[threadIdx.x]);
    printf("ALMOST END OF LOWER TRIANGLE, x1 value at thread %d is %f\n", iGrid, x1[threadIdx.x]);

    float leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x0[threadIdx.x - 1];
    float centerX = x0[threadIdx.x];
    float rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x0[threadIdx.x + 1];
    //float leftX = x0[threadIdx.x - 1];
    //float centerX = x0[threadIdx.x];
    //float rightX = x0[threadIdx.x + 1];
    if (iGrid == 0) {
       leftX = 0.0;    
    }
    if (iGrid == nGrids-1) {
        rightX = 0.0;
    }
    if (method == 0) {
        x1[threadIdx.x] = jacobiGrid(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x]);
    }
    else if (method == 1) {
	x1[threadIdx.x] = RBGSGrid(leftMatrixBlock[threadIdx.x], centerMatrixBlock[threadIdx.x], rightMatrixBlock[threadIdx.x], leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid); 
	printf("lower red iteration end, iGrid %d, x1 value %f\n",  iGrid, x1[threadIdx.x]);
	__syncthreads();
        //leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x1[threadIdx.x - 1];
        centerX = x1[threadIdx.x];
        rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x1[threadIdx.x + 1];
	leftX = (threadIdx.x == 0) ? x1[blockDim.x - 1] : x1[threadIdx.x - 1]; // - HERE IS THE NEW leftX for GS - check it!!!
        //centerX = x1[threadIdx.x];
        //rightX = (threadIdx.x == blockDim.x-1) ? x1[0] : x1[threadIdx.x + 1];
	if (iGrid == 0) {
	    leftX = 0.0f;
	}
	if (iGrid == nGrids-1) {
	    rightX = 0.0f;
	}
        printf("end: iGrid %d, thread id %d, leftX %f, rightX %f\n", iGrid, threadIdx.x, leftX, rightX);
	x1[threadIdx.x] = RBGSGrid(leftMatrixBlock[threadIdx.x], centerMatrixBlock[threadIdx.x], rightMatrixBlock[threadIdx.x], leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid+1); 
	printf("lower iteration end, iGrid %d, x1 value %f\n", iGrid, x1[threadIdx.x]);
        }
    else {
	x1[threadIdx.x] = SORGrid(leftMatrixBlock[threadIdx.x], centerMatrixBlock[threadIdx.x], rightMatrixBlock[threadIdx.x], leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid, relaxation); 
	x1[threadIdx.x] = SORGrid(leftMatrixBlock[threadIdx.x], centerMatrixBlock[threadIdx.x], rightMatrixBlock[threadIdx.x], leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid+1, relaxation); 
    }
       float * tmp = x1; x1 = x0; x0 = tmp; 

    printf("END OF LOWER TRIANGLE, x0 value at thread %d is %f\n", iGrid, x0[threadIdx.x]);
    printf("END OF LOWER TRIANGLE, x1 value at thread %d is %f\n", iGrid, x1[threadIdx.x]);
}

__global__
void _jacobiGpuLowerTriangle(float * x0Gpu, float *xLeftGpu,
                             float * xRightGpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu, int nGrids, int method, float relaxation)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    float * x0Block = x0Gpu + blockShift;
    float * rhsBlock = rhsGpu + blockShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _jacobiGpuShiftedDiamond(float * xLeftGpu, float * xRightGpu,
                              float * rhsGpu, 
			      float * leftMatrixGpu, float * centerMatrixGpu,
                              float * rightMatrixGpu, int nGrids, int method, float relaxation)
{

    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xRightGpu + blockShift;
    float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);  

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);

}

__global__
void _jacobiGpuDiamond(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu, int nGrids, int method, float relaxation)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;

    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];

    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);
    
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method, relaxation);
}
float * jacobiGpuSwept(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids, int nIters, const int threadsPerBlock, const int method, const float relaxation) { 
    
    // Determine number of threads and blocks 
    const int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);

    // Allocate memory for solution and inputs
    float *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(float) * threadsPerBlock * nBlocks);
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rhsGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&leftMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&centerMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rightMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    cudaMemcpy(x0Gpu + nGrids, initX, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu + nGrids, leftMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu + nGrids, centerMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu + nGrids, rightMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;

    _jacobiGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(float) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, method, relaxation);
    _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method, relaxation);

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _jacobiGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method, relaxation); 
    _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method, relaxation); 
    }

    _jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method, relaxation);

    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}

int main(int argc, char *argv[])
{
    // Ask user for inputs
    const int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    const int nIters = atoi(argv[3]);
    const int method = atoi(argv[4]); // 0 for Jacobi, 1 for GS, 2 for SOR
    float relaxation = 1.0; // if SOR is used, this corresponds to G-S

    // Check that the methods provided are valid. If not, throw error.
    if (argc < 4) 
    {    
        std::cout << "Please specify at least 4 inputs: nGrid ThreadsPerBlock nIters Method\n" << std::endl;
	std::cout << "Method should be an int corresponding to one of the methods: Jacobi(0), GS(1), SOR(2)\n" << std::endl;
    }

    // Check that if SOR is selected, that a relaxation parameter is specified
    if (method == 2 && (argc == 4)) {
        // std::cout << "Since you selected 2 (SOR), you need a fifth argument specifying the relaxation parameter\n" << std:endl;
    }
    else if (method == 2) {	
        relaxation = atoi(argv[5]);
    }
    
    // Declare arrays and population with values for Poisson equation
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    float * leftMatrix = new float[nGrids];
    float * centerMatrix = new float[nGrids];
    float * rightMatrix = new float[nGrids];
    float dx = 1.0f / (nGrids + 1);
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        initX[iGrid] = 1.0f; 
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
    }

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Run the CPU Implementation and measure the time required
    clock_t cpuStartTime = clock();
    float * solutionCpu = jacobiCpu(initX, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, method, relaxation);
    clock_t cpuEndTime = clock();
    double cpuTime = (cpuEndTime - cpuStartTime) / (double) CLOCKS_PER_SEC;

    // Run the Classic GPU Implementation and measure the time required
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    float * solutionGpuClassic = jacobiGpuClassic(initX, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method, relaxation);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    // Run the Swept GPU Implementation and measure the time required
    cudaEvent_t startSwept, stopSwept;
    float timeSwept;
    cudaEventCreate( &startSwept );
    cudaEventCreate( &stopSwept );
    cudaEventRecord( startSwept, 0);
    float * solutionGpuSwept = jacobiGpuSwept(initX, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method, relaxation);
    cudaEventRecord(stopSwept, 0);
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);

    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIters);
    printf("\n");

    // Print out results to the screen, notify if any GPU Classic or Swept values differ significantly
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%d %f %f \n",iGrid, //solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
	//assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-2) {
	    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	}
    } 

    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = timeClassic / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration);

    // Compute the residual of the resulting solution (|b-Ax|)
    float residual = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the converged solution is %f\n", residual);
  
    /*float exactSolution[nGrids];
    std::ifstream input("exactSolution.txt");
    ifor (int i = 0; i < nGrids; i++)
    {
        input >> exactSolution[i];
        // printf("Data is %f\n", exactSolution[i]);
    }
    
    //exactSolution = readExactSolution(nGrids);

    float error = solutionError(solutionGpuSwept, exactSolution, nGrids);

    printf("The error is %f\n", error); */

    // Save Results to a file "N tpb Iterations CPUTime/perstep ClassicTime/perstep SweptTime/perStep ClassicTime/SweptTime"
    std::ofstream timings;
    timings.open("time.txt",std::ios_base::app);
    timings << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << cpuTimePerIteration << "\t" << classicTimePerIteration << "\t" << sweptTimePerIteration << "\t" << timeMultiplier << "\n";
    timings.close();

    // Free memory
    cudaEventDestroy(startClassic);
    cudaEventDestroy(startSwept);
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
}
